"""Tests for the cached/analytic solver path and the solve-budget accounting.

The reference finite-difference solver remains the run reference: these tests
prove the optimized paths reproduce its contact residuals within tolerance
while the budget counters explain exactly where the saved work comes from.
"""

import numpy as np
import pytest

from fingering.skeleton import HandSkeleton
from fingering.flute import Flute
from fingering.timeline import default_timeline
from fingering.animation import PoseSampler
from fingering.ik import (
    ContactTask,
    solve_ik,
    solve_ik_fast,
    JACOBIAN_HYBRID,
    JACOBIAN_ANALYTIC,
    JACOBIAN_FD,
    PRIORITY_CONTACT,
    PRIORITY_HOVER,
    _stack,
    _jacobian_fd,
    _jacobian_analytic,
)
from fingering.budget import SolveBudget, FULL_TREE_NODES, FINGER_CHAIN_NODES
from fingering import benchmark as bm


# ------------------------------------------------------------- FK cache core
def test_fk_cache_hits_on_identical_q_and_rebuilds_on_change():
    sk = HandSkeleton()
    q = sk.initial_guess_q()
    fk_a, hit_a = sk.fk_with_status(q)
    assert not hit_a
    fk_b, hit_b = sk.fk_with_status(q.copy())
    assert hit_b and fk_b is fk_a
    fk_c, hit_c = sk.fk_with_status(sk.clamp_q(q + 1e-4))
    assert not hit_c and fk_c is not fk_a


def test_geometry_revision_invalidates_fk_cache():
    sk = HandSkeleton()
    q = sk.initial_guess_q()
    sk.fk(q)
    _, hit = sk.fk_with_status(q)
    assert hit
    sk.invalidate_geometry_cache()
    _, hit_after = sk.fk_with_status(q)
    assert not hit_after


def test_sampler_explicit_invalidation_recovers_after_external_edit():
    # Simulates an embedding app moving the instrument without touching the
    # flute object: the explicit hook must still drop stale kinematics.
    sk = HandSkeleton()
    sampler = PoseSampler(sk, Flute(), default_timeline(), backend="fast",
                          jacobian_mode=JACOBIAN_ANALYTIC)
    q = sk.initial_guess_q()
    sk.fk(q)
    assert sk.fk_with_status(q)[1]
    sampler.invalidate_target_cache()
    assert not sk.fk_with_status(q)[1]
    # Poses remain deterministic and solvable after invalidation.
    a = sampler.pose_at(3.0).fk.pads
    b = sampler.pose_at(3.0).fk.pads
    assert np.allclose(a, b)


def test_partial_fk_matches_full_fk_and_reuses_other_chains():
    sk = HandSkeleton()
    rng = np.random.default_rng(3)
    q = sk.clamp_q(sk.initial_guess_q() + rng.normal(scale=0.1, size=sk.n_q))
    base = sk.fk(q)
    finger = 2
    dq = np.zeros(sk.n_q)
    dq[6 + finger * 4 + 1] = 0.08
    qp = sk.clamp_q(q + dq)

    partial = sk.fk_finger_partial(qp, finger, base, q)
    full = sk.fk(qp)
    assert np.allclose(partial.R, full.R, atol=1e-12)
    assert np.allclose(partial.p, full.p, atol=1e-12)
    assert np.allclose(partial.pads, full.pads, atol=1e-12)

    for f in range(4):
        if f == finger:
            continue
        for node in sk.chain_nodes[f]:
            assert np.array_equal(partial.p[node], base.p[node])
            assert np.array_equal(partial.R[node], base.R[node])


def test_partial_fk_rejects_wrist_changes_and_falls_back_to_full():
    sk = HandSkeleton()
    q = sk.initial_guess_q()
    base = sk.fk(q)
    qp = q.copy()
    qp[1] += 0.01  # wrist translation moves every chain
    out = sk.fk_finger_partial(qp, 0, base, q)
    assert np.allclose(out.pads, sk.fk(qp).pads, atol=1e-12)


# ------------------------------------------------------------- analytic math
def test_analytic_jacobian_matches_finite_differences():
    sk = HandSkeleton()
    flute = Flute()
    tasks = [
        ContactTask(0, flute.hole_position(1), PRIORITY_CONTACT, "c0"),
        ContactTask(1, flute.hover_position(2), PRIORITY_HOVER, "h1"),
        ContactTask(3, flute.hole_position(0), PRIORITY_CONTACT, "c3"),
    ]
    rng = np.random.default_rng(11)
    eye = np.eye(sk.n_q)
    for _ in range(5):
        q = sk.clamp_q(sk.initial_guess_q() + rng.normal(scale=0.12, size=sk.n_q))
        r0, _, W = _stack(sk, q, tasks)
        fd = _jacobian_fd(sk, q, tasks, r0, W, SolveBudget())
        an = _jacobian_analytic(sk, q, tasks, sk.fk(q), W, SolveBudget())
        free = [
            j
            for j in range(sk.n_q)
            if not np.array_equal(sk.clamp_q(q + eye[j] * 1e-7), q)
        ]
        err = np.abs(fd[:, free] - an[:, free])
        # Free (un-clamped) columns agree to FD truncation level everywhere.
        assert err.max() < 1e-5


# ------------------------------------------------------------- solver parity
def _home_tasks(flute: Flute):
    return [
        ContactTask(f, flute.hole_position(f + 1), PRIORITY_CONTACT, "home")
        for f in range(4)
    ]


@pytest.mark.parametrize("mode", [JACOBIAN_HYBRID, JACOBIAN_ANALYTIC])
def test_fast_solver_matches_reference_residuals_and_limits(mode):
    sk = HandSkeleton()
    flute = Flute()
    tasks = _home_tasks(flute)
    ref = solve_ik(sk, tasks)
    fast = solve_ik_fast(sk, tasks, jacobian_mode=mode)
    assert fast.converged == ref.converged
    assert np.allclose(fast.residuals_m, ref.residuals_m, atol=2e-5)
    assert fast.limit_overshoot == 0.0
    assert ref.limit_overshoot == 0.0
    fk = sk.fk(fast.q)
    assert np.max(np.abs(sk.actual_bone_lengths(fk) - sk.bone_lengths())) < 1e-12


@pytest.mark.parametrize("mode", [JACOBIAN_HYBRID, JACOBIAN_ANALYTIC])
def test_fast_solver_keeps_visible_residual_when_unreachable(mode):
    sk = HandSkeleton()
    far = np.array([0.0, -0.066, 0.40])
    ref = solve_ik(sk, [ContactTask(0, far, PRIORITY_CONTACT, "far")])
    fast = solve_ik_fast(
        sk, [ContactTask(0, far, PRIORITY_CONTACT, "far")], jacobian_mode=mode
    )
    assert not fast.converged
    assert fast.unreachable() == [0]
    assert fast.residuals_m[0] > 0.05
    assert fast.limit_overshoot == 0.0
    # Both paths stop at a joint limit; their terminal residuals need not be
    # bit-identical (IK is multi-modal), but they must agree to within ~1% and
    # neither may quietly reduce the visible gap (no hidden penetration).
    assert abs(fast.residuals_m[0] - ref.residuals_m[0]) / ref.residuals_m[0] < 0.01


def test_fast_solver_is_deterministic():
    sk = HandSkeleton()
    tasks = _home_tasks(Flute())
    a = solve_ik_fast(sk, tasks, jacobian_mode=JACOBIAN_ANALYTIC).q
    b = solve_ik_fast(sk, tasks, jacobian_mode=JACOBIAN_ANALYTIC).q
    assert np.allclose(a, b)


def test_invalid_fast_mode_is_rejected():
    with pytest.raises(ValueError):
        solve_ik_fast(HandSkeleton(), [], jacobian_mode=JACOBIAN_FD)


# ------------------------------------------------------------- budget counters
def test_reference_budget_counts_24_fk_calls_per_iteration():
    sk = HandSkeleton()
    result = solve_ik(sk, _home_tasks(Flute()))
    b = result.budget
    n_q = sk.n_q
    # Every LM iteration rebuilds the Jacobian (n_q FD perturbations) and does
    # one candidate FK, plus the initial and final residual stacks; a repeated
    # q is served from cache at zero node-update cost.
    jac_calls = b.iterations * n_q
    stack_calls = 2 + b.iterations
    assert b.fk_full_evals + b.fk_cache_hits == jac_calls + stack_calls
    assert b.fk_partial_evals == 0
    assert b.jacobian_builds == b.iterations
    assert b.lin_solves == b.iterations
    assert b.accepted_steps + b.rejected_steps == b.iterations
    assert b.wall_time_s > 0.0


def test_hybrid_budget_uses_partial_finger_fks_and_analytic_wrist():
    sk = HandSkeleton()
    result = solve_ik_fast(sk, _home_tasks(Flute()), jacobian_mode=JACOBIAN_HYBRID)
    b = result.budget
    it = result.iterations
    # 16 finger DOFs -> one scoped 4-node partial FK per build.
    assert b.fk_partial_evals >= 16 * it
    # Partial updates must account for four node frames each.
    assert b.node_updates() == (
        b.fk_full_evals * FULL_TREE_NODES
        + b.fk_partial_evals * FINGER_CHAIN_NODES
    )
    # The hybrid path still does strictly fewer node updates than reference.
    ref = solve_ik(HandSkeleton(), _home_tasks(Flute()))
    assert b.node_updates() < ref.budget.node_updates()


def test_analytic_budget_spends_zero_fk_inside_jacobian_builds():
    sk = HandSkeleton()
    result = solve_ik_fast(sk, _home_tasks(Flute()), jacobian_mode=JACOBIAN_ANALYTIC)
    b = result.budget
    assert b.fk_partial_evals == 0
    # Only residual/candidate stacks remain: <= 2 full FKs per iteration.
    assert b.fk_full_evals <= 2 * b.iterations + 2


# ------------------------------------------------------------- cache staleness
def test_dragged_target_invalidates_cache_between_poses():
    sk = HandSkeleton()
    flute = Flute()
    sampler = PoseSampler(
        sk, flute, default_timeline(), backend="fast",
        jacobian_mode=JACOBIAN_ANALYTIC,
    )
    sampler.pose_at(1.0)
    rev_before = flute.geometry_revision
    flute.set_override(0, flute.hole_position(1) + np.array([0.0, 0.01, 0.0]))
    assert flute.geometry_revision == rev_before + 1
    pose = sampler.pose_at(1.0)
    # The sampler observed the edit and no stale FK survived.
    assert sampler._flute_rev_seen == flute.geometry_revision
    assert pose.ik.contacts[0].target is not None
    # Clearing the override bumps the revision again and the pose still solves.
    flute.clear_override(0)
    pose2 = sampler.pose_at(1.0)
    assert sampler._flute_rev_seen == flute.geometry_revision
    assert pose2.ik.converged or pose2.ik.residuals_m.max() < 0.01


def test_target_move_actually_changes_the_solved_pad_position():
    sk = HandSkeleton()
    flute = Flute()
    sampler = PoseSampler(
        sk, flute, default_timeline(), backend="fast",
        jacobian_mode=JACOBIAN_ANALYTIC,
    )
    # t=4.0 sits in the pinky->hole5 clip where finger 0 retargets to h0.
    t = 4.0
    p0 = sampler.pose_at(t).fk.pads[0].copy()
    flute.set_override(0, flute.hole_position(0) + np.array([0.0, 0.015, 0.0]))
    moved = sampler.pose_at(t)
    p1 = moved.fk.pads[0]
    # A stale cache would leave the pad exactly where it was.
    assert np.linalg.norm(p1 - p0) > 1e-3
    # The pad follows the retargeted hole while staying inside joint limits.
    assert moved.ik.limit_overshoot == 0.0
    flute.clear_override(0)
    p2 = sampler.pose_at(t).fk.pads[0]
    assert np.allclose(p2, p0, atol=1e-6)


# ------------------------------------------------------------- benchmark gates
def test_benchmark_scenarios_run_full_budget_and_pass_gates(tmp_path):
    # Smaller repeat count keeps the test fast; thresholds stay at the
    # benchmark's explicit contract values.
    bm.TIMING_REPEATS = 1
    report = bm.run_benchmark(out=tmp_path)
    assert set(report["scenarios"]) == {"fast_retarget", "moving_target"}
    for name, sc in report["scenarios"].items():
        rows = {r["backend"]: r for r in sc["rows"]}
        ref = rows["reference"]
        for label in ("fast_hybrid", "fast_analytic"):
            row = rows[label]
            assert float(row["max_limit_overshoot_deg"]) == 0.0
            assert float(row["max_bone_drift_mm"]) < 1e-9
            assert float(row["max_contact_residual_mm"]) <= float(
                ref["max_contact_residual_mm"]
            ) + 1e-9 + bm.CONTACT_THRESHOLD_M * 1000
            assert float(row["converged_fraction"]) >= float(
                ref["converged_fraction"]
            )
            assert sc["speedup_vs_reference"][label] >= 1.5
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "benchmark.csv").exists()


def test_benchmark_guard_fails_on_penetration_regression():
    # The guard must refuse to celebrate speed when contact degrades: simulate
    # a regressed stats object with visible penetration past the threshold.
    good = bm.BackendStats(
        backend="reference", jacobian_mode=JACOBIAN_FD, wall_time_s=1.0,
        solve_time_s=1.0, max_contact_residual_mm=0.1,
        mean_contact_residual_mm=0.1, max_hover_residual_mm=0.0,
        max_limit_overshoot_deg=0.0, max_bone_drift_mm=0.0,
        converged_fraction=1.0, fk_full=1, fk_partial=0, fk_cache_hits=0,
        node_updates=FULL_TREE_NODES, iterations_total=1,
    )
    bad = bm.BackendStats(
        backend="fast_analytic", jacobian_mode=JACOBIAN_ANALYTIC, wall_time_s=0.1,
        solve_time_s=0.1, max_contact_residual_mm=2.0,
        mean_contact_residual_mm=2.0, max_hover_residual_mm=0.0,
        max_limit_overshoot_deg=0.0, max_bone_drift_mm=0.0,
        converged_fraction=1.0, fk_full=1, fk_partial=0, fk_cache_hits=0,
        node_updates=FULL_TREE_NODES, iterations_total=1,
    )
    with pytest.raises(AssertionError, match="residual"):
        bm._guard("moving_target", good, bad)
    bad.max_contact_residual_mm = 0.1
    bad.max_limit_overshoot_deg = 1.5
    with pytest.raises(AssertionError, match="overshoot"):
        bm._guard("moving_target", good, bad)


def test_fast_retarget_load_keeps_switch_residual_bounded():
    scenario = bm.fast_retarget_scenario()
    sk = HandSkeleton()
    sampler = PoseSampler(
        sk, Flute(), scenario.timeline, backend="fast",
        jacobian_mode=JACOBIAN_HYBRID,
        contact_threshold=bm.CONTACT_THRESHOLD_M,
    )
    times = np.linspace(0.0, scenario.timeline.duration, 41)
    poses = [sampler.pose_at(float(t)) for t in times]
    residuals = [p.ik.residuals_m.max() for p in poses]
    assert np.max(residuals) <= bm.SWITCH_RESIDUAL_GUARD_MM / 1000.0
    assert all(p.ik.limit_overshoot == 0.0 for p in poses)
