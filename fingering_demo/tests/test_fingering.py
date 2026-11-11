"""Tests for the fingering animation teaching tool."""

import numpy as np
import pytest

from fingering.skeleton import HandSkeleton, FINGER_NAMES
from fingering.flute import Flute
from fingering.timeline import default_timeline, Timeline, Clip
from fingering.animation import PoseSampler
from fingering.ik import (
    ContactTask,
    solve_ik,
    PRIORITY_CONTACT,
    PRIORITY_HOVER,
)
from fingering.measurement import measure_poses


@pytest.fixture
def rig():
    sk = HandSkeleton()
    flute = Flute()
    timeline = default_timeline()
    return sk, flute, timeline, PoseSampler(sk, flute, timeline)


# ------------------------------------------------------------- skeleton / FK
def test_bone_lengths_are_invariant_under_fk(rig):
    sk, *_ = rig
    rest = sk.bone_lengths()
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = sk.rest_q()
        q[6:] += rng.uniform(-0.3, 0.3, size=sk.n_q - 6)
        q = sk.clamp_q(q)
        fk = sk.fk(q)
        drift = np.max(np.abs(sk.actual_bone_lengths(fk) - rest))
        assert drift < 1e-12


def test_local_joint_axes_do_not_share_one_world_axis(rig):
    sk, *_ = rig
    # Use a rotated-wrist pose so local axes are not trivially the world axes.
    q = sk.initial_guess_q()
    q[3:6] = 0.2
    fk = sk.fk(sk.clamp_q(q))
    i_flex = sk.dof_names.index("index_mcp_flex")
    i_abd = sk.dof_names.index("index_mcp_abd")
    i_pip = sk.dof_names.index("index_pip_flex")
    flex_world = sk.joint_world_axis(fk, i_flex)
    abd_world = sk.joint_world_axis(fk, i_abd)
    pip_world = sk.joint_world_axis(fk, i_pip)
    # Flexion and abduction are perpendicular local axes -> distinct world axes.
    assert abs(np.dot(flex_world, abd_world)) < 1e-9
    # The local axis is expressed in world coordinates (not pinned to a fixed
    # world cardinal direction).
    assert not np.allclose(flex_world, np.array([0.0, 1.0, 0.0]))
    # Parallel flex joints on the same finger stay aligned by construction.
    assert np.allclose(flex_world, pip_world)


def test_positive_flexion_moves_pad_toward_tube(rig):
    sk, *_ = rig
    rest = sk.rest_q()
    q = rest.copy()
    i = sk.dof_names.index("index_mcp_flex")
    q[i] = np.deg2rad(50)
    z0 = sk.fk(rest).pads[0][2]
    z1 = sk.fk(sk.clamp_q(q)).pads[0][2]
    assert z1 < z0  # tube surface is below the rest pad


def test_joint_limits_are_local_and_enforced(rig):
    sk, *_ = rig
    q = sk.rest_q()
    q[6:] = sk.limits_hi[6:] + 10.0
    clamped = sk.clamp_q(q)
    assert np.all(clamped <= sk.limits_hi + 1e-12)
    assert sk.limit_overshoot(clamped) == 0.0


# ------------------------------------------------------------------------- IK
def test_known_static_fingering_reaches_home_holes(rig):
    sk, flute, _, _ = rig
    tasks = [
        ContactTask(f, flute.hole_position(f + 1), PRIORITY_CONTACT, "home")
        for f in range(4)
    ]
    result = solve_ik(sk, tasks)
    assert result.converged
    assert np.all(result.residuals_m <= result.contact_threshold)
    assert result.limit_overshoot == 0.0


def test_hover_tasks_are_lower_priority_and_also_tracked(rig):
    sk, flute, _, _ = rig
    tasks = [
        ContactTask(0, flute.hole_position(1), PRIORITY_CONTACT, "c0"),
        ContactTask(1, flute.hover_position(2), PRIORITY_HOVER, "h1"),
    ]
    result = solve_ik(sk, tasks)
    assert result.priorities[1] == PRIORITY_HOVER
    assert result.residuals_m[0] <= result.contact_threshold
    # Hover is softer: just require it moved most of the way, no fake contact.
    assert result.residuals_m[1] < 0.01


def test_unreachable_target_keeps_residual_without_stretching(rig):
    sk, flute, _, _ = rig
    far = np.array([0.0, -0.066, 0.40])
    result = solve_ik(sk, [ContactTask(0, far, PRIORITY_CONTACT, "far")])
    assert not result.converged
    assert result.unreachable() == [0]
    assert result.residuals_m[0] > 0.05
    assert result.limit_overshoot == 0.0
    fk = sk.fk(result.q)
    assert np.max(np.abs(sk.actual_bone_lengths(fk) - sk.bone_lengths())) < 1e-12


def test_solver_is_deterministic(rig):
    sk, flute, _, _ = rig
    tasks = [
        ContactTask(f, flute.hole_position(f + 1), PRIORITY_CONTACT, "c")
        for f in range(4)
    ]
    a = solve_ik(sk, tasks).q
    b = solve_ik(sk, tasks).q
    assert np.allclose(a, b)


# ------------------------------------------------------------------ timeline
def test_time_scrubbing_returns_a_determined_pose(rig):
    *_, sampler = rig
    p1 = sampler.pose_at(2.37).fk.pads
    p2 = sampler.pose_at(2.37).fk.pads
    assert np.allclose(p1, p2)
    # Jumping to the same time after visiting other times is identical.
    sampler.pose_at(0.1)
    sampler.pose_at(5.0)
    p3 = sampler.pose_at(2.37).fk.pads
    assert np.allclose(p1, p3)


def test_transition_blends_contacts_smoothly():
    tl = default_timeline()
    tl.set_transition(0.4)
    # At a boundary the closed weight must be strictly between 0 and 1.
    _, closed, _ = tl.sample(tl.boundaries[1])
    assert 0.0 < closed[0] < 1.0
    # Zero transition gives an immediate (hard) switch.
    tl.set_transition(0.0)
    _, closed0, _ = tl.sample(tl.boundaries[1] - 1e-6)
    _, closed1, _ = tl.sample(tl.boundaries[1] + 1e-6)
    assert closed0[0] != closed1[0]


def test_adjacent_hole_switch_continuous(rig):
    sk, flute, timeline, sampler = rig
    # Drive index finger from h1 to h0; residual must stay bounded throughout.
    tl = Timeline(
        [
            Clip("a", 1.0, (1, 2, 3, 4), (True, True, True, True)),
            Clip("b", 1.0, (0, 2, 3, 4), (True, True, True, True)),
        ],
        transition=0.5,
    )
    local = PoseSampler(sk, flute, tl)
    residuals = []
    for t in np.linspace(0.0, tl.duration, 41):
        residuals.append(local.pose_at(t).ik.residuals_m.max())
    assert np.max(residuals) <= 0.006


def test_contact_establishment_is_smooth(rig):
    sk, flute, _, _ = rig
    tl = Timeline(
        [
            Clip("open", 1.0, (1, 2, 3, 4), (False, False, False, False)),
            Clip("closed", 1.0, (1, 2, 3, 4), (True, False, False, False)),
        ],
        transition=0.4,
    )
    sampler = PoseSampler(sk, flute, tl)
    boundary = tl.boundaries[1]
    ts = np.linspace(boundary - 0.2, boundary + 0.2, 17)
    weights, pad_z = [], []
    for t in ts:
        pose = sampler.pose_at(t)
        weights.append(pose.closed_weight[0])
        pad_z.append(pose.fk.pads[0][2])
    weights = np.array(weights)
    # Monotonic smoothstep blend, strictly interior at the boundary midpoint.
    assert np.all(np.diff(weights) >= -1e-12)
    assert 0.0 < weights[len(weights) // 2] < 1.0
    # No positional jump larger than the sampling resolution.
    assert np.max(np.abs(np.diff(pad_z))) < 0.005


# ------------------------------------------------------------- hand proportions
@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
def test_hand_scale_variation_measurement(scale):
    sk = HandSkeleton(scale=scale)
    sampler = PoseSampler(sk, Flute(), default_timeline())
    m = measure_poses(f"scale={scale}", sampler, np.linspace(0.0, 2.0, 15))
    # Bone integrity must hold regardless of proportions.
    assert m.max_bone_length_drift_mm < 1e-9
    assert m.max_limit_overshoot_deg == 0.0


def test_measurement_reports_residuals(rig):
    *_, sampler = rig
    m = measure_poses("smoke", sampler, [0.05, 3.4])
    row = m.as_row()
    assert "smoke" in row and "mm" in row
    assert m.max_contact_residual_mm >= 0.0
    assert m.max_bone_length_drift_mm < 1e-9
    assert m.max_limit_overshoot_deg == 0.0
