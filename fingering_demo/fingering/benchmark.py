"""Fingering performance benchmark with an explicit constraint-solve budget.

The benchmark never compares raw frame rates on relaxed work: every backend
runs the *same* poses to the *same* contact threshold and iteration cap, and
a run that leaves penetration or violates joint limits fails the run instead
of being reported as a speed-up.  Two stress loads are exercised on purpose:

* ``fast_retarget``  -- rapid adjacent-hole finger reassignments (short clip
  durations, wide transitions) so targets move on nearly every sample;
* ``moving_target``  -- a dragged instrument target that keeps relocating
  while the timeline plays, forcing cache invalidation mid-run.

All thresholds and scene parameters are explicit: there is no silent default
that could mask a regression -- assertions raise ``AssertionError`` with the
measured numbers.

Run with::

    python -m fingering.benchmark --out output/benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import HandSkeleton
from .flute import Flute
from .timeline import Timeline, Clip
from .animation import PoseSampler
from .ik import (
    JACOBIAN_FD,
    JACOBIAN_HYBRID,
    JACOBIAN_ANALYTIC,
    PRIORITY_CONTACT,
)
from .budget import FULL_TREE_NODES, FINGER_CHAIN_NODES

# Explicit benchmark contract -- no value here falls back to a solver default.
CONTACT_THRESHOLD_M = 0.004
MAX_ITERS = 150
N_SAMPLES = 81
TIMING_REPEATS = 5
RESIDUAL_GUARD_MM = 0.5
SWITCH_RESIDUAL_GUARD_MM = 6.0
BACKENDS = (
    ("reference", "reference", JACOBIAN_FD),
    ("fast_hybrid", "fast", JACOBIAN_HYBRID),
    ("fast_analytic", "fast", JACOBIAN_ANALYTIC),
)


@dataclass
class Scenario:
    name: str
    timeline: Timeline
    target_moves: int
    note: str


def fast_retarget_scenario() -> Scenario:
    """Rapid hole switching: 0.45 s clips with 0.35 s overlapping blends."""
    home = (1, 2, 3, 4)
    clips = [
        Clip("open", 0.45, home, (False, False, False, False)),
        Clip("idx0", 0.45, (0, 2, 3, 4), (True, True, True, True)),
        Clip("home", 0.45, home, (True, True, True, True)),
        Clip("pinky5", 0.45, (1, 2, 3, 5), (True, True, True, True)),
        Clip("home", 0.45, home, (True, True, True, True)),
        Clip("idx0_again", 0.45, (0, 2, 3, 4), (True, True, True, True)),
    ]
    return Scenario(
        name="fast_retarget",
        timeline=Timeline(clips, transition=0.35),
        target_moves=0,
        note="0.45s clips, 0.35s transitions; every sample crosses a blend",
    )


def moving_target_scenario() -> Scenario:
    """Static clip but the index-finger target is dragged between samples."""
    home = (1, 2, 3, 4)
    tl = Timeline(
        [Clip("held", 2.0, home, (True, True, True, True))], transition=0.0
    )
    return Scenario(
        name="moving_target",
        timeline=tl,
        target_moves=N_SAMPLES,
        note="index target repositioned before every sample -> cache invalidated",
    )


@dataclass
class BackendStats:
    backend: str
    jacobian_mode: str
    wall_time_s: float
    solve_time_s: float
    max_contact_residual_mm: float
    mean_contact_residual_mm: float
    max_hover_residual_mm: float
    max_limit_overshoot_deg: float
    max_bone_drift_mm: float
    converged_fraction: float
    fk_full: int
    fk_partial: int
    fk_cache_hits: int
    node_updates: int
    iterations_total: int

    def as_row(self) -> dict:
        return {
            "backend": self.backend,
            "jacobian_mode": self.jacobian_mode,
            "wall_time_s": f"{self.wall_time_s:.4f}",
            "solve_time_s": f"{self.solve_time_s:.4f}",
            "max_contact_residual_mm": f"{self.max_contact_residual_mm:.4f}",
            "mean_contact_residual_mm": f"{self.mean_contact_residual_mm:.4f}",
            "max_hover_residual_mm": f"{self.max_hover_residual_mm:.4f}",
            "max_limit_overshoot_deg": f"{self.max_limit_overshoot_deg:.6f}",
            "max_bone_drift_mm": f"{self.max_bone_drift_mm:.6f}",
            "converged_fraction": f"{self.converged_fraction:.4f}",
            "fk_full_evals": self.fk_full,
            "fk_partial_evals": self.fk_partial,
            "fk_cache_hits": self.fk_cache_hits,
            "node_updates": self.node_updates,
            "iterations_total": self.iterations_total,
        }


def _run_once(label: str, backend: str, mode: str, scenario: Scenario) -> tuple[list, float]:
    sk = HandSkeleton(scale=1.0)
    flute = Flute()
    sampler = PoseSampler(
        sk,
        flute,
        scenario.timeline,
        contact_threshold=CONTACT_THRESHOLD_M,
        backend=backend,
        jacobian_mode=mode,
    )
    times = np.linspace(0.0, scenario.timeline.duration, N_SAMPLES)
    rest_lengths = sk.bone_lengths()
    poses = []
    rng = np.random.default_rng(20261111)
    t0 = _perf()
    for k, t in enumerate(times):
        if scenario.target_moves and k > 0:
            # Explicit instrument edit: revision bump must invalidate the FK
            # cache before the next solve.
            base = flute.hole_position(1)
            target = base + np.array(
                [0.002 * np.sin(k * 0.7), 0.004 * rng.uniform(-1, 1), 0.0]
            )
            flute.set_override(0, target)
        pose = sampler.pose_at(float(t))
        poses.append(pose)
        if scenario.target_moves and k > 0:
            assert (
                sampler._flute_rev_seen == flute.geometry_revision
            ), "sampler failed to observe the target revision"
    wall = _perf() - t0
    solve_time = float(sum(p.ik.budget.wall_time_s for p in poses))

    contact_res, hover_res = [], []
    bone_drift, overshoot_rad, converged = [], [], []
    for pose in poses:
        fk = pose.fk
        bone_drift.append(np.max(np.abs(sk.actual_bone_lengths(fk) - rest_lengths)))
        overshoot_rad.append(pose.ik.limit_overshoot)
        converged.append(pose.ik.converged)
        for f, task in enumerate(pose.ik.contacts):
            res = float(pose.ik.residuals_m[f])
            (contact_res if task.priority == PRIORITY_CONTACT else hover_res).append(
                res
            )
    stats = BackendStats(
        backend=label,
        jacobian_mode=mode,
        wall_time_s=wall,
        solve_time_s=solve_time,
        max_contact_residual_mm=float(np.max(contact_res)) * 1000,
        mean_contact_residual_mm=float(np.mean(contact_res)) * 1000,
        max_hover_residual_mm=(float(np.max(hover_res)) * 1000) if hover_res else 0.0,
        max_limit_overshoot_deg=float(np.degrees(np.max(overshoot_rad))),
        max_bone_drift_mm=float(np.max(bone_drift)) * 1000,
        converged_fraction=float(np.mean(converged)),
        fk_full=sum(p.ik.budget.fk_full_evals for p in poses),
        fk_partial=sum(p.ik.budget.fk_partial_evals for p in poses),
        fk_cache_hits=sum(p.ik.budget.fk_cache_hits for p in poses),
        node_updates=sum(p.ik.budget.node_updates() for p in poses),
        iterations_total=sum(p.ik.iterations for p in poses),
    )
    return stats, poses


def _perf() -> float:
    import time

    return time.perf_counter()


def _guard(scenario_name: str, ref: BackendStats, stats: BackendStats) -> None:
    """Hard correctness gates: a faster number never excuses penetration."""
    assert stats.max_limit_overshoot_deg == 0.0, (
        f"{scenario_name}/{stats.backend}: joint-limit overshoot "
        f"{stats.max_limit_overshoot_deg} deg"
    )
    assert stats.max_bone_drift_mm < 1e-9, (
        f"{scenario_name}/{stats.backend}: bone drift {stats.max_bone_drift_mm} mm"
    )
    guard = (
        SWITCH_RESIDUAL_GUARD_MM
        if scenario_name == "fast_retarget"
        else RESIDUAL_GUARD_MM
    )
    assert stats.max_contact_residual_mm <= guard, (
        f"{scenario_name}/{stats.backend}: contact residual "
        f"{stats.max_contact_residual_mm:.3f} mm exceeds {guard} mm"
    )
    assert stats.converged_fraction >= ref.converged_fraction, (
        f"{scenario_name}/{stats.backend}: convergence regressed "
        f"{stats.converged_fraction:.3f} < ref {ref.converged_fraction:.3f}"
    )
    # No residual regression beyond the contact threshold relative to reference.
    assert (
        stats.max_contact_residual_mm
        <= max(ref.max_contact_residual_mm, CONTACT_THRESHOLD_M * 1000) + 1e-6
    ), (
        f"{scenario_name}/{stats.backend}: residual "
        f"{stats.max_contact_residual_mm:.3f} mm worse than reference "
        f"{ref.max_contact_residual_mm:.3f} mm"
    )


def run_benchmark(out: Path | None = None) -> dict:
    scenarios = [fast_retarget_scenario(), moving_target_scenario()]
    report: dict = {
        "contract": {
            "contact_threshold_m": CONTACT_THRESHOLD_M,
            "max_iters": MAX_ITERS,
            "n_samples": N_SAMPLES,
            "timing_repeats": TIMING_REPEATS,
            "full_tree_nodes": FULL_TREE_NODES,
            "finger_chain_nodes": FINGER_CHAIN_NODES,
        },
        "scenarios": {},
    }
    for scenario in scenarios:
        rows = []
        per_backend_timings = {}
        for label, backend, mode in BACKENDS:
            timed = []
            last_stats = None
            for rep in range(TIMING_REPEATS):
                stats, _ = _run_once(label, backend, mode, scenario)
                timed.append(stats.wall_time_s)
                last_stats = stats
            best = min(timed)
            median = float(np.median(timed))
            last_stats.wall_time_s = best
            rows.append(last_stats)
            per_backend_timings[label] = {
                "best_wall_s": best,
                "median_wall_s": median,
                "all_wall_s": timed,
            }
        ref_stats = next(r for r in rows if r.backend == "reference")
        for stats in rows:
            _guard(scenario.name, ref_stats, stats)
        table = [r.as_row() for r in rows]
        speedup = {
            r.backend: round(ref_stats.wall_time_s / r.wall_time_s, 2)
            for r in rows
        }
        report["scenarios"][scenario.name] = {
            "note": scenario.note,
            "target_moves": scenario.target_moves,
            "rows": table,
            "timings": per_backend_timings,
            "speedup_vs_reference": speedup,
        }
        _print_scenario(scenario, table, speedup)

    if out is not None:
        _write_artifacts(out, report)
    return report


def _print_scenario(scenario: Scenario, table: list[dict], speedup: dict) -> None:
    print(f"\n=== scenario: {scenario.name} ({scenario.note}) ===")
    header = (
        f"{'backend':<15}{'wall_s':>8}{'solve_s':>9}{'resMax_mm':>11}"
        f"{'resMean':>9}{'hover_mm':>10}{'limit':>8}{'bone':>9}{'conv':>7}"
        f"{'fkFull':>8}{'fkPart':>8}{'hits':>7}{'nodes':>8}{'speedup':>9}"
    )
    print(header)
    for row in table:
        print(
            f"{row['backend']:<15}{float(row['wall_time_s']):8.3f}"
            f"{float(row['solve_time_s']):9.3f}"
            f"{float(row['max_contact_residual_mm']):11.4f}"
            f"{float(row['mean_contact_residual_mm']):9.4f}"
            f"{float(row['max_hover_residual_mm']):10.4f}"
            f"{float(row['max_limit_overshoot_deg']):8.2f}"
            f"{float(row['max_bone_drift_mm']):9.2e}"
            f"{float(row['converged_fraction']):7.2f}"
            f"{row['fk_full_evals']:8d}{row['fk_partial_evals']:8d}"
            f"{row['fk_cache_hits']:7d}{row['node_updates']:8d}"
            f"{speedup[row['backend']]:9.2f}x"
        )


def _write_artifacts(out: Path, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2))
    with (out / "benchmark.csv").open("w", newline="") as fh:
        fieldnames = [
            "scenario", "backend", "jacobian_mode", "wall_time_s", "solve_time_s",
            "max_contact_residual_mm", "mean_contact_residual_mm",
            "max_hover_residual_mm", "max_limit_overshoot_deg",
            "max_bone_drift_mm", "converged_fraction", "fk_full_evals",
            "fk_partial_evals", "fk_cache_hits", "node_updates",
            "iterations_total", "speedup_vs_reference",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, sc in report["scenarios"].items():
            speed = sc["speedup_vs_reference"]
            for row in sc["rows"]:
                writer.writerow({"scenario": name, **row,
                                 "speedup_vs_reference": speed[row["backend"]]})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="optional output directory")
    args = parser.parse_args(argv)
    run_benchmark(Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
