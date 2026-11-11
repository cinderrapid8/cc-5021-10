"""Headless entry point: render frames, run measurements and write reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .skeleton import HandSkeleton
from .flute import Flute
from .timeline import default_timeline
from .animation import PoseSampler
from .render import render_pose
from .measurement import measure_poses
from .export import export_joint_curves, export_contact_markers
from .ik import JACOBIAN_ANALYTIC, JACOBIAN_HYBRID


def render_frames(out: Path, sampler: PoseSampler, n_frames: int = 9):
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_frames):
        t = sampler.timeline.duration * k / max(n_frames - 1, 1)
        pose = sampler.pose_at(t)
        fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(12, 5))
        render_pose(ax_side, ax_top, sampler.sk, sampler.flute, pose)
        path = out / f"frame_{k:02d}_t{t:.2f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def run_report(
    out: Path,
    scales=(0.85, 1.0, 1.15),
    backend: str = "reference",
    jacobian_mode: str = JACOBIAN_ANALYTIC,
):
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# backend={backend} jacobian={jacobian_mode}"]
    for scale in scales:
        sk = HandSkeleton(scale=scale)
        flute = Flute()
        timeline = default_timeline()
        sampler = PoseSampler(
            sk, flute, timeline, backend=backend, jacobian_mode=jacobian_mode
        )
        times = np.linspace(0.0, timeline.duration, 61)
        m = measure_poses(f"scale={scale}", sampler, times)
        lines.append(m.as_row())

    # Adjacent-hole continuous switching on the default hand.
    sk = HandSkeleton()
    sampler = PoseSampler(
        sk, Flute(), default_timeline(),
        backend=backend, jacobian_mode=jacobian_mode,
    )
    switch_times = np.linspace(3.6, 5.6, 25)
    lines.append(measure_poses("adjacent_switch", sampler, switch_times).as_row())

    # Solve-budget summary: contact residuals and solve time on one shared run,
    # so any speed number stays attached to the work that produced it.
    times = np.linspace(0.0, sampler.timeline.duration, 61)
    poses = [sampler.pose_at(float(t)) for t in times]
    total_solve_s = float(sum(p.ik.budget.wall_time_s for p in poses))
    total_iters = sum(p.ik.iterations for p in poses)
    lines.append(
        f"budget: iters={total_iters} solve_time={total_solve_s:.3f}s "
        f"fk_full={sum(p.ik.budget.fk_full_evals for p in poses)} "
        f"fk_partial={sum(p.ik.budget.fk_partial_evals for p in poses)} "
        f"cache_hits={sum(p.ik.budget.fk_cache_hits for p in poses)} "
        f"node_updates={sum(p.ik.budget.node_updates() for p in poses)}"
    )

    report = out / "measurements.txt"
    report.write_text("\n".join(lines) + "\n")
    export_joint_curves(sampler, out / "joint_curves.csv")
    export_contact_markers(sampler, out / "contact_markers.csv")
    render_frames(out / "frames", sampler)
    print("\n".join(lines))
    print(f"\nreport + data + frames written under {out}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument(
        "--backend",
        choices=("reference", "fast"),
        default="reference",
        help="reference FD solver (default) or cached/analytic fast path",
    )
    parser.add_argument(
        "--jacobian",
        choices=(JACOBIAN_HYBRID, JACOBIAN_ANALYTIC),
        default=JACOBIAN_ANALYTIC,
        help="Jacobian assembly used by the fast backend",
    )
    args = parser.parse_args(argv)
    run_report(Path(args.out), backend=args.backend, jacobian_mode=args.jacobian)


if __name__ == "__main__":
    main()
