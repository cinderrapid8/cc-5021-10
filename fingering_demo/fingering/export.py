"""Export genuine joint-angle curves and contact markers (CSV)."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .animation import PoseSampler
from .skeleton import FINGER_NAMES


def sample_curve(sampler: PoseSampler, n_samples: int = 121):
    """Evaluate the pure pose function on a uniform grid."""
    ts = np.linspace(0.0, sampler.timeline.duration, n_samples)
    poses = [sampler.pose_at(float(t)) for t in ts]
    return ts, poses


def export_joint_curves(sampler: PoseSampler, path: str | Path, n_samples: int = 121):
    """Write every joint variable actually driven by the solver (SI units)."""
    ts, poses = sample_curve(sampler, n_samples)
    sk = sampler.sk
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s"] + sk.dof_names + ["limit_overshoot_rad"])
        for t, pose in zip(ts, poses):
            row = [f"{t:.6f}"] + [f"{v:.8f}" for v in pose.ik.q]
            row.append(f"{pose.ik.limit_overshoot:.8f}")
            writer.writerow(row)
    return path


def export_contact_markers(sampler: PoseSampler, path: str | Path, n_samples: int = 121):
    """One row per finger per sample: target, pad, signed distance and state."""
    ts, poses = sample_curve(sampler, n_samples)
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "time_s",
                "finger",
                "hole",
                "closed_weight",
                "target_x",
                "target_y",
                "target_z",
                "pad_x",
                "pad_y",
                "pad_z",
                "distance_m",
                "contact_state",
                "priority",
            ]
        )
        for t, pose in zip(ts, poses):
            for f in range(4):
                task = pose.ik.contacts[f]
                pad = pose.fk.pads[f]
                dist = pose.ik.residuals_m[f]
                if task.priority == 1:
                    state = "contact" if dist <= pose.ik.contact_threshold else "contact_missed"
                else:
                    # A hover target sits above the hole; tracking it means the
                    # finger is lifted, never in contact with the hole.
                    state = "hover" if dist <= pose.ik.contact_threshold else "hover_missed"
                writer.writerow(
                    [
                        f"{t:.6f}",
                        FINGER_NAMES[f],
                        int(round(pose.holes[f])),
                        f"{pose.closed_weight[f]:.4f}",
                        *(f"{v:.6f}" for v in task.target),
                        *(f"{v:.6f}" for v in pad),
                        f"{dist:.6f}",
                        state,
                        int(task.priority),
                    ]
                )
    return path
