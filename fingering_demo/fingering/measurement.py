"""Quantitative checks: contact distances, joint-limit and bone integrity."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .animation import PoseSampler
from .skeleton import FINGER_NAMES
from .ik import PRIORITY_CONTACT


@dataclass
class Measurement:
    name: str
    max_contact_residual_mm: float
    mean_contact_residual_mm: float
    max_hover_residual_mm: float
    max_limit_overshoot_deg: float
    max_bone_length_drift_mm: float
    unreachable_fingers: list

    def as_row(self) -> str:
        un = ",".join(FINGER_NAMES[f] for f in self.unreachable_fingers) or "-"
        return (
            f"{self.name:<22} contact max {self.max_contact_residual_mm:6.2f} mm | "
            f"mean {self.mean_contact_residual_mm:5.2f} mm | "
            f"hover max {self.max_hover_residual_mm:6.2f} mm | "
            f"limit {self.max_limit_overshoot_deg:5.2f} deg | "
            f"bone drift {self.max_bone_length_drift_mm:5.3f} mm | "
            f"unreachable: {un}"
        )


def measure_poses(name: str, sampler: PoseSampler, times) -> Measurement:
    contact_res, hover_res, overshoots, bone_drifts, unreachable = [], [], [], [], set()
    rest_lengths = sampler.sk.bone_lengths()
    for t in times:
        pose = sampler.pose_at(float(t))
        actual = sampler.sk.actual_bone_lengths(pose.fk)
        bone_drifts.append(np.max(np.abs(actual - rest_lengths)))
        overshoots.append(pose.ik.limit_overshoot)
        for f, task in enumerate(pose.ik.contacts):
            res = pose.ik.residuals_m[f]
            if task.priority == PRIORITY_CONTACT:
                contact_res.append(res)
                if res > pose.ik.contact_threshold:
                    unreachable.add(f)
            else:
                hover_res.append(res)
    return Measurement(
        name=name,
        max_contact_residual_mm=float(np.max(contact_res)) * 1000 if contact_res else 0.0,
        mean_contact_residual_mm=float(np.mean(contact_res)) * 1000 if contact_res else 0.0,
        max_hover_residual_mm=float(np.max(hover_res)) * 1000 if hover_res else 0.0,
        max_limit_overshoot_deg=float(np.degrees(np.max(overshoots))),
        max_bone_length_drift_mm=float(np.max(bone_drifts)) * 1000,
        unreachable_fingers=sorted(unreachable),
    )
