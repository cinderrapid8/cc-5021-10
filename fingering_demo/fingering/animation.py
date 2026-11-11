"""Turn timeline samples into deterministic skeleton poses via IK."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .skeleton import HandSkeleton, FINGER_NAMES
from .flute import Flute
from .timeline import Timeline
from .ik import (
    ContactTask,
    IKResult,
    PRIORITY_CONTACT,
    PRIORITY_HOVER,
    solve_ik,
)


@dataclass
class Pose:
    t: float
    fk: object
    ik: IKResult
    holes: np.ndarray
    closed_weight: np.ndarray
    clip_index: int


class PoseSampler:
    """Every pose is solved from the same rest start.

    Therefore ``pose_at(t)`` is a pure function of t and the current model:
    scrubbing the animation backwards or jumping in time always yields the
    identical pose instead of depending on playback history.
    """

    def __init__(
        self,
        sk: HandSkeleton,
        flute: Flute,
        timeline: Timeline,
        contact_threshold: float = 0.004,
    ):
        self.sk = sk
        self.flute = flute
        self.timeline = timeline
        self.contact_threshold = contact_threshold

    def tasks_at(self, t: float) -> tuple[list[ContactTask], np.ndarray, np.ndarray, int]:
        holes, closed, clip_idx = self.timeline.sample(t)
        tasks: list[ContactTask] = []
        for f in range(4):
            hole = int(round(holes[f]))
            # Fractional closed weight -> target slides from hover to contact.
            contact = self.flute.hole_position(hole)
            hover = self.flute.hover_position(hole)
            target = (1.0 - closed[f]) * hover + closed[f] * contact
            priority = PRIORITY_CONTACT if closed[f] >= 0.5 else PRIORITY_HOVER
            tasks.append(
                ContactTask(
                    finger=f,
                    target=target,
                    priority=priority,
                    label=f"{FINGER_NAMES[f]}->h{hole}",
                )
            )
        return tasks, holes, closed, clip_idx

    def pose_at(self, t: float) -> Pose:
        tasks, holes, closed, clip_idx = self.tasks_at(t)
        ik = solve_ik(self.sk, tasks, q0=None, contact_threshold=self.contact_threshold)
        return Pose(
            t=t,
            fk=self.sk.fk(ik.q),
            ik=ik,
            holes=holes,
            closed_weight=closed,
            clip_index=clip_idx,
        )
