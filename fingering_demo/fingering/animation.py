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
    JACOBIAN_ANALYTIC,
    solve_ik,
    solve_ik_fast,
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

    Two solver backends share the exact same LM numerics:

    * ``"reference"`` -> full finite-difference Jacobian (the run reference);
    * ``"fast"``      -> analytic or hybrid Jacobian over cached, scoped FK.

    Target/instrument edits are detected through geometry revision counters;
    the moment a finger target or the instrument changes, the skeleton FK
    cache is invalidated so no stale pad position can survive into a solve.
    """

    def __init__(
        self,
        sk: HandSkeleton,
        flute: Flute,
        timeline: Timeline,
        contact_threshold: float = 0.004,
        backend: str = "reference",
        jacobian_mode: str = JACOBIAN_ANALYTIC,
    ):
        if backend not in ("reference", "fast"):
            raise ValueError(f"unknown backend {backend!r}")
        self.sk = sk
        self.flute = flute
        self.timeline = timeline
        self.contact_threshold = contact_threshold
        self.backend = backend
        self.jacobian_mode = jacobian_mode
        self._flute_rev_seen = flute.geometry_revision

    def invalidate_target_cache(self) -> None:
        """Drop kinematic caches after an explicit target/instrument edit."""
        self.sk.invalidate_geometry_cache()
        self._flute_rev_seen = self.flute.geometry_revision

    def _sync_geometry(self) -> None:
        if self.flute.geometry_revision != self._flute_rev_seen:
            self.invalidate_target_cache()

    def tasks_at(self, t: float) -> tuple[list[ContactTask], np.ndarray, np.ndarray, int]:
        holes, closed, clip_idx = self.timeline.sample(t)
        tasks: list[ContactTask] = []
        for f in range(4):
            hole = int(round(holes[f]))
            # Fractional closed weight -> target slides from hover to contact.
            # Editor overrides (dragged targets) replace the hole-derived
            # target through Flute.finger_target instead of being bypassed.
            if f in self.flute.overrides:
                target = self.flute.finger_target(f, hole, closed[f] >= 0.5)
            else:
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
        # Edits made outside the sampler (dragged targets, moved instrument)
        # are caught here before any cached FK could be reused.
        self._sync_geometry()
        tasks, holes, closed, clip_idx = self.tasks_at(t)
        if self.backend == "fast":
            ik = solve_ik_fast(
                self.sk,
                tasks,
                q0=None,
                contact_threshold=self.contact_threshold,
                jacobian_mode=self.jacobian_mode,
            )
        else:
            ik = solve_ik(
                self.sk, tasks, q0=None, contact_threshold=self.contact_threshold
            )
        return Pose(
            t=t,
            fk=self.sk.fk(ik.q),
            ik=ik,
            holes=holes,
            closed_weight=closed,
            clip_index=clip_idx,
        )
