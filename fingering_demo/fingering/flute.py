"""Simplified flute tube with labelled tone holes (teaching geometry only).

World layout: the tube runs along the Y axis; its top surface faces +Z.
The hand hangs over the tube, knuckles fan along Y, so the four finger pads
land on a row of holes when the local +y flexion joints curl toward -Z.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Flute:
    radius: float = 0.010
    length: float = 0.300
    hole_radius: float = 0.0035
    # y positions of the six holes along the tube axis.  Holes 1..4 are the
    # home holes (index -> pinky); 0 and 5 are stretch/adjacent-switch holes.
    hole_y: tuple[float, ...] = (
        -0.100,
        -0.066,
        -0.022,
        0.022,
        0.066,
        0.100,
    )
    # Editor overrides: finger -> world target position (None = use hole).
    overrides: dict | None = None

    def __post_init__(self):
        if self.overrides is None:
            self.overrides = {}

    def n_holes(self) -> int:
        return len(self.hole_y)

    def hole_position(self, hole: int) -> np.ndarray:
        """Pad contact point: on the +Z (top) surface of the tube."""
        return np.array([0.0, self.hole_y[hole], self.radius])

    def hover_position(self, hole: int, lift: float = 0.025) -> np.ndarray:
        """Where a lifted finger rests above its hole."""
        pos = self.hole_position(hole)
        pos[2] += lift
        return pos

    def finger_target(self, finger: int, hole: int, closed: bool) -> np.ndarray:
        if finger in self.overrides:
            return np.asarray(self.overrides[finger], dtype=float).copy()
        if closed:
            return self.hole_position(hole)
        return self.hover_position(hole)

    def tube_outline(self, n: int = 40):
        """Side-view samples along the tube for drawing (y, z_top, z_bottom)."""
        half = self.length / 2.0
        y = np.linspace(-half, half, n)
        return y, np.full_like(y, -self.radius), np.full_like(y, self.radius)
