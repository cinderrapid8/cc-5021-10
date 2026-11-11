"""Fingering clips and a non-cyclical, jump-safe timeline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Clip:
    name: str
    duration: float
    # Per finger: assigned hole index and whether the hole is covered.
    holes: tuple[int, int, int, int]
    closed: tuple[bool, bool, bool, bool]


def default_timeline() -> Timeline:
    # Six-hole simplified flute: (1,2,3,4) are the four home holes,
    # 0 and 5 are reach/stretch holes used for adjacent-hole switching.
    home = (1, 2, 3, 4)
    clips = [
        Clip("all_open", 1.0, home, (False, False, False, False)),
        Clip("index_down", 1.0, home, (True, False, False, False)),
        Clip("three_down", 1.2, home, (True, True, True, False)),
        Clip("all_closed", 1.2, home, (True, True, True, True)),
        Clip("pinky_to_hole5", 1.2, home, (True, True, True, True)),
        Clip("index_to_hole0", 1.2, home, (True, True, True, True)),
    ]
    # Stretch-hole clips actually retarget the relevant finger.
    clips[4].holes = (1, 2, 3, 5)
    clips[5].holes = (0, 2, 3, 4)
    return Timeline(clips, transition=0.25)


class Timeline:
    def __init__(self, clips: list[Clip], transition: float = 0.2):
        self.clips = clips
        self.transition = float(transition)
        self.boundaries = np.concatenate([[0.0], np.cumsum([c.duration for c in clips])])

    @property
    def duration(self) -> float:
        return float(self.boundaries[-1])

    def set_transition(self, transition: float) -> None:
        self.transition = max(0.0, float(transition))

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hole per finger, closed weight per finger, active clip id).

        A transition is centered on each boundary: over its window both the
        target position and the open/closed contact weight are blended with a
        smoothstep, so contact establishment and release are continuous.
        """
        t = float(np.clip(t, 0.0, self.duration))
        idx = int(np.searchsorted(self.boundaries[1:], t, side="right"))
        idx = min(idx, len(self.clips) - 1)
        clip = self.clips[idx]
        holes = np.array(clip.holes, dtype=float)
        closed = np.array(clip.closed, dtype=float)

        b0, b1 = self.boundaries[idx], self.boundaries[idx + 1]
        half = self.transition / 2.0
        if half > 0.0 and idx > 0 and t < b0 + half:
            s = self._smoothstep((t - (b0 - half)) / self.transition)
            prev = self.clips[idx - 1]
            holes = s * holes + (1.0 - s) * np.array(prev.holes, dtype=float)
            closed = s * closed + (1.0 - s) * np.array(prev.closed, dtype=float)
        if half > 0.0 and idx < len(self.clips) - 1 and t > b1 - half:
            s = self._smoothstep((t - (b1 - half)) / self.transition)
            nxt = self.clips[idx + 1]
            holes = (1.0 - s) * holes + s * np.array(nxt.holes, dtype=float)
            closed = (1.0 - s) * closed + s * np.array(nxt.closed, dtype=float)
        return holes, closed, idx

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)
