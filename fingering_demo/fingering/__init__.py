"""Instrument fingering animation demo (simplified teaching model)."""

from .skeleton import HandSkeleton, rodrigues, FINGER_NAMES
from .flute import Flute
from .timeline import Timeline, Clip
from .ik import solve_ik, ContactTask, IKResult, PRIORITY_CONTACT, PRIORITY_HOVER
from .animation import PoseSampler, Pose
from .export import export_joint_curves, export_contact_markers

__all__ = [
    "HandSkeleton",
    "rodrigues",
    "FINGER_NAMES",
    "Flute",
    "Timeline",
    "Clip",
    "solve_ik",
    "ContactTask",
    "IKResult",
    "PRIORITY_CONTACT",
    "PRIORITY_HOVER",
    "PoseSampler",
    "Pose",
    "export_joint_curves",
    "export_contact_markers",
]
