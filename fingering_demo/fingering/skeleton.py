"""Self-contained hand skeleton.

Every joint rotates about an axis defined in that joint's *local* frame and is
clamped to a local-coordinate range.  Forward kinematics only compose rigid
rotations/translations, so bone lengths are constants of the model and can
never be stretched by the solver.

Local frame convention (right handed, identical for wrist and finger nodes):
    +x = distal (along the tube, toward +X world)
    +y = lateral (along +Y world; fingers fan along this axis)
    +z = palmar (toward the tube, -Z world); the finger pad is on the +z side.
A positive Rodrigues rotation about local +y sends distal +x toward palmar
+z, which is anatomical flexion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Finger DOFs start after the 6 wrist DOFs; each finger owns 4 DOFs in the
# build order (mcp_flex, mcp_abd, pip_flex, dip_flex).
FINGER_DOF_STRIDE = 4

FINGER_NAMES = ("index", "middle", "ring", "pinky")
HOME_HOLES = (1, 2, 3, 4)

AXIS_FLEX = np.array([0.0, 1.0, 0.0])
AXIS_ABD = np.array([0.0, 0.0, 1.0])

# Per-finger rest link lengths (proximal, middle, distal) in metres.
_BASE_LENGTHS = {
    "index": (0.040, 0.026, 0.020),
    "middle": (0.044, 0.029, 0.022),
    "ring": (0.041, 0.028, 0.021),
    "pinky": (0.033, 0.020, 0.017),
}
# Distal stagger of each knuckle along local +x at rest.
_BASE_STAGGER = (0.004, 0.000, 0.003, 0.009)
FINGER_SPACING = 0.044

# Local joint ranges (radians).  Flexion positive, zero = straight finger.
MCP_FLEX = (0.0, np.deg2rad(85.0))
MCP_ABD = (np.deg2rad(-30.0), np.deg2rad(30.0))
PIP_FLEX = (0.0, np.deg2rad(100.0))
DIP_FLEX = (0.0, np.deg2rad(80.0))

WRIST_POS_TRAVEL = 0.035
WRIST_ROT_TRAVEL = 0.30


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` about local ``axis``."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues: the 3-vector whose direction/length encode R."""
    cc = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cc)
    if angle < 1e-10:
        return np.zeros(3)
    skew = (R - R.T) / (2.0 * np.sin(angle))
    return angle * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


@dataclass
class DOF:
    name: str
    node: int
    axis: np.ndarray
    lo: float
    hi: float


@dataclass
class Node:
    name: str
    parent: int | None
    rest_p: np.ndarray
    rest_R: np.ndarray
    dof_indices: list[int] = field(default_factory=list)


@dataclass
class FKResult:
    q: np.ndarray
    R: np.ndarray  # (N, 3, 3) world orientations
    p: np.ndarray  # (N, 3) world positions
    pads: np.ndarray  # (4, 3) finger pad world positions

    def finger_chain(self, sk: "HandSkeleton", finger: int) -> np.ndarray:
        return self.p[sk.chain_nodes[finger]]


class HandSkeleton:
    """Four-finger simplified hand with a 6-DOF wrist."""

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self.nodes: list[Node] = []
        self.dofs: list[DOF] = []

        # Wrist rests above the tube; fingers fan along Y over the hole row
        # and curl about local +y toward -Z onto the +Z tube surface.
        self.wrist_rest_p = np.array([-0.030, 0.0, 0.090])
        self.wrist_rest_R = np.eye(3)

        self._add_node("wrist", None, np.zeros(3), np.eye(3))
        spacing = FINGER_SPACING * scale

        self.chain_nodes: list[list[int]] = []
        self.pad_offsets: list[np.ndarray] = []
        for f, fname in enumerate(FINGER_NAMES):
            L1, L2, L3 = (v * scale for v in _BASE_LENGTHS[fname])
            base = np.array(
                [_BASE_STAGGER[f] * scale, (f - 1.5) * spacing, 0.0]
            )
            mcp = self._add_node(f"{fname}_mcp", 0, base, np.eye(3))
            i_flex = self._add_dof(f"{fname}_mcp_flex", mcp, AXIS_FLEX, *MCP_FLEX)
            i_abd = self._add_dof(f"{fname}_mcp_abd", mcp, AXIS_ABD, *MCP_ABD)
            self.nodes[mcp].dof_indices = [i_flex, i_abd]

            pip = self._add_node(f"{fname}_pip", mcp, np.array([L1, 0, 0]), np.eye(3))
            i_pip = self._add_dof(f"{fname}_pip_flex", pip, AXIS_FLEX, *PIP_FLEX)
            self.nodes[pip].dof_indices = [i_pip]

            dip = self._add_node(f"{fname}_dip", pip, np.array([L2, 0, 0]), np.eye(3))
            i_dip = self._add_dof(f"{fname}_dip_flex", dip, AXIS_FLEX, *DIP_FLEX)
            self.nodes[dip].dof_indices = [i_dip]

            tip = self._add_node(f"{fname}_tip", dip, np.array([L3, 0, 0]), np.eye(3))
            self.chain_nodes.append([mcp, pip, dip, tip])
            # Pad just beyond the tip on the palmar (+z) side.
            self.pad_offsets.append(np.array([0.004 * scale, 0.0, 0.006 * scale]))

        # q = [wrist xyz, wrist rotation vector (3), finger DOFs ...]
        self.wrist_pos_idx = slice(0, 3)
        self.wrist_rot_idx = slice(3, 6)
        self.n_q = 6 + len(self.dofs)
        self.dof_names = ["wrist_x", "wrist_y", "wrist_z",
                          "wrist_rx", "wrist_ry", "wrist_rz"] + [
            d.name for d in self.dofs
        ]

        lo = np.full(self.n_q, -np.inf)
        hi = np.full(self.n_q, np.inf)
        lo[0:3] = self.wrist_rest_p - WRIST_POS_TRAVEL
        hi[0:3] = self.wrist_rest_p + WRIST_POS_TRAVEL
        lo[3:6] = -WRIST_ROT_TRAVEL
        hi[3:6] = WRIST_ROT_TRAVEL
        for k, dof in enumerate(self.dofs):
            lo[6 + k] = dof.lo
            hi[6 + k] = dof.hi
        self.limits_lo = lo
        self.limits_hi = hi
        self._rest_q = None
        # FK memoization: one entry, keyed by geometry revision + q contents.
        self._geometry_revision = 0
        self._fk_cache_key = None
        self._fk_cache = None

    # ------------------------------------------------------------------ build
    def _add_node(self, name, parent, p, R) -> int:
        self.nodes.append(Node(name, parent, np.array(p, float), np.array(R, float)))
        return len(self.nodes) - 1

    def _add_dof(self, name, node, axis, lo, hi) -> int:
        self.dofs.append(DOF(name, node, np.array(axis, float), lo, hi))
        return 6 + len(self.dofs) - 1

    # ------------------------------------------------------------- cache ctrl
    def invalidate_geometry_cache(self) -> None:
        """Drop memoized kinematics after an external geometry change.

        Bone lengths and joint frames are model constants, so the only events
        that can stale a cached FK are edits the embedding application makes
        explicitly (e.g. moving an instrument target).  Callers signal such an
        edit through this hook instead of the cache guessing on its own.
        """
        self._geometry_revision += 1
        self._fk_cache_key = None
        self._fk_cache = None
        self._rest_q = None

    def finger_dof_indices(self, finger: int) -> list[int]:
        start = 6 + finger * FINGER_DOF_STRIDE
        return list(range(start, start + FINGER_DOF_STRIDE))

    # ------------------------------------------------------------------- rest
    def rest_q(self) -> np.ndarray:
        if self._rest_q is None:
            q = np.zeros(self.n_q)
            q[self.wrist_pos_idx] = self.wrist_rest_p
            self._rest_q = q
        return self._rest_q.copy()

    def initial_guess_q(self) -> np.ndarray:
        """Deterministic half-flexed start: keeps the solver away from the
        straight-finger singularity without depending on playback history."""
        q = self.rest_q()
        for dof in self.dofs:
            if dof.name.endswith("_mcp_flex"):
                q[self.dof_names.index(dof.name)] = np.deg2rad(45.0)
            elif dof.name.endswith("_pip_flex"):
                q[self.dof_names.index(dof.name)] = np.deg2rad(55.0)
            elif dof.name.endswith("_dip_flex"):
                q[self.dof_names.index(dof.name)] = np.deg2rad(30.0)
        return self.clamp_q(q)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.limits_lo, self.limits_hi)

    def limit_overshoot(self, q: np.ndarray) -> float:
        """Maximum excursion past a local joint limit (0.0 means none)."""
        return float(
            max(
                0.0,
                np.max(self.limits_lo - q, initial=0.0),
                np.max(q - self.limits_hi, initial=0.0),
            )
        )

    # --------------------------------------------------------------------- FK
    def _update_node(self, i: int, q: np.ndarray, R: np.ndarray, p: np.ndarray) -> None:
        node = self.nodes[i]
        Rp, pp = R[node.parent], p[node.parent]
        R_dof = np.eye(3)
        for di in node.dof_indices:
            dof = self.dofs[di - 6]
            R_dof = R_dof @ rodrigues(dof.axis, q[di])
        # Local-frame composition: rest placement, then joint rotation.
        R[i] = Rp @ node.rest_R @ R_dof
        p[i] = pp + Rp @ node.rest_p

    def _build_fk(self, q: np.ndarray) -> FKResult:
        n = len(self.nodes)
        R = np.zeros((n, 3, 3))
        p = np.zeros((n, 3))

        delta = rodrigues(np.array([1.0, 0, 0]), q[3])
        delta = delta @ rodrigues(np.array([0, 1.0, 0]), q[4])
        delta = delta @ rodrigues(np.array([0, 0, 1.0]), q[5])
        R[0] = self.wrist_rest_R @ delta
        p[0] = q[0:3]

        for i in range(1, n):
            self._update_node(i, q, R, p)

        pads = np.zeros((4, 3))
        for f in range(4):
            tip = self.chain_nodes[f][-1]
            pads[f] = p[tip] + R[tip] @ self.pad_offsets[f]
        return FKResult(q=q.copy(), R=R, p=p, pads=pads)

    def fk(self, q: np.ndarray) -> FKResult:
        """Forward kinematics with a one-entry exact-content cache.

        The cache stores the q it was built from, so it can only hit when the
        joint configuration is genuinely unchanged; stale geometry is ruled
        out by the revision counter bumped in ``invalidate_geometry_cache``.
        """
        q = np.asarray(q, dtype=float)
        key = (self._geometry_revision, q.shape, q.dtype.str)
        if (
            self._fk_cache is not None
            and self._fk_cache_key == key
            and np.array_equal(self._fk_cache.q, q)
        ):
            return self._fk_cache
        fk = self._build_fk(q)
        self._fk_cache_key = key
        self._fk_cache = fk
        return fk

    def fk_with_status(self, q: np.ndarray) -> tuple[FKResult, bool]:
        """Same memoized FK as :meth:`fk`, plus whether the cache was hit."""
        q = np.asarray(q, dtype=float)
        key = (self._geometry_revision, q.shape, q.dtype.str)
        if (
            self._fk_cache is not None
            and self._fk_cache_key == key
            and np.array_equal(self._fk_cache.q, q)
        ):
            return self._fk_cache, True
        fk = self._build_fk(q)
        self._fk_cache_key = key
        self._fk_cache = fk
        return fk, False

    def fk_finger_partial(
        self,
        q: np.ndarray,
        finger: int,
        base_fk: FKResult,
        base_q: np.ndarray,
    ) -> FKResult:
        """Recompute only one finger chain when its own four DOFs changed.

        Wrist DOFs and the other three fingers are untouched, so their frames
        are reused from ``base_fk`` instead of running a full 17-node FK.  The
        wrist block of the Jacobian never takes this path: a wrist perturbation
        moves every chain and requires a full rebuild.
        """
        q = np.asarray(q, dtype=float)
        if np.any(q[0:6] != base_q[0:6]):
            return self.fk(q)
        R = base_fk.R.copy()
        p = base_fk.p.copy()
        for i in self.chain_nodes[finger]:
            self._update_node(i, q, R, p)
        pads = base_fk.pads.copy()
        tip = self.chain_nodes[finger][-1]
        pads[finger] = p[tip] + R[tip] @ self.pad_offsets[finger]
        return FKResult(q=q.copy(), R=R, p=p, pads=pads)

    # ------------------------------------------------------------- inspection
    def joint_world_axis(self, fk: FKResult, dof_index: int) -> np.ndarray:
        """World direction of a *local* joint axis after FK (proof that joints
        do not all share one world rotation axis)."""
        dof = self.dofs[dof_index - 6]
        return fk.R[dof.node] @ dof.axis

    def bone_lengths(self) -> np.ndarray:
        """Rest bone lengths for every parent->child link."""
        return np.array([np.linalg.norm(nd.rest_p) for nd in self.nodes[1:]])

    def actual_bone_lengths(self, fk: FKResult) -> np.ndarray:
        return np.array(
            [
                np.linalg.norm(fk.p[i] - fk.p[node.parent])
                for i, node in enumerate(self.nodes[1:], start=1)
            ]
        )
