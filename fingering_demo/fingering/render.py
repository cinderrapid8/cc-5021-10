"""2D rendering of the tube, skeleton, targets and contact markers.

Two complementary views are used because the teaching geometry is 3D:
  * side view: Y (tube axis) horizontal, Z vertical -- shows fingers lifting
    off and pressing onto the holes;
  * top view:  Y horizontal, X vertical -- shows finger spread along the tube.
All drawing is headless-safe (uses an injected Axes, never pyplot itself).
"""

from __future__ import annotations

import numpy as np

from .skeleton import HandSkeleton, FINGER_NAMES
from .flute import Flute
from .ik import PRIORITY_CONTACT

_FINGER_COLORS = ("#d62728", "#ff7f0e", "#2ca02c", "#9467bd")


def _draw_tube_side(ax, flute: Flute):
    y, z_bot, z_top = flute.tube_outline()
    ax.fill_between(y, z_bot, z_top, color="#cfd8dc", edgecolor="#455a64", zorder=1)
    for h in range(flute.n_holes()):
        pos = flute.hole_position(h)
        c = plt_circle(ax, pos[1], pos[2], flute.hole_radius, "#263238", zorder=3)
        ax.text(pos[1], pos[2] - 0.006, f"h{h}", ha="center", va="top", fontsize=7)


def plt_circle(ax, x, r, radius, color, zorder):
    from matplotlib.patches import Circle

    patch = Circle((x, r), radius, color=color, zorder=zorder)
    ax.add_patch(patch)
    return patch


def _draw_skeleton(ax, sk: HandSkeleton, fk, view: str):
    for f, chain in enumerate(sk.chain_nodes):
        pts = fk.p[chain]
        if view == "side":
            xs, ys = pts[:, 1], pts[:, 2]
        else:
            xs, ys = pts[:, 1], pts[:, 0]
        ax.plot(xs, ys, "-o", color=_FINGER_COLORS[f], lw=2, ms=4, zorder=5)
        pad = fk.pads[f]
        px, py = (pad[1], pad[2]) if view == "side" else (pad[1], pad[0])
        ax.plot(px, py, "s", color=_FINGER_COLORS[f], ms=7, zorder=6)
    wrist = fk.p[0]
    wx, wy = (wrist[1], wrist[2]) if view == "side" else (wrist[1], wrist[0])
    ax.plot(wx, wy, "k^", ms=9, zorder=7)


def _draw_targets(ax, pose, view: str):
    from matplotlib.patches import Circle

    for f, task in enumerate(pose.ik.contacts):
        t = task.target
        tx, ty = (t[1], t[2]) if view == "side" else (t[1], t[0])
        reached = pose.ik.residuals_m[f] <= pose.ik.contact_threshold
        if task.priority == PRIORITY_CONTACT:
            style = dict(facecolor="none", edgecolor="#1b5e20", lw=1.8)
        else:
            style = dict(facecolor="none", edgecolor="#9e9e9e", lw=1.2, ls="--")
        ring = Circle((tx, ty), 0.0045, zorder=4, picker=True, **style)
        ring.finger = f
        ax.add_patch(ring)
        if not reached and task.priority == PRIORITY_CONTACT:
            ax.plot(tx, ty, "x", color="#c62828", ms=8, mew=2, zorder=8)


def render_pose(ax_side, ax_top, sk: HandSkeleton, flute: Flute, pose):
    for ax, view in ((ax_side, "side"), (ax_top, "top")):
        ax.clear()
        _draw_tube_side(ax, flute) if view == "side" else _draw_tube_top(ax, flute)
        _draw_skeleton(ax, sk, pose.fk, view)
        _draw_targets(ax, pose, view)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        if view == "side":
            ax.set_xlabel("tube axis Y (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title("side view: lift / press")
        else:
            ax.set_xlabel("tube axis Y (m)")
            ax.set_ylabel("X (m)")
            ax.set_title("top view: finger spread")
    _annotate_residuals(ax_side, pose)


def _draw_tube_top(ax, flute: Flute):
    y, _, _ = flute.tube_outline()
    half = flute.radius
    ax.fill_between(y, -half, half, color="#eceff1", edgecolor="#455a64", zorder=1)
    for h in range(flute.n_holes()):
        pos = flute.hole_position(h)
        plt_circle(ax, pos[1], pos[0], flute.hole_radius, "#263238", zorder=3)
        ax.text(pos[1], pos[0] + 0.006, f"h{h}", ha="center", va="bottom", fontsize=7)


def _annotate_residuals(ax, pose):
    lines = [f"clip {pose.clip_index}  t={pose.t:.2f}s"]
    for f, task in enumerate(pose.ik.contacts):
        r = pose.ik.residuals_m[f] * 1000
        mark = "OK " if r <= pose.ik.contact_threshold * 1000 else "!! "
        lines.append(
            f"{mark}P{task.priority} {FINGER_NAMES[f]:<6} {r:6.2f} mm"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#90a4ae"),
    )
