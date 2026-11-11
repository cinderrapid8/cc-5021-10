"""Priority-based damped least-squares inverse kinematics.

The solver never modifies bone lengths: the only state is joint variables q
(6 wrist DOFs + local finger DOFs).  When a fingertip target is out of reach
the residual stays visible instead of being hidden by stretching a bone.

Stability comes from Levenberg-Marquardt adaptive damping with a monotonic
cost acceptance rule; the fixed initial guess makes every call deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .skeleton import HandSkeleton

PRIORITY_WRIST = 0
PRIORITY_CONTACT = 1
PRIORITY_HOVER = 2

_WEIGHTS = {PRIORITY_WRIST: 0.005, PRIORITY_CONTACT: 1.0, PRIORITY_HOVER: 0.35}
_POS_SCALE = 0.010  # 1 cm of position error -> unit residual
_WRIST_SCALE = 0.05
_DAMPING0 = 5e-3
_DAMPING_UP = 3.0
_DAMPING_DOWN = 0.6
_MAX_ITERS = 150
_FD_EPS = 1e-7


@dataclass
class ContactTask:
    finger: int
    target: np.ndarray
    priority: int
    label: str = ""


@dataclass
class IKResult:
    q: np.ndarray
    contacts: list[ContactTask] = field(default_factory=list)
    residuals_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    priorities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    converged: bool = False
    iterations: int = 0
    wrist_drift: float = 0.0
    limit_overshoot: float = 0.0
    contact_threshold: float = 0.004

    def residual_table(self) -> list[str]:
        rows = []
        for task, res, pr in zip(self.contacts, self.residuals_m, self.priorities):
            rows.append(
                f"P{pr} {task.label:<14} finger={task.finger} "
                f"residual={res * 1000:6.2f} mm"
            )
        return rows

    def reached(self) -> np.ndarray:
        return self.residuals_m <= self.contact_threshold

    def unreachable(self) -> list[int]:
        out = []
        for task, res, pr in zip(self.contacts, self.residuals_m, self.priorities):
            if pr == PRIORITY_CONTACT and res > self.contact_threshold:
                out.append(task.finger)
        return out


def _stack(sk: HandSkeleton, q: np.ndarray, tasks: list[ContactTask]):
    fk = sk.fk(q)
    rest = sk.rest_q()
    # Regularize only the 6 wrist DOFs; finger joints are free to satisfy
    # contacts.  Kept weak so it never overrides a reachable target.
    wrist_err = np.zeros(sk.n_q)
    wrist_err[0:6] = (q[0:6] - rest[0:6]) / _WRIST_SCALE
    blocks = [wrist_err]
    weights = [_WEIGHTS[PRIORITY_WRIST]]
    sizes = [sk.n_q]
    residuals_m = np.zeros(len(tasks))
    for k, task in enumerate(tasks):
        err = fk.pads[task.finger] - np.asarray(task.target, dtype=float)
        residuals_m[k] = float(np.linalg.norm(err))
        blocks.append(err / _POS_SCALE)
        weights.append(_WEIGHTS[task.priority])
        sizes.append(3)
    r = np.concatenate(blocks)
    W = np.concatenate([np.full(sz, np.sqrt(w)) for sz, w in zip(sizes, weights)])
    return r, residuals_m, W


def _jacobian(sk, q, tasks, r0, W):
    n, m = sk.n_q, len(r0)
    J = np.zeros((m, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = _FD_EPS
        r1, _, _ = _stack(sk, sk.clamp_q(q + dq), tasks)
        J[:, j] = (r1 - r0) / _FD_EPS
    return J * W[:, None]


def _cost(rw):
    return float(rw @ rw)


def solve_ik(
    sk: HandSkeleton,
    tasks: list[ContactTask],
    q0: np.ndarray | None = None,
    contact_threshold: float = 0.004,
    max_iters: int = _MAX_ITERS,
) -> IKResult:
    q = sk.initial_guess_q() if q0 is None else sk.clamp_q(np.asarray(q0, dtype=float))
    n = sk.n_q
    result = IKResult(
        q=q.copy(), contacts=list(tasks), contact_threshold=contact_threshold
    )

    r0, _, W = _stack(sk, q, tasks)
    rw = r0 * W
    damping = _DAMPING0
    it = 0
    for it in range(max_iters):
        Jw = _jacobian(sk, q, tasks, r0, W)
        A = Jw.T @ Jw + (damping**2) * np.eye(n)
        step = np.linalg.solve(A, -Jw.T @ rw)
        q_new = sk.clamp_q(q + step)
        r_new, _, W_new = _stack(sk, q_new, tasks)
        rw_new = r_new * W_new
        if _cost(rw_new) < _cost(rw):
            q, r0, W, rw = q_new, r_new, W_new, rw_new
            damping = max(damping * _DAMPING_DOWN, 1e-6)
            if np.linalg.norm(step) < 1e-5:
                break
        else:
            damping *= _DAMPING_UP
            if damping > 1e3:
                break

    _, residuals_m, _ = _stack(sk, q, tasks)
    result.q = q
    result.iterations = it + 1
    result.residuals_m = residuals_m
    result.priorities = np.array([t.priority for t in tasks])
    contact_res = [
        r for r, t in zip(residuals_m, tasks) if t.priority == PRIORITY_CONTACT
    ]
    result.converged = all(r <= contact_threshold for r in contact_res)
    result.wrist_drift = float(np.linalg.norm(q[0:6] - sk.rest_q()[0:6]))
    result.limit_overshoot = sk.limit_overshoot(q)
    return result
