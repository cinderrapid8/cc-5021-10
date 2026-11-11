"""Priority-based damped least-squares inverse kinematics.

Two mathematically equivalent solver paths are provided:

* :func:`solve_ik` is the reference implementation: a full finite-difference
  Jacobian rebuilt from 22 perturbed FK evaluations per iteration.  It is kept
  as the run reference -- its residuals, limits and determinism are what the
  optimized path must reproduce.
* :func:`solve_ik_fast` reuses the same LM loop but assembles the Jacobian
  either fully analytically from one cached FK, or in a hybrid mode where the
  six wrist columns are analytic and the sixteen finger columns use scoped
  single-chain finite differences.

The solver never modifies bone lengths: the only state is joint variables q
(6 wrist DOFs + local finger DOFs).  When a fingertip target is out of reach
the residual stays visible instead of being hidden by stretching a bone.

Stability comes from Levenberg-Marquardt adaptive damping with a monotonic
cost acceptance rule; the fixed initial guess makes every call deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .skeleton import HandSkeleton, rodrigues as _rodrigues
from .budget import SolveBudget, BudgetTimer

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

JACOBIAN_FD = "fd"
JACOBIAN_HYBRID = "hybrid"
JACOBIAN_ANALYTIC = "analytic"
_JACOBIAN_MODES = (JACOBIAN_FD, JACOBIAN_HYBRID, JACOBIAN_ANALYTIC)


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
    jacobian_mode: str = JACOBIAN_FD
    budget: SolveBudget = field(default_factory=SolveBudget)

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


# ------------------------------------------------------------- residual stack
def _weights_and_sizes(sk: HandSkeleton, tasks: list[ContactTask]):
    sizes = [sk.n_q]
    weights = [_WEIGHTS[PRIORITY_WRIST]]
    for task in tasks:
        weights.append(_WEIGHTS[task.priority])
        sizes.append(3)
    W = np.concatenate([np.full(sz, np.sqrt(w)) for sz, w in zip(sizes, weights)])
    return W


def _residuals_from_fk(
    sk: HandSkeleton,
    q: np.ndarray,
    tasks: list[ContactTask],
    fk,
    rest: np.ndarray,
):
    # Regularize only the 6 wrist DOFs; finger joints are free to satisfy
    # contacts.  Kept weak so it never overrides a reachable target.
    wrist_err = np.zeros(sk.n_q)
    wrist_err[0:6] = (q[0:6] - rest[0:6]) / _WRIST_SCALE
    blocks = [wrist_err]
    residuals_m = np.zeros(len(tasks))
    for k, task in enumerate(tasks):
        err = fk.pads[task.finger] - np.asarray(task.target, dtype=float)
        residuals_m[k] = float(np.linalg.norm(err))
        blocks.append(err / _POS_SCALE)
    return np.concatenate(blocks), residuals_m


def _stack(sk: HandSkeleton, q: np.ndarray, tasks: list[ContactTask]):
    fk = sk.fk(q)
    r, residuals_m = _residuals_from_fk(sk, q, tasks, fk, sk.rest_q())
    W = _weights_and_sizes(sk, tasks)
    return r, residuals_m, W


def _cost(rw):
    return float(rw @ rw)


# ------------------------------------------------------------- Jacobian: FD
def _jacobian_fd(sk, q, tasks, r0, W, budget: SolveBudget):
    """Reference Jacobian: one clamped full-FK perturbation per DOF."""
    n, m = sk.n_q, len(r0)
    J = np.zeros((m, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = _FD_EPS
        qp = sk.clamp_q(q + dq)
        fk, hit = sk.fk_with_status(qp)
        if hit:
            budget.fk_cache_hits += 1
        else:
            budget.fk_full_evals += 1
        r1, _ = _residuals_from_fk(sk, qp, tasks, fk, sk.rest_q())
        J[:, j] = (r1 - r0) / _FD_EPS
    return J * W[:, None]


# ------------------------------------------------------ Jacobian: analytic
def _wrist_axis_twists(fk):
    """World angular-velocity axes for the three wrist rotation-vector DOFs.

    The wrist frame is built as R = R_x(q3) R_y(q4) R_z(q5) (each Rodrigues
    about a local axis).  Differentiating that product gives, for a unit rate
    of each DOF, the world-frame twist axes below -- not simply R e_i.
    """
    q = fk.q
    Rx = _rodrigues(np.array([1.0, 0.0, 0.0]), q[3])
    Rxy = Rx @ _rodrigues(np.array([0.0, 1.0, 0.0]), q[4])
    return (
        Rx @ np.array([1.0, 0.0, 0.0]),
        Rxy @ np.array([0.0, 1.0, 0.0]),
        Rxy @ np.array([0.0, 0.0, 1.0]),
    )


def _jacobian_analytic(sk, q, tasks, fk, W, budget: SolveBudget):
    """Closed-form Jacobian built from a single cached FK (zero extra FKs).

    Wrist position columns are unit translations; rotational columns use the
    spatial Jacobian v = omega x r with omega the DOF's world angular-velocity
    axis (see :func:`_wrist_axis_twists`).  Finger columns touch only their own
    pad rows because every finger chain is a separate branch of the tree --
    all other task rows stay exactly zero.
    """
    n = sk.n_q
    m = n + 3 * len(tasks)
    J = np.zeros((m, n))

    # Wrist regularization block: d(wrist_err)/d(q) on the first 6 rows only;
    # finger rows of the regularizer are zero because wrist_err is zero there.
    J[0:6, 0:6] = np.diag([1.0 / _WRIST_SCALE] * 6)

    wrist_omega = _wrist_axis_twists(fk)
    for k, task in enumerate(tasks):
        f = task.finger
        rows = slice(n + 3 * k, n + 3 * k + 3)
        pad = fk.pads[f]
        # Wrist translation: pad moves 1:1 with wrist xyz.
        J[rows, 0:3] = np.eye(3) / _POS_SCALE
        # Wrist rotation-vector DOFs: world twist axes of the composed frame.
        for axis_i in range(3):
            J[rows, 3 + axis_i] = (
                np.cross(wrist_omega[axis_i], pad - q[0:3]) / _POS_SCALE
            )
        # Finger DOFs: local axis expressed in world at the joint node.  For a
        # joint carrying two stacked rotations (MCP flex+abduction) the flex
        # axis must be propagated through the preceding rotation.
        for dof_index in sk.finger_dof_indices(f):
            omega_world = _dof_world_axis(sk, fk, dof_index)
            J[rows, dof_index] = (
                np.cross(omega_world, pad - fk.p[sk.dofs[dof_index - 6].node])
                / _POS_SCALE
            )
    return J * W[:, None]


def _dof_world_axis(sk, fk, dof_index: int) -> np.ndarray:
    """World angular-velocity axis of one finger DOF at the current pose.

    A node composes its DOF rotations in declaration order, so a DOF preceded
    by a sibling rotation on the same node rotates its local axis through that
    earlier rotation first.
    """
    dof = sk.dofs[dof_index - 6]
    node_idx = dof.node
    node = sk.nodes[node_idx]
    R_prefix = np.eye(3)
    for earlier in node.dof_indices:
        if earlier == dof_index:
            break
        earlier_dof = sk.dofs[earlier - 6]
        R_prefix = R_prefix @ _rodrigues(earlier_dof.axis, fk.q[earlier])
    # fk.R[node] = R_parent @ rest_R @ R_dof, so R_parent @ rest_R is the
    # frame in which the node-local pre-DOF axis is expressed.
    R_dof_full = np.eye(3)
    for di in node.dof_indices:
        d = sk.dofs[di - 6]
        R_dof_full = R_dof_full @ _rodrigues(d.axis, fk.q[di])
    R_frame = fk.R[node_idx] @ np.linalg.inv(R_dof_full)
    return R_frame @ R_prefix @ dof.axis


def _jacobian_hybrid(sk, q, tasks, r0, fk, W, budget: SolveBudget):
    """Analytic wrist columns + chain-scoped finite differences for fingers.

    Each finger perturbation leaves q[0:6] and every other chain untouched, so
    four node frames (mcp..tip) are recomputed instead of the full 17-node
    tree.  This is the conservative fast path: it stays numerically adjacent to
    the reference FD while cutting both the FK scope and the row work.
    """
    n = sk.n_q
    m = n + 3 * len(tasks)
    J = np.zeros((m, n))
    J[0:6, 0:6] = np.diag([1.0 / _WRIST_SCALE] * 6)
    # Wrist task columns analytically (same twist-axis derivation as the full
    # analytic J); finger columns stay finite-differenced below.
    wrist_omega = _wrist_axis_twists(fk)
    for k, task in enumerate(tasks):
        rows = slice(n + 3 * k, n + 3 * k + 3)
        pad = fk.pads[task.finger]
        J[rows, 0:3] = np.eye(3) / _POS_SCALE
        for axis_i in range(3):
            J[rows, 3 + axis_i] = (
                np.cross(wrist_omega[axis_i], pad - q[0:3]) / _POS_SCALE
            )

    rest = sk.rest_q()
    # Finger FD columns: partial FK scoped to the perturbed finger's chain.
    finger_of_dof = {j: (j - 6) // 4 for j in range(6, n)}
    for j in range(6, n):
        f = finger_of_dof[j]
        dq = np.zeros(n)
        dq[j] = _FD_EPS
        qp = sk.clamp_q(q + dq)
        fk_p = sk.fk_finger_partial(qp, f, fk, q)
        budget.fk_partial_evals += 1
        r1, _ = _residuals_from_fk(sk, qp, tasks, fk_p, rest)
        J[:, j] = (r1 - r0) / _FD_EPS
    return J * W[:, None]


# ------------------------------------------------------------- shared LM loop
def _run_lm(
    sk: HandSkeleton,
    tasks: list[ContactTask],
    q0: np.ndarray,
    contact_threshold: float,
    max_iters: int,
    jacobian_mode: str,
) -> IKResult:
    if jacobian_mode not in _JACOBIAN_MODES:
        raise ValueError(f"unknown jacobian mode {jacobian_mode!r}")
    n = sk.n_q
    budget = SolveBudget()
    result = IKResult(
        q=q0.copy(),
        contacts=list(tasks),
        contact_threshold=contact_threshold,
        jacobian_mode=jacobian_mode,
        budget=budget,
    )

    with BudgetTimer(budget):
        fk0, hit = sk.fk_with_status(q0)
        if hit:
            budget.fk_cache_hits += 1
        else:
            budget.fk_full_evals += 1
        rest = sk.rest_q()
        W = _weights_and_sizes(sk, tasks)
        r0, _ = _residuals_from_fk(sk, q0, tasks, fk0, rest)
        rw = r0 * W
        damping = _DAMPING0
        q = q0
        it = 0
        for it in range(max_iters):
            budget.jacobian_builds += 1
            if jacobian_mode == JACOBIAN_FD:
                Jw = _jacobian_fd(sk, q, tasks, r0, W, budget)
            elif jacobian_mode == JACOBIAN_ANALYTIC:
                fk_cur, hit = sk.fk_with_status(q)
                if hit:
                    budget.fk_cache_hits += 1
                else:
                    budget.fk_full_evals += 1
                Jw = _jacobian_analytic(sk, q, tasks, fk_cur, W, budget)
            else:
                fk_cur, hit = sk.fk_with_status(q)
                if hit:
                    budget.fk_cache_hits += 1
                else:
                    budget.fk_full_evals += 1
                Jw = _jacobian_hybrid(sk, q, tasks, r0, fk_cur, W, budget)

            budget.lin_solves += 1
            A = Jw.T @ Jw + (damping**2) * np.eye(n)
            step = np.linalg.solve(A, -Jw.T @ rw)
            q_new = sk.clamp_q(q + step)

            fk_new, hit = sk.fk_with_status(q_new)
            if hit:
                budget.fk_cache_hits += 1
            else:
                budget.fk_full_evals += 1
            r_new, _ = _residuals_from_fk(sk, q_new, tasks, fk_new, rest)
            rw_new = r_new * W
            if _cost(rw_new) < _cost(rw):
                q, r0, rw = q_new, r_new, rw_new
                budget.accepted_steps += 1
                damping = max(damping * _DAMPING_DOWN, 1e-6)
                if np.linalg.norm(step) < 1e-5:
                    break
            else:
                budget.rejected_steps += 1
                damping *= _DAMPING_UP
                if damping > 1e3:
                    break

        fk_final, hit = sk.fk_with_status(q)
        if hit:
            budget.fk_cache_hits += 1
        else:
            budget.fk_full_evals += 1
        _, residuals_m = _residuals_from_fk(sk, q, tasks, fk_final, rest)
        budget.iterations = it + 1

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


def solve_ik(
    sk: HandSkeleton,
    tasks: list[ContactTask],
    q0: np.ndarray | None = None,
    contact_threshold: float = 0.004,
    max_iters: int = _MAX_ITERS,
) -> IKResult:
    """Reference FD solver: unchanged numerics, now with a budget report."""
    q = sk.initial_guess_q() if q0 is None else sk.clamp_q(np.asarray(q0, float))
    return _run_lm(sk, tasks, q, contact_threshold, max_iters, JACOBIAN_FD)


def solve_ik_fast(
    sk: HandSkeleton,
    tasks: list[ContactTask],
    q0: np.ndarray | None = None,
    contact_threshold: float = 0.004,
    max_iters: int = _MAX_ITERS,
    jacobian_mode: str = JACOBIAN_ANALYTIC,
) -> IKResult:
    """Cached-analytic / hybrid solver with the same LM semantics.

    Same initial guess, damping schedule, monotonic acceptance, joint-limit
    clamping and contact threshold as :func:`solve_ik`; only the Jacobian
    assembly and FK evaluation scope change.
    """
    if jacobian_mode not in _JACOBIAN_MODES or jacobian_mode == JACOBIAN_FD:
        raise ValueError(
            f"solve_ik_fast requires {JACOBIAN_HYBRID!r} or {JACOBIAN_ANALYTIC!r}"
        )
    q = sk.initial_guess_q() if q0 is None else sk.clamp_q(np.asarray(q0, float))
    return _run_lm(sk, tasks, q, contact_threshold, max_iters, jacobian_mode)
