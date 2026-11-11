"""Constraint-solve budget: explicit cost decomposition for one IK solve.

Every speed claim in this project is backed by these counters instead of a
bare frame-rate number.  The categories are counted *in the order the work
happens* inside one Levenberg-Marquardt iteration, so the per-iteration
budget can be reasoned about without reading the solver:

    1. residual stack at the current q
         -> one full-tree FK (an exact cache hit costs zero node updates)
    2. Jacobian assembly
         reference solver : n_q forward-FD perturbations, each a full FK
         hybrid solver    : 6 analytic wrist columns (zero FK) plus
                            16 finger perturbations scoped to one 4-node chain
         analytic solver  : zero FK; columns derived from the cached FK
    3. one damped normal-equation linear solve
    4. candidate step: clamp to joint limits, then one residual stack
         -> one full-tree FK
    5. monotonic accept/reject and the damping update

With n_q = 22 the reference therefore spends 1 + 22 + 1 = 24 full FK
evaluations per iteration (34,982 over the 61-pose reference timeline).
The optimizations below reduce both the FK count and the node-update scope
while keeping the same LM acceptance rule, contact threshold and limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

# Node counts of the four-finger skeleton: wrist + 4 chains * 4 nodes = 17.
FULL_TREE_NODES = 17
# One finger chain updates mcp -> pip -> dip -> tip = 4 nodes.
FINGER_CHAIN_NODES = 4


@dataclass
class SolveBudget:
    iterations: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    jacobian_builds: int = 0
    lin_solves: int = 0
    fk_full_evals: int = 0
    fk_partial_evals: int = 0
    fk_cache_hits: int = 0
    wall_time_s: float = 0.0

    def node_updates(self) -> int:
        """Equivalent number of single-node transform updates.

        Puts full-tree and single-finger evaluations on one comparable scale:
        a full FK rewrites 17 node frames, a partial update only 4.
        """
        return (
            self.fk_full_evals * FULL_TREE_NODES
            + self.fk_partial_evals * FINGER_CHAIN_NODES
        )

    def as_row(self) -> str:
        return (
            f"iters={self.iterations:3d} (acc {self.accepted_steps:3d}/"
            f"rej {self.rejected_steps:3d}) J={self.jacobian_builds:3d} "
            f"lin={self.lin_solves:3d} fk_full={self.fk_full_evals:4d} "
            f"fk_partial={self.fk_partial_evals:4d} cache_hits="
            f"{self.fk_cache_hits:4d} node_updates={self.node_updates():6d} "
            f"time={self.wall_time_s * 1e3:7.2f} ms"
        )


class BudgetTimer:
    """Attach wall-clock timing to an explicitly counted solve."""

    def __init__(self, budget: SolveBudget):
        self.budget = budget
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self.budget

    def __exit__(self, *exc):
        self.budget.wall_time_s += time.perf_counter() - self._t0
        return False
