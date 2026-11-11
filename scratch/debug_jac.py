import sys
sys.path.insert(0, "/workspace/fingering_demo")
import numpy as np
from fingering.skeleton import HandSkeleton
from fingering.flute import Flute
from fingering.ik import (
    ContactTask, PRIORITY_CONTACT, PRIORITY_HOVER,
    _jacobian_analytic, _jacobian_fd, _stack,
)
from fingering.budget import SolveBudget

sk = HandSkeleton()
flute = Flute()
tasks = [
    ContactTask(0, flute.hole_position(1), PRIORITY_CONTACT, "c0"),
    ContactTask(2, flute.hole_position(3), PRIORITY_CONTACT, "c2"),
]
rng = np.random.default_rng(7)
q = sk.initial_guess_q()
q += rng.normal(scale=0.15, size=sk.n_q)
q = sk.clamp_q(q)
r0, _, W = _stack(sk, q, tasks)
fd = _jacobian_fd(sk, q, tasks, r0, W, SolveBudget())
an = _jacobian_analytic(sk, q, tasks, sk.fk(q), W, SolveBudget())

n = sk.n_q
for j in range(n):
    d = np.abs(fd[:, j] - an[:, j]).max()
    flag = "  <-- LIMIT" if np.array_equal(sk.clamp_q(q + np.eye(n)[j] * 1e-7), q) else ""
    print(f"col {j:2d} ({sk.dof_names[j]:<20}) max abs diff = {d:.3e}{flag}")
# Show a concrete mismatch: wrist rot col 3 vs task 0
print("\nFD col 3 (wrist rx) task0 rows:", np.round(fd[n:n+3, 3], 4))
print("AN col 3 (wrist rx) task0 rows:", np.round(an[n:n+3, 3], 4))
print("\nFD col 6 (index mcp flex) task0 rows:", np.round(fd[n:n+3, 6], 4))
print("AN col 6 (index mcp flex) task0 rows:", np.round(an[n:n+3, 6], 4))
print("\nFD col 6 ring task rows (must be ~0):", np.round(fd[n+3:n+6, 6], 6))
print("AN col 6 ring task rows:", np.round(an[n+3:n+6, 6], 6))
print("\nwrist reg rows FD[0:6,0:6] diag:", np.round(np.diag(fd[0:6, 0:6]), 4))
print("wrist reg rows AN[0:6,0:6] diag:", np.round(np.diag(an[0:6, 0:6]), 4))
