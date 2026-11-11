"""Validate the analytic Jacobian against dense finite differences."""
import sys

sys.path.insert(0, "/workspace/fingering_demo")

import numpy as np

from fingering.skeleton import HandSkeleton
from fingering.flute import Flute
from fingering.ik import (
    ContactTask,
    PRIORITY_CONTACT,
    PRIORITY_HOVER,
    _jacobian_analytic,
    _jacobian_fd,
    _stack,
)
from fingering.budget import SolveBudget

sk = HandSkeleton()
flute = Flute()
tasks = [
    ContactTask(0, flute.hole_position(1), PRIORITY_CONTACT, "c0"),
    ContactTask(1, flute.hover_position(2), PRIORITY_HOVER, "h1"),
    ContactTask(2, flute.hole_position(3), PRIORITY_CONTACT, "c2"),
    ContactTask(3, flute.hole_position(0), PRIORITY_CONTACT, "c3"),
]

rng = np.random.default_rng(7)
worst = 0.0
eye = np.eye(sk.n_q)
for trial in range(6):
    q = sk.initial_guess_q()
    q += rng.normal(scale=0.15, size=sk.n_q)
    q = sk.clamp_q(q)
    r0, _, W = _stack(sk, q, tasks)
    fd = _jacobian_fd(sk, q, tasks, r0, W, SolveBudget())
    fk = sk.fk(q)
    an = _jacobian_analytic(sk, q, tasks, fk, W, SolveBudget())
    # At an active joint limit the clamped FD derivative is 0 by construction
    # while the analytic derivative stays nonzero (the un-clamped kinematic
    # slope); mask those columns rather than treating clamping as an error.
    free = np.ones(sk.n_q, dtype=bool)
    for j in range(sk.n_q):
        if np.array_equal(sk.clamp_q(q + eye[j] * 1e-7), q):
            free[j] = False
    err = np.abs(fd[:, free] - an[:, free])
    scale = np.maximum(np.abs(fd[:, free]), 1e-2)
    rel = err / scale
    # Absolute guard for near-zero FD entries.
    bad = (rel > 5e-3) & (err > 1e-5)
    worst = max(worst, err.max())
    print(f"trial {trial}: free cols {free.sum()}/{sk.n_q}  "
          f"max abs J err={err.max():.3e}  entries over tol={bad.sum()}")
    assert not bad.any()
print("analytic Jacobian matches finite differences (relative tolerance 5e-3)")

# Cache identity: repeated identical q returns the same object; new q rebuilds.
fk_a, hit_a = sk.fk_with_status(q)
fk_b, hit_b = sk.fk_with_status(q.copy())
assert hit_a and hit_b and fk_a is fk_b
fk_c, hit_c = sk.fk_with_status(sk.clamp_q(q + 1e-4))
assert not hit_c and fk_c is not fk_a
print("FK cache: identical q hits, changed q rebuilds")

# Partial FK must match a full FK for a single-finger DOF perturbation.
finger = 2
dq = np.zeros(sk.n_q)
dq[6 + finger * 4 + 1] = 0.07
qp = sk.clamp_q(q + dq)
full = sk.fk(qp)
part = sk.fk_finger_partial(qp, finger, fk_a, q)
assert np.allclose(full.R, part.R, atol=1e-12)
assert np.allclose(full.p, part.p, atol=1e-12)
assert np.allclose(full.pads, part.pads, atol=1e-12)
# And it must leave other finger chains byte-identical to the base.
other = [f for f in range(4) if f != finger]
for f in other:
    for node in sk.chain_nodes[f]:
        assert np.array_equal(part.p[node], fk_a.p[node])
print("partial FK equals full FK on changed chain, reuses others exactly")
