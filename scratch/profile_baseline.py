"""Profile current FD-based solver: FK evaluations dominate the budget."""
import sys
import time

sys.path.insert(0, "/workspace/fingering_demo")

import numpy as np

from fingering.skeleton import HandSkeleton
from fingering.flute import Flute
from fingering.timeline import default_timeline
from fingering.animation import PoseSampler
from fingering import ik as ik_mod

sk = HandSkeleton()
flute = Flute()
timeline = default_timeline()
sampler = PoseSampler(sk, flute, timeline)

# Instrument the FK to count evaluations.
counts = {"fk": 0}
orig_fk = sk.fk


def counting_fk(q):
    counts["fk"] += 1
    return orig_fk(q)


sk.fk = counting_fk

times = np.linspace(0.0, timeline.duration, 61)
t0 = time.perf_counter()
iters = []
for t in times:
    counts["fk"] = 0
    pose = sampler.pose_at(float(t))
    iters.append((pose.ik.iterations, counts["fk"], pose.ik.residuals_m.max()))
elapsed = time.perf_counter() - t0

n_q = sk.n_q
print(f"n_q = {n_q}, tasks per pose = 4, residual rows = {1 + 4 * 3} = 13")
print(f"61 poses total wall time: {elapsed:.3f}s ({elapsed/61*1000:.1f} ms/pose)")
print("per-pose (LM iters, full FK evals, max residual mm):")
for t, (it, fk, res) in zip(times[::6], iters[::6]):
    print(f"  t={t:4.2f}  iters={it:3d}  fk={fk:4d}  res={res*1000:5.2f}")
total_it = sum(i for i, _, _ in iters)
total_fk = sum(f for _, f, _ in iters)
print(f"total iters={total_it}, total FK evals={total_fk}")
print(f"FK per iter ~= 1 + {n_q} (FD columns) + 1 (candidate) = {n_q + 2}")
