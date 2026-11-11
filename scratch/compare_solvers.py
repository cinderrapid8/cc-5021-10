"""Compare reference FD vs hybrid/analytic solver on the full timeline."""
import sys
import time

sys.path.insert(0, "/workspace/fingering_demo")

import numpy as np

from fingering.skeleton import HandSkeleton
from fingering.flute import Flute
from fingering.timeline import default_timeline
from fingering.animation import PoseSampler
from fingering.ik import JACOBIAN_HYBRID, JACOBIAN_ANALYTIC


def run(backend, mode=None):
    sk = HandSkeleton()
    sampler = PoseSampler(sk, Flute(), default_timeline(), backend=backend,
                          jacobian_mode=mode or JACOBIAN_ANALYTIC)
    times = np.linspace(0.0, sampler.timeline.duration, 61)
    rows = []
    t0 = time.perf_counter()
    for t in times:
        p = sampler.pose_at(float(t))
        rows.append((p.ik.residuals_m.copy(), p.ik.q.copy(), p.ik.converged,
                     p.ik.budget, p.ik.limit_overshoot))
    wall = time.perf_counter() - t0
    return rows, wall


ref, t_ref = run("reference")
hyb, t_hyb = run("fast", JACOBIAN_HYBRID)
ana, t_ana = run("fast", JACOBIAN_ANALYTIC)

max_res_ref = max(r[0].max() for r in ref)
print(f"{'mode':<10} {'wall_s':>7} {'maxres_mm':>10} {'max|dq|':>10} "
      f"{'full_fk':>8} {'part_fk':>8} {'hits':>7} {'nodes':>8} {'overshoot':>10}")
for name, data, wall in (("reference", ref, t_ref), ("hybrid", hyb, t_hyb),
                         ("analytic", ana, t_ana)):
    full = sum(r[3].fk_full_evals for r in data)
    part = sum(r[3].fk_partial_evals for r in data)
    hits = sum(r[3].fk_cache_hits for r in data)
    nodes = sum(r[3].node_updates() for r in data)
    maxres = max(r[0].max() for r in data)
    over = max(r[4] for r in data)
    dq = max(np.max(np.abs(r[1] - rr[1])) for r, rr in zip(data, ref))
    print(f"{name:<10} {wall:7.3f} {maxres*1000:10.4f} {dq:10.2e} "
          f"{full:8d} {part:8d} {hits:7d} {nodes:8d} {over:10.2e}")

# Residual agreement: fast paths must not produce visibly worse contact.
for name, data in (("hybrid", hyb), ("analytic", ana)):
    dres = max(np.max(np.abs(r[0] - rr[0])) for r, rr in zip(data, ref))
    conv_mismatch = sum(r[2] != rr[2] for r, rr in zip(data, ref))
    print(f"{name}: max residual delta vs ref = {dres*1000:.4f} mm, "
          f"convergence mismatches = {conv_mismatch}")
