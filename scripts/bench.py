"""Device benchmark: steps/second on the production-shaped workload.

Run once with JAX_PLATFORMS=cpu and once with JAX_PLATFORMS=cuda (inside the
granted lease window, XLA_PYTHON_CLIENT_MEM_FRACTION as granted) and compare.
The shape is deliberately the heaviest production state: d=3, H=8, eta=0.35,
C=64 chains — Nmax ~ 1300, one lockstep move per chain-step.
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from mcax import make_spec, init_state, run

PI = np.pi
eta = 0.35
rho = 6.0 * eta / PI
etaf = lambda x: x * (4 - 3 * x) / (1 - x) ** 2  # not used; CS below
mu_ex = (8 * eta - 9 * eta ** 2 + 3 * eta ** 3) / (1 - eta) ** 3
mu = float(np.log(rho) + mu_ex)

spec = make_spec(3, H=8.0, z_act=float(np.exp(mu)), Lperp=10.0, slit=True)
n0 = int(0.9 * rho * spec.H * spec.Lperp ** 2)
C, NSTEP = 64, 20_000

print(f"device: {jax.devices()[0].platform}, spec Nmax={spec.Nmax}, C={C}")
st = init_state(spec, C, seed=0, n0=n0)
t0 = time.time()
st, hist, Ns = run(spec, st, 2_000, 500, 160)     # compile + warm
jax.block_until_ready(hist)
t1 = time.time()
st, hist, Ns = run(spec, st, NSTEP, 500, 160)
jax.block_until_ready(hist)
t2 = time.time()
sps = NSTEP / (t2 - t1)
print(f"compile+warm {t1 - t0:.1f}s; {NSTEP} steps in {t2 - t1:.1f}s "
      f"= {sps:,.0f} steps/s (chain-steps/s = {sps * C:,.0f})")
print(f"<N> = {float(np.asarray(Ns).mean()):.1f}")
