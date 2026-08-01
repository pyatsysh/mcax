"""Device benchmark: steps/second on the production-shaped workload.

Run once with JAX_PLATFORMS=cpu and once with JAX_PLATFORMS=cuda (inside the
granted lease window, XLA_PYTHON_CLIENT_MEM_FRACTION as granted) and compare.
The shape is deliberately the heaviest production state: d = 3, H = 8,
eta = 0.35, C = 64 chains, Nmax ~ 1300, one lockstep move per chain-step.

The figure of merit is **chain-steps per second**, not steps per second. A step
advances all C chains at once, so the batch is where the device earns its
keep; quoting bare steps/s understates the throughput by a factor of C.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp

from mcax import make_spec, init_state, run, eos

p = argparse.ArgumentParser()
p.add_argument("--eta", type=float, default=0.35)
p.add_argument("--chains", type=int, default=64)
p.add_argument("--steps", type=int, default=20_000)
args = p.parse_args()

rho = args.eta / eos.B[3]
spec = make_spec(3, H=8.0, z_act=float(eos.z_of_rho(3, rho)), Lperp=10.0,
                 slit=True)
# Seat ~90% of the target N on a lattice: a grand-canonical fill from empty
# takes as long as the burn-in at this density, and would be timed as work.
n0 = int(0.9 * rho * spec.H * spec.Lperp ** 2)
C, NSTEP = args.chains, args.steps

print(f"device: {jax.devices()[0].platform}   d=3 eta={args.eta} "
      f"Nmax={spec.Nmax} C={C} n0={n0}")
st = init_state(spec, C, seed=0, n0=n0)

t0 = time.time()
st, ac = run(spec, st, 2_000, 500, 160)                # compile + warm
jax.block_until_ready(ac.hist)
t1 = time.time()

st, ac = run(spec, st, NSTEP, 500, 160)
jax.block_until_ready(ac.hist)
t2 = time.time()

sps = NSTEP / (t2 - t1)
print(f"compile+warm {t1 - t0:.1f}s; {NSTEP} steps in {t2 - t1:.1f}s "
      f"= {sps:,.0f} steps/s = {sps * C:,.0f} chain-steps/s")
print(f"<N> = {float(onp.asarray(Ns).mean()):.1f}  "
      f"n_hi = {int(onp.max(onp.asarray(st.nhi)))} of {spec.Nmax}")
