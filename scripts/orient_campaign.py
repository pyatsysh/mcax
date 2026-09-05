"""Starter dataset for FREELY ROTATING superballs (the orientable act).

Deliberately small and honest: two shapes, bulk activity ladders plus a
handful of slit states with orientation-resolved observables, produced by
the validated orientable engine (orient_validate.py, V0-V3 green). The
consumer is the dimint-dft orientational extension, which does not exist
yet — this dataset is what its design work will be grounded on, and every
state records the same convergence evidence as the parallel campaign
(replica error, drift, cubatic monitor) so nothing has to be re-run to be
trusted.

Scope guards carried over: fluid-branch certification by the cubatic
monitor s4 (a plastic crystal shows up there before anywhere else near
the sphere end), conservative packing caps, save-as-you-go.

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 \
      <python> scripts/orient_campaign.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)

from mcax import bodies, geometry, orient
from mcax.bodies import Superball

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "dimint-dft", "data", "orientable")
os.makedirs(OUT, exist_ok=True)

P_GRID = (3.0, 4.0)
C = 16
N_BURN, N_RUN, THIN = 30_000, 90_000, 50
Z_LADDER = list(onp.geomspace(0.05, 12.0, 9)) + [22.0, 40.0]
# the two appended rungs (2026-08-08): the geomspace top lands at
# eta ~ 0.245, just under the 0.25 slit targets; appending preserves the
# tags of every completed rung so the resume logic re-runs nothing
SLITS = ((5.0, 0.15), (5.0, 0.25), (8.0, 0.15), (8.0, 0.25))
ETA_CAP = 0.38            # conservative: free bodies order earlier
S4_TRIP = 0.12            # cubatic monitor threshold


def chain_err(Ns):
    v = onp.asarray(Ns, dtype=float).mean(axis=1)
    return float(v.std(ddof=1) / onp.sqrt(v.size))


def drift_rel(Ns):
    v = onp.asarray(Ns, dtype=float).mean(axis=0)
    third = len(v) // 3
    return abs(float(v[-third:].mean() - v[:third].mean())) \
        / max(float(v.mean()), 1e-12)


def run_state(tag, p, geom, H, z_act, rows, n0_frac=0.85):
    body = Superball(p)
    spec = orient.make_spec(H=H, Lperp=(H if geom == "bulk" else 6.0),
                            z_act=z_act, body=body, geom=geom, dtheta=0.35)
    v = bodies.volume(body, 3)
    n0 = int(n0_frac * min(0.5, z_act * v / (1 + z_act * v) * 3) / v
             * geometry.volume(spec))
    t0 = time.time()
    # aligned=True always: a Haar-oriented prefill can seat persistent
    # overlaps (see burn_and_sample's docstring); the burn randomises
    r = orient.burn_and_sample(spec, C=C, seed=11, n_burn=N_BURN,
                               n_run=N_RUN, thin=THIN,
                               nbins=max(int(H / 0.1), 20), n0=n0,
                               aligned=True)
    vol = geometry.volume(spec)
    eta = r.n_mean * v / vol
    s4m = float(onp.nanmax(onp.abs(r.s4)))
    row = dict(tag=tag, p=p, geom=geom, H=H, Lperp=float(spec.Lperp),
               z_act=float(z_act), mu=float(onp.log(z_act)),
               v_particle=float(v), volume=float(vol),
               n_mean=float(r.n_mean), n_err=chain_err(r.Ns),
               eta_mean=float(eta), drift_rel=drift_rel(r.Ns),
               s4_max=s4m, ordered=bool(s4m > S4_TRIP),
               acc=[float(x) for x in r.acc],
               rho=onp.asarray(r.rho).tolist(),
               s4_profile=onp.asarray(r.s4).tolist(),
               s6_profile=onp.asarray(r.s6).tolist(),
               z_bins=onp.asarray(r.z).tolist(),
               saturation=float(r.saturation),
               chains=C, n_run=N_RUN, n_burn=N_BURN,
               wall_time_s=time.time() - t0)
    rows.append(row)
    onp.save(os.path.join(OUT, "orient_states.npy"),
             onp.array(rows, dtype=object), allow_pickle=True)
    flag = "  ORDERED" if row["ordered"] else ""
    print(f"{tag:<30s} eta {eta:.3f}  <N> {r.n_mean:7.1f} "
          f"({row['n_err']:.2f})  drift {row['drift_rel']:.2%}  "
          f"s4 {s4m:.3f}{flag}  {row['wall_time_s']:.0f}s", flush=True)
    return row


def main():
    rows = []
    path = os.path.join(OUT, "orient_states.npy")
    if os.path.exists(path):
        rows = list(onp.load(path, allow_pickle=True))
        print(f"resuming with {len(rows)} states done")
    done = {r["tag"] for r in rows}

    # stage 1: bulk activity ladders per shape
    for p in P_GRID:
        for z in Z_LADDER:
            tag = f"bulk_p{p:g}_z{z:.3f}"
            if tag in done:
                continue
            row = run_state(tag, p, "bulk", 8.0, float(z), rows)
            if row["eta_mean"] > ETA_CAP or row["ordered"]:
                print(f"  ladder for p={p:g} capped at eta "
                      f"{row['eta_mean']:.3f}")
                break

    # stage 2: slit states at target reservoir packings, activity read off
    # the measured ladder by local interpolation of ln z against eta
    for p in P_GRID:
        lad = sorted([r for r in rows
                      if r["geom"] == "bulk" and r["p"] == p
                      and not r["ordered"]], key=lambda r: r["eta_mean"])
        e = onp.array([r["eta_mean"] for r in lad])
        lz = onp.array([onp.log(r["z_act"]) for r in lad])
        for H, eta_t in SLITS:
            tag = f"slit_p{p:g}_H{H:g}_eta{eta_t:.2f}"
            if tag in done or eta_t > e.max():
                continue
            z = float(onp.exp(onp.interp(eta_t, e, lz)))
            run_state(tag, p, "slit", H, z, rows)

    with open(os.path.join(OUT, "MANIFEST.md"), "w") as f:
        f.write("# Orientable superball starter dataset\n\n"
                f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
                "orient_campaign.py (engine: mcax.orient, validation "
                "V0-V3 in mcax/out/orient_validation.json).\n\n"
                "Freely rotating superballs, d = 3, shapes p in "
                f"{list(P_GRID)}; bulk activity ladders plus slit states "
                "with rho(z) and the cubatic moment profile s4(z). The "
                "cubatic monitor certifies the fluid branch per state "
                f"(trip at s4 > {S4_TRIP}); `ordered` rows are audit "
                "records, not data. mu = ln z is an exact input.\n\n"
                f"states: {len(rows)}\n")
    print(f"done: {len(rows)} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
