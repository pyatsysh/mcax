"""mcax validation against exact and quasi-exact equations of state.

Thin CLI over the same checks as `tests/test_validation.py`, kept because a
validation run is something you want to WATCH, with numbers and timings on
stdout, rather than reduced to a pass/fail dot.

    python scripts/validate.py                  # everything
    python scripts/validate.py --dims 1         # just the exact d = 1 cases
    JAX_PLATFORMS=cuda python scripts/validate.py
    python scripts/validate.py --markdown       # emit the README table rows

Periodic (bulk) runs in d = 1, 2, 3 at a set activity: the measured <rho> must
match the EOS inversion rho(z). Tonks is exact in d = 1, the razor where any
detailed-balance or bookkeeping bug has nowhere to hide, while d = 2 and d = 3
are held to Henderson and Carnahan-Starling to their own accuracy. Then a d = 1
slit run checks the wall contact theorem rho(0+) = beta P against exact Tonks
pressure, quadratic-extrapolated to contact from the first three bins.

Split R-hat is reported and gated for every case, not just the density. A mean
can be right for the wrong reason: the 3-D case once matched Carnahan-Starling
to 0.1% with its chains at R-hat 1.23, having spent the whole burn-in filling
from empty.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp

from mcax import (make_spec, burn_and_sample, summary, format_summary,
                  eos, potentials)

# (d, eta, chains, n_burn, n_run, reference). The 3-D case costs 10x the others
# because N decorrelates over ~70 draws there; see tests/test_validation.py.
CASES = [
    (1, 0.5, 16, 100_000, 200_000, "Tonks and Percus"),
    (2, 0.4, 16, 100_000, 200_000, "Henderson"),
    (3, 0.3, 32, 500_000, 2_000_000, "Carnahan and Starling"),
]

p = argparse.ArgumentParser()
p.add_argument("--dims", type = int, nargs = "+", default = [1, 2, 3],
               help = "which dimensions to run (default: all)")
p.add_argument("--scale", type = float, default = 1.0,
               help = "multiply every step count (0.1 for a quick look)")
p.add_argument("--attractive", action = "store_true",
               help = "also run the square-well cases against Takahashi")
p.add_argument("--markdown", action = "store_true",
               help = "also emit the README and site table rows")
args = p.parse_args()

print(f"device: {jax.devices()[0].platform}   mcax validation")
fails = 0
rows = []

print("== bulk EOS checks (periodic) ==")
for d, eta, C, n_burn, n_run, ref in CASES:
    if d not in args.dims:
        continue
    n_burn = int(n_burn * args.scale)
    n_run = max(int(n_run * args.scale), 1000)
    rho_t = eta / eos.B[d]
    spec = make_spec(d, H = 8.0, z_act = float(eos.z_of_rho(d, rho_t)),
                     Lperp = 8.0, slit = False)
    V = spec.H * spec.Lperp ** (d - 1)
    n0 = int(0.9 * rho_t * V)
    t0 = time.time()
    res = burn_and_sample(spec, C = C, seed = d, n_burn = n_burn,
                          n_run = n_run, thin = 500, nbins = 80, n0 = n0)
    rho = res.n_mean / V
    rel = abs(rho - rho_t) / rho_t
    s = summary(res.Ns)
    exact = " (exact)" if d == 1 else ""
    print(f"d={d} eta={eta}{exact}: <rho>={rho:.5f} vs EOS {rho_t:.5f} "
          f"(rel {rel:.5f}) acc(disp/ins/del)={onp.round(res.acc, 3)} "
          f"[{time.time() - t0:.0f}s]")
    print(f"    {format_summary(s)}")
    if res.capacity_warning:
        print(f"    !! {res.capacity_warning}")
        fails += 1
    fails += rel > 0.02
    fails += s["split_rhat"] > 1.05
    acc = "**exact**" if d == 1 else "0.1%"
    rows.append(f"| d = {d} bulk, eta = {eta} | {ref} | {acc} | "
                f"{rel:.4f} | {s['split_rhat']:.3f} |")

if 1 in args.dims:
    print("== d=1 slit: wall contact theorem vs exact Tonks ==")
    rho_b = 0.5
    P = eos.p_of_rho(1, rho_b)
    spec = make_spec(1, H = 8.0, z_act = float(eos.z_of_rho(1, rho_b)),
                     slit = True)
    t0 = time.time()
    res = burn_and_sample(spec, C = 48, seed = 7,
                          n_burn = int(100_000 * args.scale),
                          n_run = max(int(600_000 * args.scale), 1000),
                          thin = 500, nbins = 160)
    # Both walls are equivalent, so mirror-averaging halves the counting
    # variance before extrapolating. Even so the extrapolation amplifies bin
    # noise to sigma(contact) ~ 0.03, so the gate sits at ~4 sigma: a nominally
    # strict 5% gate at this statistics was a 43% chance of a spurious failure.
    rho_sym = 0.5 * (res.rho + res.rho[::-1])
    contact = onp.polyval(onp.polyfit(res.z[:3], rho_sym[:3], 2), 0.0)
    rel = abs(contact - P) / P
    s = summary(res.Ns)
    print(f"contact rho(0+)={contact:.5f} vs beta P={P:.5f} (rel {rel:.5f}) "
          f"[{time.time() - t0:.0f}s]")
    print(f"    {format_summary(s)}")
    if res.capacity_warning:
        print(f"    !! {res.capacity_warning}")
        fails += 1
    fails += rel > 0.12
    fails += s["split_rhat"] > 1.05
    rows.append(f"| d = 1 slit contact, eta = 0.5 | rho(0+) = beta P, exact "
                f"Tonks | **exact** | {rel:.4f} | {s['split_rhat']:.3f} |")

# The attractive cases. Takahashi's nearest-neighbour solution is exact for
# lam <= 2, so these carry the same weight as the Tonks rows above: a failure is
# a bug in the energy bookkeeping and not a reference equation's error bar.
SW_CASES = [
    (0.5, 1.5, 0.30, "sw-shallow"),
    (1.0, 1.5, 0.30, "sw-standard"),
    (1.0, 1.5, 0.45, "sw-dense"),
    (1.5, 1.8, 0.25, "sw-deep-wide"),
]

if 1 in args.dims and args.attractive:
    print("== d=1 square well: bulk density vs exact Takahashi ==")
    for eps, lam, rho_t, label in SW_CASES:
        spec = make_spec(1, H = 60.0, slit = False,
                         z_act = eos.sw1d_z_of_rho(eps, lam, rho_t),
                         pair = potentials.SquareWell(eps, lam))
        t0 = time.time()
        res = burn_and_sample(spec, C = 32, seed = 11,
                              n_burn = int(200_000 * args.scale),
                              n_run = max(int(600_000 * args.scale), 1000),
                              thin = 500, nbins = 40,
                              n0 = int(0.9 * rho_t * spec.H))
        rho = res.n_mean / spec.H
        rel = abs(rho - rho_t) / rho_t
        s = summary(res.Ns)
        print(f"eps={eps} lam={lam}: <rho>={rho:.5f} vs exact {rho_t:.5f} "
              f"(rel {rel:.5f}) acc={onp.round(res.acc, 3)} "
              f"[{time.time() - t0:.0f}s]")
        print(f"    {format_summary(s)}")
        if res.capacity_warning:
            print(f"    !! {res.capacity_warning}")
            fails += 1
        fails += rel > 0.02
        fails += s["split_rhat"] > 1.05
        rows.append(f"| d = 1 bulk square well, beta eps = {eps}, "
                    f"lam = {lam}, rho = {rho_t} | Takahashi | **exact** | "
                    f"{rel:.4f} | {s['split_rhat']:.3f} |")

    print("== d=1 slit square well: contact theorem with attraction ==")
    eps, lam, rho_b = 1.0, 1.5, 0.35
    P = eos.sw1d_p_of_rho(eps, lam, rho_b)
    spec = make_spec(1, H = 10.0, slit = True,
                     z_act = eos.sw1d_z_of_rho(eps, lam, rho_b),
                     pair = potentials.SquareWell(eps, lam))
    t0 = time.time()
    res = burn_and_sample(spec, C = 48, seed = 13,
                          n_burn = int(150_000 * args.scale),
                          n_run = max(int(600_000 * args.scale), 1000),
                          thin = 500, nbins = 200)
    rho_sym = 0.5 * (res.rho + res.rho[::-1])
    contact = onp.polyval(onp.polyfit(res.z[:3], rho_sym[:3], 2), 0.0)
    rel = abs(contact - P) / P
    s = summary(res.Ns)
    print(f"contact rho(0+)={contact:.5f} vs beta P={P:.5f} (rel {rel:.5f}) "
          f"[{time.time() - t0:.0f}s]")
    print(f"    {format_summary(s)}")
    fails += rel > 0.12
    fails += s["split_rhat"] > 1.05
    rows.append(f"| d = 1 slit contact, square well, rho = {rho_b} | "
                f"rho(0+) = beta P, exact Takahashi | **exact** | "
                f"{rel:.4f} | {s['split_rhat']:.3f} |")

if args.markdown:
    print("\n== table rows ==")
    print("| Case | Reference | Reference accuracy | Measured rel. error | split R-hat |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(r)

print("VALIDATE", "FAIL" if fails else "PASS", f"({fails} failures)")
sys.exit(1 if fails else 0)
