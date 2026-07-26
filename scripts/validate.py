"""mcax validation against exact/quasi-exact bulk equations of state.

Periodic (bulk) runs in d = 1, 2, 3 at a set activity; the measured <rho> must
match the EOS inversion rho(z_act): Tonks exactly in d=1 (the razor — any
detailed-balance or bookkeeping bug fails here), scaled-particle in d=2 and
Carnahan-Starling in d=3 to their own accuracy. Then a d=1 slit run checks the
wall contact theorem rho(0+) = beta P against exact Tonks pressure
(quadratic-extrapolated contact from the first three bins).
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from mcax import make_spec, burn_and_sample

PI = np.pi
B = {1: 1.0, 2: PI / 4.0, 3: PI / 6.0}          # eta = B[d] * rho at sigma = 1


def f_ex(d, rho):
    """beta f_ex per d-volume: Tonks / SPT / CS."""
    eta = B[d] * rho
    if d == 1:
        return -rho * np.log(1.0 - eta)
    if d == 2:
        return rho * (-np.log(1.0 - eta) + eta / (1.0 - eta))
    return rho * (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2


def mu_of_rho(d, rho, h=1e-7):
    fp = (f_ex(d, rho + h) - f_ex(d, rho - h)) / (2 * h)
    return np.log(rho) + fp


def p_of_rho(d, rho, h=1e-7):
    fp = (f_ex(d, rho + h) - f_ex(d, rho - h)) / (2 * h)
    return rho + rho * fp - f_ex(d, rho)


def rho_of_mu(d, mu, lo=1e-6, hi=None):
    hi = hi or 0.95 / B[d]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mu_of_rho(d, mid) < mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


fails = 0
print("== bulk EOS checks (periodic) ==")
for d, eta, C, steps in ((1, 0.5, 16, 200_000), (2, 0.4, 16, 200_000),
                         (3, 0.3, 16, 200_000)):
    rho_t = eta / B[d]
    mu = mu_of_rho(d, rho_t)
    spec = make_spec(d, H=8.0, z_act=float(np.exp(mu)), Lperp=8.0,
                     slit=False)
    t0 = time.time()
    z, rho, Ns, acc = burn_and_sample(spec, C=C, seed=d, n_burn=steps // 4,
                                      n_run=steps, thin=500, nbins=80)
    V = spec.H * spec.Lperp ** (d - 1)
    rho_meas = Ns.mean() / V
    rel = abs(rho_meas - rho_t) / rho_t
    print(f"d={d} eta={eta}: <rho>={rho_meas:.4f} vs EOS {rho_t:.4f} "
          f"(rel {rel:.3f}) acc(disp/ins/del)={np.round(acc, 2)} "
          f"[{time.time() - t0:.0f}s]")
    fails += rel > 0.02

print("== d=1 slit: wall contact theorem vs exact Tonks ==")
eta = 0.5
mu = mu_of_rho(1, eta)
P = p_of_rho(1, eta)
spec = make_spec(1, H=8.0, z_act=float(np.exp(mu)), slit=True)
z, rho, Ns, acc = burn_and_sample(spec, C=32, seed=7, n_burn=100_000,
                                  n_run=400_000, thin=500, nbins=160)
# quadratic extrapolation of the first three bins to z = 0+
c = np.polyfit(z[:3], rho[:3], 2)
contact = np.polyval(c, 0.0)
rel = abs(contact - P) / P
print(f"contact rho(0+)={contact:.4f} vs beta P={P:.4f} (rel {rel:.3f})")
fails += rel > 0.05

print("VALIDATE", "FAIL" if fails else "PASS", f"({fails} failures)")
sys.exit(1 if fails else 0)
