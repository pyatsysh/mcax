"""Attractive pair tails: the w(r) a mean-field cDFT functional approximates.

Classical DFT almost always splits the free energy into a hard-sphere reference
and a perturbative tail,

    beta F[rho] = beta F_hs[rho] + (1/2) int int rho(r) rho(r') beta w(|r-r'|),

with the second term treated in mean field, meaning no correlations in the tail
at all. The exact ground truth for that functional is a simulation of exactly
u(r) = hard core sigma PLUS the same w. Then the discrepancy between the two is
the mean-field approximation itself, measured cleanly, with no mismatch of
Hamiltonians to argue about. That is what this module is for, and it is why the
tails here are the ones that literature uses rather than whatever is convenient.

Every w is expressed OUTSIDE the hard core and returns zero inside it: the core
is the engine's overlap test and is not this module's business. Strengths are
beta epsilon, so temperature enters only through them and T* = 1/eps.

    SquareWell(eps, lam)          -eps for sigma < r < lam sigma
    Yukawa(eps, kappa)            -eps sigma exp(-kappa (r-sigma)) / r
    LennardJonesWCA(eps)          the WCA split: -eps inside r_min, LJ beyond
    LennardJonesBH(eps)           the Barker-Henderson split: LJ for r > sigma

**The square well is the one to start with.** It is a second INDICATOR, so the
energy of a configuration is just minus eps times a neighbour count and the
kernel keeps its integer character. It is also the only one with an exact
answer: in one dimension with lam <= 2 only nearest neighbours interact and
Takahashi's solution gives the equation of state in closed form, so d = 1 stays
the razor with attraction switched on exactly as it is without. See
`mcax.eos.sw1d_state` and the slow validation tier.

**What this does NOT do, and it matters.** An attractive fluid has a
liquid-vapour transition, and below T_c the muVT distribution of N at
coexistence is bimodal with a barrier that grows with system size. A plain muVT
sampler does not cross that barrier: it sticks in one phase and reports a
converged-looking mean that is wrong, and R-hat will not always catch it because
every chain can stick in the SAME phase. Nothing here does histogram
reweighting, umbrella sampling or finite-size scaling. Stay supercritical or
dilute, watch the N-histogram for bimodality, and treat anything near
coexistence as out of scope for now.

**Hashing.** Same reasoning as `mcax.fields`: these are NamedTuples so that two
identically-parameterised potentials are one jit cache key rather than two.
"""
from typing import NamedTuple

import numpy as onp
import jax.numpy as np


class SquareWell(NamedTuple):
    """beta w = -eps for sigma < r < lam sigma, zero outside.

    `lam` is the well width in diameters; 1.5 is the literature default. Keep
    it at or below 2 if you want the exact 1-D solution to apply, because past
    that second neighbours interact and the nearest-neighbour construction
    stops being exact.
    """
    eps: float
    lam: float = 1.5

    def __call__(self, r2, sigma):
        inside = (r2 > sigma ** 2) & (r2 < (self.lam * sigma) ** 2)
        return np.where(inside, -self.eps, 0.0)


class Yukawa(NamedTuple):
    """beta w = -eps sigma exp(-kappa (r - sigma)) / r, zero inside the core.

    The hard-core Yukawa tail, chosen in a good deal of analytic DFT work
    because its mean-field integral is available in closed form. `kappa` is the
    inverse screening length in units of 1/sigma; kappa = 1.8 is the common
    comparison point.
    """
    eps: float
    kappa: float = 1.8
    rcut: float = 6.0

    def __call__(self, r2, sigma):
        # Clamp before dividing. A stale slot can sit at r = 0 and the mask
        # would discard the result anyway, but only after inf has already been
        # formed, and inf * 0 downstream is a nan.
        r = np.maximum(np.sqrt(r2), sigma)
        w = -self.eps * sigma * np.exp(-self.kappa * (r - sigma)) / r
        return np.where((r2 > sigma ** 2) & (r2 < (self.rcut * sigma) ** 2),
                        w, 0.0)


class LennardJonesWCA(NamedTuple):
    """The Weeks-Chandler-Andersen attractive part of Lennard-Jones:

        beta w(r) = -eps                       r < r_min = 2^(1/6) sigma
                  = 4 eps [(s/r)^12 - (s/r)^6] r >= r_min

    which is the split that keeps w purely attractive and everywhere smooth.
    The repulsion it leaves behind is replaced here by the hard core, so
    `sigma` is doing the job the WCA reference sphere would do, and the
    correspondence with a full Lennard-Jones fluid is approximate by exactly
    that much.
    """
    eps: float
    rcut: float = 3.0

    def __call__(self, r2, sigma):
        r = np.maximum(np.sqrt(r2), 1e-12)
        s6 = (sigma / r) ** 6
        lj = 4.0 * self.eps * (s6 ** 2 - s6)
        rmin = 2.0 ** (1.0 / 6.0) * sigma
        w = np.where(r < rmin, -self.eps, lj)
        return np.where((r2 > sigma ** 2) & (r2 < (self.rcut * sigma) ** 2),
                        w, 0.0)


class LennardJonesBH(NamedTuple):
    """The Barker-Henderson attractive part: the full Lennard-Jones beyond
    sigma, nothing inside it.

        beta w(r) = 4 eps [(s/r)^12 - (s/r)^6]     r > sigma

    Barker-Henderson pairs this with a temperature-dependent effective hard
    diameter, `bh_diameter` below, rather than with sigma itself. Use that
    diameter as `spec.sigma` if you are reproducing a Barker-Henderson
    comparison; use sigma directly if you only want a well-defined Hamiltonian.
    """
    eps: float
    rcut: float = 3.0

    def __call__(self, r2, sigma):
        r = np.maximum(np.sqrt(r2), 1e-12)
        s6 = (sigma / r) ** 6
        lj = 4.0 * self.eps * (s6 ** 2 - s6)
        return np.where((r2 > sigma ** 2) & (r2 < (self.rcut * sigma) ** 2),
                        lj, 0.0)


def bh_diameter(eps, sigma = 1.0, n = 20_000):
    """Barker-Henderson effective hard diameter,

        d(T) = int_0^sigma [1 - exp(-beta u_LJ(r))] dr,

    with u_LJ the full Lennard-Jones. Host-side and one-off, so a fine
    trapezoid costs nothing. `eps` is beta epsilon, so d shrinks towards sigma
    as the temperature falls and grows as it rises.
    """
    r = onp.linspace(1e-6, sigma, n)
    s6 = (sigma / r) ** 6
    u = 4.0 * eps * (s6 ** 2 - s6)
    return float(onp.trapezoid(1.0 - onp.exp(-onp.clip(u, -700.0, 700.0)), r))


def evaluate(pair, r2, sigma):
    """beta w for every entry of `r2`, or exact zeros when there is no tail."""
    if pair is None:
        return np.zeros_like(r2)
    return pair(r2, sigma)


def mean_field_integral(pair, d, sigma = 1.0, rmax = None, n = 40_000):
    """int w(r) d^d r, the number a mean-field DFT actually consumes.

    Written down here because it is the bridge between a run and the functional
    the run is meant to test: the mean-field excess free energy per unit volume
    is (1/2) rho^2 times this integral, so comparing against a cDFT calculation
    starts by checking that both sides use the same number.
    """
    rmax = (rmax if rmax is not None else
            getattr(pair, "rcut", 2.0) * sigma) if pair is not None else sigma
    r = onp.linspace(sigma * (1.0 + 1e-9), rmax, n)
    w = onp.asarray(evaluate(pair, np.asarray(r ** 2), sigma))
    shell = {1: 2.0 * onp.ones_like(r),
             2: 2.0 * onp.pi * r,
             3: 4.0 * onp.pi * r ** 2}[d]
    return float(onp.trapezoid(w * shell, r))
