"""Reference equations of state for hard rods, discs and spheres.

The pressure of a hard-particle fluid at packing fraction eta is written as a
compressibility factor Z = beta P / rho:

    d = 1   Z = 1 / (1 - eta)                        Tonks, EXACT
    d = 2   Z = (1 + eta^2/8) / (1 - eta)^2          Henderson
    d = 3   Z = (1 + eta + eta^2 - eta^3)/(1-eta)^3  Carnahan-Starling

and everything else here follows from the excess free energy that integrates
these, via mu = ln rho + d(f_ex)/d(rho) and beta P = rho + rho f' - f_ex.

These are the TARGETS mcax is validated against, and they are also what you
need to set a run up: the sampler takes an activity z = exp(beta mu), but a
physicist thinks in packing fraction, so mu_of_rho / rho_of_mu are the
translation layer.

Accuracy by dimension, which matters when reading a validation tolerance:

  d = 1 (rods)    Tonks/Percus, EXACT, in closed form. This is why the 1-D case
                  is kept: a detailed-balance or bookkeeping error has nowhere
                  to hide.
  d = 2 (discs)   Henderson, good to about 0.1% over the stable range. Note
                  that scaled-particle theory is NOT used here: its mu-inversion
                  is about 1.1% off at eta = 0.4, which would eat half the error
                  budget of any check built on it.
  d = 3 (spheres) Carnahan-Starling, good to about 0.1% up to freezing.

All expressions are dimensionless: beta = 1, Lambda = 1, lengths in units of
the hard-core diameter sigma, densities per unit d-volume.
"""
import numpy as onp

# eta = B[d] * rho at sigma = 1: the packing fraction per unit density.
B = {1: 1.0, 2: onp.pi / 4.0, 3: onp.pi / 6.0}


def f_ex(d, rho):
    """Excess free energy density beta f_ex (per unit d-volume)."""
    if d not in B:
        raise ValueError(f"d must be 1, 2 or 3, got {d}")
    eta = B[d] * onp.asarray(rho, dtype = float)
    if d == 1:
        return -rho * onp.log1p(-eta)
    if d == 2:
        # No closed form, so integrate (Z - 1)/eta d(eta). The integrand is
        # smooth and the quadrature is cheap and one-off.
        e = onp.atleast_1d(eta)
        out = onp.empty_like(e)
        for i, ei in enumerate(e):
            x = onp.linspace(1e-9, ei, 4000)
            zx = (1.0 + x ** 2 / 8.0) / (1.0 - x) ** 2
            out[i] = onp.trapezoid((zx - 1.0) / x, x)
        out = out.reshape(onp.shape(eta)) if onp.shape(eta) else out[0]
        return rho * out
    return rho * (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2


def _dfex(d, rho, h = 1e-7):
    """d(f_ex)/d(rho): analytic for rods, central difference otherwise.

    Rods get the closed form so the razor case carries no finite-difference
    error at all, which is the whole point of keeping d = 1.
    """
    if d == 1:
        return -onp.log1p(-rho) + rho / (1.0 - rho)
    return (f_ex(d, rho + h) - f_ex(d, rho - h)) / (2.0 * h)


def mu_of_rho(d, rho):
    """Chemical potential beta mu at density rho (ideal plus excess)."""
    return onp.log(rho) + _dfex(d, rho)


def z_of_rho(d, rho):
    """Activity z = exp(beta mu), which is what `mcax.make_spec` wants.

    For rods this is z = rho/(1-rho) exp(rho/(1-rho)), exactly.
    """
    return onp.exp(mu_of_rho(d, rho))


def p_of_rho(d, rho):
    """Pressure beta P. For rods this is exactly rho / (1 - rho)."""
    if d == 1:
        return rho / (1.0 - rho)
    return rho + rho * _dfex(d, rho) - f_ex(d, rho)


def rho_of_mu(d, mu, lo = 1e-9, hi = None):
    """Invert `mu_of_rho` by bisection. mu is monotone in rho, so this always
    converges; 200 halvings take the bracket well below any tolerance we use."""
    hi = hi if hi is not None else 0.95 / B[d]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mu_of_rho(d, mid) < mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rho_of_z(d, z, **kw):
    """Density at activity z, the inverse of `z_of_rho`."""
    return rho_of_mu(d, float(onp.log(z)), **kw)


def contact_density(d, rho):
    """Contact density at a hard wall, from the contact theorem.

    For hard particles against a hard wall the exact sum rule is
    rho(0+) = beta P, with CENTRE exclusion at the wall. That is the same
    convention mcax's slit geometry uses, so it applies literally rather than
    at a profile shifted by sigma/2.
    """
    return p_of_rho(d, rho)
