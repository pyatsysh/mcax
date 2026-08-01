"""Reference equations of state for hard rods, discs and spheres.

The pressure of a hard-particle fluid at packing fraction eta is written as a
compressibility factor Z = beta P / rho:

    d = 1   Z = 1 / (1 - eta)                        Tonks, EXACT
    d = 2   Z = (1 + eta^2/8) / (1 - eta)^2          Henderson
    d = 3   Z = (1 + eta + eta^2 - eta^3)/(1-eta)^3  Carnahan-Starling

Everything else here follows from the excess free energy that integrates these.
Write beta f_ex = rho g(eta), so that g is the excess free energy PER PARTICLE.
Integrating (Z - 1)/eta gives g in closed form in every dimension, and then

    beta mu_ex = d(f_ex)/d(rho) = g + eta g'(eta),     Z = 1 + eta g'(eta),

because rho d/d(rho) = eta d/d(eta) at fixed sigma. So one pair of functions per
dimension, `_g` and `_eta_dg`, generates the whole module, and `p_of_rho`
reproduces the three Z expressions above identically rather than by quadrature.

The d = 2 integral is elementary once the integrand is simplified:

    (Z - 1)/eta = (2 - 7 eta/8) / (1 - eta)^2
    =>  g(eta)  = (9/8) eta/(1 - eta) - (7/8) ln(1 - eta)

which is worth writing down because the obvious move (quadrature, then finite
differences for mu) costs about 200x more and is accurate to 1e-7 rather than to
roundoff.

These are the TARGETS mcax is validated against, and they are also what you
need to set a run up: the sampler takes an activity z = exp(beta mu), but a
physicist thinks in packing fraction, so mu_of_rho / rho_of_mu are the
translation layer.

Accuracy by dimension, which matters when reading a validation tolerance:

  d = 1 (rods)    Tonks/Percus, EXACT, in closed form. This is why the 1-D case
                  is kept: a detailed-balance or bookkeeping error has nowhere
                  to hide. `sw1d` below extends that guarantee to the attractive
                  square well.
  d = 2 (discs)   Henderson, good to about 0.1% over the stable range. Note
                  that scaled-particle theory is NOT used here: its mu-inversion
                  is about 1.1% off at eta = 0.4, which would eat half the error
                  budget of any check built on it.
  d = 3 (spheres) Carnahan-Starling, good to about 0.1% up to freezing.

All expressions are dimensionless: beta = 1, Lambda = 1, lengths in units of
the hard-core diameter sigma, densities per unit d-volume. Attraction strengths
are therefore beta epsilon, and temperature enters only through them.
"""
import numpy as onp

# eta = B[d] * rho at sigma = 1: the packing fraction per unit density.
B = {1: 1.0, 2: onp.pi / 4.0, 3: onp.pi / 6.0}


def _check_d(d):
    if d not in B:
        raise ValueError(f"d must be 1, 2 or 3, got {d}")
    return d


def _g(d, eta):
    """Excess free energy per particle, beta f_ex / rho, as a function of eta."""
    if d == 1:
        return -onp.log1p(-eta)
    if d == 2:
        return 1.125 * eta / (1.0 - eta) - 0.875 * onp.log1p(-eta)
    return (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2


def _eta_dg(d, eta):
    """eta g'(eta), which is both Z - 1 and the excess part of mu beyond g."""
    if d == 1:
        return eta / (1.0 - eta)
    if d == 2:
        return eta * (1.125 / (1.0 - eta) ** 2 + 0.875 / (1.0 - eta))
    return eta * (4.0 - 2.0 * eta) / (1.0 - eta) ** 3


def f_ex(d, rho):
    """Excess free energy density beta f_ex (per unit d-volume)."""
    _check_d(d)
    rho = onp.asarray(rho, dtype = float)
    return rho * _g(d, B[d] * rho)


def _dfex(d, rho):
    """d(f_ex)/d(rho) = beta mu_ex, in closed form in every dimension."""
    _check_d(d)
    eta = B[d] * onp.asarray(rho, dtype = float)
    return _g(d, eta) + _eta_dg(d, eta)


def mu_of_rho(d, rho):
    """Chemical potential beta mu at density rho (ideal plus excess)."""
    return onp.log(rho) + _dfex(d, rho)


def z_of_rho(d, rho):
    """Activity z = exp(beta mu), which is what `mcax.make_spec` wants.

    For rods this is z = rho/(1-rho) exp(rho/(1-rho)), exactly.
    """
    return onp.exp(mu_of_rho(d, rho))


def p_of_rho(d, rho):
    """Pressure beta P = rho (1 + eta g'), i.e. rho times the Z above."""
    _check_d(d)
    rho = onp.asarray(rho, dtype = float)
    return rho * (1.0 + _eta_dg(d, B[d] * rho))


def _d_eta_Z(d, eta):
    """d(eta Z)/d(eta), which is d(beta P)/d(rho) because rho = eta / B[d].

    Written out per dimension rather than differentiated from `_eta_dg`,
    and then checked against a finite difference of `p_of_rho` in the tests,
    which is what stops the two drifting apart. The d = 3 form
    (1 + 4eta + 4eta^2 - 4eta^3 + eta^4)/(1-eta)^4 is the usual
    Carnahan-Starling compressibility.
    """
    if d == 1:
        return 1.0 / (1.0 - eta) ** 2
    if d == 2:
        return (1.0 + eta + 0.375 * eta ** 2 - 0.125 * eta ** 3) \
            / (1.0 - eta) ** 3
    return (1.0 + 4.0 * eta + 4.0 * eta ** 2 - 4.0 * eta ** 3 + eta ** 4) \
        / (1.0 - eta) ** 4


def dp_drho(d, rho):
    """d(beta P)/d(rho): the inverse of the reduced compressibility."""
    _check_d(d)
    return _d_eta_Z(d, B[d] * onp.asarray(rho, dtype = float))


def compressibility(d, rho):
    """rho kT chi_T, which a muVT run measures as Var(N) / <N>.

    This is the exact target for the fluctuation estimators in
    `mcax.observables`. The chain is short and worth stating: the grand
    ensemble gives d<N>/d(beta mu) = Var(N), Gibbs-Duhem gives
    d(beta P)/d(rho) = rho d(beta mu)/d(rho), and together

        Var(N) / <N>  =  [d(beta P)/d(rho)]^{-1}  =  rho kT chi_T  =  S(0),

    so one number checks the sampler's fluctuations, its equation of state and
    (through S(0) = 1 + rho int [g(r) - 1] dr) its pair correlations at once.
    For rods it is exactly (1 - rho)^2.
    """
    return 1.0 / dp_drho(d, rho)


def rho_of_mu(d, mu, lo = 1e-9, hi = None):
    """Invert `mu_of_rho` by bisection. mu is monotone in rho, so this always
    converges; 200 halvings take the bracket well below any tolerance we use."""
    _check_d(d)
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

    The sum rule constrains the WALL, not the fluid: it holds whatever the
    particles do to each other, so it survives switching an attraction on, and
    remains a check on any geometry whose confinement is purely hard. It does
    NOT survive an attractive external field, where the general sum rule
    picks up an integral of rho grad V_ext instead.
    """
    return p_of_rho(d, rho)


# ---- the attractive razor: 1-D square well, exactly ----------------------- #
#
# Everything above is hard-particle only, and only d = 1 of it is exact. Turning
# an attraction on would therefore cost the razor, except that the 1-D fluid
# with NEAREST-NEIGHBOUR interactions is also exactly solvable (Takahashi's
# transfer-integral construction in the isobaric ensemble). A square well of
# range lambda <= 2 qualifies, because two particles that are not nearest
# neighbours are separated by at least two hard cores and so never see each
# other's well. The whole solution is one Laplace transform:
#
#     Omega(p) = int_0^inf dr exp(-beta u(r) - p r),        p = beta P
#     beta mu  = -ln Omega(p)
#     1 / rho  = -d ln Omega / dp
#
# parametric in the pressure. At eps = 0 it collapses to Tonks, which is the
# first thing the tests check.


def _sw1d_reduced(eps, lam, p):
    """(A_hat, A_hat') where Omega(p) = exp(-p) A_hat(p) / p.

    Pulling the hard-core factor exp(-p) out is not cosmetic. Written directly,
    Omega is a difference of exponentials that both underflow to zero by about
    p = 700, so the high-pressure end of the bisection evaluates 0/0 and takes
    the log of zero. The reduced form

        A_hat = e^eps + (1 - e^eps) e^{-(lam-1) p}

    is monotone between 1 and e^eps and therefore bounded away from zero for
    every p and either sign of eps, so the same bisection runs to p = 1e6
    without a single warning.
    """
    w = onp.exp(-(lam - 1.0) * p)
    ae = onp.exp(eps)
    return ae + (1.0 - ae) * w, -(lam - 1.0) * (1.0 - ae) * w


def sw1d_state(eps, lam, p):
    """(rho, beta mu) of the 1-D square-well fluid at pressure beta P = p.

    `eps` is beta epsilon, positive for attraction. In the reduced variables

        1/rho = 1 + 1/p - A_hat'/A_hat,      beta mu = p + ln p - ln A_hat

    and setting eps = 0 gives A_hat = 1, hence 1/rho = 1 + 1/p and
    beta mu = p + ln p, which is Tonks. The tests start there.
    """
    if not 1.0 <= lam <= 2.0:
        raise ValueError(f"lambda must lie in [1, 2] for the nearest-neighbour "
                         f"solution to be exact, got {lam}")
    p = onp.asarray(p, dtype = float)
    a, da = _sw1d_reduced(eps, lam, p)
    ell = 1.0 + 1.0 / p - da / a                    # length per particle
    return 1.0 / ell, p + onp.log(p) - onp.log(a)


def sw1d_p_of_mu(eps, lam, mu, lo = 1e-12, hi = 1e6):
    """Invert beta mu(p) by bisection. d(mu)/dp is the length per particle, so
    mu is strictly increasing in p and the bracket never fails."""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if sw1d_state(eps, lam, mid)[1] < mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sw1d_p_of_rho(eps, lam, rho, lo = 1e-12, hi = 1e6):
    """Invert rho(p) by bisection; rho increases with pressure."""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if sw1d_state(eps, lam, mid)[0] < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sw1d_mu_of_rho(eps, lam, rho):
    """beta mu at density rho: the setter for an attractive 1-D run."""
    return float(sw1d_state(eps, lam, sw1d_p_of_rho(eps, lam, rho))[1])


def sw1d_z_of_rho(eps, lam, rho):
    """Activity for `make_spec`, the attractive counterpart of `z_of_rho`."""
    return float(onp.exp(sw1d_mu_of_rho(eps, lam, rho)))


def sw1d_rho_of_z(eps, lam, z):
    """Density at activity z: the exact target for an attractive muVT run."""
    p = sw1d_p_of_mu(eps, lam, float(onp.log(z)))
    return float(sw1d_state(eps, lam, p)[0])
