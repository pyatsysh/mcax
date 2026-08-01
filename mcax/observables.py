"""Fluctuation response functions and pair correlations from a muVT run.

The engine cannot be differentiated. Accept/reject is discrete, the log-density
of a hard particle is an indicator with no gradient anywhere, and the ensemble
is trans-dimensional, so no amount of JAX makes d<rho>/d(beta mu) fall out of an
autodiff pass. That is permanent and is stated as such in the README.

What replaces it is exact rather than approximate. In the grand ensemble a
derivative with respect to beta mu IS a covariance with the particle number,

    d<A> / d(beta mu)  =  <A N> - <A> <N>,

for any observable A, because differentiating z^N / N! brings down a factor of
N. Two cases carry almost all of the value:

    A = N       ->   d<N>/d(beta mu) = Var(N)                  (a susceptibility)
    A = rho(z)  ->   d rho(z)/d(beta mu) = <rho(z) N> - rho(z) <N>

The first needs nothing but the N-series a run already returns. The second needs
the density-number cross-covariance accumulated during the run, which is why
`run` carries a second histogram alongside the first. Neither costs a single
extra sweep of the sampler: these are estimators, computed from samples already
drawn, not new simulations.

The reduced compressibility ties them to the equation of state,

    Var(N) / <N>  =  rho kT chi_T  =  S(0)  =  1 + rho int [g(r) - 1] dr,

and `mcax.eos.compressibility` gives the exact value to check against. That
identity is the most useful test in the library: it closes a loop between the
fluctuations, the pressure and the pair correlations, so a single number
disagreeing means one of the three is wrong and the other two say which.

Error bars come from the chain batch. Chains are independent by construction
(separate PRNG keys), so the scatter of a per-chain estimate across chains is a
legitimate standard error with no autocorrelation modelling in it at all. That
is worth more here than for a mean: a variance is a noisier estimator than a
mean of the same series, and quoting one without an error bar invites the reader
to over-read it.
"""
import numpy as onp

from .core import _default_rmax


def _as_chains(x):
    """Coerce to a 2-D (chain, draw) float array."""
    a = onp.asarray(x, dtype = float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected (chain, draw) or (draw,), got shape {a.shape}")
    return a


def _chain_mean_and_error(per_chain):
    """(mean, standard error) of a per-chain estimate, across chains."""
    v = onp.asarray(per_chain, dtype = float)
    m = float(v.mean())
    if v.size < 2:
        return m, float("nan")
    return m, float(v.std(ddof = 1) / onp.sqrt(v.size))


def susceptibility(Ns):
    """d<N>/d(beta mu) = Var(N), with an error bar. `Ns` is `result.Ns`.

    The variance is taken WITHIN each chain and then averaged, not pooled over
    the whole batch. The two differ by exactly the between-chain scatter, which
    is the numerator of R-hat: pooling would quietly fold a convergence failure
    into the physics and report a too-large susceptibility for a batch that had
    simply not met yet. Check `split_rhat` and read this second.
    """
    a = _as_chains(Ns)
    return _chain_mean_and_error(a.var(axis = 1, ddof = 1))


def compressibility(Ns):
    """rho kT chi_T = Var(N)/<N> = S(0), with an error bar.

    Compare against `mcax.eos.compressibility(d, rho)`, which is exact for rods
    and good to about 0.1% for discs and spheres.
    """
    a = _as_chains(Ns)
    per_chain = a.var(axis = 1, ddof = 1) / onp.maximum(a.mean(axis = 1), 1e-300)
    return _chain_mean_and_error(per_chain)


def response_profile(spec, result):
    """(z, d rho(z)/d(beta mu), error) from the density-number covariance.

    The local counterpart of `susceptibility`, and the one that matters for
    DFT: it is the response of the whole profile to the reservoir, so it says
    which layer of a confined fluid fills first as the chemical potential is
    raised. Its integral over the box must come back to Var(N), which
    `tests/test_observables.py` asserts and which is the cheapest way to catch a
    normalisation slip.
    """
    h = onp.asarray(result.hist, dtype = float)             # (C, nbins)
    hn = onp.asarray(result.hist_n, dtype = float)          # (C, nbins)
    n = _as_chains(result.Ns).mean(axis = 1)                # (C,)
    nsamples = result.nsamples
    dz = spec.H / h.shape[-1]
    cell = spec.Lperp ** (spec.d - 1) * dz                  # bin volume

    cov = hn / nsamples - (h / nsamples) * n[:, None]       # per chain
    chi = cov.mean(axis = 0) / cell
    err = (cov.std(axis = 0, ddof = 1) / onp.sqrt(cov.shape[0]) / cell
           if cov.shape[0] > 1 else onp.full(chi.shape, onp.nan))
    z = (onp.arange(h.shape[-1]) + 0.5) * dz
    return z, chi, err


def pair_correlation(spec, result, rmax = None):
    """(r, g(r), error) from the accumulated pair-distance histogram.

    Normalised with <N>^2, which is the grand-canonical definition:

        <n_pairs(r)>  =  (1/2) <N>^2 g(r) V_shell(r) / V.

    The canonical <N(N-1)> normalisation, which is what most closed-system
    codes use, is WRONG here and wrong in an invisible way. It forces
    rho int [g - 1] dr to zero whenever the shells tile the minimum-image
    cell, which sets S(0) to exactly 1 at every density and quietly destroys
    the one identity these estimators exist to check. With <N>^2 the tail sits
    at 1 + (Var(N) - <N>)/<N>^2 instead of at 1, and that offset is not an
    artefact: it IS the compressibility, seen at order 1/N.

    The shell volume is the d-dimensional one: 2 dr for rods, 2 pi r dr for
    discs, 4 pi r^2 dr for spheres.

    Only meaningful in `bulk`, and `run` refuses to accumulate it otherwise:
    against a wall the pair distribution depends on both centres separately,
    g(r1, r2), and averaging that over the profile gives a number that is not
    the g(r) anybody means.
    """
    g = onp.asarray(result.gr, dtype = float)               # (C, nbins_g)
    nb = g.shape[-1]
    if nb == 0:
        raise ValueError("no pair histogram in this result: re-run with "
                         "nbins_g > 0")
    nsamples = result.nsamples
    rmax = _default_rmax(spec) if rmax is None else rmax
    dr = rmax / nb
    r = (onp.arange(nb) + 0.5) * dr

    if spec.d == 1:
        shell = onp.full(nb, 2.0 * dr)
    elif spec.d == 2:
        shell = 2.0 * onp.pi * r * dr
    else:
        shell = 4.0 * onp.pi * r ** 2 * dr

    vol = spec.H * spec.Lperp ** (spec.d - 1)
    n2 = _as_chains(result.Ns).mean(axis = 1) ** 2          # <N>^2 per chain
    norm = onp.maximum(n2, 1e-300)[:, None] * shell[None, :] / (2.0 * vol)
    per_chain = (g / nsamples) / norm
    mean = per_chain.mean(axis = 0)
    err = (per_chain.std(axis = 0, ddof = 1) / onp.sqrt(per_chain.shape[0])
           if per_chain.shape[0] > 1 else onp.full(mean.shape, onp.nan))
    return r, mean, err


def structure_factor_zero(spec, result, rho, rmax = None):
    """S(0) = 1 + rho int [g(r) - 1] dr, the compressibility route.

    The same number `compressibility(Ns)` estimates from the N-series, by a
    different path: this one integrates the pair correlations and never looks
    at the particle number, that one looks at nothing else. They agree only if
    the sampler and both normalisations are right.

    How tight the agreement should be depends on the dimension, and it is worth
    knowing which regime you are in before reading a discrepancy. For rods the
    shells tile the minimum-image cell exactly, so with rmax = L/2 the two
    routes coincide algebraically and any disagreement beyond a per cent is a
    bug. For discs and spheres the ball of radius L/2 does not fill the cubic
    cell, so pairs beyond rmax are missing from the integral and S(0) comes out
    low by a finite-size deficit of a few per cent.
    """
    r, g, _ = pair_correlation(spec, result, rmax = rmax)
    dr = r[1] - r[0] if len(r) > 1 else 0.0
    if spec.d == 1:
        shell = 2.0
    elif spec.d == 2:
        shell = 2.0 * onp.pi * r
    else:
        shell = 4.0 * onp.pi * r ** 2
    return 1.0 + rho * float(onp.sum((g - 1.0) * shell) * dr)


def summary(spec, result, rho_target = None):
    """Everything the fluctuations say, as a dict, for logs and scripts."""
    chi_n, chi_n_err = susceptibility(result.Ns)
    kappa, kappa_err = compressibility(result.Ns)
    out = {
        "n_mean": float(onp.mean(result.Ns)),
        "susceptibility": chi_n,
        "susceptibility_err": chi_n_err,
        "compressibility": kappa,
        "compressibility_err": kappa_err,
    }
    if rho_target is not None:
        from . import eos
        out["compressibility_exact"] = float(eos.compressibility(spec.d,
                                                                 rho_target))
    return out
