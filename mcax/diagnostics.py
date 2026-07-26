"""Convergence diagnostics for the chain batch: split R-hat, ESS, MCSE.

The estimator behind ESS is the integrated autocorrelation time

    tau = 1 + 2 sum_{t>=1} rho_t,        ESS = (chains * draws) / tau

summed by Geyer's initial positive sequence: walk the lags in PAIRS, stop at
the first pair that does not sum positive, and clamp the pair sequence to be
non-increasing. The pairing matters because the odd and even lags of an AR
process are individually noisy but sum positively for as long as there is real
signal left; past that point the estimator is summing noise, and truncating is
what keeps ESS from being inflated. Definitions follow Stan (Vehtari et al.
2021, rank-normalisation, folding and localisation), so the numbers are
comparable with what ArviZ and NumPyro print. These are the CLASSICAL,
non-rank-normalised estimators, which is the honest choice for a particle
number: N is a bounded count, not a heavy-tailed posterior draw.

Deliberately dependency-free (old numpy only, host-side, on small arrays).
mcax runs many short independent chains rather than one long one, so the
diagnostics that matter are exactly the cross-chain ones, and they are cheap.

Why these live here and are not delegated to NumPyro or ArviZ: the natural
output of a batched sampler is a (chain, draw) array, which is precisely what
those libraries consume. So mcax produces that layout and stops. If you have
ArviZ, az.summary(az.convert_to_dataset(res.Ns[None])) works on the same array;
if you have NumPyro, so does its summary machinery. Interop by data shape costs
nothing, whereas depending on a probabilistic-programming stack to compute
forty lines of autocovariance would be a poor trade for a library whose install
is otherwise just JAX.

Rules of thumb for a hard-particle GCMC run:

  split_rhat above 1.01   the chains have not met. Usually the burn-in was too
                          short for them to forget a shared lattice prefill,
                          not that anything is wrong with the sampler.
  ess                     counts INDEPENDENT draws. Insertion acceptance falls
                          off a cliff with density (about 1% at eta = 0.3 in
                          3D), so at high activity N decorrelates slowly and
                          ESS per draw drops hard. Thin more, or add chains.

Chains are exactly independent by construction (separate PRNG keys), so unlike
a single-chain sampler, adding chains buys ESS linearly and for free.
"""
import numpy as onp


def _as_chains(x):
    """Coerce to a 2-D (chain, draw) float array."""
    a = onp.asarray(x, dtype = float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected (chain, draw) or (draw,), got shape {a.shape}")
    return a


def _autocov(v):
    """Biased autocovariance (divided by n) of a 1-D series, via FFT."""
    n = v.size
    c = v - v.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = onp.fft.rfft(c, nfft)
    ac = onp.fft.irfft(f * onp.conjugate(f), nfft)[:n]
    return ac / n


def split_rhat(x):
    """Split R-hat: the potential scale reduction over half-chains.

    Each chain is halved FIRST, so a single chain that drifts monotonically is
    caught; plain R-hat over whole chains would not see it, because a lone
    drifting chain has no cross-chain scatter to give it away. Returns nan if
    there are too few draws to split meaningfully.
    """
    a = _as_chains(x)
    m, n = a.shape
    h = n // 2
    if h < 2:
        return float("nan")
    s = onp.concatenate([a[:, :h], a[:, n - h:]], axis = 0)     # (2m, h)
    return _rhat_of(s)


def _rhat_of(s):
    m, n = s.shape
    within = s.var(axis = 1, ddof = 1)
    w = within.mean()
    if w <= 0.0:
        # Every chain constant: either perfectly converged, or every chain is
        # stuck somewhere different, which is the opposite. Tell them apart.
        return 1.0 if onp.allclose(s, s[0, 0]) else float("inf")
    b = n * s.mean(axis = 1).var(ddof = 1)
    var_plus = (n - 1) / n * w + b / n
    return float(onp.sqrt(var_plus / w))


def _rho_hat(a):
    """Cross-chain autocorrelation estimate, and the variance it is scaled by."""
    m, n = a.shape
    acov = onp.stack([_autocov(r) for r in a])                  # (m, n)
    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = float(chain_var.mean())
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += float(a.mean(axis = 1).var(ddof = 1))
    if var_plus <= 0.0:
        return onp.zeros(n), var_plus
    rho = 1.0 - (mean_var - acov.mean(axis = 0)) / var_plus
    rho[0] = 1.0
    return rho, var_plus


def ess(x):
    """Effective sample size over the whole batch (Stan's bulk-ESS estimator)."""
    a = _as_chains(x)
    m, n = a.shape
    if n < 4:
        return float("nan")
    rho, var_plus = _rho_hat(a)
    if var_plus <= 0.0:
        return float("nan")

    pairs = []
    t = 1
    while t + 1 < n - 4:
        p = rho[t] + rho[t + 1]
        if p <= 0.0:
            break
        pairs.append(p)
        t += 2

    if pairs:
        # Geyer's monotone estimator: the true pair sequence is non-increasing,
        # so clamp the estimate to be so too.
        mono = onp.minimum.accumulate(onp.asarray(pairs))
        # tau = -1 + 2 sum_{t>=0} rho_t, and the sum is rho_0 + sum(pairs) with
        # rho_0 = 1, hence 1 + 2 sum(pairs).
        tau = 1.0 + 2.0 * float(mono.sum())
    else:
        tau = 1.0

    total = m * n
    tau = max(tau, 1.0 / onp.log10(max(total, 11)))
    return float(total / tau)


def mcse(x):
    """Monte Carlo standard error of the mean: sd / sqrt(ESS)."""
    a = _as_chains(x)
    n_eff = ess(a)
    if not onp.isfinite(n_eff) or n_eff <= 0.0:
        return float("nan")
    return float(a.std(ddof = 1) / onp.sqrt(n_eff))


def summary(x, name = "N"):
    """Everything above, plus the mean and sd, as a dict.

    `x` is the (chain, draw) array: for a mcax run, `result.Ns`.
    """
    a = _as_chains(x)
    n_eff = ess(a)
    return {
        "name": name,
        "chains": a.shape[0],
        "draws": a.shape[1],
        "mean": float(a.mean()),
        "sd": float(a.std(ddof = 1)),
        "mcse": mcse(a),
        "ess": n_eff,
        "ess_per_draw": n_eff / a.size if a.size else float("nan"),
        "split_rhat": split_rhat(a),
    }


def format_summary(s):
    """One-line rendering of `summary`, for logs and scripts."""
    return (f"{s['name']}: mean {s['mean']:.4f} +/- {s['mcse']:.4f} "
            f"(sd {s['sd']:.4f}) ess {s['ess']:.0f}/{s['chains'] * s['draws']} "
            f"split-Rhat {s['split_rhat']:.4f}")
