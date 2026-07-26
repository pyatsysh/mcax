"""Diagnostics checked against processes whose answers are known analytically.

The point of these tests: an ESS estimator that is quietly wrong is worse than
none at all, because it manufactures confidence. So each case here has a
closed-form target rather than a regression value.
"""
import numpy as onp
import pytest

from mcax.diagnostics import (split_rhat, ess, mcse, summary, format_summary,
                              _as_chains)


def _ar1(r, chains, draws, seed=0):
    """AR(1) with unit stationary variance: tau = (1 + r) / (1 - r)."""
    rng = onp.random.default_rng(seed)
    x = onp.zeros((chains, draws))
    s = onp.sqrt(1.0 - r * r)
    for t in range(1, draws):
        x[:, t] = r * x[:, t - 1] + s * rng.standard_normal(chains)
    return x


# ---- ESS ------------------------------------------------------------------ #

def test_ess_of_iid_is_almost_all_draws():
    x = onp.random.default_rng(1).standard_normal((8, 2000))
    assert ess(x) > 0.9 * x.size


@pytest.mark.parametrize("r,draws", [(0.5, 8000), (0.8, 8000), (0.9, 20000)])
def test_ess_recovers_ar1_autocorrelation_time(r, draws):
    """total / ESS must recover tau = (1 + r) / (1 - r).

    Geyer truncation biases tau slightly low on finite chains, so the window is
    asymmetric: generous below, tight above. Over-estimating ESS is the
    dangerous direction and is held to 10%.
    """
    x = _ar1(r, 8, draws, seed=2)
    tau_true = (1.0 + r) / (1.0 - r)
    tau_est = x.size / ess(x)
    assert 0.80 * tau_true < tau_est < 1.10 * tau_true


def test_ess_never_exceeds_total_draws_for_correlated_data():
    x = _ar1(0.7, 4, 4000, seed=3)
    assert ess(x) < x.size


def test_ess_nan_for_too_few_draws():
    assert onp.isnan(ess(onp.zeros((4, 3))))


# ---- split R-hat ---------------------------------------------------------- #

def test_rhat_of_converged_chains_is_one():
    x = onp.random.default_rng(4).standard_normal((8, 2000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)


def test_rhat_detects_offset_chains():
    rng = onp.random.default_rng(5)
    x = rng.standard_normal((4, 1000)) + onp.arange(4)[:, None] * 3.0
    assert split_rhat(x) > 1.5


def test_rhat_detects_drift_within_a_single_chain():
    """The reason for *split* R-hat: one drifting chain has no cross-chain
    scatter to give it away, but its two halves disagree."""
    rng = onp.random.default_rng(6)
    x = onp.linspace(0.0, 10.0, 2000)[None, :] + 0.1 * rng.standard_normal((1, 2000))
    assert split_rhat(x) > 1.5


def test_rhat_of_constant_chains_is_one():
    assert split_rhat(onp.full((4, 100), 7.0)) == pytest.approx(1.0)


def test_rhat_of_chains_stuck_at_different_constants_is_infinite():
    x = onp.repeat(onp.arange(4.0)[:, None], 100, axis=1)
    assert not onp.isfinite(split_rhat(x))


def test_rhat_nan_for_too_few_draws():
    assert onp.isnan(split_rhat(onp.zeros((4, 2))))


# ---- MCSE ----------------------------------------------------------------- #

def test_mcse_of_iid_matches_standard_error():
    x = onp.random.default_rng(7).standard_normal((8, 2000))
    assert mcse(x) == pytest.approx(1.0 / onp.sqrt(x.size), rel=0.15)


def test_mcse_grows_with_correlation():
    """Same number of draws, more correlation, larger error on the mean."""
    lo = mcse(_ar1(0.2, 8, 4000, seed=8))
    hi = mcse(_ar1(0.9, 8, 4000, seed=8))
    assert hi > 2.0 * lo


# ---- shape handling and summary ------------------------------------------- #

def test_one_dimensional_input_is_one_chain():
    assert _as_chains(onp.arange(10.0)).shape == (1, 10)


def test_three_dimensional_input_rejected():
    with pytest.raises(ValueError):
        _as_chains(onp.zeros((2, 3, 4)))


def test_summary_fields_and_formatting():
    x = onp.random.default_rng(9).standard_normal((6, 500))
    s = summary(x, name="N")
    assert s["chains"] == 6 and s["draws"] == 500
    assert set(s) == {"name", "chains", "draws", "mean", "sd", "mcse",
                      "ess", "ess_per_draw", "split_rhat"}
    assert 0.0 < s["ess_per_draw"] <= 1.0
    assert "split-Rhat" in format_summary(s)


def test_arviz_layout_contract():
    """The interop promise: (chain, draw) in, and we never transpose it.

    If this ever flips, ArviZ and NumPyro would silently read draws as chains
    and report nonsense, so pin it: 3 chains of 100 stays 3 chains of 100.
    """
    s = summary(onp.zeros((3, 100)) + onp.arange(3)[:, None])
    assert (s["chains"], s["draws"]) == (3, 100)
