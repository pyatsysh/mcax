"""Fluctuation response functions and pair correlations.

Two of these are exact rather than statistical, and they are the ones worth
reading first. ``test_response_profile_integrates_to_the_variance`` is an
algebraic identity between two accumulators of the same run, so it holds to
floating point whatever the physics did, and it catches every normalisation
slip in the covariance path. ``test_ideal_gas_number_is_poisson`` pins the
susceptibility against a known distribution with the overlap test switched off.

The rest are statistical and deliberately loose here; the tight versions of the
same checks live in ``test_validation.py`` behind the ``slow`` marker.
"""
import numpy as onp
import pytest

from mcax import make_spec, burn_and_sample, observables, eos


def _bulk(d, rho, **kw):
    return make_spec(d=d, z_act=float(eos.z_of_rho(d, rho)), slit=False, **kw)


# ---- the exact identities ------------------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3])
def test_response_profile_integrates_to_the_variance(d):
    """int d rho(z)/d(beta mu) dV = Var(N), exactly, not statistically.

    Both sides come from the same draws: the left is the density-number
    covariance summed over bins, the right is the variance of the number
    itself, and every particle lands in exactly one bin. So the identity is
    algebra and any disagreement is a bug in the normalisation, never noise.
    """
    spec = _bulk(d, 0.25 / eos.B[d], H=6.0, Lperp=6.0)
    res = burn_and_sample(spec, C=4, seed=0, n_burn=2_000, n_run=8_000,
                          thin=200, nbins=12)
    z, chi, err = observables.response_profile(spec, res)

    cell = spec.Lperp ** (spec.d - 1) * (spec.H / len(z))
    lhs = float((chi * cell).sum())
    rhs = float(res.Ns.var(axis=1, ddof=0).mean())
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_response_profile_is_flat_in_bulk():
    """Nothing breaks translational symmetry, so the response is uniform too."""
    spec = _bulk(1, 0.4, H=20.0)
    res = burn_and_sample(spec, C=16, seed=3, n_burn=20_000, n_run=60_000,
                          thin=200, nbins=10)
    _, chi, _ = observables.response_profile(spec, res)
    assert chi.std() / chi.mean() < 0.25


@pytest.mark.parametrize("d", [1, 2, 3])
def test_ideal_gas_number_is_poisson(d):
    """With no core, N is Poisson: Var(N) = <N>, so the compressibility is 1.

    The overlap test is out of the picture at sigma = 1e-6, which leaves the
    muVT acceptances and the estimator alone in the frame.
    """
    L = {1: 12.0, 2: 5.0, 3: 3.5}[d]
    z = 0.7
    spec = make_spec(d=d, H=L, z_act=z, Lperp=L, sigma=1e-6, slit=False,
                     Nmax=int(3 * z * L ** d) + 20)
    res = burn_and_sample(spec, C=16, seed=1, n_burn=6_000, n_run=40_000,
                          thin=100, nbins=8)
    kappa, err = observables.compressibility(res.Ns)
    assert kappa == pytest.approx(1.0, rel=0.08), f"Var(N)/<N> = {kappa}"
    assert 0.0 < err < 0.2

    chi_n, _ = observables.susceptibility(res.Ns)
    assert chi_n == pytest.approx(res.Ns.mean(), rel=0.08)


# ---- against the equation of state ---------------------------------------- #

def test_compressibility_of_rods_matches_tonks():
    """rho kT chi_T = (1 - rho)^2 for hard rods, which is exact.

    Loose here because Var(N) is a noisier estimator than <N> and this is the
    fast tier; ``test_validation.py`` runs the same check to 5%.
    """
    rho = 0.4
    spec = _bulk(1, rho, H=30.0)
    res = burn_and_sample(spec, C=32, seed=11, n_burn=40_000, n_run=120_000,
                          thin=200, nbins=8)
    kappa, err = observables.compressibility(res.Ns)
    exact = float(eos.compressibility(1, rho))
    assert kappa == pytest.approx(exact, rel=0.20), (
        f"measured {kappa:.4f} +/- {err:.4f} against exact {exact:.4f}")


def test_attraction_would_raise_the_compressibility_of_rods():
    """A sanity check on the exact targets themselves, with no sampling.

    An attraction makes the fluid easier to compress at fixed density, so the
    square-well compressibility must exceed the hard-rod one. This is the
    property the attractive runs will be validated on.
    """
    rho, h = 0.4, 1e-5
    dp = (eos.sw1d_p_of_rho(1.0, 1.5, rho + h)
          - eos.sw1d_p_of_rho(1.0, 1.5, rho - h)) / (2 * h)
    assert 1.0 / dp > float(eos.compressibility(1, rho))


# ---- pair correlations ---------------------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3])
def test_g_of_r_vanishes_inside_the_hard_core(d):
    """No pair may sit closer than sigma, so g(r < sigma) is identically zero."""
    spec = _bulk(d, 0.25 / eos.B[d], H=8.0, Lperp=8.0)
    res = burn_and_sample(spec, C=4, seed=2, n_burn=4_000, n_run=12_000,
                          thin=300, nbins=8, nbins_g=80)
    r, g, err = observables.pair_correlation(spec, res)
    inside = r < spec.sigma - 0.02
    assert inside.sum() > 3, "need some bins inside the core to test"
    assert onp.all(g[inside] == 0.0)


@pytest.mark.parametrize("d,eta,L", [(1, 0.3, 40.0), (2, 0.3, 14.0),
                                     (3, 0.2, 6.0)])
def test_g_of_r_tends_to_one_at_large_separation(d, eta, L):
    """The tail sits at 1 + (Var(N) - <N>)/<N>^2, so it approaches 1 only as
    the box grows. Boxes here hold <N> of order 12, 75 and 82, which puts the
    offset well inside the tolerance; a canonical normalisation would pin the
    tail at 1 exactly and pass this while failing the S(0) check below."""
    spec = _bulk(d, eta / eos.B[d], H=L, Lperp=L)
    res = burn_and_sample(spec, C=8, seed=4, n_burn=6_000, n_run=30_000,
                          thin=300, nbins=8, nbins_g=60)
    r, g, err = observables.pair_correlation(spec, res)
    tail = g[r > 0.8 * r.max()]
    assert tail.mean() == pytest.approx(1.0, abs=0.15), f"tail {tail.mean()}"


def test_g_of_r_has_a_contact_peak():
    """Hard particles pile up at contact: g(sigma+) > 1 at any real density."""
    spec = _bulk(1, 0.5, H=20.0)
    res = burn_and_sample(spec, C=8, seed=6, n_burn=20_000, n_run=60_000,
                          thin=200, nbins=8, nbins_g=100)
    r, g, err = observables.pair_correlation(spec, res)
    contact = g[(r > spec.sigma) & (r < 1.3 * spec.sigma)]
    assert contact.max() > 1.2


def test_structure_factor_agrees_with_the_number_fluctuations():
    """S(0) two ways: integrate g(r) - 1, or take Var(N)/<N>.

    In one dimension the shells tile the minimum-image cell exactly, so with
    rmax = L/2 these two routes are algebraically the same number and the
    tolerance can be tight. That is what makes this the sharpest test in the
    file: it fails outright if g(r) is normalised the canonical way, which pins
    S(0) at 1 whatever the density.
    """
    rho = 0.4
    spec = _bulk(1, rho, H=30.0)
    res = burn_and_sample(spec, C=16, seed=8, n_burn=30_000, n_run=90_000,
                          thin=200, nbins=8, nbins_g=120)
    s0_pairs = observables.structure_factor_zero(spec, res, rho)
    s0_number, _ = observables.compressibility(res.Ns)
    assert s0_pairs == pytest.approx(s0_number, rel=0.05), (
        f"S(0) from g(r) = {s0_pairs:.4f}, from Var(N) = {s0_number:.4f}")


def test_structure_factor_in_three_dimensions_survives_the_truncation():
    """The same check where the ball of radius L/2 does not fill the cell.

    Pairs between L/2 and the cell corner are never histogrammed, so the
    integral is missing a shell. Which way that pushes S(0) is not obvious and
    not worth asserting: the tail of g sits slightly BELOW one for a repulsive
    fluid, so the missing region carries negative weight and the truncated
    integral runs high, but how high depends on how far the corners are from
    the asymptote. Measured at eta = 0.2 in a box of 7 sigma it is about 5%.
    """
    eta = 0.2
    rho = eta / eos.B[3]
    spec = _bulk(3, rho, H=7.0, Lperp=7.0)
    res = burn_and_sample(spec, C=8, seed=9, n_burn=20_000, n_run=60_000,
                          thin=300, nbins=8, nbins_g=80)
    s0_pairs = observables.structure_factor_zero(spec, res, rho)
    s0_number, _ = observables.compressibility(res.Ns)
    assert s0_pairs == pytest.approx(s0_number, rel=0.25), (
        f"S(0) from g(r) = {s0_pairs:.4f}, from Var(N) = {s0_number:.4f}")


# ---- refusals and shapes -------------------------------------------------- #

def test_pair_correlation_refuses_a_slit():
    """g(r) is not a function of r alone against a wall, so `run` will not
    pretend to accumulate one."""
    spec = make_spec(d=1, H=8.0, z_act=1.0, slit=True)
    with pytest.raises(ValueError, match="bulk"):
        burn_and_sample(spec, C=2, seed=0, n_burn=200, n_run=400, thin=100,
                        nbins=8, nbins_g=20)


def test_pair_correlation_refuses_a_result_without_one():
    spec = _bulk(1, 0.3, H=10.0)
    res = burn_and_sample(spec, C=2, seed=0, n_burn=500, n_run=1_000,
                          thin=200, nbins=8)
    with pytest.raises(ValueError, match="nbins_g"):
        observables.pair_correlation(spec, res)


def test_single_chain_reports_nan_rather_than_a_fake_error_bar():
    """One chain gives no cross-chain scatter, so there is no honest error to
    quote. Returning nan is better than returning zero."""
    spec = _bulk(1, 0.3, H=10.0)
    res = burn_and_sample(spec, C=1, seed=0, n_burn=1_000, n_run=4_000,
                          thin=200, nbins=8)
    _, err = observables.susceptibility(res.Ns)
    assert onp.isnan(err)


def test_summary_carries_the_exact_target_when_given_one():
    spec = _bulk(1, 0.3, H=12.0)
    res = burn_and_sample(spec, C=4, seed=0, n_burn=2_000, n_run=8_000,
                          thin=200, nbins=8)
    s = observables.summary(spec, res, rho_target=0.3)
    assert set(s) >= {"n_mean", "susceptibility", "compressibility",
                      "compressibility_exact"}
    assert s["compressibility_exact"] == pytest.approx(0.49, rel=1e-9)
