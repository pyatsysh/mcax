"""The reference equations of state, checked against closed forms and limits."""
import numpy as onp
import pytest

from mcax import eos


# ---- d = 1: Tonks is exact, so these are equalities not tolerances -------- #

@pytest.mark.parametrize("rho", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
def test_tonks_pressure_closed_form(rho):
    assert eos.p_of_rho(1, rho) == pytest.approx(rho / (1.0 - rho), rel=1e-12)


@pytest.mark.parametrize("rho", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
def test_tonks_activity_closed_form(rho):
    # z = rho/(1-rho) * exp(rho/(1-rho)) for hard rods at sigma = 1
    want = rho / (1.0 - rho) * onp.exp(rho / (1.0 - rho))
    assert eos.z_of_rho(1, rho) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("rho", [0.05, 0.25, 0.5, 0.75])
def test_tonks_thermodynamic_consistency(rho):
    """beta P = rho + rho f' - f_ex must agree with the closed form."""
    p = rho + rho * eos._dfex(1, rho) - eos.f_ex(1, rho)
    assert p == pytest.approx(rho / (1.0 - rho), rel=1e-10)


# ---- the closed forms integrate the equations of state they claim to ------ #
#
# `p_of_rho` and `mu_of_rho` are both built from the same pair of functions of
# eta, so checking one against the other proves only that the algebra is
# self-consistent. These two check them against something OUTSIDE the module:
# a quadrature of the literature Z, and the Gibbs-Duhem relation.

Z_REFERENCE = {
    1: lambda e: 1.0 / (1.0 - e),
    2: lambda e: (1.0 + e ** 2 / 8.0) / (1.0 - e) ** 2,
    3: lambda e: (1.0 + e + e ** 2 - e ** 3) / (1.0 - e) ** 3,
}


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.05, 0.2, 0.4, 0.6])
def test_fex_integrates_the_reference_compressibility(d, eta):
    """beta f_ex / rho = int_0^eta (Z - 1)/x dx, by independent quadrature.

    The point of keeping a quadrature in the TEST and not in the library: this
    is the derivation being checked, and doing it a second way is the check.
    Simpson on 200k points is far more accurate than the 1e-9 asserted here.
    """
    x = onp.linspace(1e-12, eta, 200_001)
    integrand = (Z_REFERENCE[d](x) - 1.0) / x
    want = onp.trapezoid(integrand, x)
    assert eos._g(d, eta) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.1, 0.3, 0.5])
def test_gibbs_duhem(d, eta):
    """dP/drho = rho dmu/drho, the relation that ties the two together."""
    rho, h = eta / eos.B[d], 1e-6
    dp = (eos.p_of_rho(d, rho + h) - eos.p_of_rho(d, rho - h)) / (2 * h)
    dmu = (eos.mu_of_rho(d, rho + h) - eos.mu_of_rho(d, rho - h)) / (2 * h)
    assert dp == pytest.approx(rho * dmu, rel=1e-6)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_compressibility_factor_is_exact_not_quadrature(d):
    """Z = 1 + eta g' must reproduce the reference to MACHINE precision.

    d = 2 used to come from a 4000-point trapezoid and was right to about 1e-7.
    Asserting 1e-14 here is what stops that regressing.
    """
    for eta in (0.05, 0.2, 0.4, 0.6):
        rho = eta / eos.B[d]
        assert eos.p_of_rho(d, rho) / rho == pytest.approx(
            Z_REFERENCE[d](eta), rel=1e-14)


# ---- ideal-gas limits, all dimensions ------------------------------------ #

@pytest.mark.parametrize("d", [1, 2, 3])
def test_ideal_gas_limit(d):
    """As rho -> 0 the fluid is ideal: z -> rho and beta P -> rho."""
    rho = 1e-8
    assert eos.z_of_rho(d, rho) == pytest.approx(rho, rel=1e-4)
    assert eos.p_of_rho(d, rho) == pytest.approx(rho, rel=1e-4)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_fex_vanishes_at_zero_density(d):
    assert eos.f_ex(d, 1e-10) == pytest.approx(0.0, abs=1e-8)


# ---- monotonicity and invertibility -------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3])
def test_mu_and_p_are_monotone(d):
    rho = onp.linspace(1e-4, 0.6 / eos.B[d], 60)
    mu = onp.array([eos.mu_of_rho(d, r) for r in rho])
    p = onp.array([eos.p_of_rho(d, r) for r in rho])
    assert onp.all(onp.diff(mu) > 0), "mu must increase with density"
    assert onp.all(onp.diff(p) > 0), "pressure must increase with density"


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.05, 0.2, 0.4])
def test_round_trip_rho_mu_rho(d, eta):
    rho = eta / eos.B[d]
    assert eos.rho_of_mu(d, eos.mu_of_rho(d, rho)) == pytest.approx(rho, rel=1e-6)
    assert eos.rho_of_z(d, eos.z_of_rho(d, rho)) == pytest.approx(rho, rel=1e-6)


# ---- literature spot values ---------------------------------------------- #

def test_carnahan_starling_compressibility_factor():
    """Z = P/(rho kT) = (1 + eta + eta^2 - eta^3)/(1 - eta)^3 for CS."""
    for eta in (0.1, 0.2, 0.3, 0.4):
        rho = eta / eos.B[3]
        z_cs = (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3
        assert eos.p_of_rho(3, rho) / rho == pytest.approx(z_cs, rel=1e-5)


def test_henderson_compressibility_factor():
    """Z = (1 + eta^2/8)/(1 - eta)^2 for Henderson's disc equation."""
    for eta in (0.1, 0.2, 0.3, 0.4):
        rho = eta / eos.B[2]
        z_h = (1 + eta ** 2 / 8.0) / (1 - eta) ** 2
        assert eos.p_of_rho(2, rho) / rho == pytest.approx(z_h, rel=1e-12)


def test_contact_density_is_the_pressure():
    """The hard-wall sum rule, with centre exclusion: rho(0+) = beta P."""
    for d in (1, 2, 3):
        rho = 0.3 / eos.B[d]
        assert eos.contact_density(d, rho) == eos.p_of_rho(d, rho)


def test_bad_dimension_rejected():
    with pytest.raises(ValueError):
        eos.f_ex(4, 0.1)


# ---- the attractive razor: 1-D square well ------------------------------- #

@pytest.mark.parametrize("rho", [0.05, 0.2, 0.5, 0.8])
def test_square_well_at_zero_depth_is_tonks(rho):
    """eps = 0 must collapse the Takahashi solution onto hard rods, exactly."""
    p = eos.sw1d_p_of_rho(0.0, 1.5, rho)
    rho_back, mu = eos.sw1d_state(0.0, 1.5, p)
    assert rho_back == pytest.approx(rho, rel=1e-9)
    assert p == pytest.approx(eos.p_of_rho(1, rho), rel=1e-8)
    assert mu == pytest.approx(eos.mu_of_rho(1, rho), rel=1e-8)


@pytest.mark.parametrize("lam", [1.2, 1.5, 2.0])
def test_square_well_zero_depth_independent_of_range(lam):
    """With no depth the well is invisible, whatever its range."""
    assert eos.sw1d_z_of_rho(0.0, lam, 0.4) == pytest.approx(
        float(eos.z_of_rho(1, 0.4)), rel=1e-8)


@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("lam", [1.3, 1.8])
def test_square_well_second_virial_limit(eps, lam):
    """beta P -> rho + B2 rho^2 with B2 = 1 + (lam - 1)(1 - e^eps) in 1-D.

    The attraction makes B2 smaller than the hard-rod 1, and negative once the
    well is deep enough, which is the whole point: this is the term a mean-field
    cDFT functional is approximating.
    """
    b2 = 1.0 + (lam - 1.0) * (1.0 - onp.exp(eps))
    rho = 1e-4
    p = eos.sw1d_p_of_rho(eps, lam, rho)
    assert p == pytest.approx(rho + b2 * rho ** 2, rel=1e-3)


@pytest.mark.parametrize("eps", [0.5, 1.5])
def test_square_well_attraction_lowers_the_pressure(eps):
    """At fixed density an attraction can only reduce the pressure."""
    rho = 0.4
    assert eos.sw1d_p_of_rho(eps, 1.5, rho) < eos.p_of_rho(1, rho)


@pytest.mark.parametrize("eps", [0.0, 0.8, 1.6])
def test_square_well_round_trip_through_activity(eps):
    """rho -> z -> rho, the path an attractive run actually takes."""
    rho = 0.35
    z = eos.sw1d_z_of_rho(eps, 1.5, rho)
    assert eos.sw1d_rho_of_z(eps, 1.5, z) == pytest.approx(rho, rel=1e-7)


@pytest.mark.parametrize("eps", [0.4, 1.2])
def test_square_well_gibbs_duhem(eps):
    """dP/drho = rho dmu/drho for the exact solution too."""
    rho, h = 0.4, 1e-5
    p = lambda r: eos.sw1d_p_of_rho(eps, 1.5, r)
    mu = lambda r: eos.sw1d_mu_of_rho(eps, 1.5, r)
    dp = (p(rho + h) - p(rho - h)) / (2 * h)
    dmu = (mu(rho + h) - mu(rho - h)) / (2 * h)
    assert dp == pytest.approx(rho * dmu, rel=1e-4)


def test_square_well_rejects_a_range_beyond_two():
    """Past lambda = 2 second neighbours interact and the solution is no longer
    exact, so it refuses rather than returning a plausible wrong number."""
    with pytest.raises(ValueError):
        eos.sw1d_state(1.0, 2.5, 1.0)
