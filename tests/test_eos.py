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
    """beta P = rho + rho f' - f_ex must agree with the closed form.

    This is not circular: p_of_rho short-circuits to the closed form for d = 1,
    so this checks that f_ex and its analytic derivative are mutually
    consistent with that pressure via the Gibbs-Duhem route.
    """
    p = rho + rho * eos._dfex(1, rho) - eos.f_ex(1, rho)
    assert p == pytest.approx(rho / (1.0 - rho), rel=1e-10)


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
        assert eos.p_of_rho(2, rho) / rho == pytest.approx(z_h, rel=1e-3)


def test_contact_density_is_the_pressure():
    """The hard-wall sum rule, with centre exclusion: rho(0+) = beta P."""
    for d in (1, 2, 3):
        rho = 0.3 / eos.B[d]
        assert eos.contact_density(d, rho) == eos.p_of_rho(d, rho)


def test_bad_dimension_rejected():
    with pytest.raises(ValueError):
        eos.f_ex(4, 0.1)
