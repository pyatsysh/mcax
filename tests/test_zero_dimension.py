"""The zero-dimensional limit: a cavity too small to hold two particles.

Not a fourth value of `d`, because there are no positions to sample. It is a
geometry, and it is reachable in every dimension: a spherical pore of radius
below sigma/2 admits at most one centre, since two centres inside it are at most
2R apart.

It is worth more than its size. Fundamental measure theory is constructed so
that dimensional crossover reproduces this limit exactly, which is where the
logarithm in the Rosenfeld functional comes from, and it is the sharpest of the
dimensional reductions a functional can be tested against. So a hard-particle
sampler that can produce it is producing the reference data for the hardest
check in the theory.

Everything here is exact, because the whole system is two states. With
Xi = 1 + zV,

    <N>    = zV / (1 + zV)        saturating at 1 however large z
    Var(N) = <N> (1 - <N>)        Bernoulli, no freedom left
    F_ex   = eta + (1-eta) ln(1-eta)
    mu_ex  = -ln(1 - eta)

The variance is the sharper of the first two: the mean could come out right for
a sampler that merely never inserted twice, but the variance also requires the
fluctuation estimators to be right.
"""
import numpy as onp
import pytest

from mcax import (make_spec, burn_and_sample, geometry, observables,
                  summary, eos)
from conftest import alive_positions


def _cavity(d, R=0.4, z=5.0, **kw):
    return make_spec(d, geom="sphere", H=R, z_act=z, Nmax=4, **kw)


# ---- is it actually zero-dimensional -------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3])
def test_small_sphere_is_recognised_as_zero_dimensional(d):
    assert geometry.is_zero_dimensional(_cavity(d, R=0.4))
    assert not geometry.is_zero_dimensional(_cavity(d, R=0.6))


def test_the_boundary_is_strict():
    """At exactly 2R = sigma two centres sit at contact, which the overlap test
    allows, so the cavity holds two and the limit does not apply."""
    assert not geometry.is_zero_dimensional(_cavity(3, R=0.5))
    assert geometry.is_zero_dimensional(_cavity(3, R=0.5 - 1e-9))


def test_a_narrow_one_dimensional_slit_is_also_zero_dimensional():
    assert geometry.is_zero_dimensional(
        make_spec(1, H=0.8, z_act=1.0, slit=True))
    assert not geometry.is_zero_dimensional(
        make_spec(1, H=1.2, z_act=1.0, slit=True))


def test_extended_geometries_are_not():
    """Anything with a periodic or extended axis holds arbitrarily many."""
    assert not geometry.is_zero_dimensional(
        make_spec(3, H=0.5, Lperp=8.0, z_act=1.0, slit=True))
    assert not geometry.is_zero_dimensional(
        make_spec(1, H=0.5, z_act=1.0, slit=False))


# ---- the sampler in the limit --------------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("z", [0.5, 5.0, 50.0])
def test_occupancy_is_the_two_state_result(d, z):
    """<N> = zV/(1+zV), exactly, and the cavity never holds two.

    The z = 50 case is the one that matters: an ordinary muVT box would fill up,
    and this one cannot go past a single particle however hard the reservoir
    pushes. If `n_hi` ever reached 2 the geometry would not be what it claims.

    The gate is four standard errors of THIS run rather than a fixed percentage,
    because the right percentage depends on the occupancy. N here is Bernoulli
    with sd sqrt(eta(1-eta)), so a low-occupancy cavity carries a large relative
    error at any affordable step count: at eta = 0.12 and 51200 draws the
    standard error alone is 1.2% of the mean, and a nominal 2% gate is 1.7
    sigma, which fails about one run in six. That is a bad test, not a bad
    sampler; measured over 2.6e8 steps this case agrees with the exact answer to
    0.03%.
    """
    spec = _cavity(d, z=z)
    V = geometry.volume(spec)
    res = burn_and_sample(spec, C=32, seed=0, n_burn=20_000, n_run=80_000,
                          thin=50, nbins=6)
    exact = float(eos.eta_0d(z, V))
    err = summary(res.Ns)["mcse"]
    assert res.n_hi == 1, f"cavity held {res.n_hi} particles at once"
    assert abs(res.n_mean - exact) < 4.0 * err, (
        f"<N> {res.n_mean:.6f} against exact {exact:.6f}, "
        f"off by {abs(res.n_mean - exact) / err:.1f} standard errors")


@pytest.mark.parametrize("d", [1, 2, 3])
def test_the_occupancy_is_bernoulli(d):
    """Var(N) = eta(1 - eta), which needs the fluctuation estimator to be right
    as well as the sampler."""
    spec = _cavity(d)
    eta = float(eos.eta_0d(spec.z_act, geometry.volume(spec)))
    res = burn_and_sample(spec, C=32, seed=1, n_burn=20_000, n_run=80_000,
                          thin=50, nbins=6)
    var, err = observables.susceptibility(res.Ns)
    exact = float(eos.var_n_0d(eta))
    assert abs(var - exact) < 4.0 * err, (
        f"Var(N) {var:.5f} +/- {err:.5f} against Bernoulli {exact:.5f}, "
        f"off by {abs(var - exact) / err:.1f} standard errors")


@pytest.mark.parametrize("d", [1, 2, 3])
def test_excess_chemical_potential_comes_back(d):
    """beta mu_ex = -ln(1 - eta), recovered from the measured occupancy.

    This is the quantity a functional has to reproduce under dimensional
    crossover, so getting it out of a run is the point of the whole exercise.
    """
    spec = _cavity(d)
    res = burn_and_sample(spec, C=32, seed=2, n_burn=20_000, n_run=80_000,
                          thin=50, nbins=6)
    eta = float(eos.eta_0d(spec.z_act, geometry.volume(spec)))
    # Propagate the occupancy error through mu_ex = -ln(1 - eta), whose slope
    # is 1/(1 - eta), rather than guessing a percentage.
    err = summary(res.Ns)["mcse"] / (1.0 - eta)
    measured = float(eos.mu_ex_0d(res.n_mean))
    assert abs(measured - float(eos.mu_ex_0d(eta))) < 4.0 * err


def test_a_full_cavity_saturates():
    """z -> inf gives eta -> 1 and Var -> 0: the cavity is always occupied."""
    spec = _cavity(3, z=5_000.0)
    res = burn_and_sample(spec, C=16, seed=3, n_burn=20_000, n_run=40_000,
                          thin=50, nbins=6)
    assert res.n_hi == 1
    assert res.n_mean > 0.99
    var, _ = observables.susceptibility(res.Ns)
    assert var < 0.02


@pytest.mark.parametrize("d", [1, 2, 3])
def test_every_configuration_holds_at_most_one_centre(d):
    """The structural version of the same statement, read off the final state
    rather than off the counter."""
    spec = _cavity(d, z=50.0)
    res = burn_and_sample(spec, C=32, seed=4, n_burn=20_000, n_run=40_000,
                          thin=50, nbins=6)
    for c in range(32):
        assert len(alive_positions(res, c)) <= 1


# ---- what "zero-dimensional" actually means -------------------------------- #

# Cavities of wildly different size, shape and dimension, every one of which
# admits at most one particle. V spans a factor of 27 across this list.
CAVITIES = [
    ("sphere-d3-R020", dict(d=3, geom="sphere", H=0.20)),
    ("sphere-d3-R040", dict(d=3, geom="sphere", H=0.40)),
    ("sphere-d3-R049", dict(d=3, geom="sphere", H=0.49)),
    ("sphere-d2-R040", dict(d=2, geom="sphere", H=0.40)),
    ("sphere-d1-R040", dict(d=1, geom="sphere", H=0.40)),
    ("slit-d1-H090", dict(d=1, H=0.90, slit=True)),
]


@pytest.mark.parametrize("label,kw", CAVITIES, ids=[c[0] for c in CAVITIES])
def test_excess_free_energy_forgets_the_cavity(label, kw):
    """At fixed occupancy, mu_ex is the same number whatever the cavity is.

    This is what the name means, and it is worth being clear that it is NOT
    "the region is a point". These cavities have perfectly ordinary volumes,
    from 0.034 to 0.90, in one, two and three dimensions. What is
    zero-dimensional is the excess free energy: all the spatial dependence
    cancels out of it and it collapses to a function of the single scalar eta.

    The ideal part keeps the geometry, which is why the ACTIVITY needed to
    reach eta = 1/2 differs from cavity to cavity as 1/V. Only the excess is
    universal.
    """
    probe = make_spec(z_act=1.0, Nmax=4, **kw)
    V = geometry.volume(probe)
    assert geometry.is_zero_dimensional(probe)
    spec = make_spec(z_act=1.0 / V, Nmax=4, **kw)      # so that eta = 1/2
    res = burn_and_sample(spec, C=64, seed=0, n_burn=30_000, n_run=200_000,
                          thin=50, nbins=6)
    assert res.n_hi == 1
    eta = res.n_mean
    err = summary(res.Ns)["mcse"] / (1.0 - eta)
    assert abs(float(eos.mu_ex_0d(eta)) - float(eos.mu_ex_0d(0.5))) < 4.0 * err, (
        f"{label}: V = {V:.5f} gave mu_ex {float(eos.mu_ex_0d(eta)):+.5f} "
        f"against {float(eos.mu_ex_0d(0.5)):+.5f}")


@pytest.mark.parametrize("d", [2, 3])
def test_the_occupied_cavity_is_uniformly_filled(d):
    """The one particle is spread uniformly over the accessible region.

    It has nothing to interact with, since there is never a second particle, so
    its distribution is the flat one an ideal particle in a box would have. The
    profile is binned radially and divided by shell volumes, so a flat answer
    also confirms those shells: dividing a ball by slabs would tilt it like
    1/r^(d-1) and look like structure that is not there.
    """
    spec = _cavity(d, z=50.0)
    res = burn_and_sample(spec, C=64, seed=6, n_burn=30_000, n_run=200_000,
                          thin=50, nbins=8)
    rho = res.rho[1:]              # the innermost shell holds a tiny volume
    assert rho.std() / rho.mean() < 0.10, f"not uniform: {res.rho}"


# ---- the reference free energy -------------------------------------------- #

def test_zero_dimensional_free_energy_limits():
    """f_ex rises from 0 at an empty cavity to 1 at a full one."""
    assert eos.f_ex_0d(0.0) == pytest.approx(0.0, abs=1e-12)
    assert eos.f_ex_0d(1.0 - 1e-12) == pytest.approx(1.0, abs=1e-9)
    assert eos.f_ex_0d(0.5) == pytest.approx(0.5 + 0.5 * onp.log(0.5), rel=1e-12)


@pytest.mark.parametrize("eta", [0.1, 0.35, 0.6, 0.85])
def test_mu_ex_is_the_derivative_of_f_ex(eta):
    """d(f_ex)/d(eta) = -ln(1 - eta), checked numerically so the two cannot
    drift apart."""
    h = 1e-7
    fd = (eos.f_ex_0d(eta + h) - eos.f_ex_0d(eta - h)) / (2 * h)
    assert fd == pytest.approx(float(eos.mu_ex_0d(eta)), rel=1e-6)


@pytest.mark.parametrize("eta", [0.05, 0.3, 0.7])
def test_free_energy_is_convex_and_positive(eta):
    """Exclusion costs free energy, and the cost accelerates."""
    assert eos.f_ex_0d(eta) > 0.0
    h = 1e-4
    second = (eos.f_ex_0d(eta + h) - 2 * eos.f_ex_0d(eta)
              + eos.f_ex_0d(eta - h)) / h ** 2
    assert second > 0.0


def test_occupancy_inverts_the_activity():
    """eta_0d and mu_ex_0d are consistent: beta mu = ln(eta/V) + mu_ex."""
    z, V = 3.0, 0.4
    eta = float(eos.eta_0d(z, V))
    assert onp.log(eta / V) + float(eos.mu_ex_0d(eta)) == pytest.approx(
        onp.log(z), rel=1e-12)
