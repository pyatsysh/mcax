"""Attractive pair tails, and the exact 1-D answer they are checked against.

The tails themselves are ordinary functions and are tested as such: shape,
continuity at the joins, the cutoffs, and the mean-field integral each one
contributes to a cDFT functional.

The sampler side rests on one thing. In one dimension a square well of range
lam <= 2 reaches only nearest neighbours, so Takahashi's isobaric construction
gives the equation of state in closed form and the attractive engine has an
EXACT target, not a literature value with its own error bar. That is the same
role Tonks plays for the hard-particle engine, and it is why the square well is
the tail to implement first.
"""
import numpy as onp
import pytest

import jax.numpy as jnp

from mcax import make_spec, burn_and_sample, potentials, eos, observables


def _u(pair, r, sigma=1.0):
    """beta w at a single separation, as a float."""
    return float(potentials.evaluate(pair, jnp.asarray(r ** 2), sigma))


# ---- the tails as functions ----------------------------------------------- #

def test_square_well_is_a_second_indicator():
    w = potentials.SquareWell(eps=1.2, lam=1.5)
    assert _u(w, 0.5) == 0.0, "inside the core is the overlap test's business"
    assert _u(w, 1.2) == -1.2
    assert _u(w, 1.49) == -1.2
    assert _u(w, 1.51) == 0.0
    assert _u(w, 4.0) == 0.0


def test_yukawa_is_attractive_and_decays():
    w = potentials.Yukawa(eps=1.0, kappa=1.8)
    assert _u(w, 0.5) == 0.0
    vals = [_u(w, r) for r in (1.01, 1.5, 2.0, 3.0)]
    assert all(v < 0 for v in vals)
    assert all(a < b for a, b in zip(vals, vals[1:])), "must decay towards zero"
    assert _u(w, 6.1) == 0.0, "beyond the cutoff"
    # against the closed form
    r = 1.7
    want = -1.0 * onp.exp(-1.8 * (r - 1.0)) / r
    assert _u(w, r) == pytest.approx(want, rel=1e-12)


def test_wca_split_is_continuous_at_its_minimum():
    """w = -eps inside r_min and Lennard-Jones beyond, and the two agree there
    because (sigma/r_min)^6 = 1/2 exactly."""
    eps = 0.9
    w = potentials.LennardJonesWCA(eps)
    rmin = 2.0 ** (1.0 / 6.0)
    assert _u(w, 1.05) == pytest.approx(-eps, rel=1e-12)
    assert _u(w, rmin - 1e-9) == pytest.approx(-eps, rel=1e-9)
    assert _u(w, rmin + 1e-9) == pytest.approx(-eps, rel=1e-6)
    assert _u(w, 3.1) == 0.0, "beyond the cutoff"


def test_barker_henderson_tail_is_lennard_jones_beyond_sigma():
    eps = 0.7
    w = potentials.LennardJonesBH(eps)
    assert _u(w, 0.9) == 0.0
    assert _u(w, 1.0 + 1e-9) == pytest.approx(0.0, abs=1e-6), \
        "LJ vanishes at sigma, so the tail joins continuously"
    r = 1.4
    s6 = (1.0 / r) ** 6
    assert _u(w, r) == pytest.approx(4 * eps * (s6 ** 2 - s6), rel=1e-12)
    assert _u(w, 1.12) == pytest.approx(-eps, rel=0.01), "minimum near 2^(1/6)"


@pytest.mark.parametrize("eps", [0.2, 1.0, 3.0])
def test_barker_henderson_diameter_sits_below_sigma_and_grows_with_eps(eps):
    """d(T) = int_0^sigma [1 - exp(-beta u)] dr is bounded by sigma because the
    integrand is a probability, and it approaches sigma as T falls."""
    d = potentials.bh_diameter(eps)
    assert 0.8 < d < 1.0
    assert d < potentials.bh_diameter(eps * 2)


def test_mean_field_integral_of_a_square_well_is_its_volume():
    """The number a mean-field functional consumes, in closed form:
    int w d^3r = -eps (4/3) pi [(lam sigma)^3 - sigma^3]."""
    eps, lam = 1.3, 1.5
    w = potentials.SquareWell(eps, lam)
    want = -eps * (4.0 / 3.0) * onp.pi * (lam ** 3 - 1.0)
    assert potentials.mean_field_integral(w, 3, rmax=lam) == pytest.approx(
        want, rel=1e-4)


def test_mean_field_integral_is_negative_for_every_tail():
    """All four are net attractive, which is the whole point of them."""
    for w in (potentials.SquareWell(1.0), potentials.Yukawa(1.0),
              potentials.LennardJonesWCA(1.0), potentials.LennardJonesBH(1.0)):
        for d in (1, 2, 3):
            assert potentials.mean_field_integral(w, d) < 0.0


def test_potentials_hash_by_value():
    assert potentials.SquareWell(1.0, 1.5) == potentials.SquareWell(1.0, 1.5)
    assert hash(potentials.Yukawa(1.0)) == hash(potentials.Yukawa(1.0))
    assert potentials.SquareWell(1.0) != potentials.SquareWell(1.1)


def test_evaluate_of_no_tail_is_zero():
    assert float(potentials.evaluate(None, jnp.asarray(2.0), 1.0)) == 0.0


def test_mean_field_integral_of_no_tail_is_zero():
    """Not a degenerate quadrature over an empty interval: exactly zero."""
    for d in (1, 2, 3):
        assert potentials.mean_field_integral(None, d) == 0.0


# ---- the sampler with a tail switched on ---------------------------------- #

def test_zero_depth_is_bitwise_the_hard_particle_run():
    """eps = 0 must not perturb the hard-particle path at all."""
    kw = dict(d=1, H=12.0, z_act=1.5, slit=False)
    run = lambda spec: burn_and_sample(spec, C=4, seed=0, n_burn=2_000,
                                       n_run=6_000, thin=200, nbins=10)
    a = run(make_spec(**kw))
    b = run(make_spec(pair=potentials.SquareWell(0.0, 1.5), **kw))
    assert onp.array_equal(a.rho, b.rho)
    assert a.n_mean == b.n_mean


@pytest.mark.parametrize("eps", [0.5, 1.0])
def test_one_dimensional_square_well_against_takahashi(eps):
    """The razor, with attraction on: measured rho against the exact solution.

    Nothing here is fitted and nothing is a literature value. If the energy
    bookkeeping has a sign error, a double count or a missing exclusion of the
    particle from its own neighbour list, this is where it shows.
    """
    lam, rho_target = 1.5, 0.3
    z_act = eos.sw1d_z_of_rho(eps, lam, rho_target)
    spec = make_spec(d=1, H=40.0, z_act=z_act, slit=False,
                     pair=potentials.SquareWell(eps, lam))
    res = burn_and_sample(spec, C=32, seed=7, n_burn=60_000, n_run=200_000,
                          thin=250, nbins=8)
    rho = res.n_mean / spec.H
    assert res.capacity_warning is None
    assert rho == pytest.approx(rho_target, rel=0.05), (
        f"eps {eps}: measured {rho:.4f} against exact {rho_target}")


def test_attraction_raises_the_density_at_fixed_activity():
    """At the same z an attractive fluid is denser, and the exact solution says
    by how much, so this is a direction AND a magnitude."""
    eps, lam, z_act = 1.0, 1.5, 2.0
    hard = make_spec(d=1, H=40.0, z_act=z_act, slit=False)
    soft = make_spec(d=1, H=40.0, z_act=z_act, slit=False,
                     pair=potentials.SquareWell(eps, lam))
    kw = dict(C=32, seed=3, n_burn=40_000, n_run=120_000, thin=250, nbins=8)
    a = burn_and_sample(hard, **kw)
    b = burn_and_sample(soft, **kw)
    assert b.n_mean > a.n_mean
    assert a.n_mean / hard.H == pytest.approx(
        float(eos.rho_of_z(1, z_act)), rel=0.05)
    assert b.n_mean / soft.H == pytest.approx(
        eos.sw1d_rho_of_z(eps, lam, z_act), rel=0.05)


def test_square_well_leaves_a_step_in_g_of_r():
    """The well edge at lam sigma is a discontinuity in the potential, so g(r)
    carries a visible step there. A tail applied at the wrong range would put
    the step somewhere else."""
    eps, lam = 1.2, 1.5
    z_act = eos.sw1d_z_of_rho(eps, lam, 0.35)
    spec = make_spec(d=1, H=30.0, z_act=z_act, slit=False,
                     pair=potentials.SquareWell(eps, lam))
    res = burn_and_sample(spec, C=16, seed=5, n_burn=40_000, n_run=120_000,
                          thin=250, nbins=8, nbins_g=120)
    r, g, err = observables.pair_correlation(spec, res)
    just_inside = g[(r > lam - 0.15) & (r < lam)].mean()
    just_outside = g[(r > lam) & (r < lam + 0.15)].mean()
    assert just_inside > 1.25 * just_outside, (
        f"no step at lam: inside {just_inside:.3f}, outside {just_outside:.3f}")


def test_attraction_raises_the_compressibility_of_rods():
    """Var(N)/<N> against the exact square-well compressibility.

    The fluctuation route and the density route are independent, so passing
    both means the attractive ensemble is right and not merely centred right.
    """
    eps, lam, rho = 1.0, 1.5, 0.3
    z_act = eos.sw1d_z_of_rho(eps, lam, rho)
    spec = make_spec(d=1, H=40.0, z_act=z_act, slit=False,
                     pair=potentials.SquareWell(eps, lam))
    res = burn_and_sample(spec, C=32, seed=9, n_burn=60_000, n_run=200_000,
                          thin=250, nbins=8)
    kappa, err = observables.compressibility(res.Ns)
    h = 1e-5
    exact = 1.0 / ((eos.sw1d_p_of_rho(eps, lam, rho + h)
                    - eos.sw1d_p_of_rho(eps, lam, rho - h)) / (2 * h))
    assert kappa == pytest.approx(exact, rel=0.20), (
        f"measured {kappa:.4f} +/- {err:.4f} against exact {exact:.4f}")
    assert exact > float(eos.compressibility(1, rho)), "attraction softens"
