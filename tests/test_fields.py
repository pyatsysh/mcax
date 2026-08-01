"""External fields.

The test that decides this file is the barometric one. Switch the core off and
the muVT density in ANY external field is exactly

    rho(r) = z exp(-beta V_ext(r))

point by point, with no approximation, no fitting and no reference data. So a
sign error, a missing factor or a field applied to the wrong axis all show up as
a profile that is the wrong shape rather than as a number that is slightly off.
Everything else in this file is a consistency check on top of that.
"""
import numpy as onp
import pytest

from mcax import make_spec, burn_and_sample, fields, eos


def _ideal(field, d=1, H=8.0, z_act=0.8, nbins=16, **kw):
    """An ideal gas in a slit with `field` switched on."""
    spec = make_spec(d=d, H=H, z_act=z_act, sigma=1e-6, slit=True,
                     Nmax=int(4 * z_act * H * 10.0 ** (d - 1)) + 40,
                     field=field, **kw)
    res = burn_and_sample(spec, C=24, seed=1, n_burn=10_000, n_run=120_000,
                          thin=100, nbins=nbins)
    return spec, res


def _barometric(field, spec, nbins, nsub=400):
    """z_act <exp(-beta V)> averaged ACROSS each bin, not sampled at its centre.

    The histogram reports a bin average, so the reference has to be one too.
    For a slowly varying field the difference is a fraction of a per cent and
    the distinction looks pedantic; across the first bin of a 9-3 wall
    exp(-V) runs from 0.017 to 1.48 and the centre value is meaningless.
    """
    import jax.numpy as jnp
    edges = onp.linspace(0.0, spec.H, nbins + 1)
    out = onp.empty(nbins)
    for b in range(nbins):
        zs = onp.linspace(edges[b], edges[b + 1], nsub)
        v = onp.array([float(fields.evaluate(field, jnp.array([z]))) for z in zs])
        out[b] = onp.trapezoid(onp.exp(-v), zs) / (edges[b + 1] - edges[b])
    return spec.z_act * out


# ---- the barometric law --------------------------------------------------- #

def test_no_field_is_exactly_the_hard_particle_path():
    """field = None must not perturb anything, to the last digit."""
    kw = dict(d=1, H=8.0, z_act=1.5, slit=True)
    a = burn_and_sample(make_spec(**kw), C=4, seed=0, n_burn=2_000,
                        n_run=6_000, thin=200, nbins=10)
    b = burn_and_sample(make_spec(field=None, **kw), C=4, seed=0, n_burn=2_000,
                        n_run=6_000, thin=200, nbins=10)
    assert onp.allclose(a.rho, b.rho, rtol=0, atol=0)


def test_gravity_gives_the_barometric_profile():
    """rho(z) = z_act exp(-g z), the textbook case, checked bin by bin."""
    g = 0.4
    field = fields.Gravity(g)
    spec, res = _ideal(field)
    want = _barometric(field, spec, len(res.z))
    assert onp.allclose(res.rho, want, rtol=0.08, atol=0.01 * spec.z_act), (
        f"measured {res.rho}\nwanted   {want}")


@pytest.mark.parametrize("k", [0.15, 0.4])
def test_harmonic_gives_a_gaussian_profile(k):
    z0, field = 4.0, fields.Harmonic(k, 4.0)
    spec, res = _ideal(field)
    want = _barometric(field, spec, len(res.z))
    assert onp.allclose(res.rho, want, rtol=0.10, atol=0.01 * spec.z_act)


def test_square_well_wall_gives_a_step():
    """A step in V is a step in rho, of exactly exp(eps)."""
    eps, width = 0.9, 2.0
    spec, res = _ideal(fields.SquareWellWall(eps, width))
    near = res.rho[res.z < width - 0.5].mean()
    far = res.rho[res.z > width + 1.5].mean()
    assert near / far == pytest.approx(onp.exp(eps), rel=0.08)


def test_exponential_wall_matches_its_own_exponential():
    field = fields.ExpWall(1.1, 1.5)
    spec, res = _ideal(field)
    want = _barometric(field, spec, len(res.z))
    assert onp.allclose(res.rho, want, rtol=0.10, atol=0.01 * spec.z_act)


def test_lj93_wall_matches_its_own_boltzmann_factor():
    """The 9-3 wall is the awkward one: it diverges at contact and is
    attractive just beyond, so the profile carries a hard zero and a peak
    inside one sigma. It is also where the bin average earns its keep, since
    exp(-V) runs over two orders of magnitude across the first bin. This tests
    the clipping as much as it tests the field."""
    w = fields.LJ93Wall(eps=0.8, sigma_w=1.0)
    spec, res = _ideal(w, H=10.0, nbins=20)
    want = _barometric(w, spec, len(res.z))
    assert onp.allclose(res.rho, want, rtol=0.12, atol=0.02 * spec.z_act), (
        f"measured {res.rho}\nwanted   {want}")
    assert res.rho[0] == 0.0, "the divergent core must be empty"
    assert res.rho.max() > 1.5 * spec.z_act, "the well must show a peak"


def test_mirror_puts_the_same_wall_on_the_far_side():
    """A field mirrored about H/2 must give a profile symmetric about H/2."""
    H = 10.0
    w = fields.SquareWellWall(0.8, 1.5)
    spec, res = _ideal(fields.both_walls(w, H), H=H)
    asym = onp.abs(res.rho - res.rho[::-1]).max() / res.rho.mean()
    assert asym < 0.12, f"asymmetry {asym:.3f}"


# ---- composition and hashing ---------------------------------------------- #

def test_sum_of_fields_adds_the_potentials():
    both = fields.Sum((fields.Gravity(0.3), fields.Harmonic(0.2, 0.0)))
    spec, res = _ideal(both)
    want = _barometric(both, spec, len(res.z))
    assert onp.allclose(res.rho, want, rtol=0.10, atol=0.01 * spec.z_act)


def test_fields_hash_by_value():
    """The reason they are NamedTuples: identical fields must be one jit key."""
    assert fields.Gravity(1.0) == fields.Gravity(1.0)
    assert hash(fields.Gravity(1.0)) == hash(fields.Gravity(1.0))
    assert fields.Gravity(1.0) != fields.Gravity(2.0)
    s = fields.Sum((fields.Gravity(1.0), fields.Harmonic(0.5, 1.0)))
    assert hash(s) == hash(fields.Sum((fields.Gravity(1.0),
                                       fields.Harmonic(0.5, 1.0))))


def test_evaluate_clips_rather_than_returning_an_infinity():
    """A divergent field must come back large and FINITE, or a difference of
    two divergences downstream is a nan and quietly poisons an acceptance."""
    import jax.numpy as jnp
    v = fields.evaluate(fields.LJ93Wall(1.0), jnp.array([1e-12]))
    assert onp.isfinite(float(v))
    assert float(v) == pytest.approx(fields.CLIP)


def test_evaluate_of_no_field_is_zero():
    import jax.numpy as jnp
    assert float(fields.evaluate(None, jnp.array([3.0]))) == 0.0


# ---- with the hard core back on ------------------------------------------- #

def test_attractive_wall_raises_the_contact_density():
    """With a real core and an attractive wall the fluid should pile up more
    than the hard wall alone would give."""
    rho_b = 0.3
    z_act = float(eos.z_of_rho(1, rho_b))
    plain = make_spec(d=1, H=12.0, z_act=z_act, slit=True)
    sticky = make_spec(d=1, H=12.0, z_act=z_act, slit=True,
                       field=fields.both_walls(
                           fields.SquareWellWall(1.0, 1.0), 12.0))
    a = burn_and_sample(plain, C=16, seed=2, n_burn=20_000, n_run=60_000,
                        thin=200, nbins=24)
    b = burn_and_sample(sticky, C=16, seed=2, n_burn=20_000, n_run=60_000,
                        thin=200, nbins=24)
    assert b.rho[0] > 1.5 * a.rho[0]
    assert b.n_mean > a.n_mean, "an attractive wall must adsorb"
