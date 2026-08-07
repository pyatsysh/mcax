"""The superball hard core: does the p-norm really mean what it claims to.

The overlap predicate in `mcax.shapes` rests on one geometric identity, that
two translates of a centrally symmetric convex body meet exactly when their
separation lies in the body doubled. If that identity is right the predicate is
exact and there is nothing statistical to check; if it is wrong the sampler
draws from a different ensemble and every downstream number is quietly false.
So the predicate is checked against a BRUTE-FORCE intersection of the two
bodies, on a grid, with no algebra shared between the two sides.

The rest is arithmetic that a formula can get wrong in a way physics will not
reveal: the volume of a superball, its second virial coefficient, and the
claim that one dimension is p-independent.
"""
import numpy as onp
import pytest

import jax.numpy as np

from mcax import make_spec, burn_and_sample, eos, geometry, order, shapes
from mcax.shapes import Superball, CUBE, SPHERE

PS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, onp.inf]


# --------------------------------------------------------------------------- #
#  A0: the sphere path is the sphere path                                      #
# --------------------------------------------------------------------------- #

def test_none_and_p2_are_the_same_trajectory():
    """`shape = None` and `Superball(2.0)` must not merely agree, they must be
    the same run: the same branch, the same arithmetic, the same PRNG draws.

    This is acceptance test A0 in miniature. Its point is that adding shapes
    cost the hard-sphere engine NOTHING, so every number the library published
    before shapes existed still stands without being re-measured.
    """
    kw = dict(d=3, H=6.0, Lperp=6.0, z_act=2.0, geom="slit")
    a = burn_and_sample(make_spec(**kw), C=4, seed=3, n_burn=400,
                        n_run=2_000, thin=50, nbins=24, n0=20)
    b = burn_and_sample(make_spec(shape=SPHERE, **kw), C=4, seed=3, n_burn=400,
                        n_run=2_000, thin=50, nbins=24, n0=20)
    assert onp.array_equal(onp.asarray(a.state.pos), onp.asarray(b.state.pos))
    assert onp.array_equal(onp.asarray(a.state.alive),
                           onp.asarray(b.state.alive))
    assert onp.array_equal(a.Ns, b.Ns)
    assert onp.array_equal(a.rho, b.rho)


# --------------------------------------------------------------------------- #
#  The overlap predicate against a brute-force intersection                    #
# --------------------------------------------------------------------------- #

def _brute_force_overlap(p, delta, sigma=1.0, n=None):
    """Do two aligned superballs at separation `delta` share a point.

    Deliberately naive: lay a grid over the region that could hold a shared
    point and ask whether any node is inside both bodies. No Minkowski
    argument, no norms of the separation, nothing the predicate under test
    could also be wrong about.

    The grid is only ever asked about separations a clear 10% off the contact
    surface, so its spacing has to resolve 0.1 sigma and no more: 64 nodes per
    axis in three dimensions is a factor of thirty finer than that and keeps
    the whole battery inside a few seconds.
    """
    d = len(delta)
    n = n if n is not None else (200 if d == 2 else 64)
    R = 0.5 * sigma
    lo = onp.minimum(0.0, delta) - R
    hi = onp.maximum(0.0, delta) + R
    axes = [onp.linspace(lo[i], hi[i], n) for i in range(d)]
    grid = onp.stack([g.ravel() for g in onp.meshgrid(*axes, indexing="ij")],
                     axis=-1)
    inA = shapes.contains_point(Superball(p), grid, sigma)
    inB = shapes.contains_point(Superball(p), grid - delta[None, :], sigma)
    return bool(onp.any(inA & inB))


def _contact_separation(p, u, sigma=1.0):
    """The separation along direction `u` at which the two bodies just touch:
    where the p-norm of the separation equals sigma, by the claim under test."""
    u = onp.asarray(u, dtype=float)
    if onp.isinf(p):
        nrm = onp.max(onp.abs(u))
    else:
        nrm = onp.sum(onp.abs(u) ** p) ** (1.0 / p)
    return sigma * u / nrm


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 6.0, onp.inf])
@pytest.mark.parametrize("d", [2, 3])
def test_predicate_matches_brute_force_across_directions(p, d):
    """Straddle the predicted contact surface in many directions at once.

    Just inside it the bodies must share a point and just outside they must
    not. A predicate with the wrong exponent, or one testing the norm of the
    separation against sigma/2 rather than sigma, fails this in the very first
    off-axis direction even though it would pass every on-axis check.
    """
    rng = onp.random.default_rng(7)
    dirs = onp.vstack([onp.eye(d), onp.ones((1, d)), rng.normal(size=(6, d))])
    for u in dirs:
        if onp.allclose(u, 0.0):
            continue
        contact = _contact_separation(p, u)
        for scale, expect in ((0.90, True), (1.10, False)):
            delta = scale * contact
            got = bool(shapes.overlaps(Superball(p), np.asarray(delta), 1.0))
            assert got is expect, (
                f"p={p} d={d} u={u} scale={scale}: predicate said {got}")
            assert _brute_force_overlap(p, delta) is expect, (
                f"p={p} d={d} u={u} scale={scale}: brute force disagrees "
                f"with the contact surface itself")


@pytest.mark.parametrize("p", [1.0, 2.0, 4.0, onp.inf])
def test_predicate_matches_brute_force_on_random_pairs(p):
    """Adversarial rather than constructed: random separations, no reference to
    where the contact surface is supposed to be.

    Separations within a few per cent of contact are skipped, and the reason is
    the REFERENCE rather than the predicate. Just inside contact the two bodies
    share a sliver thinner than the grid spacing, so the brute force reports no
    overlap where there is one: it is the approximate side of this comparison
    and it has to be kept away from the case it cannot resolve. Everything
    else is fair game, and the discarded shell is exactly where the constructed
    straddle test above does its work.
    """
    rng = onp.random.default_rng(19)
    n_tested = 0
    for delta in rng.uniform(-1.3, 1.3, size=(40, 3)):
        nrm = float(onp.max(onp.abs(delta)) if onp.isinf(p)
                    else onp.sum(onp.abs(delta) ** p) ** (1.0 / p))
        if abs(nrm - 1.0) < 0.08:
            continue
        got = bool(shapes.overlaps(Superball(p), np.asarray(delta), 1.0))
        assert got == _brute_force_overlap(p, delta), f"p={p} delta={delta}"
        n_tested += 1
    assert n_tested > 20


def test_cube_is_the_max_norm_not_a_large_p_limit():
    """p = inf is implemented directly, so it must be EXACTLY the max norm and
    not a large-p approximation of it. Two cubes separated by 0.999 along one
    axis and 0 along the others overlap; at 1.001 they do not, whatever the
    other components do."""
    for other in (0.0, 0.5, 0.999):
        assert bool(shapes.overlaps(CUBE, np.asarray([0.999, other, other]), 1.0))
        assert not bool(shapes.overlaps(CUBE, np.asarray([1.001, other, other]),
                                        1.0))


def test_p_below_one_is_refused():
    """Concave bodies break the Minkowski identity, and the failure is
    invisible downstream, so it raises rather than warns."""
    with pytest.raises(ValueError, match="p >= 1"):
        shapes.check(Superball(0.5))
    with pytest.raises(ValueError, match="p >= 1"):
        make_spec(d=3, H=6.0, z_act=1.0, shape=Superball(0.8))


# --------------------------------------------------------------------------- #
#  Volume and the second virial coefficient                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_volume_formula_against_monte_carlo_integration(p, d):
    """The closed form sigma^d Gamma(1+1/p)^d / Gamma(1+d/p), checked by
    throwing points into the enclosing cube. Nothing clever, but it is the
    number every packing fraction in the campaign is divided by."""
    rng = onp.random.default_rng(101)
    x = rng.uniform(-0.5, 0.5, size=(400_000, d))
    frac = shapes.contains_point(Superball(p), x, 1.0).mean()
    got = shapes.volume(Superball(p), d, 1.0)
    assert abs(got - frac) < 4.0 * onp.sqrt(frac * (1 - frac) / 400_000) + 1e-3


def test_volume_recovers_the_known_members():
    assert shapes.volume(SPHERE, 3) == pytest.approx(onp.pi / 6.0)
    assert shapes.volume(SPHERE, 2) == pytest.approx(onp.pi / 4.0)
    assert shapes.volume(CUBE, 3) == pytest.approx(1.0)
    assert shapes.volume(Superball(1.0), 3) == pytest.approx(1.0 / 6.0)
    for p in PS:                        # d = 1 is a rod for every exponent
        assert shapes.volume(Superball(p), 1) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [1.5, 2.0, 4.0, onp.inf])
@pytest.mark.parametrize("d", [2, 3])
def test_b2_is_half_the_excluded_volume(p, d):
    """B2 = (1/2) integral [1 - exp(-beta u)] d^d r, which for a hard body is
    half the volume of the set of separations that overlap. Integrated here by
    Monte Carlo over the PREDICATE, so this ties the closed form 2^(d-1) v to
    the same predicate the sampler uses rather than to the volume formula."""
    rng = onp.random.default_rng(23)
    box = 2.2
    x = rng.uniform(-0.5 * box, 0.5 * box, size=(300_000, d))
    hit = onp.asarray(shapes.overlaps(Superball(p), np.asarray(x), 1.0))
    excluded = hit.mean() * box ** d
    assert 0.5 * excluded == pytest.approx(shapes.b2(Superball(p), d, 1.0),
                                           rel=0.02)


# --------------------------------------------------------------------------- #
#  One dimension is p-independent, and that is the razor                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", PS)
def test_one_dimension_is_a_rod_whatever_p(p):
    """The predicate itself, on a dense sweep of separations: in d = 1 the
    p-norm of a single component is its absolute value for every p, so the
    overlap test must be |dr| < sigma exactly and identically."""
    dr = onp.linspace(-2.0, 2.0, 4001)[:, None]
    got = onp.asarray(shapes.overlaps(Superball(p), np.asarray(dr), 1.0))
    assert onp.array_equal(got, onp.abs(dr[:, 0]) < 1.0)


@pytest.mark.parametrize("p", [1.0, 3.0, onp.inf])
def test_rods_hit_tonks_for_every_p(p, tol):
    """And the sampler agrees, which is the razor doing its job: Tonks is exact
    in one dimension, so a bookkeeping error anywhere in the shape layer shows
    up against a closed-form answer rather than against another simulation."""
    rho = 0.5
    spec = make_spec(d=1, H=40.0, z_act=eos.z_of_rho(1, rho), geom="bulk",
                     shape=Superball(p))
    r = burn_and_sample(spec, C=16, seed=5, n_burn=20_000, n_run=60_000,
                        thin=50, nbins=40)
    got = r.n_mean / geometry.volume(spec)
    assert abs(got - rho) < tol * rho


# --------------------------------------------------------------------------- #
#  Cavities, boxes, and where the zero-dimensional limit sits                  #
# --------------------------------------------------------------------------- #

def test_a_cubic_cavity_is_zero_dimensional_for_cubes_before_spheres():
    """The sharpest statement the p-norm makes about geometry.

    A cavity 0.9 on a side cannot hold two cube centres, because escaping needs
    a full sigma along SOME axis and no axis is that long. It can easily hold
    two sphere centres, which need only 1.0 along the diagonal and have 1.56 of
    it. Getting this backwards would mislabel which states are 0-D.
    """
    kw = dict(d=3, H=0.9, Lperp=0.9, z_act=1.0, geom="box", Nmax=4)
    assert geometry.is_zero_dimensional(make_spec(shape=CUBE, **kw))
    assert not geometry.is_zero_dimensional(make_spec(shape=SPHERE, **kw))


def test_box_geometry_confines_every_axis():
    """A box has no periodic axis at all, so nothing may leave it and the
    minimum image must not wrap anything back in."""
    spec = make_spec(d=3, H=4.0, Lperp=4.0, z_act=3.0, geom="box")
    per, flag = geometry.periods(spec)
    assert not flag.any()
    r = burn_and_sample(spec, C=4, seed=2, n_burn=2_000, n_run=6_000,
                        thin=50, nbins=20)
    pos = onp.asarray(r.state.pos)[onp.asarray(r.state.alive)]
    assert pos.min() >= 0.0 and pos.max() <= 4.0


def test_zero_dimensional_cavity_saturates_for_every_p():
    """A cavity that admits one centre admits one centre whatever its shape:
    <N> = zV/(1+zV) exactly, and the shape enters only through V."""
    for p in (2.0, 4.0, onp.inf):
        spec = make_spec(d=3, geom="sphere", H=0.3, z_act=40.0, Nmax=4,
                         shape=Superball(p))
        assert geometry.is_zero_dimensional(spec)
        r = burn_and_sample(spec, C=32, seed=8, n_burn=3_000, n_run=30_000,
                            thin=20, nbins=8)
        want = eos.eta_0d(40.0, geometry.volume(spec))
        assert abs(r.n_mean - want) < 0.02


# --------------------------------------------------------------------------- #
#  The freezing alarm                                                          #
# --------------------------------------------------------------------------- #

def _s_max_over_n(spec, kv, pts):
    """S(k)max / N for one configuration. |rho_k|^2 is divided by N TWICE: once
    to make S(k), once more for the size-independent ratio the guard reads."""
    n = spec.Nmax
    pos = onp.zeros((n, spec.d)); pos[:len(pts)] = pts
    alive = onp.zeros(n, bool); alive[:len(pts)] = True
    s = onp.asarray(order.sk_chain(np.asarray(pos), np.asarray(alive),
                                   np.asarray(kv)))
    return float(s.max()) / len(pts) ** 2


def test_structure_factor_separates_a_lattice_from_a_gas():
    """S(k)max/N is the discriminator the freezing guard reads, so it has to be
    O(1) for a crystal and O(1/N) for a disordered configuration. Checked on
    configurations built by hand, not sampled, so the answer is known."""
    spec = make_spec(d=3, H=6.0, Lperp=6.0, z_act=1.0, geom="bulk", Nmax=256)
    kv = order.kvectors(spec, nmax=4)
    assert len(kv) > 10

    a = onp.arange(6) + 0.5
    lat = onp.stack([g.ravel() for g in onp.meshgrid(a, a, a, indexing="ij")],
                    axis=-1)                            # 216 sites, spacing 1
    gas = onp.random.default_rng(4).uniform(0.0, 6.0, size=(216, 3))

    assert _s_max_over_n(spec, kv, lat) > 0.9   # one peak carries the whole lot
    assert _s_max_over_n(spec, kv, gas) < 0.1   # nothing coherent anywhere


def test_the_monitor_reaches_the_bragg_peak_of_a_campaign_sized_cell():
    """The failure this exists to prevent is the monitor being SILENT.

    A crystal in a cell of edge L with lattice constant a puts its first Bragg
    peak at n = L/a, which for the campaign's L = 8 is n = 7 or 8. A ball of
    wavevectors out to |n| = 4 never samples it, so an ordered configuration
    comes back looking like a fluid and the freezing guard passes a crystal.
    Nothing else in the pipeline would catch that, so it is checked directly,
    against jittered crystals at the two lattice constants the cell admits.
    """
    spec = make_spec(d=3, H=8.0, Lperp=8.0, z_act=1.0, geom="bulk", Nmax=600,
                     shape=CUBE)
    kv = order.kvectors(spec, nmax=4, kmax=4.0 * onp.pi)
    rng = onp.random.default_rng(0)
    for ncell in (7, 8):
        a = (onp.arange(ncell) + 0.5) * (8.0 / ncell)
        lat = onp.stack([g.ravel() for g in
                         onp.meshgrid(a, a, a, indexing="ij")], axis=-1)
        lat = lat + 0.02 * rng.normal(size=lat.shape)   # thermal jitter
        got = _s_max_over_n(spec, kv, lat)
        assert got > 0.5, f"{ncell}^3 crystal read as S/N = {got:.3f}"
    gas = rng.uniform(0.0, 8.0, size=(512, 3))
    assert _s_max_over_n(spec, kv, gas) < order.ORDER_TRIP


def test_kvectors_respect_which_axes_are_periodic():
    """A hard box has no reciprocal lattice, and a slit has one only in plane.
    Returning wavevectors along a walled axis would report layering as
    freezing, which is exactly the false alarm this must not raise."""
    box = make_spec(d=3, H=4.0, Lperp=4.0, z_act=1.0, geom="box")
    assert len(order.kvectors(box)) == 0
    slit = make_spec(d=3, H=4.0, Lperp=8.0, z_act=1.0, geom="slit")
    kv = order.kvectors(slit, nmax=3)
    assert len(kv) > 0
    assert onp.allclose(kv[:, -1], 0.0)
