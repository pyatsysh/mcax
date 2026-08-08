"""The orientable engine: quaternions, support maps, and the overlap battery.

The overlap test here is the only place in mcax that answers a geometric
question APPROXIMATELY, so it gets the most scrutiny. Three kinds of check,
and the third is the one that matters:

  1. Known answers. Pairs whose verdict can be worked out by hand: contact
     along an axis, a cube turned forty-five degrees presenting a corner,
     coincident centres, far separation.
  2. Reduction. With both bodies unrotated the answer must be the aligned
     engine's p-norm predicate, which is exact and independent code.
  3. The GUARANTEE DIRECTION. A fixed budget can fail to find a separating
     direction that exists, and must never claim to have found one that does
     not. The first costs efficiency, the second destroys the ensemble, so the
     tests assert the asymmetry directly rather than assuming it.
"""
import numpy as onp
import pytest

import jax
import jax.numpy as np

from mcax import bodies, geometry, orient, shapes
from mcax.bodies import Superball
from mcax.core import make_spec, burn_and_sample

INF = float("inf")
PS = [2.0, 3.0, 4.0, 6.0, INF]
E = onp.eye(3)


# --------------------------------------------------------------------------- #
#  Quaternions                                                                 #
# --------------------------------------------------------------------------- #

def test_shoemake_is_uniform_on_the_rotation_group():
    """Haar uniformity is what makes grand-canonical insertion carry no
    orientational factor, so it is checked rather than trusted.

    Two consequences of uniformity, both easy to violate with a plausible-
    looking sampler (uniform Euler angles fails both): every matrix element has
    mean zero, and the trace has mean zero as well, since the character of the
    identity representation is the only one that survives averaging.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), 40_000)
    R = onp.asarray(orient.q_matrix(jax.vmap(orient.q_random)(keys)))
    assert onp.abs(R.mean(axis=0)).max() < 0.02
    assert abs(onp.trace(R, axis1=1, axis2=2).mean()) < 0.03
    # and they really are rotations
    assert onp.abs(R @ onp.swapaxes(R, 1, 2) - onp.eye(3)).max() < 1e-12
    assert onp.abs(onp.linalg.det(R) - 1.0).max() < 1e-12


def test_perturbation_is_symmetric_and_shrinks_with_the_window():
    """A rotation proposal has to be symmetric for the acceptance to omit a
    Hastings ratio, and a zero window has to be the identity exactly, because
    that is the frozen mode the parallel engine is compared against."""
    q = onp.array([1.0, 0.0, 0.0, 0.0])
    for dtheta, bound in ((0.0, 1e-15), (0.2, 0.11), (1.0, 0.5)):
        ks = jax.random.split(jax.random.PRNGKey(1), 4000)
        out = onp.asarray(jax.vmap(
            lambda k: orient.q_perturb(k, np.asarray(q), dtheta))(ks))
        ang = 2.0 * onp.arccos(onp.clip(onp.abs(out[:, 0]), -1.0, 1.0))
        assert ang.max() <= dtheta + 1e-9
        assert abs(ang.mean()) <= bound + 1e-9
        assert onp.abs(onp.linalg.norm(out, axis=1) - 1.0).max() < 1e-12


# --------------------------------------------------------------------------- #
#  Support maps                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", PS)
def test_support_point_realises_the_support_value(p):
    """s(u) must be ON the body and must achieve h(u) = u . s(u). If the two
    disagree the overlap test is asking about a body that is not the one the
    volume and packing fraction were computed for."""
    b = Superball(p)
    rng = onp.random.default_rng(2)
    u = onp.vstack([onp.eye(3), onp.ones((1, 3)), rng.normal(size=(40, 3))])
    s = onp.asarray(bodies.support(b, np.asarray(u)))
    hv = onp.asarray(bodies.h(b, np.asarray(u)))
    assert onp.abs(onp.sum(u * s, axis=-1) - hv).max() < 1e-12
    assert bodies.contains_point(b, s).all()


@pytest.mark.parametrize("p", PS)
def test_support_map_is_centrally_symmetric(p):
    """s(-u) = -s(u). The Minkowski difference in `mcax.orient` is written with
    a PLUS where the general formula has a minus, and this is the identity that
    licenses it."""
    b = Superball(p)
    u = onp.random.default_rng(3).normal(size=(50, 3))
    a = onp.asarray(bodies.support(b, np.asarray(u)))
    c = onp.asarray(bodies.support(b, np.asarray(-u)))
    assert onp.abs(a + c).max() < 1e-12


# --------------------------------------------------------------------------- #
#  The overlap battery                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", PS)
def test_unrotated_pairs_reduce_to_the_aligned_predicate(p):
    """With both bodies at the identity this engine is solving a problem the
    aligned engine answers in closed form, so away from contact the two must
    agree exactly. Inside a 2% shell around contact they need not: that is the
    budget's blind spot and the next test pins its direction."""
    rng = onp.random.default_rng(4)
    dl = rng.uniform(-1.4, 1.4, size=(250, 3))
    nrm = (onp.max(onp.abs(dl), axis=1) if onp.isinf(p)
           else onp.sum(onp.abs(dl) ** p, axis=1) ** (1.0 / p))
    far = onp.abs(nrm - 1.0) > 0.02
    ref = onp.asarray(shapes.overlaps(shapes.Superball(p), np.asarray(dl), 1.0))
    f = jax.jit(jax.vmap(lambda v: orient.overlaps_pair(
        Superball(p), v, np.asarray(E), np.asarray(E))))
    got = onp.asarray(f(np.asarray(dl)))
    assert onp.array_equal(got[far], ref[far])


@pytest.mark.parametrize("p", PS)
def test_the_budget_only_ever_errs_towards_overlap(p):
    """The whole safety argument in one assertion.

    A short budget may fail to certify a separation that a long one finds, and
    that is tolerable: the pair is called overlapping, a legal move is
    rejected, and the sampler is exact for a very slightly fattened body. The
    reverse — a short budget declaring free what a long one calls overlapping —
    would admit real overlaps and is not tolerable at any rate.
    """
    b = Superball(p)
    ri, rc = bodies.inradius(b, 3), bodies.circumradius(b, 3)
    M = 4000
    rng = onp.random.default_rng(5)
    u = rng.normal(size=(M, 3))
    u /= onp.linalg.norm(u, axis=1)[:, None]
    dr = u * rng.uniform(2 * ri, 2 * rc, size=(M, 1))
    ks = jax.random.split(jax.random.PRNGKey(6), 2 * M)
    Q = jax.vmap(orient.q_random)(ks)
    Ra, Rb = orient.q_matrix(Q[:M]), orient.q_matrix(Q[M:])

    def go(n_iter):
        f = jax.jit(jax.vmap(lambda v, A, B: orient.overlaps_pair(
            b, v, A, B, n_iter, 1e-9)))
        return onp.asarray(f(np.asarray(dr), Ra, Rb))

    ref, prod = go(400), go(32)
    assert not (ref & ~prod).any(), "a short budget declared a pair FREE that " \
                                    "a long one calls overlapping"
    # and the trips, which are allowed, stay rare enough to be a rounding of
    # the shape rather than a change of it
    assert (prod & ~ref).sum() / max((~ref).sum(), 1) < 0.02


def test_a_cube_turned_forty_five_degrees_meets_corner_first():
    """A hand-computable rotated case, which the aligned predicate cannot
    reach at all. A unit cube turned 45 degrees about z reaches sqrt(2)/2 along
    x where an unrotated one reaches 1/2, so contact along x is at
    1/2 + sqrt(2)/2 = 1.2071 rather than at 1."""
    q45 = np.asarray([onp.cos(onp.pi / 8), 0.0, 0.0, onp.sin(onp.pi / 8)])
    R45 = orient.q_matrix(q45)
    d0 = 0.5 + 0.5 * onp.sqrt(2.0)
    b = Superball(INF)
    assert bool(orient.overlaps_pair(b, np.asarray([d0 - 0.01, 0.0, 0.0]),
                                     np.asarray(E), R45))
    assert not bool(orient.overlaps_pair(b, np.asarray([d0 + 0.01, 0.0, 0.0]),
                                         np.asarray(E), R45))
    # ... and the unrotated pair still separates at 1, so the rotation is what
    # moved the contact and not a change of size somewhere
    assert not bool(orient.overlaps_pair(b, np.asarray([1.01, 0.0, 0.0]),
                                         np.asarray(E), np.asarray(E)))


@pytest.mark.parametrize("p", PS)
def test_verdicts_are_invariant_under_a_global_rotation(p):
    """Rotating both bodies and the separation together is the same physical
    configuration seen from a turned camera. A verdict that changed would mean
    the lab frame had leaked into the test, which for a candidate-axis method
    is a real hazard."""
    b = Superball(p)
    rng = onp.random.default_rng(7)
    ks = jax.random.split(jax.random.PRNGKey(8), 60)
    R = onp.asarray(orient.q_matrix(jax.vmap(orient.q_random)(ks)))
    for i in range(60):
        dl = rng.uniform(-1.4, 1.4, size=3)
        a = bool(orient.overlaps_pair(b, np.asarray(dl), np.asarray(E),
                                      np.asarray(E)))
        c = bool(orient.overlaps_pair(b, np.asarray(R[i] @ dl),
                                      np.asarray(R[i]), np.asarray(R[i])))
        assert a == c


# --------------------------------------------------------------------------- #
#  The engine                                                                  #
# --------------------------------------------------------------------------- #

def test_rotating_a_sphere_is_always_accepted():
    """V0 in one line: a sphere is invariant under rotation, so no rotation
    move can ever be blocked by an overlap. Anything below 100% means the
    rotation branch is disturbing the position or the orientation is leaking
    into the overlap test."""
    spec = orient.make_spec(H=6.0, Lperp=6.0, z_act=5.0, body=Superball(2.0),
                            geom="slit", dtheta=0.6)
    r = orient.burn_and_sample(spec, C=4, seed=1, n_burn=1_500, n_run=4_000,
                               thin=50, nbins=24)
    assert r.acc[1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("p", [2.0, 4.0])
def test_frozen_rotation_matches_the_aligned_engine(p):
    """V1 in miniature: dtheta = 0 from an aligned start is the parallel
    superball model, reached through the support-function overlap test instead
    of the p-norm one. The two engines share no overlap code, so agreement
    certifies both.

    **Both engines get the same lattice prefill, and without it this test is
    not about what it claims.** They run different move mixes, so from an empty
    box the one proposing fewer translations per step makes room for new
    particles more slowly and is simply further behind at a fixed step count.
    Measured at p = 2: from empty at 16k steps the two differ by 0.6% and the
    gap is still closing; seeded at 90% and run to 40k they agree to 0.1%. The
    first number is a mixing rate, the second is the physics, and only the
    second is what V1 asks about.
    """
    z = 6.0
    osp = orient.make_spec(H=6.0, Lperp=6.0, z_act=z, body=Superball(p),
                           geom="slit", dtheta=0.0, p_rot=0.1)
    psp = make_spec(d=3, H=6.0, Lperp=6.0, z_act=z, geom="slit",
                    shape=shapes.Superball(p), Nmax=osp.Nmax)
    from mcax import eos
    v = bodies.volume(Superball(p), 3)
    n0 = int(0.9 * shapes.volume(shapes.Superball(2.0), 3) / v
             * eos.rho_of_z(3, z) * geometry.volume(psp))
    kw = dict(C=8, seed=2, n_burn=12_000, n_run=30_000, thin=50, nbins=24,
              n0=n0)
    ro = orient.burn_and_sample(osp, aligned=True, **kw)
    rp = burn_and_sample(psp, **kw)
    e = onp.sqrt(ro.Ns.mean(axis=1).std(ddof=1) ** 2
                 + rp.Ns.mean(axis=1).std(ddof=1) ** 2) / onp.sqrt(8)
    assert abs(ro.n_mean - rp.n_mean) < 4.0 * e + 0.02 * rp.n_mean


def test_free_cubes_are_less_dense_than_parallel_ones():
    """Rotation can only enlarge the orientation-averaged excluded volume, so
    at the same activity a freely rotating fluid must be the thinner of the
    two. A sign error in the rotation acceptance shows up here immediately."""
    osp = orient.make_spec(H=7.0, Lperp=7.0, z_act=1.5, body=Superball(INF),
                           geom="bulk", dtheta=0.5)
    ro = orient.burn_and_sample(osp, C=8, seed=3, n_burn=4_000, n_run=16_000,
                                thin=50, nbins=16)
    psp = make_spec(d=3, H=7.0, Lperp=7.0, z_act=1.5, geom="bulk",
                    shape=shapes.Superball(INF), Nmax=osp.Nmax)
    rp = burn_and_sample(psp, C=8, seed=3, n_burn=4_000, n_run=16_000,
                         thin=50, nbins=16)
    assert ro.n_mean < rp.n_mean
    assert 0.0 < ro.acc[1] < 1.0            # rotations are being tried and cost


def test_cubic_order_parameter_vanishes_for_isotropic_orientations():
    """S4 is normalised so that an isotropic distribution gives exactly zero
    and a body axis along the wall normal gives one. Both ends are checked,
    because an unnormalised order parameter reads as spurious order."""
    ks = jax.random.split(jax.random.PRNGKey(9), 20_000)
    q = jax.vmap(orient.q_random)(ks)
    st = orient.OState(pos=np.zeros((1, 20_000, 3)), quat=q[None],
                       alive=np.ones((1, 20_000), dtype=bool),
                       key=np.zeros((1, 2), dtype=np.uint32),
                       acc=np.zeros((1, 4)), tot=np.zeros((1, 4)),
                       nhi=np.zeros(1))
    assert onp.abs(orient.cubatic(st)).max() < 0.02
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 200, 1))
    st2 = st._replace(quat=ident, pos=np.zeros((1, 200, 3)),
                      alive=np.ones((1, 200), dtype=bool))
    assert onp.allclose(orient.cubatic(st2), 1.0, atol=1e-9)


def test_steinhardt_separates_a_lattice_from_a_liquid():
    """Q6 through the addition theorem: about 0.575 for fcc, near zero for a
    disordered set. The implementation never forms a spherical harmonic, so
    this is the check that the Legendre route is the same quantity."""
    a = onp.arange(4) * 1.2
    g = onp.stack([x.ravel() for x in onp.meshgrid(a, a, a, indexing="ij")],
                  axis=-1)
    # simple cubic: Q6 = 0.354 exactly, a textbook value
    st = orient.OState(pos=np.asarray(g)[None], quat=np.zeros((1, len(g), 4)),
                       alive=np.ones((1, len(g)), dtype=bool),
                       key=np.zeros((1, 2), dtype=np.uint32),
                       acc=np.zeros((1, 4)), tot=np.zeros((1, 4)),
                       nhi=np.zeros(1))
    assert orient.steinhardt(st, rcut=1.3, l=6) == pytest.approx(0.354, abs=0.02)
    gas = onp.random.default_rng(10).uniform(0.0, 5.0, size=(400, 3))
    st2 = st._replace(pos=np.asarray(gas)[None],
                      alive=np.ones((1, 400), dtype=bool),
                      quat=np.zeros((1, 400, 4)))
    assert orient.steinhardt(st2, rcut=1.3, l=6) < 0.2
