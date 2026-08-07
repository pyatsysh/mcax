"""Confining geometries: bulk, slit, spherical pore, cylindrical pore, wedge.

Every geometry answers the same six questions, and the sampler asks nothing
else. Two are traced (they run inside the kernel), four are plain Python read
once at trace time:

    contains(spec, r)      is this centre inside the accessible region
    profile_coord(spec, r) which coordinate the density profile is binned in
    ---
    bbox(spec)             the box uniform proposals are drawn from
    periods(spec)          period per axis, and which axes are periodic at all
    volume(spec)           the true accessible volume, for reporting rho
    bin_volume(spec, n)    the volume of each profile bin

Adding a geometry means writing those six and nothing else, which is the point
of the split.

**The proposal trick.** A sphere is not a box, and rejection-sampling a uniform
point inside one needs an unbounded loop, which `jit` will not take. So
insertions are proposed uniformly in the BOUNDING BOX and rejected if they land
outside the region. That is not an approximation and it is not a bias: with a
uniform proposal density 1/V_box the Metropolis-Hastings ratio for an insertion
is z V_box/(N+1), and a proposal outside the region is a hard-wall rejection
like any other. The acceptances therefore carry V_box, not V, while the
reported density carries V. The only cost is efficiency, and it is bounded and
known: V/V_box is pi/6 for a sphere in 3-D, pi/4 for a disc or a cylinder, and
exactly 1/2 for a wedge of any angle.

**Which parameter means what.** `H` and `Lperp` are reused rather than
multiplied into a parameter per geometry, so it is worth writing down once:

    geom        H                    Lperp                  psi
    bulk        box edge (periodic)  transverse edge        .
    slit        wall separation      transverse edge        .
    box         wall separation      transverse EDGE, hard  .
    sphere      RADIUS               .                      .
    cylinder    axial period         RADIUS                 .
    wedge       height above apex    periodic edge (d = 3)  opening half-angle

`box` is `slit` with the transverse periodicity taken away: a cuboid cavity
with hard walls on every face. It exists for the small-cavity end of the ladder,
where a periodic transverse axis would be a lie about the geometry, and it is
the only bounded region here whose 0-D limit is reached along a DIAGONAL rather
than an axis.

`geometry.describe(spec)` prints this back for a given spec, which is quicker
than counting axes by hand.

**Conventions that do not change.** Walls always act on CENTRES, never on
surfaces, in every geometry: the same centre-exclusion convention the DFT
reference data uses. The profile coordinate is the wall axis in a slit or a
wedge and the RADIUS in a sphere or a cylinder, so `result.z` is a radius for
the curved pores.

**The contact theorem does not survive curvature.** rho(contact) = beta P is a
planar-wall sum rule. Against a sphere or a cylinder the curvature contributes,
and the equality fails at order sigma/R. Use it as a check in `slit` and
`wedge` (away from the apex) only.
"""
import numpy as onp
import jax.numpy as np

from . import shapes

GEOMETRIES = ("bulk", "slit", "box", "sphere", "cylinder", "wedge")

# Smallest dimension each geometry makes sense in. A cylinder needs a radial
# plane and an axis; a wedge needs a width axis and a height axis.
_MIN_D = {"bulk": 1, "slit": 1, "box": 1, "sphere": 1, "cylinder": 2,
          "wedge": 2}


def check(geom, d):
    """Reject a geometry that has no meaning in this dimension."""
    if geom not in GEOMETRIES:
        raise ValueError(f"geom must be one of {GEOMETRIES}, got {geom!r}")
    if d < _MIN_D[geom]:
        raise ValueError(f"geom {geom!r} needs d >= {_MIN_D[geom]}, got d = {d}")


def _ball(d, r):
    """Volume of the d-dimensional ball: 2r, pi r^2, (4/3) pi r^3."""
    from math import gamma, pi
    return pi ** (0.5 * d) * r ** d / gamma(0.5 * d + 1.0)


# ---- the four static answers ---------------------------------------------- #

def bbox(spec):
    """(lo, hi), each (d,): the box insertions are proposed uniformly in."""
    d, g = spec.d, spec.geom
    if g in ("bulk", "slit", "box"):
        lo = onp.zeros(d)
        hi = onp.array([spec.Lperp] * (d - 1) + [spec.H])
    elif g == "sphere":
        lo = onp.full(d, -spec.H)
        hi = onp.full(d, spec.H)
    elif g == "cylinder":
        lo = onp.array([-spec.Lperp] * (d - 1) + [0.0])
        hi = onp.array([spec.Lperp] * (d - 1) + [spec.H])
    else:
        w = spec.H * onp.tan(spec.psi)
        lo = onp.array([0.0] * (d - 2) + [-w, 0.0])
        hi = onp.array([spec.Lperp] * (d - 2) + [w, spec.H])
    return lo, hi


def bbox_volume(spec):
    """The V that goes into the insertion and deletion acceptances."""
    lo, hi = bbox(spec)
    return float(onp.prod(hi - lo))


def volume(spec):
    """The TRUE accessible volume, which is what rho is measured against."""
    d, g = spec.d, spec.geom
    if g in ("bulk", "slit", "box"):
        return spec.H * spec.Lperp ** (d - 1)
    if g == "sphere":
        return _ball(d, spec.H)
    if g == "cylinder":
        return _ball(d - 1, spec.Lperp) * spec.H
    return spec.H ** 2 * onp.tan(spec.psi) * spec.Lperp ** (d - 2)


def periods(spec):
    """(period (d,), periodic (d,) bool). Aperiodic axes carry a period of 1.0
    so that the minimum-image arithmetic never sees an infinity."""
    d, g = spec.d, spec.geom
    if g == "bulk":
        per = onp.array([spec.Lperp] * (d - 1) + [spec.H])
        flag = onp.ones(d, dtype = bool)
    elif g == "slit":
        per = onp.array([spec.Lperp] * (d - 1) + [1.0])
        flag = onp.array([True] * (d - 1) + [False])
    elif g == "box":
        per = onp.ones(d)
        flag = onp.zeros(d, dtype = bool)
    elif g == "sphere":
        per = onp.ones(d)
        flag = onp.zeros(d, dtype = bool)
    elif g == "cylinder":
        per = onp.array([1.0] * (d - 1) + [spec.H])
        flag = onp.array([False] * (d - 1) + [True])
    else:
        per = onp.array([spec.Lperp] * (d - 2) + [1.0, 1.0])
        flag = onp.array([True] * (d - 2) + [False, False])
    return per, flag


def bin_volume(spec, nbins):
    """(nbins,) volume of each profile bin, exactly, not to leading order.

    For a slit these are all equal and this is overkill. For a spherical pore
    they are shell volumes growing like r^{d-1}, and getting them wrong tilts
    the whole profile in a way that looks like physics.
    """
    d, g = spec.d, spec.geom
    edges = onp.linspace(0.0, extent(spec), nbins + 1)
    if g in ("bulk", "slit", "box"):
        return onp.full(nbins, spec.Lperp ** (d - 1) * (spec.H / nbins))
    if g == "sphere":
        return _ball(d, edges[1:]) - _ball(d, edges[:-1])
    if g == "cylinder":
        return (_ball(d - 1, edges[1:]) - _ball(d - 1, edges[:-1])) * spec.H
    # Wedge: the slab between z and z+dz is a strip 2 z tan(psi) wide.
    return (onp.tan(spec.psi) * (edges[1:] ** 2 - edges[:-1] ** 2)
            * spec.Lperp ** (d - 2))


def max_separation(spec):
    """Largest centre-to-centre distance the region admits, or inf.

    Finite only where every axis is bounded and short. Returned as inf wherever
    an axis is periodic or extended, since two centres can then be arbitrarily
    far apart (or, under the minimum image, as far as half the period).
    """
    if spec.geom == "sphere":
        return 2.0 * spec.H                     # antipodes of the ball
    if spec.geom == "box":
        lo, hi = bbox(spec)
        return float(onp.sqrt(onp.sum((hi - lo) ** 2)))     # the long diagonal
    if spec.geom == "slit" and spec.d == 1:
        return spec.H                           # the two walls
    return float("inf")


def max_p_separation(spec):
    """The same bound read in the p-norm the overlap test actually uses.

    Not a rescaling of `max_separation`, because which separation is extremal
    depends on the norm. In a cuboid the furthest pair sits at the corners and
    the answer is the p-norm of the edge vector directly; in a ball the
    constraint is Euclidean and the worst direction is the one maximising
    ||x||_p at fixed ||x||_2, which `shapes.max_norm_ratio` gives.
    """
    if spec.geom == "box":
        lo, hi = bbox(spec)
        return float(shapes.norm(spec.shape, onp.asarray(hi - lo)))
    return max_separation(spec) * shapes.max_norm_ratio(spec.shape, spec.d)


def is_zero_dimensional(spec):
    """True if the region cannot hold two particles at once.

    The 0-D limit of fundamental measure theory, reached through geometry
    rather than through `d`: see the note in `mcax.eos`. A spherical pore of
    radius below sigma/2 qualifies in any dimension, and so does a
    one-dimensional slit narrower than sigma.

    The comparison is in the p-norm, because that is the metric the overlap
    test is written in, and for a non-spherical shape the two disagree. A
    cuboid cavity is the sharpest case: parallel CUBES escape each other as
    soon as one axis separates them by sigma, so a cavity is 0-D for them only
    if every edge is shorter than sigma, whereas spheres in the same cavity are
    still excluded out to the diagonal.

    Strict, not `<=`: at exactly 2R = sigma two centres sit at contact, which
    the overlap test allows, and the cavity would hold two.
    """
    return max_p_separation(spec) < spec.sigma


def extent(spec):
    """Upper end of the profile coordinate."""
    if spec.geom == "sphere":
        return spec.H                       # the radius
    if spec.geom == "cylinder":
        return spec.Lperp                   # the radius
    return spec.H


# ---- the two traced answers ----------------------------------------------- #

def contains(spec, r):
    """True if the centre `r` (shape (d,)) is inside the accessible region.

    Called on the trial position of every move, so it has to be branch-free in
    the traced arguments; the geometry itself is static, so the Python `if`
    below is resolved once at trace time.
    """
    g = spec.geom
    if g == "bulk":
        return np.asarray(True)
    if g == "slit":
        return (r[-1] >= 0.0) & (r[-1] <= spec.H)
    if g == "box":
        hi = np.asarray(bbox(spec)[1])
        return np.all((r >= 0.0) & (r <= hi))
    if g == "sphere":
        return np.sum(r * r) <= spec.H ** 2
    if g == "cylinder":
        return np.sum(r[:-1] * r[:-1]) <= spec.Lperp ** 2
    w = r[-1] * np.tan(spec.psi)
    return (r[-1] >= 0.0) & (r[-1] <= spec.H) & (np.abs(r[-2]) <= w)


def profile_coord(spec, pos):
    """The coordinate the density profile is binned in, for every particle.

    `pos` is (Nmax, d); the return is (Nmax,). Radial for the curved pores,
    the wall axis otherwise.
    """
    if spec.geom == "sphere":
        return np.sqrt(np.sum(pos * pos, axis = -1))
    if spec.geom == "cylinder":
        return np.sqrt(np.sum(pos[:, :-1] * pos[:, :-1], axis = -1))
    return pos[:, -1]


def wrap(spec, r):
    """Fold a trial position back through the periodic axes only."""
    per, flag = periods(spec)
    return np.where(np.asarray(flag), np.mod(r, np.asarray(per)), r)


def min_image(spec, dr):
    """Minimum-image separation, applied on the periodic axes only."""
    per, flag = periods(spec)
    per, flag = np.asarray(per), np.asarray(flag)
    return np.where(flag, dr - per * np.round(dr / per), dr)


# ---- teaching ------------------------------------------------------------- #

def describe(spec):
    """One-line reading of what this spec's parameters mean. E.G.

        >>> describe(make_spec(d = 3, geom = "cylinder", H = 10.0,
        ...                    Lperp = 4.0, z_act = 1.0))
        'cylinder in d = 3: radius 4.0, axial period 10.0, volume 502.65'
    """
    g, d = spec.geom, spec.d
    v = volume(spec)
    if g == "bulk":
        return f"bulk in d = {d}: box {spec.Lperp} x .. x {spec.H}, volume {v:.2f}"
    if g == "slit":
        return (f"slit in d = {d}: walls on centres at 0 and {spec.H}, "
                f"transverse {spec.Lperp}, volume {v:.2f}")
    if g == "box":
        return (f"hard box in d = {d}: edges {spec.Lperp} x .. x {spec.H}, "
                f"walls on every face, volume {v:.2f}")
    if g == "sphere":
        return f"spherical pore in d = {d}: radius {spec.H}, volume {v:.2f}"
    if g == "cylinder":
        return (f"cylinder in d = {d}: radius {spec.Lperp}, axial period "
                f"{spec.H}, volume {v:.2f}")
    return (f"wedge in d = {d}: half-angle {onp.degrees(spec.psi):.1f} deg, "
            f"height {spec.H}, volume {v:.2f}")
