"""Particle shapes: the hard core generalised from a sphere to a superball.

A superball is the body

    |x_1|^p + ... + |x_d|^p  <=  (sigma/2)^p,

that is, a ball of the p-norm. The exponent sweeps a family that is worth having
in one place: p = 1 is a cross-polytope (an octahedron in d = 3), p = 2 is the
sphere the rest of this library was written for, p -> infinity is a cube, and
everything between is a rounded cube or a rounded octahedron. `sigma` is the
width along a coordinate axis in every case, so it is the sphere diameter at
p = 2 and the cube edge at p = infinity.

**Particles here are ALIGNED and do not rotate.** Every superball has the same
orientation, fixed to the coordinate axes, so the configuration space is still
positions alone and the sampler needs no rotational move. That is not a
simplification of an orientational problem, it is a different (and standard)
model: the parallel hard cube fluid of Cuesta and Martinez-Raton is the
p -> infinity member of exactly this family.

**Why the overlap test is one line.** For two translates of a convex body K,

    (K + a) and (K + b) intersect   <=>   a - b  in  K + (-K),

and a superball is centrally symmetric, so K + (-K) = K + K = 2K. Two aligned
superballs therefore overlap precisely when their centre separation lies in the
superball scaled by two, which is

    |dr_1|^p + ... + |dr_d|^p  <  sigma^p,

the p-norm written against sigma exactly as the Euclidean test is written
against sigma^2. Nothing about the sampler changes: the core is still an
indicator, the geometry is still hard, and at p = 2 the arithmetic is the old
arithmetic to the last bit.

**Convexity is the whole of the assumption, so p >= 1 is enforced.** Below p = 1
the body is concave, K + K is strictly larger than 2K, and the criterion above
silently admits configurations that overlap. That failure is invisible
downstream (the profiles stay plausible), so it raises rather than warns.

**In d = 1 every p is the same fluid.** The criterion collapses to |dr| < sigma
whatever p is, which is Tonks for the whole family. That makes rods the razor
for the shape layer just as they are for everything else: any p-dependence
appearing in one dimension is a bug, and `tests/test_shapes.py` asserts its
absence.
"""
from math import gamma, inf
from typing import NamedTuple

import numpy as onp
import jax.numpy as np


class Superball(NamedTuple):
    """The p-norm hard core. `p = 2.0` is the sphere, `p = inf` is the cube.

    Held as a NamedTuple of floats so that two identically-parameterised shapes
    hash equal and share one jit cache entry, the same reasoning as
    `mcax.fields` and `mcax.potentials`.
    """
    p: float

    @property
    def is_sphere(self):
        return self.p == 2.0

    @property
    def is_cube(self):
        return self.p == inf


SPHERE = Superball(2.0)
CUBE = Superball(inf)
OCTAHEDRON = Superball(1.0)


def check(shape):
    """Reject a shape the Minkowski argument above does not cover."""
    if shape is None:
        return
    if not isinstance(shape, Superball):
        raise TypeError(f"shape must be a Superball or None, got {shape!r}")
    if not (shape.p >= 1.0):
        raise ValueError(
            f"superball exponent must be p >= 1, got {shape.p}. Below 1 the "
            f"body is concave, so the overlap test (a p-norm against sigma) "
            f"stops being exact and admits real overlaps. Nothing downstream "
            f"would reveal that, hence the refusal.")


def _p_of(shape):
    """The exponent, with `None` reading as the sphere."""
    return 2.0 if shape is None else float(shape.p)


# ---- the traced answer ---------------------------------------------------- #

def overlaps(shape, dr, sigma):
    """True where the minimum-image separation `dr` (..., d) is an overlap.

    `shape = None` and `Superball(2.0)` both take the Euclidean branch, which is
    the sum of squares compared against sigma^2 and therefore bit-identical to
    the hard-sphere engine that predates this module. The general branch costs
    one `abs` and one power per component, and the cube costs neither.
    """
    p = _p_of(shape)
    if p == 2.0:
        return np.sum(dr * dr, axis = -1) < sigma ** 2
    if p == inf:
        # The infinity-norm, i.e. parallel hard cubes: two cubes of edge sigma
        # miss each other as soon as ONE axis separates them by a full edge.
        return np.max(np.abs(dr), axis = -1) < sigma
    # An INTEGER exponent lowers to repeated multiplication, a float one to a
    # library pow, and the difference is not marginal: the overlap test is the
    # innermost line of the sampler and a general pow costs about an order of
    # magnitude more than a multiply. A measured d = 3 state took 114 s with
    # p = 4.0 and 66 s with p = 4, for identical arithmetic. So the exponent is
    # narrowed to an int wherever it is one, and an EVEN one skips the abs as
    # well since a negative base is squared away.
    ip = int(p) if float(p).is_integer() else None
    if ip is not None:
        s = np.sum(dr ** ip if ip % 2 == 0 else np.abs(dr) ** ip, axis = -1)
        return s < sigma ** ip
    if float(2.0 * p).is_integer():
        # Half-integer, which p = 2.5 is: |x|^(n + 1/2) = |x|^n sqrt(|x|), and
        # a square root is a hardware instruction where a general power is a
        # polynomial evaluation. Worth the branch because the shape grid has a
        # half-integer in it and it would otherwise be the slowest run of the
        # campaign by a factor of three.
        n, a = int(p - 0.5), np.abs(dr)
        s = np.sum((a ** n) * np.sqrt(a), axis = -1)
        return s < sigma ** p
    # abs() first, so a negative base never reaches the power. Raising a
    # negative float to a non-integer exponent is nan, and a nan compared with
    # anything is False, which would read as "no overlap" and quietly sample
    # the wrong ensemble.
    return np.sum(np.abs(dr) ** p, axis = -1) < sigma ** p


def norm(shape, dr):
    """The p-norm of `dr` itself, in the units sigma is measured in.

    Not needed by the kernel (which compares the p-th power and skips the root)
    but wanted whenever a separation has to be reported or histogrammed in the
    metric the particles actually see.
    """
    p = _p_of(shape)
    if p == 2.0:
        return np.sqrt(np.sum(dr * dr, axis = -1))
    if p == inf:
        return np.max(np.abs(dr), axis = -1)
    return np.sum(np.abs(dr) ** p, axis = -1) ** (1.0 / p)


# ---- the static answers --------------------------------------------------- #

def volume(shape, d, sigma = 1.0):
    """Volume of one superball,

        v = sigma^d Gamma(1 + 1/p)^d / Gamma(1 + d/p),

    which is sigma^d pi/6 at p = 2 in d = 3 (the sphere), sigma^d at
    p = infinity (the cube) and sigma^d / d! at p = 1 (the cross-polytope).

    The spread is the reason `sigma` alone does not fix a state point in this
    family: at fixed sigma and fixed density the packing fraction changes by a
    factor of six between an octahedron and a cube in three dimensions. Quote
    eta, or fix the volume with `sigma_for_volume`.
    """
    p = _p_of(shape)
    if p == inf:
        return float(sigma) ** d
    return float(sigma) ** d * gamma(1.0 + 1.0 / p) ** d / gamma(1.0 + d / p)


def packing(shape, d, sigma = 1.0):
    """eta per unit density: the shape's entry in the table `mcax.eos.B` holds
    for spheres. Identically 1 in d = 1 for every p, as it must be."""
    return volume(shape, d, sigma)


def b2(shape, d, sigma = 1.0):
    """Second virial coefficient, EXACT for this family and any p >= 1.

        B_2 = (1/2) vol(K + K) = (1/2) 2^d v = 2^(d-1) v,

    because the excluded region of two aligned convex centrally-symmetric
    bodies is the body itself doubled. In packing-fraction form B_2 rho =
    2^(d-1) eta, which is 4 eta for spheres in three dimensions, the familiar
    number. This is the one non-trivial equation of state coefficient available
    in closed form across the whole family, so it is what a measured rho(z)
    should be checked against at low density.
    """
    return 2.0 ** (d - 1) * volume(shape, d, sigma)


def sigma_for_volume(shape, d, v = 1.0):
    """The sigma at which one particle occupies volume `v`.

    The alternative parameterisation of a sweep over p: hold the particle
    volume fixed rather than the axial width, so that a given density is the
    same packing fraction at every p and the shape is the only thing varying.
    Which of the two is wanted depends on what the comparison is for, so both
    are available and neither is the default.
    """
    return float(v / volume(shape, d, 1.0)) ** (1.0 / d)


def support_radius(shape, d, sigma = 1.0):
    """Largest Euclidean distance from the centre to the surface.

    The body's circumradius: sigma/2 times d^(1/2 - 1/p) for p <= 2 (the
    octahedron's vertices lie on the axes and this is 1) and sigma/2 times
    d^(1/2) for p >= 2 with the cube's corners furthest out. Wanted for
    bounding boxes and for deciding whether a cavity is zero-dimensional.
    """
    p = _p_of(shape)
    if p >= 2.0:
        # Vertices of the enclosing cube, approached as p -> infinity; for
        # finite p >= 2 the corner sits at (sigma/2) d^(1/p) along the diagonal
        # in p-norm terms, i.e. at Euclidean radius (sigma/2) d^(1/2 - 1/p).
        return 0.5 * sigma * d ** (0.5 - 1.0 / p)
    return 0.5 * sigma


def inradius(shape, d, sigma = 1.0):
    """Largest Euclidean ball inside the body: sigma/2 for p >= 2, and
    (sigma/2) d^(1/2 - 1/p) for p <= 2, where the faces cut closest in."""
    p = _p_of(shape)
    if p >= 2.0:
        return 0.5 * sigma
    return 0.5 * sigma * d ** (0.5 - 1.0 / p)


def max_norm_ratio(shape, d):
    """sup ||x||_p / ||x||_2 over directions, which converts a Euclidean bound
    on a separation into the p-norm bound the overlap test reads.

    Equal to d^(1/p - 1/2) when p <= 2 (worst along the diagonal) and to 1 when
    p >= 2 (worst along an axis). Used by `mcax.geometry` to decide whether a
    cavity can hold two centres at all.
    """
    p = _p_of(shape)
    return 1.0 if p >= 2.0 else float(d) ** (1.0 / p - 0.5)


def describe(shape, d = 3, sigma = 1.0):
    """One line naming the shape and its volume, for logs and run headers."""
    p = _p_of(shape)
    name = ("sphere" if p == 2.0 else "cube" if p == inf else
            "cross-polytope" if p == 1.0 else
            "rounded octahedron" if p < 2.0 else "rounded cube")
    ps = "inf" if p == inf else f"{p:g}"
    return (f"superball p = {ps} ({name}) in d = {d}: axial width {sigma:g}, "
            f"volume {volume(shape, d, sigma):.5f}, "
            f"B2 {b2(shape, d, sigma):.5f}")


# ---- host-side reference, for tests --------------------------------------- #

def contains_point(shape, x, sigma = 1.0):
    """Is the point `x` (an (..., d) host array) inside one superball centred
    at the origin. Plain numpy, never traced: this exists so that the overlap
    predicate above can be checked against a brute-force intersection of two
    bodies rather than against itself."""
    p = _p_of(shape)
    x = onp.asarray(x, dtype = float)
    if p == inf:
        return onp.max(onp.abs(x), axis = -1) <= 0.5 * sigma
    return onp.sum(onp.abs(x) ** p, axis = -1) <= (0.5 * sigma) ** p
