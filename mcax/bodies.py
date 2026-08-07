"""Convex bodies described by their support function, for rotatable particles.

A convex body K is fixed completely by its support function

    h_K(u) = max_{x in K} u . x

and by the SUPPORT MAP s_K(u) = argmax_{x in K} u . x, which is the point of K
furthest along u. Everything the orientable engine needs is one or the other:
the overlap test in `mcax.orient` never looks at the body itself, only asks it
for support points, so a new particle shape is a new support map and nothing
else. That is the interface the engine is written against.

**The superball's support map, in closed form.** Maximising u . x subject to
sum |x_i|^p <= R^p with a Lagrange multiplier gives |x_i| proportional to
|u_i|^(1/(p-1)), and writing q for the conjugate exponent, 1/p + 1/q = 1, the
normalised answer is

    s(u) = R sgn(u) |u|^(q-1) / ||u||_q^(q-1),        h(u) = R ||u||_q

so the support function is the DUAL norm, as it is for any norm ball. At p = 2
this is R u / ||u||_2, the obvious point on a sphere; at p = infinity the
conjugate is q = 1 and it collapses to R sgn(u), the corner of the cube in u's
octant. Both are taken directly rather than as limits.

**Central symmetry is used and is not incidental.** Every body here satisfies
K = -K, so s(-u) = -s(u), and the Minkowski difference of two of them loses a
sign: B - A = B + A up to the centre offset. `mcax.orient` leans on that, so a
body without central symmetry may not simply be dropped in.

**Orientation.** A rotated body Omega K has h(u) = h_K(Omega^T u) and support
map Omega s_K(Omega^T u). The rotation stays outside the body, which is what
lets one support map serve every orientation.
"""
from math import gamma, inf
from typing import NamedTuple

import numpy as onp
import jax.numpy as np


class Superball(NamedTuple):
    """|x_1|^p + ... + |x_d|^p <= R^p, with R = sigma/2 the axial half-width.

    Carried as floats so two identical bodies hash equal and share a jit cache
    entry, as everywhere else in this library. `p >= 2` is what the orientable
    engine is validated over; p = 2 is the sphere and is the case orientations
    decouple in, which is why it is the first rung of the validation ladder.
    """
    p: float
    R: float = 0.5

    @property
    def sigma(self):
        return 2.0 * self.R

    @property
    def q(self):
        """The conjugate exponent, 1/p + 1/q = 1. Equals 1 for the cube."""
        return 1.0 if self.p == inf else self.p / (self.p - 1.0)


def check(body):
    if not isinstance(body, Superball):
        raise TypeError(f"body must be a Superball, got {body!r}")
    if not (body.p >= 1.0):
        raise ValueError(f"superball exponent must be p >= 1 (convexity), "
                         f"got {body.p}")
    if not body.R > 0:
        raise ValueError(f"R must be positive, got {body.R}")


# ---- the traced answers ---------------------------------------------------- #

_EPS = 1e-300


def support(body, u):
    """(..., d) the point of the body furthest along `u`, in the body frame.

    `u` need not be normalised: the map is scale-invariant in u, as it must be,
    since it answers a question about a direction.
    """
    p, q, R = body.p, body.q, body.R
    if p == inf:
        # q = 1 and the exponent q - 1 vanishes, so this is R sgn(u) with no
        # normalisation at all: the corner of the cube in u's octant. Taken
        # directly rather than as a limit of the expression below, which would
        # be 0**0 componentwise.
        return R * np.sign(u)
    if p == 2.0:
        return R * u / np.maximum(np.sqrt(np.sum(u * u, axis = -1,
                                                 keepdims = True)), _EPS)
    a = np.abs(u)
    w = a ** (q - 1.0)
    nq = np.sum(a ** q, axis = -1, keepdims = True) ** ((q - 1.0) / q)
    return R * np.sign(u) * w / np.maximum(nq, _EPS)


def h(body, u):
    """The support VALUE, R ||u||_q. Cheaper than a support point when only the
    number is wanted, which is what the separating-axis certificates use."""
    p, q, R = body.p, body.q, body.R
    if p == inf:
        return R * np.sum(np.abs(u), axis = -1)             # q = 1
    if p == 2.0:
        return R * np.sqrt(np.sum(u * u, axis = -1))
    return R * np.sum(np.abs(u) ** q, axis = -1) ** (1.0 / q)


# ---- the static answers ---------------------------------------------------- #

def inradius(body, d):
    """Largest ball inside the body. For p >= 2 the surface comes closest along
    a coordinate AXIS and this is R; below p = 2 the faces cut in along the
    diagonal instead."""
    if body.p >= 2.0:
        return body.R
    return body.R * d ** (0.5 - 1.0 / body.p)


def circumradius(body, d):
    """Smallest ball containing the body: R d^(1/2 - 1/p) for p >= 2, where the
    corners reach furthest along the diagonal, and R below that.

    These two radii are the cheap tiers of the overlap test. Two bodies whose
    centres are closer than twice the inradius certainly overlap; further apart
    than twice the circumradius, they certainly do not. Only the shell between
    needs a real answer, and for a superball that shell is thin: at p = 4 in
    three dimensions it is 1.00 to 1.32 diameters.
    """
    if body.p >= 2.0:
        return body.R * d ** (0.5 - 1.0 / body.p)
    return body.R


def volume(body, d):
    """(2R)^d Gamma(1 + 1/p)^d / Gamma(1 + d/p), the same formula
    `mcax.shapes.volume` uses for the aligned family. Kept here too so the two
    engines can be compared without either importing the other's conventions."""
    if body.p == inf:
        return body.sigma ** d
    return (body.sigma ** d * gamma(1.0 + 1.0 / body.p) ** d
            / gamma(1.0 + d / body.p))


def describe(body, d = 3):
    ps = "inf" if body.p == inf else f"{body.p:g}"
    return (f"superball p = {ps}, sigma = {body.sigma:g} in d = {d}: "
            f"volume {volume(body, d):.5f}, inradius {inradius(body, d):.4f}, "
            f"circumradius {circumradius(body, d):.4f}")


# ---- host-side reference, for tests ---------------------------------------- #

def contains_point(body, x):
    """Is the body-frame point `x` inside. Plain numpy, never traced: exists so
    the support map can be checked against the body it claims to describe."""
    x = onp.asarray(x, dtype = float)
    if body.p == inf:
        return onp.max(onp.abs(x), axis = -1) <= body.R * (1.0 + 1e-12)
    return (onp.sum(onp.abs(x) ** body.p, axis = -1)
            <= body.R ** body.p * (1.0 + 1e-12))
