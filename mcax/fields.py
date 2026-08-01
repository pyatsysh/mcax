"""External fields: beta V_ext(r), the one-body term in the muVT weight.

With a field switched on the configurational weight picks up a factor
exp(-beta V_ext) per particle, and the three acceptances become

    displacement : min(1, exp(-[V(r') - V(r)]))
    insertion    : min(1, z V / (N+1) exp(-V(r_new)))
    deletion     : min(1, N / (z V) exp(+V(r_i)))

with the hard-core overlap test unchanged. Everything here is in units of
kT already, so `eps` means beta epsilon and temperature enters only through it.

**Why these are NamedTuples and not closures.** The field travels in the spec
as a static argument, so `jit` keys its cache on the field's hash. A closure
hashes by identity, which means building the "same" field twice compiles the
kernel twice, and a sweep over wall strengths that reuses one spec shape would
recompile at every point for no reason. A NamedTuple hashes by VALUE, so
`Gravity(2.0) == Gravity(2.0)` and the cache hits. They are callable because
they define `__call__`, and they compose because `Sum` holds a tuple of them.

**The ideal-gas limit is the exact test for any field.** Switch the core off and
the density is barometric,

    rho(r) = z exp(-beta V_ext(r)),

point by point, with no approximation at all. That holds for every field in this
module and for anything a user writes, so it is the first thing to check against
after adding one.

**Infinities are clipped, deliberately.** A 9-3 wall diverges at contact and
`V(r') - V(r)` between two divergent points is a nan, which would silently
poison an acceptance rather than reject a move. Fields are therefore clipped to
+/- 700 in the kernel: exp(-1400) is zero to a double and the arithmetic stays
finite, so a divergence rejects the way it should instead of propagating.

E.G. a slit with an attractive 9-3 wall on both sides, which is the standard
wetting geometry:

    w = fields.LJ93Wall(eps = 1.2, sigma_w = 1.0)
    spec = make_spec(d = 3, H = 12.0, z_act = ...,
                     field = fields.Sum((w, fields.Mirror(w, 12.0))))
"""
from typing import NamedTuple

import jax.numpy as np

CLIP = 700.0        # exp(-2 * CLIP) is zero to a double, and finite


class Gravity(NamedTuple):
    """beta V = g z: a linear field along the profile axis.

    The barometric law rho(z) = z_act exp(-g z) is its exact ideal-gas answer,
    which makes it the cheapest field to validate against.
    """
    g: float

    def __call__(self, r):
        return self.g * r[-1]


class Harmonic(NamedTuple):
    """beta V = k (z - z0)^2 / 2 along the profile axis."""
    k: float
    z0: float = 0.0

    def __call__(self, r):
        return 0.5 * self.k * (r[-1] - self.z0) ** 2


class IsotropicTrap(NamedTuple):
    """beta V = k |r|^2 / 2, the isotropic version. Pairs with `sphere`."""
    k: float

    def __call__(self, r):
        return 0.5 * self.k * np.sum(r * r)


class LJ93Wall(NamedTuple):
    """The 9-3 wall, from integrating Lennard-Jones over a half-space:

        beta V(z) = eps [ (2/15) (sigma_w/z)^9 - (sigma_w/z)^3 ]

    Its minimum sits at z = 0.4^(1/6) sigma_w = 0.858 sigma_w and is 1.054 eps
    deep, so `eps` is the well depth only to within that factor. This is the
    field the wetting and drying literature runs on, and the one worth reaching
    for when comparing against a cDFT surface phase diagram.
    """
    eps: float
    sigma_w: float = 1.0
    z0: float = 0.0

    def __call__(self, r):
        z = np.abs(r[-1] - self.z0)
        # A bare (sigma/z)^9 overflows to inf long before z reaches zero, and
        # inf - inf downstream is a nan. Flooring the separation keeps the
        # divergence large, finite and monotone.
        s = self.sigma_w / np.maximum(z, 1e-3 * self.sigma_w)
        return self.eps * ((2.0 / 15.0) * s ** 9 - s ** 3)


class SquareWellWall(NamedTuple):
    """beta V = -eps within `width` of the wall, zero beyond it.

    The crudest surface field that still produces wetting, and the one whose
    cDFT counterpart is a one-line term. Attractive for eps > 0.
    """
    eps: float
    width: float = 1.0
    z0: float = 0.0

    def __call__(self, r):
        z = np.abs(r[-1] - self.z0)
        return np.where(z < self.width, -self.eps, 0.0)


class ExpWall(NamedTuple):
    """beta V = -eps exp(-(z - z0)/decay): a smooth attractive surface field."""
    eps: float
    decay: float = 1.0
    z0: float = 0.0

    def __call__(self, r):
        return -self.eps * np.exp(-np.abs(r[-1] - self.z0) / self.decay)


class Mirror(NamedTuple):
    """`inner` measured from the far wall of a slit of width H.

    Reflects the profile axis, z -> H - z, so one wall field can furnish both
    walls without writing it twice and without the two drifting apart.
    """
    inner: tuple
    H: float

    def __call__(self, r):
        return self.inner(r.at[-1].set(self.H - r[-1]))


class Sum(NamedTuple):
    """Several fields at once. `parts` is a tuple, so this stays hashable."""
    parts: tuple

    def __call__(self, r):
        total = 0.0
        for p in self.parts:
            total = total + p(r)
        return total


def both_walls(inner, H):
    """`inner` applied at both walls of a slit of width H."""
    return Sum((inner, Mirror(inner, H)))


def evaluate(field, r):
    """beta V_ext(r), clipped, or exactly zero when there is no field.

    The clip is what the kernel calls, so a field written by hand gets the same
    protection as the ones above.
    """
    if field is None:
        return np.asarray(0.0)
    return np.clip(field(r), -CLIP, CLIP)
