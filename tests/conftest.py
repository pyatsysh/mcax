"""Shared fixtures and helpers.

Two things are set before JAX is touched, both deliberately:

**CPU by default.** A test suite must never contend with a real workload for an
accelerator: someone else's job on the GPU is always higher priority than our
unit tests, and these runs are small enough that CPU is the right device anyway.
Override with ``JAX_PLATFORMS=cuda pytest ...`` if you actually want the device.

**float64.** Not optional for this engine: an overlap test at
``d2 >= sigma**2`` in float32 admits marginal overlaps, and the acceptance
ratios lose their last digits at high N.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as onp
import pytest

import jax

jax.config.update("jax_enable_x64", True)


def alive_positions(result_or_state, chain: int = 0) -> onp.ndarray:
    """The ``(N, d)`` live particle positions of one chain."""
    st = getattr(result_or_state, "state", result_or_state)
    pos = onp.asarray(st.pos[chain])
    alive = onp.asarray(st.alive[chain])
    return pos[alive]


def periodic_axes(spec):
    """(period, periodic) per axis, written out here on purpose.

    This duplicates the table in ``mcax.geometry``, and the duplication is the
    test: the kernel's minimum image is only checked if the checker does not
    import the thing it is checking. Transcribed from the geometry docstring,
    not from its code.
    """
    d = spec.d
    if spec.geom == "bulk":
        return onp.array([spec.Lperp] * (d - 1) + [spec.H]), onp.ones(d, bool)
    if spec.geom == "slit":
        return (onp.array([spec.Lperp] * (d - 1) + [1.0]),
                onp.array([True] * (d - 1) + [False]))
    if spec.geom == "sphere":
        return onp.ones(d), onp.zeros(d, bool)
    if spec.geom == "cylinder":
        return (onp.array([1.0] * (d - 1) + [spec.H]),
                onp.array([False] * (d - 1) + [True]))
    return (onp.array([spec.Lperp] * (d - 2) + [1.0, 1.0]),
            onp.array([True] * (d - 2) + [False, False]))


def min_pair_separation(spec, pts: onp.ndarray) -> float:
    """Smallest centre-centre distance under the spec's boundary conventions.

    Independent re-implementation in NumPy: the point is to check the JAX
    kernel's geometry, so sharing code with it would defeat the test.
    """
    n = len(pts)
    if n < 2:
        return onp.inf
    dr = pts[:, None, :] - pts[None, :, :]
    per, wrap = periodic_axes(spec)
    folded = dr - per * onp.round(dr / per)
    dr = onp.where(wrap, folded, dr)
    d2 = onp.sum(dr * dr, axis=-1)
    onp.fill_diagonal(d2, onp.inf)
    return float(onp.sqrt(d2.min()))


def inside_region(spec, pts: onp.ndarray) -> onp.ndarray:
    """Boolean mask of which centres lie in the accessible region.

    Same reasoning as above: transcribed from the prose description of each
    geometry rather than delegated to ``geometry.contains``.
    """
    if len(pts) == 0:
        return onp.zeros(0, bool)
    if spec.geom == "bulk":
        return onp.ones(len(pts), bool)
    if spec.geom == "slit":
        return (pts[:, -1] >= -1e-12) & (pts[:, -1] <= spec.H + 1e-12)
    if spec.geom == "sphere":
        return onp.sum(pts ** 2, axis=1) <= spec.H ** 2 + 1e-9
    if spec.geom == "cylinder":
        return onp.sum(pts[:, :-1] ** 2, axis=1) <= spec.Lperp ** 2 + 1e-9
    return ((pts[:, -1] >= -1e-12) & (pts[:, -1] <= spec.H + 1e-12)
            & (onp.abs(pts[:, -2]) <= pts[:, -1] * onp.tan(spec.psi) + 1e-9))


@pytest.fixture(scope="session")
def tol():
    """Statistical tolerance for the short runs used in the fast suite."""
    return 0.05
