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


def min_pair_separation(spec, pts: onp.ndarray) -> float:
    """Smallest centre-centre distance under the spec's boundary conventions.

    Independent re-implementation in NumPy: the point is to check the JAX
    kernel's geometry, so sharing code with it would defeat the test.
    """
    n = len(pts)
    if n < 2:
        return onp.inf
    dr = pts[:, None, :] - pts[None, :, :]
    if spec.slit:
        # transverse axes periodic, wall axis direct
        if spec.d > 1:
            dr[..., :-1] -= spec.Lperp * onp.round(dr[..., :-1] / spec.Lperp)
    else:
        box = onp.array([spec.Lperp] * (spec.d - 1) + [spec.H])
        dr -= box * onp.round(dr / box)
    d2 = onp.sum(dr * dr, axis=-1)
    onp.fill_diagonal(d2, onp.inf)
    return float(onp.sqrt(d2.min()))


@pytest.fixture(scope="session")
def tol():
    """Statistical tolerance for the short runs used in the fast suite."""
    return 0.05
