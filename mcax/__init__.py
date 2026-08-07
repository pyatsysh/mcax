"""mcax: batched hard-particle grand-canonical Monte Carlo in JAX.

Two engines, sharing their geometry, fields and diagnostics:

    mcax           positions only. Spheres, or ALIGNED superballs of any
                   exponent p, whose overlap test is a p-norm against sigma.
    mcax.orient    positions and orientations. Any convex body with a support
                   map; the overlap test is a certificate search and is the one
                   approximate answer in the library. See
                   `docs/orientable-overlap.md`.

`mcax.orient` is imported on demand rather than here, so the aligned engine
carries none of its compile weight.
"""
from .core import (MCSpec, MCState, Result, Accum, make_spec, init_state,
                   lattice_fill, run, density_profile, burn_and_sample,
                   require_float64)
from .diagnostics import split_rhat, ess, mcse, summary, format_summary
from .observables import (susceptibility, compressibility, response_profile,
                          pair_correlation, structure_factor_zero)
from .shapes import Superball, SPHERE, CUBE, OCTAHEDRON
from . import bodies
from . import eos
from . import fields
from . import geometry
from . import observables
from . import order
from . import potentials
from . import shapes

__version__ = "0.2.0"
