"""mcax: batched hard-particle grand-canonical Monte Carlo in JAX."""
from .core import (MCSpec, MCState, Result, Accum, make_spec, init_state,
                   lattice_fill, run, density_profile, burn_and_sample,
                   require_float64)
from .diagnostics import split_rhat, ess, mcse, summary, format_summary
from .observables import (susceptibility, compressibility, response_profile,
                          pair_correlation, structure_factor_zero)
from . import eos
from . import observables

__version__ = "0.1.0"
