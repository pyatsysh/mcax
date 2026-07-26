"""mcax: batched hard-particle grand-canonical Monte Carlo in JAX."""
from .core import (MCSpec, MCState, Result, make_spec, init_state,
                   lattice_fill, run, density_profile, burn_and_sample)
from .diagnostics import split_rhat, ess, mcse, summary, format_summary
from . import eos

__version__ = "0.1.0"
