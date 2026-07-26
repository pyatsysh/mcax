# mcax

Batched hard-particle **grand-canonical Monte Carlo in JAX** — hard rods,
discs and spheres in bulk and slit/wall confinement, many independent chains
in lockstep on one device (CPU or GPU), everything jit/scan-compiled.

Hard cores mean no energies anywhere: an overlap test plus the exact muVT
acceptances are the whole physics, which makes the engine small, auditable
and fast. Statistics come from the chain batch (vmap), not from clever
single-chain moves — the shape accelerators actually like.

Why it exists: as of 2026 there is no JAX-native hard-particle GCMC —
jax-md offers hybrid swap MC for soft glasses, and the hard-particle
tradition (HOOMD HPMC) is excellent but neither JAX-composable nor light.
mcax fills the gap for the classical-DFT workflow that needs muVT density
profiles at walls and in slits as machine-learning ground truth, with the
d = 1 hard-rod case kept deliberately: Tonks/Percus is exact there, so any
detailed-balance or bookkeeping bug fails against an exact answer.

## Use

```python
from mcax import make_spec, burn_and_sample
import numpy as np

spec = make_spec(d=3, H=8.0, z_act=np.exp(2.0), Lperp=10.0, slit=True)
z, rho, Ns, acc = burn_and_sample(spec, C=64, seed=0,
                                  n_burn=200_000, n_run=1_000_000,
                                  thin=500, nbins=160)
```

Validation: `scripts/validate.py` — bulk EOS inversions (Tonks exact, SPT,
Carnahan-Starling) and the wall contact theorem `rho(0+) = beta P` on exact
Tonks pressure.

Requires `jax` (float64). No other dependencies.
