# mcax

Batched hard-particle **grand-canonical Monte Carlo in JAX**: hard rods, discs
and spheres in bulk and in slit confinement, many independent chains advanced in
lockstep on one device, everything jit and scan compiled.

Documentation and the longer argument: **https://pyatsysh.github.io/mcax/**

Hard cores mean there are no energies anywhere in the code. The Boltzmann factor
is an indicator function, so an overlap test plus three exact acceptances are the
whole of the physics, which is what keeps the engine small enough to audit line
by line.

## Why an accelerator changes Monte Carlo

A Markov chain cannot be parallelised inside itself: step `n+1` depends on step
`n`, and no hardware changes that. That is why Monte Carlo spent twenty years
watching molecular dynamics move onto GPUs without following.

The second axis was always there. Chains are independent of one another, so the
state becomes an array with a leading chain index, one `vmap` advances the whole
batch, and the sequential dependence stays along an axis nobody wanted to
parallelise. Two consequences are worth more than the speed: chain-to-chain
scatter is an honest independent-replica error bar, and because the chains really
are independent, adding chains buys effective sample size linearly, where extra
draws down a single chain buy nothing until that chain has mixed.

Compilation matters separately from the device. A Monte Carlo loop in NumPy
spends of order a microsecond per step inside the interpreter, more than the
arithmetic costs for a few hundred particles. `jit` and `lax.scan` remove that on
a laptop as much as on a GPU.

## What is on offer today

`jax-md` is JAX-native and excellent, but it is built for soft potentials and its
Monte Carlo is the hybrid swap kind aimed at glassy systems: hard cores in the
grand canonical ensemble are not its ensemble. HOOMD-blue HPMC is the reference
implementation of hard-particle Monte Carlo, faster per chain than anything here,
and a CUDA and C++ codebase that cannot be called from inside a training loop.
Writing your own takes an afternoon for the sampler and a month for the
validation, because the sampler is three acceptance ratios and knowing that they
are right is the actual work.

mcax fills the gap those three leave. It matters here because the consumers are
already JAX: the density-functional models, the training loops that fit them, and
the solvers underneath. A sampler in the same framework composes with them rather
than exporting to them, so grand-canonical ground truth can be generated inside a
training loop instead of shipped to it as a frozen dataset.

## Install

```bash
git clone https://github.com/pyatsysh/mcax && cd mcax
pip install -e ".[dev]"
```

Python 3.11 or newer, `jax` and `numpy`. Nothing else, and see
[Diagnostics](#diagnostics) for why that stays true. `float64` is not optional:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Use

```python
import jax; jax.config.update("jax_enable_x64", True)
from mcax import make_spec, burn_and_sample, summary, format_summary, eos

# Hard spheres at bulk packing fraction eta = 0.3, in a slit 8 diameters wide.
rho_b = 0.3 / eos.B[3]
spec = make_spec(d = 3, H = 8.0, Lperp = 10.0, slit = True,
                 z_act = float(eos.z_of_rho(3, rho_b)))

res = burn_and_sample(spec, C = 64, seed = 0, n_burn = 200_000,
                      n_run = 1_000_000, thin = 500, nbins = 160)

print(format_summary(summary(res.Ns)))     # mixing, ESS, split R-hat
assert res.capacity_warning is None        # always check this
res.z, res.rho                             # the density profile
```

`burn_and_sample` returns a `Result`: the profile, the particle-number series
`Ns` shaped `(chains, draws)`, acceptance rates per move type, the capacity
diagnostics, and the final `state` so a run can be continued.

### Geometry and conventions

`slit = True` puts hard walls on particle **centres** at `x_d = 0` and `H`, with
the transverse axes periodic. That is the centre-exclusion convention the DFT
reference data uses, which is what makes the contact theorem `rho(0+) = beta P`
apply literally rather than at a profile shifted by `sigma/2`. `slit = False` is
periodic everywhere. Lengths are in units of `sigma`, with `beta = 1` and
`Lambda = 1`.

### The three moves

```
displacement : accept iff no overlap and inside the walls
insertion    : accept with min(1, z V / (N+1)) iff no overlap
deletion     : accept with min(1, N / (z V)),        z = exp(beta mu)
```

Every step all chains draw a move type, compute all three branches
vectorised-and-masked, and select. That throws away two branches out of three and
wastes about three times the arithmetic. It is still the right trade, because it
keeps the batch free of divergence, which is what lets one compiled kernel
advance every chain at once.

## Validation

`mcax.eos` carries the reference equations of state. They are both the validation
targets and the way a run is set up, since the sampler takes an activity while a
physicist thinks in packing fraction.

The one-dimensional case is kept deliberately, and it is the reason to believe
the other two. Tonks and Percus solved hard rods in closed form, so a
detailed-balance error or a factor of the volume in the wrong place fails against
an exact number rather than against another simulation carrying its own errors.
In two and three dimensions the best references available are accurate to about
0.1%, so a marginal disagreement there is ambiguous between our bug and the
reference equation. In one dimension it is a bug.

| Case | Reference | Reference accuracy | Measured rel. error |
|---|---|---|---|
| d = 1 bulk, eta = 0.5 | Tonks and Percus | **exact** | VAL_D1 |
| d = 2 bulk, eta = 0.4 | Henderson | 0.1% | VAL_D2 |
| d = 3 bulk, eta = 0.3 | Carnahan and Starling | 0.1% | VAL_D3 |
| d = 1 slit contact, eta = 0.5 | rho(0+) = beta P, exact Tonks | **exact** | VAL_CONTACT |

Every case asserts split R-hat as well as the density, which is not decoration.
On the first full run the three-dimensional case matched Carnahan-Starling to
0.1% while its chains had not mixed at all, at R-hat 1.23: started empty at an
insertion acceptance near 1%, they spent the whole burn-in filling. The mean was
right for the wrong reason and only the mixing statistic noticed. Seating 90% of
the target occupancy on a lattice and melting it in the burn removes the
transient, which is what `lattice_fill` is for.

```bash
pytest                        # 102 fast tests, 103 s on CPU
pytest -m slow                # full EOS validation, minutes per case
python scripts/validate.py --dims 1     # watch the exact cases run
python examples/wall_profile.py         # the figure above
```

The slow tier is deselected by default so a bare `pytest` is a usable
pre-commit gate.

Alongside the statistics, the fast suite pins the invariants that catch sampler
bugs rather than confirm sampler statistics: no two live centres closer than
`sigma`, verified with a separate NumPy implementation of the boundary
conventions so a geometry error cannot hide behind itself; the ideal-gas limit
`rho = z` at vanishing core size, which tests the acceptances with the overlap
test removed from the picture entirely; accepted insertions balancing accepted
deletions at equilibrium; per-chain PRNG independence; and bit-exact
reproduction from a seed.

![Hard rods at a wall against exact Tonks](docs/wall_profile.png)

## Diagnostics

`mcax.diagnostics` gives split R-hat, ESS by Stan's estimator with Geyer
initial-positive-sequence truncation, and MCSE, in NumPy alone.

```python
from mcax import summary
summary(res.Ns)     # mean, sd, mcse, ess, split_rhat
```

The boundary is deliberate. The natural output of a batched sampler is a
`(chain, draw)` array, which is exactly what ArviZ and NumPyro consume, so mcax
produces that layout and stops there. With ArviZ installed,
`az.summary(az.convert_to_dataset(res.Ns[None]))` works on the same array.
Interop by data shape costs nothing, whereas depending on a
probabilistic-programming stack to compute forty lines of autocovariance would be
a poor trade for a library whose install is otherwise just JAX.

## Performance

The figure of merit is **chain-steps per second**, since a step advances all `C`
chains at once and quoting bare steps per second understates the throughput by a
factor of `C`. On the heaviest production shape (d = 3, H = 8, eta = 0.35,
`Nmax` about 1300, 64 chains) the GPU ran about **21 times** the CPU throughput.

```bash
JAX_PLATFORMS=cpu  python scripts/bench.py
JAX_PLATFORMS=cuda python scripts/bench.py
```

Chain count is the dial. The batch axis is what the accelerator is for, and below
about 16 chains a GPU sits mostly idle.

## What it will not do

Being written in JAX does not make this differentiable, and that is worth saying
before anything else, because the inference is natural and it is wrong.
Acceptance is a discrete accept or reject against a uniform draw, the hard-core
log-density is an indicator function with no gradient anywhere, and the
grand-canonical state space is trans-dimensional. `jax.grad` of a profile with
respect to `mu` or `sigma` is not unimplemented, it is undefined by this route.
Response functions come from fluctuation identities instead, where the derivative
with respect to `beta mu` is a sampled covariance, and that is the supported
answer.

- **Fixed capacity.** A chain holds at most `Nmax` particles, and an insertion
  into a full chain is rejected. That is not a Metropolis rejection: it truncates
  the ensemble and breaks detailed balance. Nothing in a profile would reveal it,
  so every `Result` carries `saturation` and a `capacity_warning` that fires at
  95%. Read it before believing a number.
- **The overlap test is O(N).** Each trial move is tested against every live
  particle, with no cell list, and the cost is set by `Nmax` rather than by the
  occupancy. Comfortable to a few hundred particles per chain, and the first
  thing that has to change for large boxes.
- **Per-chain speed is not competitive.** A good single-chain C code beats it.
  The win is in the batch and in living inside JAX.
- **Single-component spheres.** No mixtures, no aspherical shapes, no external
  field beyond the hard walls.

## Licence

Apache-2.0.
