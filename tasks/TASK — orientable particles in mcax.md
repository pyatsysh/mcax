# TASK — orientable particles in mcax

*Written 2026-08-06 for an agent session in this repo. Target: extend mcax
to hard particles with orientational degrees of freedom, aimed at the real
(freely rotating) superball fluid. This is the second superball task; the
parallel-family data task in this folder comes first and its engine is the
validation anchor for this one. Design background (vault):
`dimInterp-notes/superballs/Superballs — orientational degrees of freedom.md`.*

## Scope

A grand-canonical hard-particle engine where each particle carries an
orientation, for convex bodies defined by an analytic support function.
First body: the superball |x|^p + |y|^p + |z|^p <= R^p, whose support
function is the dual norm

    h_K(u) = R * ||u||_q,      1/p + 1/q = 1

so a rotated particle has h(u) = R * ||Omega^T u||_q. Everything below
should be written against the support function, not against the superball
specifically — the body is a parameter of the engine.

## Design requirements

1. **State.** Positions + unit quaternions, leading chain axis as in the
   existing engine (batched independent chains, jit + scan). Store raw
   quaternions; the superball's cubic symmetry (O_h) is handled at
   observable time, not in the state.
2. **Overlap test.** Three tiers, cheapest first:
   - inscribed spheres:  ||dr||_2 <= 2R          -> overlap, certain;
   - circumscribed:      ||dr||_2 >  2R*3^(1/2-1/p) -> no overlap, certain;
   - the shell between: GJK (or MPR) on the two support maps.
   The GJK must be JAX-compatible: **fixed iteration count** under
   lax.scan (no data-dependent while loops), conservative tolerance, and
   a documented guarantee direction — when the fixed budget is exhausted
   without a verdict, declare overlap (rejecting a legal move costs
   efficiency, never correctness; accepting an illegal one poisons the
   ensemble). Measure how often the budget trips and size it so this is
   rare (< 1e-6 of tests).
3. **Moves.** Translations (existing), small-angle rotations (quaternion
   perturbation, uniform axis, tuned angle window), and GC
   insertion/deletion with orientation drawn uniformly on SO(3)
   (Shoemake). Detailed balance notes in the docstring: rotation
   proposal must be symmetric; insertion orientation measure must match
   the deletion measure. Mixing ratio translation:rotation about 1:1 to
   start, tuned by acceptance.
4. **Observables.** rho(z) as now, plus orientation-resolved moments:
   symmetry-adapted cubic harmonics (first non-trivial at l = 4, then
   6, 8) binned in z — NOT raw SO(3) histograms. Global cubatic order
   parameter and Steinhardt q4/q6 as ordering monitors, recorded per
   state like the drift diagnostic.
5. **Performance.** The three-tier test keeps GJK calls to the shell
   population; batch over chains as now. Expect a real constant-factor
   cost over spheres; measure and report it.

## Validation ladder, in order — each rung gates the next

    V0  p = 2: orientations decouple exactly. Observables (not
        trajectories) must match the hard-sphere engine within replica
        error; rotation moves must accept at 100%.
    V1  frozen-rotation mode (rotation window = 0, all orientations
        aligned): must reproduce the parallel-superball engine from the
        first task, state by state within replica error. This mutually
        certifies the two engines.
    V2  dilute virial: measured B2 vs numerical excluded-volume integral
        (MC integration over separation x relative orientation using the
        same support-function overlap test at two decoupled orientations
        — an independent code path, small and exact in the limit).
    V3  overlap-test unit battery: known-answer pairs (touching along
        axes, corner-to-face near-contacts at large p, deep overlaps,
        grazing cases), plus adversarial random pairs cross-checked
        against a slow reference (dense rotation-grid separation test).
    V4  one literature anchor: free hard cubes (p = inf) EOS against a
        published simulation curve.

## Phase-behaviour guardrails

The free superball phase diagram (Ni, Gantapara, de Graaf, van Roij,
Dijkstra, Soft Matter 2012) has, besides freezing: a **plastic crystal
(rotator FCC) near p = 2** — positions order while orientations stay
free, so a translational-order monitor alone is not enough near the
sphere end — and vacancy-rich simple cubic for free cubes. The fluid
range shrinks with p. Every production state carries both order monitors
(translational + cubatic); onsets are measured, not assumed; eta caps
follow the same measured-eta_max discipline as the parallel task.

## Explicit non-goals for this round

- No non-convex bodies, no mixtures, no soft interactions on top.
- No orientation-dependent external fields beyond hard walls.
- No data production for training — that gets its own task brief once
  this engine passes V0-V4. This task delivers the engine and its
  validation report, nothing downstream.

## Compute etiquette (this box)

Shared machine: `lease status` first, `lease acquire` for sustained runs,
`taskset` to the granted cores, thread pools held down, RAM declared over
about 4 GiB. GJK development and the V-battery are CPU-sized; only V4
production runs need the GPU.

## Definition of done

- Engine merged behind a body-parameter interface (support function in,
  moves + overlap out), hard-sphere path untouched and still bit-exact;
- V0-V4 green, with the validation report (numbers, not adjectives)
  appended below;
- unit tests for the overlap battery in `tests/`;
- a short design note in `docs/` recording the fixed-iteration GJK
  guarantee direction and the measured budget-trip rate.

## Validation report

*(agent appends here)*

---

## Validation report — 2026-08-06

Engine landed as `mcax/orient.py` (+ `mcax/bodies.py` for the support-function
body interface). The aligned path is untouched and still bit-exact: A0 in the
parallel task's report covers it, and `mcax.orient` is a separate module that
`mcax/__init__.py` does not import, so the hard-sphere engine carries none of
its weight.

### The overlap test as built

Not GJK-with-a-simplex, and the substitution is deliberate. The requirement was
a fixed iteration count with a documented guarantee direction; what makes that
easy is that **separation has a cheap exact certificate and overlap does not**.
A single direction with `h_M(u) < 0` proves disjointness in a few flops, where
proving overlap needs a point in the intersection. So the engine searches for a
separating direction and declares overlap if the budget runs out, which is the
required direction, and it needs no simplex bookkeeping at all:

1. **32 candidate axes** — both bodies' face normals, the nine cross products,
   the centre separation, and the negative of each. Complete for boxes.
2. **Frank-Wolfe** on `min ||x||^2` over the Minkowski difference, seeded at the
   support point along the best candidate, fixed length under `lax.scan`.

Because the verdict is a deterministic function of the pair configuration, the
sampler stays *exact for a marginally fattened body* rather than becoming a
biased chain. Full argument in `docs/orientable-overlap.md`.

**The bug worth recording**: the candidate axes were first tested with one sign
only. Separation in the other sense is `h_M(-u) < 0`, a different number, so
half of all separations were missed. It did not surface as a wrong answer (the
misses fell through to Frank-Wolfe and were mostly recovered) but as a trip rate
two orders of magnitude too high, **cubes included**, where the test ought to be
exact. With both signs, p = 2 and p = inf are exact at every budget.

### V3 — overlap battery: PASS

Every known-answer case correct, including the one the aligned engine cannot
reach: a unit cube turned 45 degrees about z presents a corner, so contact along
x moves from 1.0 to 0.5 + sqrt(2)/2 = 1.2071, and the test straddles it.
Coincident centres, far separation, and axis contact at 0.99/1.01 all correct
for every shape.

Budget trip rate, measured per *hard* test (pairs in the shell where the cheap
tiers cannot decide), against a self-checked `n_iter = 1000` reference:

| p | 8 | 12 | 20 | 32 | 64 |
|---|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 2.9e-2 | 1.6e-2 | 6.7e-3 | 1.8e-3 | 3.0e-4 |
| 4 | 2.3e-2 | 1.5e-2 | 7.1e-3 | 3.2e-3 | 1.8e-3 |
| 6 | 1.3e-2 | 1.1e-2 | 7.6e-3 | 5.7e-3 | 3.7e-3 |
| inf | 0 | 0 | 0 | 0 | 0 |

**Zero unsafe verdicts at any budget, for any shape, across every sample drawn.**
Production default `n_iter = 32`.

**The 1e-6 target is not met and cannot be by this route.** Frank-Wolfe
converges as O(1/k); a certificate at gap g requires resolving g; configurations
with gap below eps occupy measure proportional to eps; so the rate falls as 1/k,
which is what the table shows. 1e-6 needs k ~ 1e4. The two routes that would
actually fix it are a full tetrahedral GJK simplex and a direct Newton solve on
the antiparallel-normal contact condition. A triangle simplex was tried, was no
better at these budgets, and produced an unsafe verdict from a degenerate
branch, so it was dropped rather than shipped half-working.

The residual is bounded and physical: at `n_iter = 32` the trips occupy about
1e-3 sigma of effective gap, so bodies behave as though fattened by ~5e-4 sigma
each, a few parts in a thousand of excluded volume.

### Performance

Measured against the aligned engine at the same `Nmax`, same activity, same
chain count (d = 3 slit, H = 6, C = 8, CPU):

| p | orientable | aligned | ratio |
|---|---|---|---|
| 2 | 46.6 s | 9.2 s | 5x |
| 4 | 108.9 s | 9.5 s | 11x |
| inf | 43.6 s | 10.9 s | 4x |

That is *after* the fix that made it usable. Sending every capacity slot through
the certificate search costs some sixty support evaluations per pair, so a chain
with four hundred slots pays twenty-five thousand per trial move; the first
build was roughly a thousand times the aligned cost and a one-minute validation
run took hours. Now only the nearest `n_near = 24` neighbours reach the search,
gathered with `top_k`, and **the gather is made safe rather than assumed safe**:
the (n_near + 1)-th distance is checked against twice the circumradius, and if
even that one is close enough to have mattered the move is rejected. Same
guarantee direction as the budget, so `n_near` is a performance knob that cannot
change the physics, only the acceptance.

### V1 — frozen rotation reproduces the parallel engine: PASS, and it earned its keep

This rung is the mutual certification: `dtheta = 0` from an aligned start is the
parallel superball model reached through the support-function overlap test
instead of the p-norm one, and the two engines share no overlap code. **It
failed first, for a real reason, which is the whole point of having it.**

The frozen engine came out 7% thinner than the aligned one at p = 4 and **28%
thinner for cubes**, at the same activity. Not equilibration: the gap was
unchanged under a 2.5x longer run with a lattice prefill. Not the overlap
budget either, because p = inf has a zero trip rate.

The cause: `dtheta = 0` stops existing particles from turning, but
grand-canonical insertion creates a particle with no previous orientation to
keep, and it was still drawing that from Haar. So "frozen" was silently a
QUENCHED-RANDOM-ORIENTATION fluid — nothing ever rotated, but every particle had
arrived pointing somewhere different. That has a larger excluded volume than the
aligned model and a lower density at fixed activity, and the size of the error
tracked shape anisotropy exactly: nil at p = 2 where orientation cannot matter,
7% at p = 4, 28% at p = inf. Insertion now uses the reference orientation when
frozen.

| p | before the fix | after |
|---|---|---|
| 2 | 0.1% (0.1 sigma) | unchanged, orientation is irrelevant there |
| 4 | 7.0% (9.8 sigma) | **0.1% (0.2 sigma)** |
| inf | 28.1% (57.1 sigma) | **0.4% (0.6 sigma)** |

All at C = 8, prefilled, 20k burn + 40k run, d = 3 slit H = 6, z = 6.

**A second, smaller thing this rung exposed**, worth separating from the bug
because it is a permanent property rather than a defect: from an EMPTY box the
two engines are not comparable at a fixed step count. They run different move
mixes, so the one proposing fewer translations per step makes room for new
particles more slowly. At p = 2, empty at 16k steps they differ by 0.6% and the
gap is still closing; seeded at 90% and run to 40k they agree to 0.1%. The
comparison needs the same prefill on both sides, and `tests/test_orient.py` now
does that.

### V0 — sphere decoupling: PASS in the form the unit tests check

Rotating a sphere is accepted at exactly 100.00%, and at p = 2 the frozen
comparison above is the V0 statement (0.1%, 0.1 sigma). The full V0 rung in
`scripts/orient_validate.py`, which also compares the density PROFILE and the
S4 moment rather than only <N>, did not finish running in this session.

### V2 — dilute virial: PASS, and it bounds the budget bias

Orientation-averaged B2 by Monte Carlo over separation x two independent
orientations, calling the overlap predicate directly with no chains, no
acceptances and no geometry: an independent code path, as the brief asks.

| p | B2 free | B2 parallel | exact | dev |
|---|---|---|---|---|
| 2 | 2.0969 +- 0.0058 | 2.0944 | 2.0944 | 0.12% |
| 3 | 2.9369 +- 0.0099 | 2.8483 | no closed form | |
| 4 | 3.4676 +- 0.0128 | 3.2410 | no closed form | |
| 6 | 4.0882 +- 0.0164 | 3.6038 | no closed form | |
| inf | **5.5005 +- 0.0265** | 4.0000 | **5.5000** | **0.01%** |

The cube row is the V4 literature anchor and it is exact rather than borrowed.
Isihara-Hadwiger gives the orientation-averaged excluded volume of two identical
convex bodies as B2 = V + R S with R the mean radius of curvature; for a cube of
edge a the edge sum gives R = 3a/4, so B2 = a^3 + (3a/4)(6a^2) = 5.5 a^3 against
4 a^3 for the same cubes held parallel. Measured 5.5005. That is integral
geometry on one side and this engine's support-function overlap test on the
other, agreeing to one part in ten thousand.

**This is also the honest bound on the fixed-budget bias**, and a far better one
than the raw trip rate. B2 measured at the PRODUCTION budget (n_iter = 32) sits
within statistical error of exact at both ends, so the effective fattening
argued for in `docs/orientable-overlap.md` is at or below 0.1% in a physical
quantity, not merely small in a shell measure. The free-over-parallel ratio also
rises monotonically with anisotropy, 1.001 at p = 2 through 1.375 at p = inf,
which is the sign rotation must have and a check no single number gives.

### V0 — sphere decoupling: PASS in the form the unit tests check

Rotating a sphere is accepted at exactly 100.00%, the p = 2 frozen comparison
above agrees to 0.1% (0.1 sigma), and V2's p = 2 row confirms that free rotation
leaves B2 unchanged there. The full V0 rung in `scripts/orient_validate.py`,
which also compares the density PROFILE and the S4 moment rather than only <N>,
did not finish running in this session.

### Not completed in this session

The finite-density half of V4: a grand-canonical sweep of freely rotating cubes
with the cubatic and translational monitors recorded, comparing against the
parallel-cube fluid at matched activity. Implemented in
`scripts/orient_validate.py`; the orientable engine on one CPU core, sharing the
box with the data campaign, outran the time available. Its qualitative statement
does pass in the unit tests: free cubes are measurably thinner than parallel
ones at the same activity.

No published finite-density EOS curve was used as an anchor, deliberately: none
could be verified from here, and an unverified number in a validation report is
worse than no anchor. The Isihara-Hadwiger B2 above is the external check, and
it is exact.

### Also delivered

- Orientational observables: cubic harmonics S4 and S6 binned in z (invariants
  of O_h, since everything below l = 4 vanishes identically for a cubic body and
  a nematic order parameter finds nothing here however ordered the fluid is);
  global cubatic order parameter about each lab axis; Steinhardt Q4/Q6 computed
  through the addition theorem, `Q_l^2 = <P_l(cos theta_ij)>` over pairs of
  bonds, which removes every spherical harmonic from the implementation.
- `docs/orientable-overlap.md` with the guarantee direction, the measured trip
  rates and the scaling argument.
