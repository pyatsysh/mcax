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

### Not completed in this session

V0, V1, V2 and V4 are implemented in `scripts/orient_validate.py` and were
running at session end but did not finish: the orientable engine on one CPU core,
sharing the box with the data campaign, is slower than the time left. They are
one command (`--quick` for a smoke, bare for production) and the script writes
`out/orient_validation.json`.

What the unit tests already establish of them, and which passed:
`tests/test_orient.py` contains miniature V0 (rotating a sphere accepts at
100.00%), miniature V1 (frozen rotation matches the aligned engine within
replica error at p = 2 and p = 4), Shoemake uniformity, proposal symmetry,
support-map correctness and central symmetry, rotation-invariance of verdicts,
the guarantee-direction assertion, and the qualitative V4 statement that free
cubes are thinner than parallel ones at the same activity.

**Known gap**: the V2 and V4 numbers against the Isihara-Hadwiger B2 = 5.5 for
free unit cubes (against 4 for parallel ones) are computed by the script but were
not read off in this session. That analytic anchor was chosen over a published
EOS curve deliberately, because no such curve could be verified from here and an
unverified number in a validation report is worse than no anchor.

### Also delivered

- Orientational observables: cubic harmonics S4 and S6 binned in z (invariants
  of O_h, since everything below l = 4 vanishes identically for a cubic body and
  a nematic order parameter finds nothing here however ordered the fluid is);
  global cubatic order parameter about each lab axis; Steinhardt Q4/Q6 computed
  through the addition theorem, `Q_l^2 = <P_l(cos theta_ij)>` over pairs of
  bonds, which removes every spherical harmonic from the implementation.
- `docs/orientable-overlap.md` with the guarantee direction, the measured trip
  rates and the scaling argument.
