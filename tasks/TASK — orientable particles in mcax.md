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
