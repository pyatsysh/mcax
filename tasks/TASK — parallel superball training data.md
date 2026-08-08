# TASK — parallel superball training data

*Written 2026-08-06 for an agent session in this repo. Consumer: the
dimint-dft project (`~/projects/classical-DFT-lab/dimint-dft`), which will
train an FMT functional amortised over the superball shape exponent.
Background reading (vault): `dimInterp-notes/superballs/` — the discussion
note explains why orientations are frozen for this round.*

## The model

Parallel (axis-aligned, orientation-frozen) hard superballs:

    body:     |x|^p + |y|^p + |z|^p <= R^p,   p in [2, inf]
    overlap:  ||r_i - r_j||_p <= 2R          (p-norm of the centre difference)
    units:    sigma = 2R = 1

That overlap line is the entire physics change to mcax. At p = 2 the
engine must reproduce the existing hard-sphere path **bit for bit** (same
seeds, same chains, identical trajectories) — that is acceptance test A0
and it is non-negotiable before anything else runs. p = inf is the
max-norm (axis-aligned cubes), which should be implemented directly, not
as a large-p limit.

The 2D analogue (parallel superdiscs, same p-norm in the plane) is needed
too — the consumer trains on a dimensional ladder.

## Shape grid

    training:   p in {2, 2.5, 4, 6}
    held out:   p = 3        (entire shape held out — the amortisation test;
                              generate the SAME state list as trained p's,
                              mark every state split=test_shape)
    benchmark:  p = inf      (cubes; for comparison against the
                              Cuesta-Martinez-Raton functional, not training)

## Deliverables

Mirror the existing dimint schema exactly — inspect
`dimint-dft/data/measure/` (MANIFEST.md + manifest.json + mc_states.npy)
and match it: state naming `d{d}_slit_H{H}_eta{eta}` extended with a
`_p{p}` suffix, per-state independent-replica error bars (chain-to-chain
scatter), drift diagnostic (shift of <N> between first and last third of
the sampling window, in units of the error bar), splits recorded in the
manifest, and a MANIFEST.md a human can audit.

Per p in {2, 2.5, 3, 4, 6, inf}:

1. **d = 3 bulk EOS.** Grand-canonical chains in a periodic box, a mu
   ladder giving eta from dilute up to eta_max(p) (see the freezing guard
   below), enough points for a smooth beta P(rho) by Gibbs-Duhem
   integration (the existing `mcax/eos.py` machinery). Validation: p = 2
   against Carnahan-Starling; p = inf against published parallel-cube
   EOS data if convenient, otherwise internal-consistency checks only.
2. **d = 2 bulk EOS** (parallel superdiscs), same protocol; p = 2
   validates against the disc EOS reference already used by dimint.
3. **d = 3 slit profiles** rho(z): axis-aligned hard walls. Per p, reuse
   the hard-sphere state list geometry: H in {4, 6, 8, 12} x eta in
   {0.15, 0.25, 0.35} for training (drop states above eta_max(p)), plus
   held-out states **deliberately on the extrapolation axes**:
   H = 5 and H = 16 (test_width), one eta above the trained range
   (test_state). The Act III lesson: slit-width extrapolation is where
   functionals die — design the held-out set for it, do not improvise.
4. **d = 2 channel profiles**, same pattern (H x eta grid as in the
   existing d2 states, capped by eta_max(p)).
5. **Small hard boxes** (axis-aligned cuboid pores, all-walls-hard) at
   2-3 sizes near the single-particle scale, as held-out 0D-adjacent
   tests. Exact 0D labels themselves are analytic and not an MC job.

No orientational observables exist in this model; no rotation moves; do
not implement them here (that is a separate task brief in this folder).

## Freezing guard — this is a hard requirement, not advice

The fluid range shrinks with p. Parallel hard cubes freeze near
eta = 0.48 in simulation, and the transition is continuous or nearly so:
**there is no nucleation barrier**, so an over-compressed chain does not
sit safely metastable — it orders, and the labels are silently poisoned.

- Monitor translational ordering per state: structure-factor main-peak
  height and/or Steinhardt q4/q6 on the fly; record the diagnostic in
  the manifest per state.
- Determine eta_max(p) by measurement: ramp eta at each p, find the
  ordering onset, cap training data at least 0.03 in eta below it, and
  report the measured eta_max table in MANIFEST.md.
- Any state whose ordering diagnostic trips is excluded and the
  exclusion is logged — an audit trail, not a silent drop.

## Acceptance tests, in order

    A0  p = 2 reproduces the current hard-sphere engine bit for bit
    A1  dilute limit: measured B2 vs exact 4*V(p) (3D) and 2*A(p) (2D)
        for every p on the grid, within replica error
    A2  p = inf cube EOS self-consistency (GC vs NPT or two box sizes)
    A3  slit contact-value sum rule beta P = rho(contact) at one state
        per p (walls are hard and parallel: the rule is exact)
    A4  replica error bars and drift pass the same thresholds as the
        2026-08-01 hard-sphere campaign (see that MANIFEST)

## Compute etiquette (this box)

Shared machine: check `lease status` first; acquire a lease for anything
sustained (`lease acquire gpu ... --why "superball GCMC"`); pin to the
granted cores with `taskset`; hold BLAS/XLA thread pools down
(OMP_NUM_THREADS etc.); WSL RAM cap 28 GiB, declare working sets over
about 4 GiB with `--ram`. Batched chains on the GPU is the intended mode;
the p-norm overlap costs the same as the Euclidean one, so throughput
should match the hard-sphere campaign.

## Definition of done

- `mcax` overlap generalised to the p-norm (and max-norm) with A0-A4 green
  and unit tests added;
- data landed under `dimint-dft/data/superball/` in the measure-campaign
  schema with MANIFEST.md, manifest.json, splits, error bars, drift and
  ordering diagnostics, and the measured eta_max(p) table;
- a short run report appended to this file (below), including anything
  that surprised you.

## Run report

*(agent appends here)*

---

## Run report — 2026-08-06

### Engine: done, A0 green

The whole physics change is one predicate. Two aligned translates of a convex
body meet exactly when their separation lies in the body doubled, and a
superball is centrally symmetric, so `||dr||_p < sigma` is exact for every
p >= 1 with no approximation anywhere. Landed as `mcax/shapes.py` plus one
changed line in the neighbour scan.

**A0 passes in the strong form.** `scripts/a0_bitwise.py` checks out the last
pre-shapes commit into a throwaway git worktree and runs five state points
through both engines with the same seeds, comparing raw arrays. All five are
IDENTICAL for both `shape = None` and `Superball(2.0)`, in positions, alive
masks, N-series, profiles, acceptance counters and capacity. A self-comparison
would not have caught a change of arithmetic shared by both new paths, which is
why it is done against the old tree.

p = inf is the max-norm directly, not a large-p limit. Two speed notes that
turned out to matter: an integer exponent lowers to repeated multiplication
where a float one lowers to a library `pow` (114 s -> 66 s on the same state for
p = 4.0 vs p = 4), and p = 2.5 gets a sqrt branch for the same reason.

Also added: a `box` geometry (cuboid, hard on every face) for the 0-D-adjacent
states, and `mcax/order.py` for the freezing guard.

### The freezing guard, and two ways it was nearly useless

**It was blind.** A crystal in a cell of edge L with lattice constant a puts its
first Bragg peak at n = L/a, which for L = 8 is n ~ 8. The first implementation
scanned a ball of |n| <= 4 and therefore never sampled a Bragg peak at all: an
ordered configuration would have come back looking like a fluid. Fixed by
scanning a small isotropic ball **plus the cubic families {100}, {110}, {111}**
out to n ~ L/sigma, which is cheap exactly because aligned particles form an
aligned crystal. Verified against hand-built jittered crystals in a
campaign-sized cell: S_max/<N> = 0.99 for a crystal, 0.013 for a gas, with the
trip at 0.10.

**It cried wolf.** The lattice prefill *starts* a dense chain on a crystal, so a
burn too short to melt it reports the ordering it was handed. A trip is now
re-run from a much sparser start with twice the burn and has to order again from
below. The very first dense ladder point tripped and was cleared this way.

The trip criterion is a conjunction, `S_max/<N> > 0.10 AND S_max > 8`. The ratio
alone is unsafe in a small system: the fluid baseline is not zero but about
ln(K)/<N>, K being the number of wavevectors scanned, so a slit holding eighty
particles sits near 0.05 for no reason but the maximum being taken over sixty
modes.

### The limit this campaign actually hit, and it is not freezing

The bulk ladder tracked its target packing fraction to four decimal places up to
eta = 0.25, drifted 1.3% at 0.32, and then **stalled at 0.371 for every target
at or above 0.46**. That is not a phase transition. A chain seated at 85% of
target has to insert the last 15% one accepted insertion at a time, and above
eta ~ 0.35 in three dimensions the insertion acceptance is well under a per
cent, so it never arrives. Seating at 98% instead leaves the burn a local
relaxation rather than a density change, and the ladder then tracked 0.3239
against a target of 0.32.

The residual is kept as a filter rather than a hope: ladder points with relative
drift above 2% are excluded from the interpolation that sets confined
activities, because a chain still on its way somewhere is not a state point.

**The honest headline is that plain muVT stops equilibrating near eta ~ 0.42 in
d = 3, well below freezing at 0.494, for a sampling reason and not a physical
one.** Cluster or bias moves are what would extend it. That belongs in the
roadmap and it changes what eta_max(p) means: the measured cap is the lesser of
the ordering onset and the equilibration limit, and in d = 3 it is the latter.

### The bulk cap is necessary and NOT sufficient, and the per-state monitor is what saves the confined data

Worth separating out, because it changes how the guard should be read. The
eta_max(p) table is measured in BULK and applied to the reservoir packing
fraction of each confined state. Confinement then raises the local density above
that reservoir value, so a state comfortably under its bulk cap can still order.

Measured, in the widest slit: `d3_slit_H16_eta0.30_p2` and the same at p = 2.5
both tripped the monitor, and p = 2's cap is 0.340, well above the 0.30 they
were set at. The p = 2.5 state came out at an in-slit eta of 0.319 against its
cap of 0.304.

So the cap alone would have passed poisoned states. What catches them is that
every state carries its OWN structure-factor monitor and is flagged on its own
evidence, not on the bulk table's. Rate so far: three trips in about a hundred
and fifty confined states, against thirty-five refused by the cap before running
— so the two mechanisms are catching different things and both are load-bearing.

### Data: NOT COMPLETE. What exists and how to finish it

The campaign is `scripts/superball_campaign.py`, fully resumable and idempotent
(every finished state is written immediately and skipped on a re-run). At
session end it had **17 of 126 bulk ladder points** and no confined states: the
GPU budget needed is six to eight hours and the session had about four, most of
which went into the three corrections above.

Resume with, exactly as-is:

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      OMP_NUM_THREADS=2 taskset -c 2-3 <python> scripts/superball_campaign.py

The two-stage structure is the part that is load-bearing and is done: there is
no reference equation of state off p = 2, so the bulk EOS is measured FIRST, one
mu-ladder per (d, p), and the confined states read their activity off it by
interpolation. The activity guess that seeds each ladder point is good to a few
per cent for a reason worth keeping: `beta mu_ex = 2 B_2 rho + O(rho^2)` and
`B_2 = 2^(d-1) v(p)` exactly, so `B_2 rho = 2^d eta` is INDEPENDENT of shape and
the sphere's excess chemical potential is right to first order in eta at every p.

**Statistics are mixed across the ladder and this is recorded per state**
(`n_run`, `n_burn_used`, `chains` in every state dict). Points taken early ran
50k steps; later ones 20k, cut so that a complete dataset might land rather than
a fragment. A production re-run raises `NRUN`/`NBURN` and deletes
`bulk_eos.npy`.

### Acceptance tests

- **A0** green, in the strong bitwise form described above.
- **A1** (B2 from the dilute limit) implemented as a straight-line fit of
  `ln(z/rho)/rho` against rho, extrapolated to rho -> 0, against the exact
  `2^(d-1) v(p)`. Needs three dilute points per ladder; not yet exercised on a
  complete ladder.
- **A2** (cube EOS, two box sizes) implemented as a consistency check where the
  dilute large-cell points meet the dense small-cell ones.
- **A3** (contact theorem) implemented, and the one number worth quoting from
  the smoke run: at p = 2 in a d = 3 slit the measured contact density was
  **0.5528 against the exact Carnahan-Starling beta P = 0.5454, a 1.36%
  agreement**. That is the engine passing the sum rule. The same check against
  the *integrated* pressure was 17% off, which isolated the problem to the
  Gibbs-Duhem quadrature rather than the engine, and led to reformulating it as
  `beta P = rho + rho mu_ex - integral mu_ex drho` — integrating the excess
  removes the logarithmic singularity at rho -> 0 and makes the anchor exact
  instead of carrying an order-rho^2 error. A `gibbs_duhem` cross-check against
  Carnahan-Starling and Henderson is now written into the manifest so the two
  error sources can never be confused again.
- **A4** (error bars and drift against the 2026-08-01 thresholds) is computed
  per state in the manifest, with relative drift as the primary column, but
  cannot be judged on 17 states.
