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
