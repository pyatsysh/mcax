"""Training data for an FMT functional amortised over the superball exponent.

Parallel (axis-aligned, orientation-frozen) hard superballs, overlap test
||r_i - r_j||_p < sigma, on the shape grid p = 2, 2.5, 3, 4, 6, infinity. The
consumer is `dimint-dft`, so the output mirrors that project's measure-campaign
schema exactly: object-array `.npy` of per-state dicts, `manifest.json`,
`MANIFEST.md`, splits, replica error bars and a drift diagnostic.

**Why the campaign has two stages and cannot have one.** For spheres the
activity that produces a wanted packing fraction is available in closed form,
so `22_measure_data.py` sets each confined state directly from
`eos.z_of_rho`. No such equation of state exists for a superball at general p,
which is most of why this data is worth taking. So the bulk EOS is measured
FIRST, one mu-ladder per (d, p), and the confined states then read their
activity off the measured curve by interpolation. Everything downstream of the
ladder depends on it, which is why it also carries the widest density range and
the ordering monitor.

**The freezing guard is the reason for the ordering monitor.** Parallel cubes
freeze near eta = 0.48 through a transition with no nucleation barrier, so an
over-compressed chain does not sit metastable, it orders, and the label is
silently a crystal's. Every state carries S(k)_max/<N> from `mcax.order`;
eta_max(p) is measured from the ladder as the onset of ordering minus a 0.03
margin, and confined states above it are dropped with a logged reason rather
than quietly.

Run (GPU, a few hours):
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      OMP_NUM_THREADS=2 taskset -c 2-3 <python> scripts/superball_campaign.py

Stages are resumable and idempotent: every finished state is written out
immediately and skipped on a re-run. `--stage bulk|confined|box|manifest`
selects one, `--cheap` gives a smoke-sized version of the whole thing.
"""
import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)

from mcax import (make_spec, burn_and_sample, eos, geometry, lattice_fill,
                  order, shapes, diagnostics)
from mcax.shapes import Superball
from mcax.campaign import (DRIFT_TOL, mu_of_eta, pname, pressure_at,
                           pressure_curve, usable_ladder)

INF = float("inf")

# --------------------------------------------------------------------------- #
#  The plan, declared up front so the manifest cannot drift from it            #
# --------------------------------------------------------------------------- #

# Shape grid. p = 3 is held out ENTIRELY (the amortisation test: the functional
# never sees it and must interpolate the shape axis), and p = infinity is a
# benchmark against the Cuesta-Martinez-Raton parallel-cube functional rather
# than training data.
P_GRID = [2.0, 2.5, 3.0, 4.0, 6.0, INF]
P_ROLE = {2.0: "train", 2.5: "train", 3.0: "test_shape", 4.0: "train",
          6.0: "train", INF: "benchmark"}

# Bulk mu-ladder, in target packing fraction. The dilute end is for the B2
# check (A1) and is cheap; the dense end is the freezing ramp and is where
# eta_max(p) gets measured, so it deliberately runs past where the data will be
# kept.
BULK_ETA = {3: [0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.46],
            2: [0.02, 0.05, 0.10, 0.18, 0.28, 0.38, 0.48, 0.56, 0.63, 0.70]}
# A dilute run in a small box holds too few particles for a 1% <N>, and the B2
# intercept is an extrapolation that magnifies whatever error is there. The
# dilute end therefore runs in a bigger box, where it is cheap anyway.
BULK_L = {3: 8.0, 2: 12.0}
BULK_L_DILUTE = {3: 12.0, 2: 18.0}
DILUTE_BELOW = 0.08

# Confined states: the hard-sphere campaign's own grid, so the shape axis is
# the only thing that changed. Held-out states sit on the EXTRAPOLATION axes
# deliberately: slit-width extrapolation is where functionals die.
SLIT_TRAIN = {3: [(H, e) for H in (4.0, 6.0, 8.0, 12.0)
                  for e in (0.15, 0.25, 0.35)],
              2: [(H, e) for H in (4.0, 6.0, 8.0, 12.0)
                  for e in (0.30, 0.45, 0.60)]}
SLIT_HELD = {3: [(5.0, 0.25, "test_width"), (16.0, 0.30, "test_width"),
                 (6.0, 0.20, "val"), (8.0, 0.42, "test_state")],
             2: [(5.0, 0.45, "test_width"), (16.0, 0.50, "test_width"),
                 (6.0, 0.37, "val"), (8.0, 0.68, "test_state")]}

# Small hard boxes: cuboid cavities with every face a wall, near the
# single-particle scale. Held out as 0-D-adjacent tests. The exact 0-D labels
# are analytic and are not an MC job, so these sit just ABOVE that limit, where
# a functional has to get the crossover right rather than the endpoint.
BOX_STATES = {3: [(1.2, 0.20), (1.6, 0.25), (2.2, 0.30)],
              2: [(1.2, 0.35), (1.6, 0.40), (2.2, 0.45)]}

ORDER_MARGIN = 0.03        # keep training data this far below the onset
# Below this packing fraction the ordering monitor is switched off rather than
# run and ignored. A fluid at eta = 0.02 is not freezing, the structure factor
# there is a reduction over a few hundred wavevectors at every thinning
# boundary, and for a dilute state in a large cell that reduction is a large
# share of the whole step cost. States below the floor carry n_kvectors = 0 in
# the manifest, which reads as "not monitored" and not as "monitored, fluid".
ORDER_FLOOR = 0.15

# Fraction of the target occupancy seated on the melt-in lattice. Was 0.85,
# following the hard-sphere campaign, and that is too low HERE for a reason
# worth recording: a chain seated at 85% has to insert the remaining 15% one
# accepted insertion at a time, and above eta about 0.35 in three dimensions
# the insertion acceptance is well under a per cent, so it never gets there.
# The measured ladder showed it plainly, tracking the target to four decimal
# places up to eta = 0.25 and then stalling at 0.371 for every target above
# 0.46. Seating at 98% leaves the burn a local relaxation rather than a
# density change, and the ordering confirmation pass is what stops the denser
# initial lattice being mistaken for a phase transition.
PREFILL = 0.98

# Attempted moves per particle in the burn-in. With PREFILL at 0.98 the burn is
# a local relaxation rather than a density change, but it still has to MELT the
# lattice it was seated on, and that is what sets the number: 13 moves per
# particle leaves the lattice standing (measured, see `burn_for`), ~110 melts it
# with room to spare. Only half of these are displacements at p_disp = 0.5.
BURN_SWEEPS = 220

DZ = 0.05                  # profile bin width, as in the hard-sphere campaign
LPERP = 8.0

ROOT = os.environ.get("MCAX_SUPERBALL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "dimint-dft",
    "data", "superball")


# --------------------------------------------------------------------------- #
#  Helpers, schema-compatible with dimint's 22_measure_data.py                 #
# --------------------------------------------------------------------------- #

def chain_error(Ns):
    """Independent-replica error on <N>: the chains really are independent, so
    their scatter is an honest standard error with no autocorrelation model."""
    per_chain = onp.asarray(Ns).mean(axis=1)
    return float(per_chain.std(ddof=1) / onp.sqrt(per_chain.size))


def drift(Ns):
    """Shift of <N> between the first and last third of the sampling window,
    in units of the replica error. The equilibration check."""
    a = onp.asarray(Ns)
    n = a.shape[1] // 3
    return float(abs(a[:, -n:].mean() - a[:, :n].mean())
                 / max(chain_error(a), 1e-12))


def seed_of(tag):
    """A seed that survives a resume. Python salts str.__hash__ per process, so
    the obvious `abs(hash(tag))` would re-seed a re-run state differently from
    its first attempt and quietly break reproducibility."""
    return zlib.crc32(tag.encode()) % (2 ** 31)


def z_guess(d, p, eta):
    """Activity expected to give reservoir packing fraction `eta` at exponent p.

    Only a starting point, and it is a good one for a reason worth recording.
    Write z = rho exp(beta mu_ex). The ideal part is exact under the change of
    shape once density is expressed through eta, since rho = eta / v(p), and
    the excess part agrees with the sphere's to first order in eta because

        beta mu_ex  =  2 B_2 rho + O(rho^2)  =  2^d eta + O(eta^2)

    is INDEPENDENT of the shape: B_2 = 2^(d-1) v for every member of the
    family, so the volume cancels against the density. The residual error is
    therefore second order in eta, and it is measured rather than assumed: the
    ladder reports the packing fraction it actually reached.
    """
    v, vs = shapes.volume(Superball(p), d), shapes.volume(Superball(2.0), d)
    return float(vs / v) * float(eos.z_of_rho(d, eta / eos.B[d]))


def sample(spec, seed, n_burn, n_run, thin, nbins, n0, kvecs, tag):
    """One state point, returned as the schema dict plus its raw profile."""
    t0 = time.time()
    # Drop the compiled-executable cache before every state. Each state is
    # its own XLA compilation (spec is a static jit argument, deliberately),
    # JAX keeps every executable for the life of the process, and a full
    # campaign is five-hundred-plus of them: the 2026-08-07 re-run died 69
    # confined states in with `Failed to materialize symbols` inside XLA —
    # the same exhaustion the test suite hit and fixed in conftest.py, which
    # the campaign process never got. Nothing is lost by clearing here: no
    # two states share a compilation, and the burn, the sampling window and
    # the confirmation re-run of THIS state all still share theirs, because
    # the clear runs before the state and not between its calls.
    jax.clear_caches()
    r = burn_and_sample(spec, C=CHAINS, seed=seed, n_burn=n_burn, n_run=n_run,
                        thin=thin, nbins=nbins, n0=n0, kvecs=kvecs)
    v1 = shapes.volume(spec.shape, spec.d, spec.sigma)
    vol = geometry.volume(spec)
    od = order.summarise(r, kvecs if kvecs is not None else onp.zeros((0, spec.d)))
    # rho(0+) and its chain-scatter error. SLIT ONLY, and the restriction is
    # the sum rule's rather than the code's: rho(contact) = beta P holds for a
    # planar wall with an otherwise homogeneous fluid beyond it. A cuboid
    # cavity's z-profile averages the bottom face together with everything the
    # four side walls are doing to it a diameter away, so the number exists and
    # is not beta P, and recording it under this name would invite exactly the
    # comparison it fails. Taken here rather than at manifest time because the
    # error bar needs the per-chain histogram and only the chain-averaged
    # profile is stored.
    contact, contact_err = (contact_with_error(spec, r) if spec.geom == "slit"
                            else (float("nan"), float("nan")))
    # What the lattice prefill ACTUALLY seated, not what was asked of it.
    # `lattice_fill` returns min(n0, sites available), silently: a cubic
    # lattice of pitch 1.05 in the L = 8 bulk box holds 7^3 = 343 sites, which
    # is eta = 0.351 for spheres, so the eta = 0.40 and 0.46 rungs START 12%
    # and 24% below target and have to close the gap one accepted insertion at
    # a time. Recording the request as the prefill hid exactly the mechanism
    # behind the stalled ladder tops; now both numbers are on the state.
    seated = len(lattice_fill(spec, n0, seed)) if n0 > 0 else 0
    # Cross-chain convergence evidence, per state. The replica error is blind
    # to a burn systematic all chains share (they all start on the same
    # prefill); split R-hat halves each chain FIRST, so drift the chains
    # undergo together still shows up as disagreement. ESS says how many
    # independent draws the window really held. Recorded, not thresholded:
    # at these run lengths a converged dense state can honestly read 1.1.
    dg = diagnostics.summary(r.Ns)
    return dict(
        tag=tag, d=spec.d, p=float(spec.shape.p), geom=spec.geom,
        H=float(spec.H), Lperp=float(spec.Lperp), sigma=float(spec.sigma),
        # The bin width this state was ACTUALLY histogrammed at. It was
        # recorded as the constant DZ, which is right for every slit and box
        # and wrong by a factor of ten for the bulk rungs (16 bins across the
        # whole cell); a schema consumer reading `dz` got a fiction there.
        z_act=float(spec.z_act), mu=float(onp.log(spec.z_act)),
        dz=float(geometry.extent(spec) / nbins),
        v_particle=float(v1), volume=float(vol),
        z=onp.asarray(r.z), rho=onp.asarray(r.rho),
        n_mean=float(r.n_mean), n_err=chain_error(r.Ns),
        rho_mean=float(r.n_mean / vol), eta_mean=float(r.n_mean * v1 / vol),
        eta_err=float(chain_error(r.Ns) * v1 / vol),
        drift_sigma=drift(r.Ns), saturation=float(r.saturation),
        n_hi=int(r.n_hi), capacity=int(r.capacity),
        acc=onp.asarray(r.acc).tolist(), chains=CHAINS, n_run=n_run,
        n_burn_used=n_burn, n0_prefill=int(n0), n0_seated=int(seated),
        ess=float(dg["ess"]), split_rhat=float(dg["split_rhat"]),
        s_max=od["s_max"], s_max_err=od["s_max_err"],
        s_max_over_n=od["s_max_over_n"], k_at_max=od["k_at_max"],
        n_kvectors=od["n_kvectors"], ordered=bool(od["ordered"]),
        contact=contact, contact_err=contact_err,
        wall_time_s=round(time.time() - t0, 1))


def store(path, rows):
    # Write-then-rename, because this campaign is DESIGNED to be killed and
    # resumed: `onp.save` straight onto the live file means a kill mid-write
    # truncates the array and the resume loses everything it was protecting.
    tmp = path + ".tmp.npy"
    onp.save(tmp, onp.array(rows, dtype=object), allow_pickle=True)
    os.replace(tmp, path)


def guard_statistics(rows, path):
    """Refuse to resume a dataset taken at different statistics.

    `--quick` and `--cheap` share tags and ROOT with production, and the skip
    logic keys on the tag alone, so an 8-chain 4k-step smoke state would be
    silently kept in a production dataset — feeding `mu_of_eta`, the virial
    fit and the eta_max table at production — and nothing in the output would
    say so except one per-state column nobody thresholds.
    """
    bad = [r for r in rows
           if r.get("chains") != CHAINS or r.get("n_run") != NRUN]
    if bad:
        b = bad[0]
        raise SystemExit(
            f"{path} holds {len(bad)} states at different statistics (e.g. "
            f"{b['tag']}: chains {b.get('chains')}, n_run {b.get('n_run')}, "
            f"against this invocation's {CHAINS}/{NRUN}). Mixing smoke and "
            f"production statistics poisons the ladder and the acceptance "
            f"tests; point MCAX_SUPERBALL_ROOT somewhere fresh or archive "
            f"the existing files first.")


def load(path):
    if not os.path.exists(path):
        return []
    return list(onp.load(path, allow_pickle=True))


def nmax_for(n_target, spec_default):
    """Capacity sized on the state's OWN occupancy, not on the dense limit.

    `make_spec` sizes for a box packed as densely as the shape allows, which is
    right when nothing is known about the activity and badly wrong when it is:
    the overlap test sweeps all Nmax slots, masked ones included, so capacity is
    paid for per step whether or not it holds a particle. A dilute point in the
    large box used for the virial fit was being given 2828 slots to hold about
    seventeen particles, and cost more per step than the densest state in the
    campaign.

    The headroom is generous where it is cheap. N fluctuates in the grand
    ensemble with a standard deviation near sqrt(<N> S(0)), so ten sqrt(N) is
    far outside anything a run will reach, and `saturation` is recorded per
    state so a bad guess would be visible rather than silent.
    """
    want = int(1.6 * n_target + 10.0 * onp.sqrt(max(n_target, 1.0))) + 64
    return int(min(want, spec_default))


def burn_for(n_target):
    """Burn-in scaled to system size, LINEARLY: a fixed number of moves per
    particle, not a fixed number of steps.

    This was a square-root rule, inherited from the hard-sphere campaign, and
    the docstring that came with it said in the same breath that the transient
    is per-particle — which is an argument for a linear rule, not a square-root
    one. One step is one attempted move on one particle, so under sqrt(N) the
    moves each particle receives FALL as 1/sqrt(N) and the largest states get
    the shortest relaxation. Measured on `d3_slit_H16_eta0.30_p2`, the widest
    slit in the campaign: 587 particles seated on a lattice, 15.8k burn steps,
    thirteen displacement attempts per particle. The lattice does not melt in
    thirteen moves, and the ordering monitor duly reported the lattice it had
    been handed — S_max = 82, S/N = 0.135, over the trip. The same state with
    the burn multiplied by eight gives S_max = 1.95 and S/N = 0.0032, which is
    a fluid, at the same density (eta 0.3116 against 0.3084) and from either a
    dense or a sparse start. The trip was the initial condition.

    So the rule is now moves per particle, and `BURN_SWEEPS` says how many.
    `NBURN` remains the floor for small states, where the per-particle count is
    generous anyway and the compile dominates.
    """
    return int(max(NBURN, BURN_SWEEPS * max(n_target, 1.0)))


# --------------------------------------------------------------------------- #
#  The ordering confirmation pass, shared by both stages                       #
# --------------------------------------------------------------------------- #

def confirm_ordering(spec, st, n_t, kv, tag, nbins):
    """Re-run a tripped state from a sparse start and see whether it orders
    again. Mutates `st`.

    A trip is not believed on one run, and this is why. The lattice prefill
    STARTS the chain on a crystal — `PREFILL` of the target seated on a cubic
    grid, so the burn does not have to fill grand-canonically at 1% acceptance
    — and a burn too short to melt that grid reports exactly the ordering it
    was handed. So a tripped state is re-run from a much sparser start with
    twice the burn, and has to order AGAIN, from below. The transition being
    barrier-free is what makes that a fair test rather than a lenient one: a
    genuinely over-compressed fluid orders on its own, with nothing to sit
    metastable behind.

    **This used to run on the bulk ladder only, and the confined states — the
    training data itself — went out on a single lattice-started run.** Measured
    afterwards on `d3_slit_H16_eta0.30_p2`, the widest slit in the campaign:
    as run it reported S_max = 82 and S/N = 0.135, over the trip, and the
    ordering was at a wavevector of 2 pi (7) / L, one Bragg order of the prefill
    grid. The same state with eight times the burn gives S_max = 1.95 and
    S/N = 0.0032 at the same density, from either a dense or a sparse start.
    That is a fluid, and six `test_width` states across the shape grid were
    flagged for a phase transition that was the initial condition. The burn is
    now long enough to melt the grid (see `burn_for`) AND the confirmation runs
    on both stages, because either alone has been shown to be insufficient.
    """
    st["ordered_lattice"] = bool(st["ordered"])
    if not st["ordered"]:
        return st
    chk = sample(spec, seed=seed_of(tag) + 1, n_burn=2 * burn_for(n_t),
                 n_run=NRUN, thin=THIN, nbins=nbins, n0=int(0.45 * n_t),
                 kvecs=kv, tag=tag + "_confirm")
    st["ordered_dilute"] = bool(chk["ordered"])
    st["s_max_over_n_dilute"] = chk["s_max_over_n"]
    st["eta_mean_dilute"] = chk["eta_mean"]
    st["ordered"], st["ordering_reason"] = resolve_ordering(st)
    st["wall_time_s"] += chk["wall_time_s"]
    print(f"      confirm from sparse start: "
          f"S/N={chk['s_max_over_n']:.4f} eta={chk['eta_mean']:.4f}"
          f" -> {st['ordering_reason']}")
    return st


# --------------------------------------------------------------------------- #
#  Stage 1: bulk equations of state, and the freezing ramp                     #
# --------------------------------------------------------------------------- #

def stage_bulk():
    path = os.path.join(ROOT, "bulk_eos.npy")
    rows = load(path)
    guard_statistics(rows, path)
    done = {r["tag"] for r in rows}
    plan = [(d, p, e) for d in (3, 2) for p in P_GRID for e in BULK_ETA[d]]
    print(f"stage bulk: {len(plan)} ladder points, {len(done)} already done")

    for i, (d, p, eta) in enumerate(plan):
        tag = f"d{d}_bulk_p{pname(p)}_eta{eta:.2f}"
        if tag in done:
            continue
        sh = Superball(p)
        v = shapes.volume(sh, d)
        L = BULK_L_DILUTE[d] if eta < DILUTE_BELOW else BULK_L[d]
        spec = make_spec(d, H=L, Lperp=L, z_act=z_guess(d, p, eta),
                         geom="bulk", shape=sh)
        n_t = eta / v * geometry.volume(spec)
        spec = spec._replace(Nmax=nmax_for(n_t, spec.Nmax))
        kv = order.kvectors(spec, nmax=4, kmax=4.0 * onp.pi) \
            if eta >= ORDER_FLOOR else None
        st = sample(spec, seed=seed_of(tag),
                    n_burn=burn_for(n_t), n_run=NRUN, thin=THIN, nbins=16,
                    n0=int(PREFILL * n_t), kvecs=kv, tag=tag)
        st["eta_target"] = float(eta)
        st["role"] = P_ROLE[p]
        confirm_ordering(spec, st, n_t, kv, tag, nbins=16)
        rows.append(st)
        store(path, rows)
        seatflag = (f"  SHORT-SEAT {st['n0_seated']}/{st['n0_prefill']}"
                    if st["n0_seated"] < st["n0_prefill"] else "")
        print(f"  [{i+1:3d}/{len(plan)}] {tag:<30s} eta={st['eta_mean']:.4f}"
              f" (aim {eta:.2f})  S/N={st['s_max_over_n']:.4f}"
              f"{'  ORDERED' if st['ordered'] else ''}"
              f"  drift={st['drift_sigma']:4.1f}s  {st['wall_time_s']:5.1f}s"
              f"{seatflag}")
    return rows


# Largest relative density gap between the lattice-start run and its sparse
# restart for the restart's verdict to be about the SAME state point.
CONFIRM_GAP = 0.05


def resolve_ordering(r):
    """(verdict, reason) for one ladder point, re-derived from stored fields.

    The confirmation pass as first written asked whether a sparse restart also
    orders, and treated "no" as proof the trip was an unmelted initial
    condition. Measured, it said "fluid after all" fourteen times out of
    fourteen, and the reason was not that fourteen crystals were mirages: the
    sparse restart could not REACH the density. Gaps ran from 8% to 26%, so the
    two runs were sitting on opposite sides of the equilibration wall and the
    comparison was between different state points.

    That failed in the unsafe direction, clearing trips that may have been
    real. So the restart's verdict now only counts when it arrived somewhere
    comparable. When it did not, the state is UNDETERMINED, which is treated as
    ordered for the purpose of capping the training range and is reported as
    its own category, because "we could not tell" and "we checked and it was
    fluid" are different claims and only one of them was earned.
    """
    if not r.get("ordered_lattice", r.get("ordered", False)):
        return False, "no trip"
    e_lat = r.get("eta_mean", 0.0)
    e_sp = r.get("eta_mean_dilute")
    if e_sp is None:
        return True, "trip, no confirmation run"
    gap = abs(e_lat - e_sp) / max(e_lat, 1e-12)
    if gap > CONFIRM_GAP:
        return True, (f"UNDETERMINED: restart reached eta {e_sp:.4f} against "
                      f"{e_lat:.4f}, a {gap:.0%} gap, so the two runs are not "
                      f"the same state point")
    if r.get("ordered_dilute"):
        return True, "trip confirmed from a sparse restart at matched density"
    return False, f"trip cleared: restart at matched density ({gap:.0%}) is fluid"


def eta_max_table(bulk):
    """eta_max(d, p): the ordering onset from the ladder, less a safety margin.

    The onset is the LOWEST density that trips the monitor, not the highest
    that did not: an ordered state at 0.45 with a fluid-looking one at 0.50
    means the monitor is noisy there and the conservative reading is the lower
    number. Where nothing trips at all the ladder's top is returned and flagged,
    because "no onset found" is not the same as "fluid everywhere" and the
    manifest should not read as though it were.
    """
    out = {}
    for d in (3, 2):
        for p in P_GRID:
            sel = sorted((r for r in bulk if r["d"] == d and r["p"] == p),
                         key=lambda r: r["eta_mean"])
            for r in sel:
                r["ordered"], r["ordering_reason"] = resolve_ordering(r)
            trip = [r["eta_mean"] for r in sel if r["ordered"]]
            top = max((r["eta_mean"] for r in sel), default=0.0)
            onset = min(trip) if trip else None
            out[f"d{d}_p{pname(p)}"] = dict(
                d=d, p=pname(p), onset=onset,
                eta_max=round((onset - ORDER_MARGIN) if onset else top, 4),
                found=onset is not None, ladder_top=round(top, 4),
                n_ordered=len(trip),
                n_undetermined=sum(1 for r in sel
                                   if "UNDETERMINED" in r.get("ordering_reason", "")))
    return out


# The ladder's analysis layer, `usable_ladder`, `mu_of_eta`, `pressure_curve`
# and `pressure_at` with the cuts they rest on, lives in `mcax.campaign`:
# `dimint-dft` reads the measured curves through those functions and a
# consumer should not have to import a campaign driver for three accessors.
# They are imported at the top of this file, so anything that imports this
# script still finds them here.


# --------------------------------------------------------------------------- #
#  Stage 2: confined profiles                                                  #
# --------------------------------------------------------------------------- #

def stage_confined(bulk, etamax):
    path = os.path.join(ROOT, "mc_states.npy")
    rows = load(path)
    guard_statistics(rows, path)
    done = {r["tag"] for r in rows}
    excluded = load(os.path.join(ROOT, "excluded.npy"))
    exdone = {r["tag"] for r in excluded}

    tier_train, tier_held, tier_box = [], [], []
    for d in (3, 2):
        for p in P_GRID:
            for H, eta in SLIT_TRAIN[d]:
                tier_train.append((d, p, "slit", H, eta,
                                   "test_shape" if P_ROLE[p] == "test_shape"
                                   else "train"))
            for H, eta, split in SLIT_HELD[d]:
                tier_held.append((d, p, "slit", H, eta, split))
            for H, eta in BOX_STATES[d]:
                tier_box.append((d, p, "box", H, eta, "test_box"))
    plan = tier_train + tier_held + tier_box
    print(f"stage confined: {len(plan)} states, {len(done)} done, "
          f"{len(exdone)} previously excluded")

    for i, (d, p, geom, H, eta, split) in enumerate(plan):
        tag = f"d{d}_{geom}_H{H:g}_eta{eta:.2f}_p{pname(p)}"
        if tag in done:
            continue
        # An exclusion is NOT permanent: it is re-derived from the current
        # ladder on every pass. The old order — skip anything in
        # excluded.npy, then check the cap — froze every verdict at whatever
        # the ladder looked like when the state was first considered, so a
        # cap that improved on a re-run could never readmit a state it had
        # refused. Now the guard runs first and the ledger records its
        # CURRENT verdict; a state it no longer refuses is generated and
        # struck off the list.
        cap = etamax[f"d{d}_p{pname(p)}"]["eta_max"]
        if eta > cap:
            if tag not in exdone:
                excluded.append(dict(tag=tag, d=d, p=pname(p), geom=geom, H=H,
                                     eta_target=eta, eta_max=cap,
                                     reason="target above measured eta_max(p)"))
                store(os.path.join(ROOT, "excluded.npy"), excluded)
                print(f"  [{i+1:3d}/{len(plan)}] {tag:<34s} EXCLUDED "
                      f"(eta {eta} > eta_max {cap})")
            continue

        sh = Superball(p)
        v = shapes.volume(sh, d)
        try:
            mu = mu_of_eta(bulk, d, p)(eta)
        except ValueError as exc:
            if tag not in exdone:
                excluded.append(dict(tag=tag, d=d, p=pname(p), geom=geom, H=H,
                                     eta_target=eta, reason=str(exc)))
                store(os.path.join(ROOT, "excluded.npy"), excluded)
                print(f"  [{i+1:3d}/{len(plan)}] {tag:<34s} EXCLUDED ({exc})")
            continue
        if tag in exdone:
            excluded = [x for x in excluded if x["tag"] != tag]
            exdone.discard(tag)
            store(os.path.join(ROOT, "excluded.npy"), excluded)
            print(f"  [{i+1:3d}/{len(plan)}] {tag:<34s} previously excluded, "
                  f"the corrected ladder now admits it")

        Lp = H if geom == "box" else LPERP
        spec = make_spec(d, H=H, Lperp=Lp, z_act=float(onp.exp(mu)), geom=geom,
                         shape=sh)
        n_t = eta / v * geometry.volume(spec)
        spec = spec._replace(Nmax=nmax_for(n_t, spec.Nmax))
        kv = order.kvectors(spec, nmax=4, kmax=4.0 * onp.pi) \
            if eta >= ORDER_FLOOR else onp.zeros((0, d))
        kv = kv if len(kv) else None
        st = sample(spec, seed=seed_of(tag),
                    n_burn=burn_for(n_t), n_run=NRUN, thin=THIN,
                    nbins=max(int(round(H / DZ)), 8),
                    n0=int(PREFILL * n_t), kvecs=kv, tag=tag)
        st.update(eta_reservoir=float(eta), split=split, role=P_ROLE[p],
                  rho_b_reservoir=float(eta / v))
        confirm_ordering(spec, st, n_t, kv, tag,
                         nbins=max(int(round(H / DZ)), 8))
        rows.append(st)
        store(path, rows)
        flag = " DRIFT" if st["drift_sigma"] > 3.0 else ""
        flag += " ORDERED" if st["ordered"] else ""
        flag += " CAPACITY" if st["saturation"] > 0.9 else ""
        print(f"  [{i+1:3d}/{len(plan)}] {tag:<34s} {split:<11s} "
              f"<N>={st['n_mean']:8.2f}+-{st['n_err']:.2f} "
              f"drift={st['drift_sigma']:5.1f}s {st['wall_time_s']:6.1f}s{flag}")
    return rows, excluded


# --------------------------------------------------------------------------- #
#  Acceptance tests A1 and A3                                                  #
# --------------------------------------------------------------------------- #

def a1_second_virial(bulk):
    """Measured B2 against the exact 2^(d-1) v, from the dilute end of each
    ladder.

    beta mu_ex = 2 B_2 rho + O(rho^2), so ln(z/rho)/rho extrapolates to 2 B_2
    as rho -> 0. Fitted as a straight line through the dilute points rather
    than read off the lowest one, because the lowest one is also the noisiest
    and a two-parameter fit uses the curvature instead of being fooled by it.
    """
    out = []
    for d in (3, 2):
        for p in P_GRID:
            sel = sorted((r for r in bulk if r["d"] == d and r["p"] == p
                          and r["eta_target"] <= 0.10),
                         key=lambda r: r["rho_mean"])
            if len(sel) < 3:
                continue
            rho = onp.array([r["rho_mean"] for r in sel])
            lz = onp.array([r["mu"] for r in sel])
            y = (lz - onp.log(rho)) / rho              # -> 2 B2 as rho -> 0
            # d y / d rho at fixed mu is -(1 + mu_ex) / rho^2, so the factor
            # (1 + mu_ex) belongs in the propagated error. It was dropped,
            # which understated the bar by ~10% at the dilute end.
            err = (1.0 + y * rho) * \
                onp.array([r["n_err"] / r["n_mean"] for r in sel]) / rho
            c = onp.polyfit(rho, y, 1)
            got = float(c[-1]) / 2.0
            exact = shapes.b2(Superball(p), d, 1.0)
            # Propagate the replica error through the intercept: the fit is
            # linear, so a plain unweighted-fit covariance is enough here.
            n = len(rho)
            sx = rho.std() * onp.sqrt(n)
            sig = float(onp.sqrt(onp.mean(err ** 2)) *
                        onp.sqrt(1.0 / n + rho.mean() ** 2 / max(sx ** 2, 1e-30)))
            out.append(dict(d=d, p=pname(p), b2_measured=got, b2_exact=exact,
                            b2_err=sig / 2.0,
                            dev_rel=abs(got - exact) / exact,
                            dev_sigma=abs(got - exact) / max(sig / 2.0, 1e-12),
                            n_points=n))
    return out


def contact_value(z, rho, nfit=6):
    """rho(0+) extrapolated to the wall, not read off the first bin.

    The first bin is an AVERAGE over [0, dz), and the profile at a hard wall is
    rising steeply there, so the bin under-reads the contact value by an amount
    that grows with the density: exactly where the sum rule is being tested. A
    quadratic through the first few bin centres, evaluated at zero, removes the
    leading part of that.
    """
    n = min(nfit, len(z))
    c = onp.polyfit(z[:n], rho[:n], 2)
    return float(onp.polyval(c, 0.0))


def contact_with_error(spec, result, nfit=6):
    """(rho(0+), standard error) with the error taken from the chain scatter.

    The mean alone is not a testable number, and A3 was reading it as one.
    Extrapolating a quadratic past the end of the data amplifies whatever noise
    is in the bins it is fitted to, and at DZ = 0.05 in a 240-bin profile that
    noise is several per cent per bin — not the 0.2% the <N> error bar suggests,
    because <N> averages 240 bins and the contact value uses six of them and
    then leans outward. So a 10% A3 deviation and a 4% one may be the same
    number twice, and without this there is no way to say.

    The chains are independent by construction, so their scatter is an honest
    standard error with no autocorrelation model in it: the same argument the
    rest of the campaign's error bars rest on, applied one bin at a time.
    """
    h = onp.asarray(result.hist, dtype=float)               # (C, nbins) counts
    cell = geometry.bin_volume(spec, h.shape[-1])
    dz = geometry.extent(spec) / h.shape[-1]
    z = (onp.arange(h.shape[-1]) + 0.5) * dz
    per_chain = onp.array([contact_value(z, h[c] / (result.nsamples * cell),
                                         nfit) for c in range(h.shape[0])])
    mean = contact_value(z, h.sum(axis=0) / (h.shape[0] * result.nsamples * cell),
                         nfit)
    err = (float(per_chain.std(ddof=1) / onp.sqrt(len(per_chain)))
           if len(per_chain) > 1 else float("nan"))
    return float(mean), err


def a3_contact_theorem(mc, bulk):
    """rho(contact) = beta P at a hard wall, exact for parallel walls and any
    particle shape, checked at one slit state per (d, p).

    The sum rule constrains the WALL rather than the fluid, so it holds for
    every member of the family and is the sharpest single check that the
    engine and the measured equation of state agree with each other. mcax's
    walls act on CENTRES, so it applies literally and no shift enters.

    Two things it is NOT, both of which it was being read as. It is not a test
    of the engine alone: beta P comes from integrating the measured ladder, so a
    deviation is shared between the sum rule and that integration, and the p = 2
    row is the only one where the two can be separated. And it is not a test at
    all without an error bar — see `contact_with_error`. `dev_sigma` is the
    number to read; `dev_rel` on its own cannot tell 10% from 4%.
    """
    out = []
    for d in (3, 2):
        for pexp in P_GRID:
            if len(usable_ladder(bulk, d, pexp)) < 4:
                continue
            cands = [s for s in mc if s["d"] == d and s["p"] == pexp
                     and s["geom"] == "slit"
                     and s["split"] in ("train", "test_shape")]
            if not cands:
                continue
            s = max(cands, key=lambda x: x["H"])
            try:
                beta_p = pressure_at(bulk, d, pexp, s["rho_b_reservoir"])
            except ValueError as exc:
                # A resume can truncate a ladder below a state generated
                # under an earlier, longer one; better no A3 row than one
                # evaluated at a clamped density.
                print(f"  A3 d={d} p={pname(pexp)}: skipped ({exc})")
                continue
            contact = s.get("contact")
            if contact is None:
                contact = contact_value(onp.asarray(s["z"]),
                                        onp.asarray(s["rho"]))
            cerr = float(s.get("contact_err", float("nan")))
            row = dict(d=d, p=pname(pexp), tag=s["tag"], contact=float(contact),
                       contact_err=cerr,
                       bin0=float(onp.asarray(s["rho"])[0]), beta_p=beta_p,
                       dev_rel=abs(contact - beta_p) / max(beta_p, 1e-12),
                       dev_sigma=(abs(contact - beta_p) / cerr
                                  if cerr == cerr and cerr > 0
                                  else float("nan")))
            if pexp == 2.0:
                # At p = 2 the pressure is known independently, so the sum rule
                # can be checked against it and the integration taken out of
                # the comparison altogether. This row is the one that says
                # whether the ENGINE satisfies the contact theorem.
                ref = float(eos.p_of_rho(d, s["rho_b_reservoir"]))
                row["beta_p_reference"] = ref
                row["dev_rel_reference"] = abs(contact - ref) / ref
            out.append(row)
    return out


def gibbs_duhem_against_reference(bulk):
    """The integration itself, checked where a reference exists.

    At p = 2 the fluid IS hard spheres and hard discs, so the pressure the
    ladder integrates to can be compared with Carnahan-Starling and Henderson.
    Any disagreement here is the numerics of the integration rather than the
    physics, which is precisely what has to be separated out before a contact
    deviation at some other p can be blamed on the engine.
    """
    out = []
    for d in (3, 2):
        if len(usable_ladder(bulk, d, 2.0)) < 4:
            continue
        rho, P = pressure_curve(bulk, d, 2.0)
        for i in range(1, len(rho)):
            out.append(dict(d=d, rho=float(rho[i]),
                            eta=float(rho[i] * eos.B[d]),
                            p_integrated=float(P[i]),
                            p_reference=float(eos.p_of_rho(d, rho[i])),
                            dev_rel=abs(P[i] - eos.p_of_rho(d, rho[i]))
                            / max(eos.p_of_rho(d, rho[i]), 1e-12)))
    return out


def a2_cube_two_box_sizes(bulk):
    """A2: the cube EOS is size-consistent. The ladder's dilute points run in a
    larger box than its dense ones, so the two overlap in density near the
    switch and the curve must not step there. Reported as the largest jump in
    beta mu across the boundary relative to the ladder's own spacing."""
    out = []
    for d in (3, 2):
        sel = sorted((r for r in bulk if r["d"] == d and r["p"] == INF),
                     key=lambda r: r["rho_mean"])
        small = [r for r in sel if r["H"] == BULK_L[d]]
        big = [r for r in sel if r["H"] == BULK_L_DILUTE[d]]
        if not small or not big:
            continue
        if len(small) < 2:
            continue
        rho = onp.array([r["rho_mean"] for r in small])
        mu = onp.array([r["mu"] for r in small])
        r0 = big[-1]                                  # densest big-box point
        # Extrapolate the EXCESS and add ln rho exactly. The densest big-box
        # rung always sits below the smallest small-box rung, so this branch
        # always extrapolates across a factor-two density gap — and a linear
        # extrapolation of the FULL mu carries the chord error of ln rho with
        # it, which is 0.22 to 0.28 in beta mu here for a PERFECT engine. A2
        # was reporting that as a size step, and a real step of that size
        # would have been shrugged off as the known bias.
        mux = mu - onp.log(rho)
        if r0["rho_mean"] < rho[0]:
            mux_p = mux[0] + (r0["rho_mean"] - rho[0]) * (mux[1] - mux[0]) / \
                (rho[1] - rho[0])
        else:
            mux_p = float(onp.interp(r0["rho_mean"], rho, mux))
        pred = float(onp.log(r0["rho_mean"]) + mux_p)
        out.append(dict(d=d, L_small=BULK_L[d], L_big=BULK_L_DILUTE[d],
                        rho=r0["rho_mean"], mu_big=r0["mu"], mu_small=pred,
                        dev=abs(r0["mu"] - pred)))
    return out


# --------------------------------------------------------------------------- #
#  Manifest                                                                    #
# --------------------------------------------------------------------------- #

def reconcile(mc, etamax):
    """Re-apply the freezing guard to states already on disk.

    `stage_confined` decides what to skip from the eta_max table it holds in
    memory, so a long run keeps whatever caps it started with. That is the right
    behaviour for a resumable campaign and the wrong one after the guard has
    been corrected mid-flight, which is what happened here: the ordering
    verdicts were being cleared by a confirmation pass that had not earned them,
    the caps were consequently far too generous, and states were generated that
    should not have been.

    Rather than discard the runs, they are RELABELLED here, from the fields each
    state already carries. A state above its corrected cap, or one whose own
    ordering monitor tripped, moves out of `train` and into `flagged` with the
    reason attached. Nothing is deleted and nothing is quietly kept: the data is
    on disk either way and the manifest says which side of the line it is on.

    **The relabelling has to reach the array, not only the manifest.** This
    mutates `mc` in place and `write_manifest` writes `mc_states.npy` back out
    afterwards. It did not, and that was the whole of the protection failing:
    the manifest declared eighteen states `flagged`, five of them out of
    `train`, while `mc_states.npy` — the file `dimint-dft` actually loads —
    still labelled all 183 as it had generated them. A consumer selecting
    `split == "train"` got 91 states including five the manifest said must not
    be trained on, and nothing in the data it loaded said so.
    """
    out = []
    for s in mc:
        cap = etamax.get(f"d{s['d']}_p{s['p'] if isinstance(s['p'], str) else pname(s['p'])}",
                         {}).get("eta_max")
        why = []
        if cap is not None and s.get("eta_reservoir", 0.0) > cap:
            why.append(f"reservoir eta {s['eta_reservoir']:.2f} above the "
                       f"corrected eta_max {cap:.3f}")
        if s.get("ordered"):
            why.append(f"ordering monitor tripped, S/N = "
                       f"{s.get('s_max_over_n', float('nan')):.3f}")
        rel = s["drift_sigma"] * s["n_err"] / max(s["n_mean"], 1e-12)
        if rel > DRIFT_TOL:
            why.append(f"relative drift {rel:.1%}")
        if why:
            # `setdefault`, not assignment: now that the relabelling is written
            # back to disk, this runs a second time over states that already
            # say `flagged`, and a plain assignment would record their original
            # split as `flagged` and lose it for good.
            s.setdefault("split_original", s["split"])
            s["split"] = "flagged"
            s["flag_reason"] = "; ".join(why)
            out.append(s)
    return out


def write_manifest(bulk, mc, excluded, etamax):
    # Rows written before `dz` was honest carry the constant DZ whatever they
    # were binned at; the bin centres do not lie, so re-derive it from them.
    for r in bulk + mc:
        z = onp.asarray(r.get("z", []))
        if len(z) > 1:
            r["dz"] = float(z[1] - z[0])
    flagged = reconcile(mc, etamax)
    if flagged:
        # `reconcile` relabelled states in memory; put them back on disk before
        # anything else, so the array and the manifest cannot disagree about
        # which states are trainable. See the note in `reconcile`.
        store(os.path.join(ROOT, "mc_states.npy"), mc)
        print(f"  reconcile: {len(flagged)} states relabelled `flagged` and "
              f"written back to mc_states.npy")
    a1 = a1_second_virial(bulk)
    a2 = a2_cube_two_box_sizes(bulk)
    a3 = a3_contact_theorem(mc, bulk)
    gd = gibbs_duhem_against_reference(bulk)
    splits = {}
    for s in mc:
        splits[s["split"]] = splits.get(s["split"], 0) + 1

    man = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"), cheap=CHEAP,
        model="parallel (axis-aligned) hard superballs, ||dr||_p < sigma",
        p_grid=[pname(p) for p in P_GRID],
        p_role={pname(k): v for k, v in P_ROLE.items()},
        chains=CHAINS, n_run=NRUN, thin=THIN, dz=DZ,
        order_trip=order.ORDER_TRIP, order_margin=ORDER_MARGIN,
        eta_max=etamax, splits=splits,
        n_bulk=len(bulk), n_states=len(mc), n_excluded=len(excluded),
        n_flagged=len(flagged),
        acceptance=dict(A1_b2=a1, A2_cube_size=a2, A3_contact=a3,
                        gibbs_duhem=gd),
        bulk=[{k: v for k, v in r.items() if k not in ("z", "rho")}
              for r in bulk],
        states=[{k: v for k, v in r.items() if k not in ("z", "rho")}
                for r in mc],
        excluded=excluded,
        total_wall_time_s=round(sum(r["wall_time_s"] for r in bulk + mc), 1))
    with open(os.path.join(ROOT, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2, default=str)

    L, A = [], None
    A = L.append
    A("# Parallel superball training data — manifest\n")
    n_want = sum(len(BULK_ETA[d]) for d in (3, 2)) * len(P_GRID)
    if len(bulk) < n_want or not mc:
        A("> **INCOMPLETE — DO NOT TRAIN ON THIS YET.**")
        A(f"> {len(bulk)} of {n_want} bulk ladder points and {len(mc)} confined "
          f"states. The confined profiles are the training data and they cannot "
          f"be generated until every ladder is finished, because each state "
          f"reads its activity off the ladder for its own (d, p).")
        A(">")
        A("> The campaign is resumable and idempotent: re-run "
          "`scripts/superball_campaign.py` with no arguments and it will skip "
          "what is here and continue. Statistics are recorded per state "
          "(`n_run`, `n_burn_used`, `chains`) and are NOT uniform across what "
          "exists, because the run length was cut mid-campaign; a production "
          "set means raising those constants and deleting `bulk_eos.npy`.\n")
    A(f"Generated {man['generated']}"
      + ("  **(--cheap: smoke statistics, not for production)**" if CHEAP else "")
      + "\n")
    A("Aligned hard superballs, body $|x|^p + |y|^p + |z|^p \\le (\\sigma/2)^p$, "
      "overlap when $\\|\\mathbf{r}_i - \\mathbf{r}_j\\|_p < \\sigma$. "
      "Orientations are frozen: there are no rotational degrees of freedom in "
      "this model and no orientational observables.\n")
    A("## Shape grid\n")
    A("| p | role | v(p), d=3 | v(p), d=2 | B2, d=3 |")
    A("|---|---|---|---|---|")
    for p in P_GRID:
        A(f"| {pname(p)} | {P_ROLE[p]} | "
          f"{shapes.volume(Superball(p), 3):.5f} | "
          f"{shapes.volume(Superball(p), 2):.5f} | "
          f"{shapes.b2(Superball(p), 3):.5f} |")
    A("")
    A("`p = 3` is held out as a whole shape — the amortisation test. It carries "
      "the same state list as the trained exponents and every one of its states "
      "is marked `test_shape`. `p = inf` is the parallel-cube benchmark, for "
      "comparison against the Cuesta-Martinez-Raton functional, and is not "
      "training data either.\n")
    A("## Measured eta_max(p) — the freezing guard\n")
    A("Parallel cubes freeze near eta = 0.48 through a transition with no "
      "nucleation barrier, so an over-compressed chain does not sit safely "
      "metastable: it orders, and nothing in the density profile says so. Each "
      "bulk ladder therefore runs past the useful range and carries "
      f"S(k)max/<N> from `mcax.order`; the onset is the lowest ladder point to "
      f"exceed {order.ORDER_TRIP}, and eta_max is that less a {ORDER_MARGIN} "
      "margin.\n")
    A("A trip is confirmed before it counts: the state is re-run from a much "
      "sparser start with twice the burn, and has to order again from below. "
      "The lattice prefill starts every dense chain on a crystal, so a single "
      "run cannot tell an unmelted initial condition from a real transition.\n")
    A("| d | p | onset | eta_max | ladder top | ordered | undetermined |")
    A("|---|---|---|---|---|---|---|")
    for k, v in etamax.items():
        on = f"{v['onset']:.4f}" if v["found"] else "none found"
        A(f"| {v['d']} | {v['p']} | {on} | {v['eta_max']:.4f} | "
          f"{v['ladder_top']:.4f} | {v['n_ordered']} | "
          f"{v['n_undetermined']} |")
    A("")
    A("An **undetermined** point tripped from a lattice start and its sparse "
      "restart could not reach a comparable density, so the restart was "
      "measuring a different state point and cleared nothing. Those count as "
      "ordered for the cap. They are the equilibration wall showing up inside "
      "the freezing guard, not a second phenomenon.")
    A("")
    if excluded:
        A(f"**{len(excluded)} states excluded** by the guard, listed rather "
          f"than silently dropped:\n")
        A("| state | reason |")
        A("|---|---|")
        for e in excluded:
            A(f"| `{e['tag']}` | {e['reason']} |")
        A("")
    A("## Acceptance tests\n")
    A("**A0** — the p = 2 path reproduces the pre-shapes hard-sphere engine "
      "bit for bit, checked against a git worktree of the previous commit by "
      "`scripts/a0_bitwise.py`, and again inside `tests/test_shapes.py`.\n")
    A("**A1** — dilute limit: B2 fitted from the ladder against the exact "
      "$2^{d-1} v(p)$.\n")
    A("| d | p | B2 measured | B2 exact | rel. dev | sigma |")
    A("|---|---|---|---|---|---|")
    for r in a1:
        A(f"| {r['d']} | {r['p']} | {r['b2_measured']:.5f} | "
          f"{r['b2_exact']:.5f} | {r['dev_rel']:.2%} | {r['dev_sigma']:.1f} |")
    A("")
    A("**A2** — cube equation of state, two box sizes. The ladder's dilute "
      "points run in a larger cell than its dense ones, so the curve must not "
      "step where they meet.\n")
    A("| d | small L | big L | rho | beta mu (big) | beta mu (small) | dev |")
    A("|---|---|---|---|---|---|---|")
    for r in a2:
        A(f"| {r['d']} | {r['L_small']} | {r['L_big']} | {r['rho']:.4f} | "
          f"{r['mu_big']:.4f} | {r['mu_small']:.4f} | {r['dev']:.4f} |")
    A("")
    A("**A3** — contact theorem $\\rho(0^+) = \\beta P$ at a hard wall, exact "
      "for parallel walls whatever the particle shape. The pressure comes from "
      "the measured ladder by Gibbs-Duhem, since no reference equation of "
      "state exists off p = 2.\n")
    A("| d | p | state | rho(contact) | err | first bin | beta P | rel. dev | "
      "sigma | exact beta P (dev) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in a3:
        ref = (f"{r['beta_p_reference']:.4f} ({r['dev_rel_reference']:.2%})"
               if "beta_p_reference" in r else "n/a")
        ce = ("n/a" if r.get("contact_err", float("nan")) != r.get("contact_err")
              else f"{r['contact_err']:.4f}")
        ds = ("n/a" if r.get("dev_sigma", float("nan")) != r.get("dev_sigma")
              else f"{r['dev_sigma']:.1f}")
        A(f"| {r['d']} | {r['p']} | `{r['tag']}` | {r['contact']:.4f} | {ce} | "
          f"{r['bin0']:.4f} | {r['beta_p']:.4f} | {r['dev_rel']:.2%} | {ds} | "
          f"{ref} |")
    A("")
    A("**Read the sigma column, not the per cent.** rho(0+) is a quadratic "
      "extrapolated past the end of the profile, which amplifies the per-bin "
      "noise; the <N> error bar does not describe it, because <N> averages "
      "every bin and this uses six of them and then leans outward. The error "
      "quoted is the chain-to-chain scatter of the same extrapolation.\n")
    A("beta P is the ladder's integral evaluated at the reservoir density on a "
      "refined grid, NOT a chord between the two ladder points that bracket it. "
      "beta P(rho) is strongly convex and the chord sits above it: at p = 2 in "
      "d = 3 that read 3.5% high against Carnahan-Starling and in d = 2 2.2% "
      "high against Henderson, the same bias at every p, and it was being "
      "reported as an engine deviation.\n")
    A("What remains is shared between the sum rule and the integration that "
      "produced beta P. The two are separated at p = 2, where the fluid is hard "
      "spheres or hard discs and the integration can be checked against "
      "Carnahan-Starling and Henderson directly:\n")
    A("| d | eta | beta P integrated | reference | rel. dev |")
    A("|---|---|---|---|---|")
    for r in gd:
        A(f"| {r['d']} | {r['eta']:.3f} | {r['p_integrated']:.4f} | "
          f"{r['p_reference']:.4f} | {r['dev_rel']:.2%} |")
    A("")
    A("## Splits\n")
    A("| split | meaning | count |")
    A("|---|---|---|")
    for k, lbl in (("train", "fitted"),
                   ("val", "unseen packing fraction at a **seen** slit width"),
                   ("test_state", "unseen packing fraction, deeper than trained"),
                   ("test_width", "unseen slit width — the extrapolation axis"),
                   ("test_shape", "**an entire unseen exponent**, p = 3"),
                   ("test_box", "cuboid cavity, hard on every face"),
                   ("flagged", "**generated, then relabelled** by the corrected "
                               "guard: above the cap, ordered, or drifting")):
        A(f"| `{k}` | {lbl} | {splits.get(k, 0)} |")
    A("")
    if flagged:
        A(f"### {len(flagged)} states relabelled `flagged`\n")
        A("Generated before the freezing guard was corrected mid-campaign, and "
          "kept rather than deleted so the decision is auditable. Do not train "
          "on these.\n")
        A("| state | was | reason |")
        A("|---|---|---|")
        for s_ in flagged:
            A(f"| `{s_['tag']}` | {s_['split_original']} | {s_['flag_reason']} |")
        A("")
    A("## Quality control\n")
    A("Every state carries an independent-replica error bar (chain-to-chain "
      "scatter), a drift diagnostic (shift of <N> between the first and last "
      "third of the window, in units of that error) and the ordering monitor. "
      "Read the RELATIVE drift, not the sigma one: the error bar shrinks as "
      "statistics improve, so a fixed sigma threshold tightens itself and "
      "misorders the states.\n")
    A("`ESS` and `split R-hat` are the cross-chain convergence evidence "
      "(`mcax.diagnostics`, Stan-style estimators on the N series). The "
      "replica error cannot see a burn systematic the chains share, because "
      "every chain starts on the same lattice prefill; split R-hat halves "
      "each chain first, so common drift still shows up. Read them as "
      "evidence, not thresholds: a converged dense state at these run "
      "lengths honestly reads R-hat near 1.1 with an ESS of a few hundred, "
      "and `n_err` remains a FLOOR on the uncertainty wherever R-hat sits "
      "well above that.\n")
    A("| state | split | <N> | err | drift (rel) | drift (sigma) | S/N | sat "
      "| ESS | split R-hat |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(mc, key=lambda x: (x["d"], x["p"], x["geom"], x["H"],
                                       x["eta_reservoir"])):
        rel = s["drift_sigma"] * s["n_err"] / max(s["n_mean"], 1e-12)
        sn = ("n/a" if s["s_max_over_n"] != s["s_max_over_n"]
              else f"{s['s_max_over_n']:.3f}")
        es = s.get("ess", float("nan"))
        rh = s.get("split_rhat", float("nan"))
        A(f"| `{s['tag']}` | {s['split']} | {s['n_mean']:.2f} | "
          f"{s['n_err']:.2f} | {rel:.2%} | {s['drift_sigma']:.1f} | {sn} | "
          f"{s['saturation']:.2f} | "
          f"{'n/a' if es != es else f'{es:.0f}'} | "
          f"{'n/a' if rh != rh else f'{rh:.3f}'} |")
    A("")
    bad = [s for s in mc
           if s["drift_sigma"] * s["n_err"] / max(s["n_mean"], 1e-12) > 0.02]
    A(f"States with relative drift above 2%: **{len(bad)}**"
      + (" — " + ", ".join(f"`{s['tag']}`" for s in bad) if bad else " (none)")
      + "\n")
    A("## Bulk ladders\n")
    A("One per (d, p). These are both the equations of state the confined "
      "states read their activity off and the freezing ramp.\n")
    A("| d | p | points | eta range | worst drift (sigma) |")
    A("|---|---|---|---|---|")
    for d in (3, 2):
        for p in P_GRID:
            sel = [r for r in bulk if r["d"] == d and r["p"] == p]
            if not sel:
                continue
            A(f"| {d} | {pname(p)} | {len(sel)} | "
              f"{min(r['eta_mean'] for r in sel):.4f} – "
              f"{max(r['eta_mean'] for r in sel):.4f} | "
              f"{max(r['drift_sigma'] for r in sel):.1f} |")
    A("")
    short = [r for r in bulk
             if r.get("n0_seated", r.get("n0_prefill", 0)) < r.get("n0_prefill", 0)]
    if short:
        A(f"**{len(short)} ladder rungs could not seat their prefill**: the "
          f"cubic lattice caps at floor(L/1.05)^d sites, eta = 0.351 for "
          f"spheres in the L = 8 box, so these rungs started well below "
          f"target and had to climb grand-canonically at sub-percent "
          f"insertion acceptance. Their stall is an initial-condition "
          f"artefact on top of the sampling wall, and the usable-ladder cuts "
          f"are what keep them out of the activity interpolation:\n")
        A("| rung | asked | seated | seat eta | reached eta |")
        A("|---|---|---|---|---|")
        for r in short:
            se = r["n0_seated"] * r["v_particle"] / r["volume"]
            A(f"| `{r['tag']}` | {r['n0_prefill']} | {r['n0_seated']} | "
              f"{se:.4f} | {r['eta_mean']:.4f} |")
        A("")
    A(f"Total sampling time {man['total_wall_time_s'] / 60:.1f} min "
      f"over {len(bulk)} ladder points and {len(mc)} confined states.\n")
    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `README.md` | the data card: schema, conventions, weighting rules. "
      "Downstream consumers start THERE. Maintained by hand, not regenerated |")
    A("| `bulk_eos.npy` | object array of bulk ladder dicts, with `z`, `rho` |")
    A("| `mc_states.npy` | object array of confined-state dicts, with profiles |")
    A("| `excluded.npy` | states the freezing guard refused, with reasons |")
    A("| `manifest.json` | everything above, machine-readable |")
    with open(os.path.join(ROOT, "MANIFEST.md"), "w") as fh:
        fh.write("\n".join(L))
    return man


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "bulk", "confined", "manifest"])
    ap.add_argument("--cheap", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="a two-shape, five-state plan: exercises every stage "
                         "and the manifest end to end in a few minutes, which "
                         "is what --cheap was too big to do")
    args = ap.parse_args()
    if args.quick:
        args.cheap = True
        global P_GRID, P_ROLE, BULK_ETA, SLIT_TRAIN, SLIT_HELD, BOX_STATES
        P_GRID = [2.0, 3.0]
        P_ROLE = {2.0: "train", 3.0: "test_shape"}
        # Six rungs, three of them below eta = 0.10, and the shape of that is
        # forced from two directions. The stall cut needs three points below it
        # before it can predict the fourth and truncates from the first failure
        # upwards, so a short ladder that loses its top has nothing left and
        # `mu_of_eta` refuses every confined state. And A1 fits its B2 intercept
        # on the rungs at eta <= 0.10, so with fewer than three of those the
        # virial table comes out empty. The first draft of this had four rungs
        # and three, and produced a manifest with no acceptance tests in it at
        # all, which is not a smoke test of anything.
        BULK_ETA = {3: [0.01, 0.04, 0.08, 0.16, 0.26, 0.34],
                    2: [0.02, 0.06, 0.10, 0.20, 0.30, 0.40]}
        SLIT_TRAIN = {3: [(6.0, 0.15)], 2: [(6.0, 0.30)]}
        SLIT_HELD = {3: [(5.0, 0.25, "test_width")],
                     2: [(5.0, 0.45, "test_width")]}
        BOX_STATES = {3: [(1.6, 0.25)], 2: [(1.6, 0.40)]}

    global CHEAP, CHAINS, NRUN, NBURN, THIN, BURN_SWEEPS
    CHEAP = args.cheap
    CHAINS = 8 if CHEAP else 32
    # `NRUN` read `20_000 if CHEAP else 20_000`, so the sampling window never
    # shrank and only the chain count and the burn floor did. The smoke run was
    # costing most of a production one, which is the sort of thing that stops a
    # smoke run being used.
    NRUN = 4_000 if CHEAP else 20_000
    NBURN = 2_000 if CHEAP else 8_000
    BURN_SWEEPS = 40 if CHEAP else BURN_SWEEPS
    THIN = 50

    os.makedirs(ROOT, exist_ok=True)
    print(f"mcax superball campaign -> {os.path.abspath(ROOT)}")
    print(f"devices: {jax.devices()}   C={CHAINS} n_run={NRUN}"
          + ("  [CHEAP]" if CHEAP else ""))

    bulk = stage_bulk() if args.stage in ("all", "bulk") \
        else load(os.path.join(ROOT, "bulk_eos.npy"))
    etamax = eta_max_table(bulk)
    print("\neta_max(p):")
    for k, v in etamax.items():
        print(f"   {k:<14s} onset "
              f"{('%.4f' % v['onset']) if v['found'] else '   none':>8s}"
              f"  -> eta_max {v['eta_max']:.4f}")

    if args.stage in ("all", "confined"):
        mc, excluded = stage_confined(bulk, etamax)
    else:
        mc = load(os.path.join(ROOT, "mc_states.npy"))
        excluded = load(os.path.join(ROOT, "excluded.npy"))

    man = write_manifest(bulk, mc, excluded, etamax)
    print(f"\nwrote {ROOT}/manifest.json + MANIFEST.md")
    print("splits: " + ", ".join(f"{k}={v}" for k, v in
                                 sorted(man["splits"].items())))
    # An empty acceptance table used to print "worst relative deviation 0.00%",
    # which reads as a pass and means the test did not run.
    for name, label in (("A1_b2", "A1 second virial"),
                        ("A3_contact", "A3 contact theorem")):
        rows = man["acceptance"][name]
        if not rows:
            print(f"{label}: NOT RUN — no rows (too few usable ladder points, "
                  f"or no confined state to check)")
            continue
        worst = max(rows, key=lambda r: r["dev_rel"])
        sig = worst.get("dev_sigma", float("nan"))
        extra = (f", {sig:.1f} sigma" if sig == sig else
                 ", no error bar on this row")
        print(f"{label}: worst relative deviation {worst['dev_rel']:.2%} "
              f"(d={worst['d']} p={worst['p']}{extra})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
