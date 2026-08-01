"""mcax: batched hard-particle grand-canonical Monte Carlo in JAX.

The configurational weight of a muVT state with N particles at positions r^N is

    P(N, r^N)  ~  z^N / N!  exp(-beta U(r^N)),        z = exp(beta mu)

and for hard particles U is 0 or infinity, so the Boltzmann factor is an
INDICATOR: 1 if no pair overlaps, 0 otherwise, and the exact acceptances follow:

    displacement : accept iff no overlap and inside the walls
    insertion    : accept with min(1, z V / (N+1)) iff no overlap
    deletion     : accept with min(1, N / (z V))

An external field beta V_ext(r) multiplies each of these by the obvious
Boltzmann factor (exp(-[V' - V]) displacing, exp(-V) inserting, exp(+V)
deleting) and changes nothing else: the overlap test is still an indicator and
the geometry is still hard. With `field = None` the factors are exactly one and
the arithmetic below is the pure hard-particle path. See `mcax.fields`.

Design: many independent chains in lockstep on one device. A chain is a
fixed-capacity particle set (positions (Nmax, d) plus an alive mask); every MC
step, all chains draw a move type, compute all three branches
vectorised-and-masked, and select. A Markov chain cannot be parallelised
internally, so the parallelism has to come from the only other axis available,
and statistics come from the chain batch (vmap) rather than from clever
single-chain moves. That is the shape accelerators actually like.

Geometry lives in `mcax.geometry` and covers bulk, slit, spherical and
cylindrical pores and a wedge. Walls act on CENTRES in every one of them, the
same centre-exclusion convention as the DFT reference data, so in a slit the
wall theorem rho(0+) = beta P applies literally rather than at a shifted
profile. Dimensions d = 1, 2, 3, and d = 1 (hard rods) is kept deliberately:
Tonks/Percus is exact there, which makes rods the engine's razor. Any detailed
balance or bookkeeping bug shows up against an exact answer.

Everything is float64 and jit/scan-compiled: a production sweep is one compiled
call per state point.
"""
from functools import partial
from typing import NamedTuple

import numpy as onp
import jax
import jax.numpy as np

from . import fields
from . import geometry


def require_float64():
    """Refuse to run in single precision, loudly, rather than be subtly wrong.

    The overlap test is `d2 >= sigma**2`, an exact comparison against a boundary
    that a float32 rounding error steps across in either direction. In single
    precision the sampler therefore admits marginal overlaps and rejects
    marginal legal moves, and NOTHING downstream reveals it: the density comes
    out plausible, the acceptance rates look ordinary, the diagnostics converge.
    A run that is silently wrong is worse than one that will not start, so this
    raises.

    JAX defaults to float32 and the flag has to be set before the first array is
    made, which is exactly the kind of requirement that gets left out of a
    script written in a hurry. Hence a guard rather than a line in the README.
    """
    if np.asarray(1.0).dtype != np.float64:
        raise RuntimeError(
            "mcax requires float64, but JAX is in single precision. Add\n\n"
            "    import jax; jax.config.update('jax_enable_x64', True)\n\n"
            "BEFORE creating any array (the flag is read once, at first use). "
            "The overlap test is an exact comparison against sigma^2, so in "
            "float32 this engine silently samples the wrong ensemble.")


class MCSpec(NamedTuple):
    d: int             # fluid dimension: 1, 2, 3
    Nmax: int          # per-chain capacity
    H: float           # confining length: see `mcax.geometry` for its meaning
    Lperp: float       # transverse edge, or the radius of a cylinder
    sigma: float       # hard-core diameter
    z_act: float       # activity exp(beta mu)
    dmax: float        # displacement half-width
    geom: str          # bulk | slit | sphere | cylinder | wedge
    p_disp: float      # move mix: displacement prob (rest split ins/del)
    psi: float         # wedge opening half-angle, radians; ignored otherwise
    field: object      # beta V_ext(r) callable, or None. See `mcax.fields`

    @property
    def slit(self):
        """Kept because it reads better than `geom == "slit"` at the call site,
        and because it was the original spelling of the whole geometry."""
        return self.geom == "slit"


def make_spec(d, H, z_act, Lperp = 10.0, sigma = 1.0, Nmax = None, dmax = 0.15,
              slit = None, p_disp = 0.5, geom = None, psi = onp.pi / 6.0,
              field = None):
    """Build the static description of a state point.

    E.G. hard spheres at bulk packing fraction eta = 0.3 in a slit 8 diameters
    wide, taking the activity from the reference equation of state:

        rho_b = 0.3 / mcax.eos.B[3]
        spec  = make_spec(d = 3, H = 8.0, z_act = eos.z_of_rho(3, rho_b))

    and the same fluid in a spherical pore of radius 5, or a wedge of opening
    half-angle 30 degrees:

        spec = make_spec(d = 3, geom = "sphere", H = 5.0, z_act = ...)
        spec = make_spec(d = 3, geom = "wedge", H = 8.0, psi = onp.pi/6, ...)

    `H` and `Lperp` mean different things in different geometries and
    `geometry.describe(spec)` will say which. Passing `slit = True/False`
    still works and selects `slit` or `bulk`.
    """
    if d != int(d) or int(d) not in (1, 2, 3):
        # Nothing in the kernel would object to d = 4: the geometry is written
        # for general d and would happily run. But `mcax.eos` has no reference
        # equation of state there, so the answer could never be checked against
        # anything, and an unvalidatable number is not worth producing. The
        # non-integer half of the test matters just as much: every dimension in
        # here goes through int(), so d = 2.5 would quietly become a 2-D run.
        raise ValueError(f"d must be 1, 2 or 3, got {d}")
    if geom is None:
        geom = "slit" if (slit is None or slit) else "bulk"
    elif slit is not None:
        raise ValueError("pass geom or slit, not both")
    geometry.check(geom, int(d))

    spec = MCSpec(int(d), 0, float(H), float(Lperp), float(sigma),
                  float(z_act), float(dmax), str(geom), float(p_disp),
                  float(psi), field)
    if Nmax is None:
        # dense-limit headroom: eta_max ~ 0.75 in any d at sigma = 1. Sized on
        # the ACCESSIBLE volume, so a spherical pore is not handed the capacity
        # of the cube around it.
        Nmax = int(1.6 * geometry.volume(spec)) + 64
    return spec._replace(Nmax = int(Nmax))


class MCState(NamedTuple):
    pos: np.ndarray        # (C, Nmax, d)
    alive: np.ndarray      # (C, Nmax) bool
    key: np.ndarray        # (C, 2) PRNG keys
    acc: np.ndarray        # (C, 3) accepted moves per type
    tot: np.ndarray        # (C, 3) attempted moves per type
    nhi: np.ndarray        # (C,) largest N ever held: capacity watchdog


class Result(NamedTuple):
    """What a sampling run produced, plus what is needed to trust it."""
    z: onp.ndarray         # (nbins,) bin centres: wall axis, or radius
    rho: onp.ndarray       # (nbins,) density profile, per unit d-volume
    Ns: onp.ndarray        # (C, nchunks) particle-number series: (chain, draw)
    acc: onp.ndarray       # (3,) acceptance rates: disp, ins, del
    n_mean: float          # <N> over chains and draws
    n_hi: int              # largest N held by any chain at any step
    capacity: int          # spec.Nmax
    saturation: float      # n_hi / capacity, see `capacity_warning`
    state: MCState         # final state, so a run can be continued
    hist: onp.ndarray      # (C, nbins) raw counts behind `rho`
    hist_n: onp.ndarray    # (C, nbins) the same, weighted by instantaneous N
    gr: onp.ndarray        # (C, nbins_g) pair separations, empty unless asked
    nsamples: int          # draws per chain, the divisor for all of the above

    @property
    def capacity_warning(self):
        """Non-None once the fixed-capacity truncation has begun to bite.

        Insertions into a full chain are rejected by `has_slot`, which is NOT a
        Metropolis rejection: it truncates the muVT ensemble at N = Nmax and
        breaks detailed balance. The headroom heuristic in `make_spec` normally
        keeps this far away, but a high-activity or small-box run can reach it,
        and nothing else in the output would show it. Check this before
        believing any number above!!
        """
        if self.saturation >= 1.0:
            return (f"capacity REACHED: some chain held all {self.capacity} "
                    f"slots. Insertions were rejected for want of a slot, so "
                    f"the ensemble is truncated and detailed balance is "
                    f"broken. Re-run with a larger Nmax.")
        if self.saturation >= 0.95:
            return (f"capacity {100 * self.saturation:.1f}% used "
                    f"(n_hi = {self.n_hi} of {self.capacity}). Not yet biased, "
                    f"but a longer run would likely truncate. Raise Nmax.")
        return None


def lattice_fill(spec, n0, seed = 0):
    """(<= n0, d) non-overlapping cubic-lattice sites inside the box.

    Chains started empty must FILL grand-canonically, one accepted insertion at
    a time. At high density the insertion acceptance is about 1%, so the fill
    takes as long as the burn-in and biases everything sampled before it
    completes. Seating ~90% of the target N on a lattice and melting it in the
    burn removes the transient. Variance reduction only: cold and warm starts
    must equilibrate to the same density, and the tests check that they do.
    """
    a = 1.05 * spec.sigma
    lo, hi = geometry.bbox(spec)
    _, wraps = geometry.periods(spec)
    # A periodic axis has to stop short by half a lattice spacing, or the last
    # site and the first are neighbours ACROSS the boundary and overlap. The
    # aperiodic axes have no such wrap and run to the wall.
    axes = [onp.arange(l + 0.5 * a, h - (0.49 * a if w else 0.01), a)
            for l, h, w in zip(lo, hi, wraps)]
    grids = onp.meshgrid(*axes, indexing = "ij")
    sites = onp.stack([g.ravel() for g in grids], axis = -1)
    # The lattice is built over the bounding box, so in a curved pore most of
    # it is outside the region. Filter on the host, where a boolean mask costs
    # nothing and no static shape has to be guessed.
    inside = onp.array([bool(geometry.contains(spec, np.asarray(s)))
                        for s in sites]) if len(sites) else onp.zeros(0, bool)
    sites = sites[inside]
    rng = onp.random.default_rng(seed)
    rng.shuffle(sites)
    return sites[:min(n0, len(sites))]


def init_state(spec, C, seed = 0, n0 = 0):
    """`n0 > 0` seats that many particles on a melt-in lattice.

    Each chain gets its OWN shuffle of the lattice, not a shared one. Seating
    every chain identically costs nothing in the mean but FLATTERS the mixing
    diagnostic: chains launched from the same configuration are perfectly
    correlated at step 0, so the between-chain variance starts at zero and
    split R-hat reports agreement the sampler has not earned.

    Measured on d = 3, eta = 0.3 (2026-07-26, same 200k burn and 400k run):
    a shared start reported R-hat 1.06, per-chain shuffles report 1.13. The
    higher number is the honest one, and it is telling the truth: N in a dense
    3-D fluid decorrelates over roughly 120 draws, so that run had about seven
    independent draws per chain and had no business looking converged. Use it
    as a signal to run longer rather than as a reason to go back.
    """
    require_float64()
    keys = jax.random.split(jax.random.PRNGKey(seed), C)
    pos = np.zeros((C, spec.Nmax, spec.d))
    alive = np.zeros((C, spec.Nmax), dtype = bool)
    if n0 > 0:
        sites = onp.stack([lattice_fill(spec, n0, seed + c) for c in range(C)])
        pos = pos.at[:, :sites.shape[1]].set(np.asarray(sites))
        alive = alive.at[:, :sites.shape[1]].set(True)
    return MCState(
        pos = pos,
        alive = alive,
        key = keys,
        acc = np.zeros((C, 3), dtype = np.int64),
        tot = np.zeros((C, 3), dtype = np.int64),
        nhi = np.sum(alive, axis = 1).astype(np.int64),
    )


# ---- per-chain kernels (vmapped over the batch by run()) ------------------ #

def _dist2(spec, a, b):
    """Squared centre distance, minimum image on whichever axes are periodic.

    Which those are is the geometry's business, not this function's: `bulk` is
    periodic everywhere, a slit on its transverse axes, a cylinder along its
    axis only, and a spherical pore nowhere at all.
    """
    return np.sum(geometry.min_image(spec, a - b) ** 2, axis = -1)


def _overlaps_any(spec, pos, alive, trial, skip_idx):
    """True if `trial` overlaps any alive particle (excluding skip_idx).

    Pass `skip_idx = spec.Nmax` to skip nobody: an out-of-bounds scatter is
    dropped by JAX, whereas -1 would wrap round and silently mask the LAST
    capacity slot, which holds a real particle in a nearly-full chain.
    """
    d2 = _dist2(spec, pos, trial[None, :])
    ok = (d2 >= spec.sigma ** 2) | ~alive
    ok = ok.at[skip_idx].set(True)
    return ~np.all(ok)


def _pick_alive(key, alive):
    """Uniform index among alive (argmax of uniforms masked to alive)."""
    u = jax.random.uniform(key, alive.shape)
    return np.argmax(np.where(alive, u, -1.0))


def _step_chain(spec, carry, _):
    pos, alive, key, acc, tot, nhi = carry
    key, k_type, k_pick, k_move, k_acc = jax.random.split(key, 5)
    N = np.sum(alive)
    # The BOUNDING-box volume, not the accessible one. Insertions are proposed
    # uniformly in the bounding box and rejected if they land outside the
    # region, so V_box is what the Metropolis-Hastings ratio carries. See the
    # proposal note in `mcax.geometry`.
    V = geometry.bbox_volume(spec)
    lo, hi = (np.asarray(x) for x in geometry.bbox(spec))
    u_type = jax.random.uniform(k_type)
    p_ins = spec.p_disp + 0.5 * (1.0 - spec.p_disp)
    move = np.where(u_type < spec.p_disp, 0, np.where(u_type < p_ins, 1, 2))

    # -- displacement branch ------------------------------------------------ #
    idx = _pick_alive(k_pick, alive)
    delta = spec.dmax * jax.random.uniform(k_move, (spec.d,), minval = -1.0,
                                           maxval = 1.0)
    trial_d = geometry.wrap(spec, pos[idx] + delta)
    # One-body energies. With no field these are exact zeros and every
    # exponential below collapses to 1, so the hard-particle path is unchanged.
    v_old = fields.evaluate(spec.field, pos[idx])
    v_new = fields.evaluate(spec.field, trial_d)
    disp_ok = alive[idx] & geometry.contains(spec, trial_d) & \
        (jax.random.uniform(k_acc) < np.exp(-(v_new - v_old))) & \
        ~_overlaps_any(spec, pos, alive, trial_d, idx)
    pos_disp = pos.at[idx].set(np.where(disp_ok, trial_d, pos[idx]))

    # -- insertion branch --------------------------------------------------- #
    slot = np.argmax(~alive)                        # first free slot
    has_slot = ~np.all(alive)
    new = jax.random.uniform(k_move, (spec.d,), minval = lo, maxval = hi)
    a_ins = spec.z_act * V / (N + 1.0) * np.exp(-fields.evaluate(spec.field, new))
    ins_ok = has_slot & geometry.contains(spec, new) & \
        (jax.random.uniform(k_acc) < np.minimum(1.0, a_ins)) & \
        ~_overlaps_any(spec, pos, alive, new, spec.Nmax)
    pos_ins = pos.at[slot].set(np.where(ins_ok, new, pos[slot]))
    alive_ins = alive.at[slot].set(np.where(ins_ok, True, alive[slot]))

    # -- deletion branch ---------------------------------------------------- #
    # The +v_old: removing a particle from a deep well costs its binding energy,
    # so a bound layer is hard to strip. Sign errors here are invisible in the
    # mean density and obvious in the profile.
    a_del = N / np.maximum(spec.z_act * V, 1e-300) * np.exp(v_old)
    del_ok = (N > 0) & (jax.random.uniform(k_acc) < np.minimum(1.0, a_del))
    alive_del = alive.at[idx].set(np.where(del_ok, False, alive[idx]))

    # -- select by move type ------------------------------------------------ #
    # All three branches are computed and two are thrown away, which wastes
    # about 3x the arithmetic per step. It is still the right trade: it keeps
    # the batch divergence-free, so one compiled kernel advances every chain.
    pos_out = np.where(move == 0, pos_disp, np.where(move == 1, pos_ins, pos))
    alive_out = np.where(move == 1, alive_ins,
                         np.where(move == 2, alive_del, alive))
    accepted = np.where(move == 0, disp_ok,
                        np.where(move == 1, ins_ok, del_ok))
    acc = acc.at[move].add(accepted.astype(np.int64))
    tot = tot.at[move].add(1)
    nhi = np.maximum(nhi, np.sum(alive_out))
    return (pos_out, alive_out, key, acc, tot, nhi), None


def _hist_chain(spec, pos, alive, nbins):
    """Histogram of alive centres along the geometry's profile coordinate."""
    zc = geometry.profile_coord(spec, pos)
    idx = np.clip((zc / geometry.extent(spec) * nbins).astype(np.int32),
                  0, nbins - 1)
    return np.zeros(nbins).at[idx].add(alive.astype(np.float64))


def _pair_hist_chain(spec, pos, alive, nbins_g, rmax):
    """Histogram of live pair separations, i < j, out to `rmax`.

    Costs an (Nmax, Nmax) distance matrix, so it is the one accumulator here
    that is not free: at Nmax = 600 and 32 chains under vmap it is a few
    hundred megabytes of transient. It runs only at thinning boundaries, never
    per step, which is what keeps that affordable, and `nbins_g = 0` switches
    it off entirely.
    """
    d2 = _dist2(spec, pos[:, None, :], pos[None, :, :])     # (Nmax, Nmax)
    r = np.sqrt(d2)
    iu = np.arange(spec.Nmax)
    ok = (iu[:, None] < iu[None, :]) & alive[:, None] & alive[None, :] \
        & (r < rmax)
    idx = np.clip((r / rmax * nbins_g).astype(np.int32), 0, nbins_g - 1)
    return np.zeros(nbins_g).at[idx].add(ok.astype(np.float64))


class Accum(NamedTuple):
    """What a scan of `run` accumulated, in raw counts.

    Kept as counts rather than normalised densities so that two runs can be
    added together, which is what continuing a run from `result.state` needs.
    `mcax.observables` does the normalising.
    """
    hist: np.ndarray       # (C, nbins) summed instantaneous z-histogram
    Ns: np.ndarray         # (C, nchunks) particle number: (chain, draw)
    hist_n: np.ndarray     # (C, nbins) the same histogram weighted by N
    gr: np.ndarray         # (C, nbins_g) pair separations, empty if switched off


@partial(jax.jit, static_argnums = (0, 2, 3, 4, 5, 6))
def run(spec, state, nsteps, thin, nbins, nbins_g = 0, rmax = None):
    """Advance all chains `nsteps`, accumulating at every thinning boundary.

    Returns (state, `Accum`). Three of the five accumulators are there for the
    fluctuation identities in `mcax.observables`: `hist_n` is the density-number
    cross-covariance that gives d rho(z)/d(beta mu), and `gr` gives the pair
    correlations. All are sampled at thinning boundaries, so they cost one
    reduction per `thin` steps and nothing per step.

    The precision guard runs at TRACE time, so it costs one comparison per
    compile and nothing per step, and it still catches a state handed in from
    somewhere other than `init_state`.
    """
    require_float64()
    C = state.pos.shape[0]
    nchunks = nsteps // thin
    if nbins_g and spec.geom == "bulk":
        rmax = _default_rmax(spec) if rmax is None else rmax
    elif nbins_g:
        raise ValueError(
            "pair correlations are only accumulated in bulk. Against a wall "
            "the pair distribution depends on both centres, g(r1, r2), and "
            "averaging that over the profile is not the g(r) anybody means.")

    def chunk(carry, _):
        st = carry

        def inner(c, _):
            return jax.vmap(lambda p, a, k, ac, t, nh: _step_chain(
                spec, (p, a, k, ac, t, nh), None)[0])(*c), None

        st, _ = jax.lax.scan(inner, st, None, length = thin)
        pos, alive, key, acc, tot, nhi = st
        h = jax.vmap(lambda p, a: _hist_chain(spec, p, a, nbins))(pos, alive)
        n = np.sum(alive, axis = 1)
        if nbins_g:
            g = jax.vmap(lambda p, a: _pair_hist_chain(
                spec, p, a, nbins_g, rmax))(pos, alive)
        else:
            g = np.zeros((C, 0))
        return st, (h, n, h * n[:, None], g)

    carry = (state.pos, state.alive, state.key, state.acc, state.tot, state.nhi)
    carry, (hists, Ns, hns, grs) = jax.lax.scan(
        chunk, carry, None, length = nchunks)
    return MCState(*carry), Accum(
        hist = np.sum(hists, axis = 0),             # (C, nbins)
        Ns = np.swapaxes(Ns, 0, 1),                 # (C, nchunks)
        hist_n = np.sum(hns, axis = 0),
        gr = np.sum(grs, axis = 0),
    )


def _default_rmax(spec):
    """Half the shortest periodic edge: past that the minimum image is
    ambiguous. Only ever asked in bulk, where every axis is periodic."""
    per, _ = geometry.periods(spec)
    return 0.5 * float(per.min())


def density_profile(spec, hist, nsamples):
    """Average over chains -> rho on bin centres, per unit d-volume.

    The divisor is the geometry's own bin volume, which for a spherical pore is
    a shell growing like r^{d-1}. Dividing by a constant slab there would tilt
    the whole profile in a way that looks exactly like a packing effect.
    """
    nbins = hist.shape[-1]
    dz = geometry.extent(spec) / nbins
    cell = geometry.bin_volume(spec, nbins)
    rho = hist.sum(axis = 0) / (hist.shape[0] * nsamples * cell)
    z = (onp.arange(nbins) + 0.5) * dz
    return z, rho


def burn_and_sample(spec, C, seed, n_burn, n_run, thin, nbins, n0 = 0,
                    nbins_g = 0):
    """Convenience: burn-in, then sample. Returns a `Result`.

    `n0` is the lattice-prefill count (see `lattice_fill`). The returned `Ns`
    is (C, nchunks), chains by draws, which is the layout ArviZ and NumPyro
    expect, so it feeds `mcax.diagnostics` (or either of those, if you have
    them) unchanged. Its drift is the equilibration diagnostic, and
    `capacity_warning` should be read before trusting `rho`.

    `nbins_g > 0` additionally accumulates pair separations, in bulk only. It
    is off by default because it is the one accumulator with a real cost.
    """
    st = init_state(spec, C, seed, n0 = n0)
    st, _ = run(spec, st, n_burn, max(n_burn // 4, 1), nbins)
    # Reset the counters but NOT nhi: capacity pressure during burn-in is just
    # as much a broken-ensemble signal as during sampling.
    st = st._replace(acc = np.zeros_like(st.acc), tot = np.zeros_like(st.tot))
    st, ac = run(spec, st, n_run, thin, nbins, nbins_g = nbins_g)
    z, rho = density_profile(spec, onp.asarray(ac.hist), n_run // thin)
    acc = onp.asarray(st.acc).sum(0) / onp.maximum(onp.asarray(st.tot).sum(0), 1)
    Ns = onp.asarray(ac.Ns)
    n_hi = int(onp.max(onp.asarray(st.nhi)))
    return Result(z = z, rho = rho, Ns = Ns, acc = acc,
                  n_mean = float(Ns.mean()), n_hi = n_hi, capacity = spec.Nmax,
                  saturation = n_hi / spec.Nmax, state = st,
                  hist = onp.asarray(ac.hist), hist_n = onp.asarray(ac.hist_n),
                  gr = onp.asarray(ac.gr), nsamples = n_run // thin)
