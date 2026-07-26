"""mcax — batched hard-particle grand-canonical Monte Carlo in JAX.

Design: **many independent chains in lockstep on one device**. A chain is a
fixed-capacity particle set (positions ``(Nmax, d)`` + alive mask); every MC
step, all chains draw a move type (displacement / insertion / deletion),
compute all three branches vectorised-and-masked, and select. Statistics come
from the chain batch (vmap), not from clever single-chain moves — the shape
that GPUs (and JAX) actually like. Hard cores mean no energies anywhere: an
overlap test and the exact muVT acceptances are the whole physics:

    displacement : accept iff no overlap and inside the walls
    insertion    : accept with min(1, z V / (N+1)) iff no overlap
    deletion     : accept with min(1, N / (z V)),   z = exp(beta mu), Lambda = 1

Geometry: ``slit`` (hard walls for CENTRES at x_d = 0 and H — the same
centre-exclusion convention as the DFT reference data, so the wall theorem
rho(0+) = beta P applies literally) or ``bulk`` (periodic everywhere).
Dimensions d = 1, 2, 3. d = 1 (hard rods) is deliberately included: Tonks/
Percus is exact there, which makes rods the engine's razor validation — any
detailed-balance or bookkeeping bug shows up against an exact answer.

Everything is float64 and jit/scan-compiled; a production sweep is one
compiled call per state point.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


class MCSpec(NamedTuple):
    d: int             # fluid dimension: 1, 2, 3
    Nmax: int          # per-chain capacity
    H: float           # wall separation (centre-accessible) along the last axis
    Lperp: float       # periodic box edge for the d-1 transverse axes
    sigma: float       # hard-core diameter
    z_act: float       # activity exp(beta mu)
    dmax: float        # displacement half-width
    slit: bool         # True: hard walls at 0, H; False: periodic (bulk)
    p_disp: float      # move mix: displacement prob (rest split ins/del)


def make_spec(d, H, z_act, Lperp=10.0, sigma=1.0, Nmax=None, dmax=0.15,
              slit=True, p_disp=0.5) -> MCSpec:
    vol = H * Lperp ** (d - 1)
    if Nmax is None:
        # dense-limit headroom: eta_max ~ 0.75 in any d at sigma = 1
        Nmax = int(1.6 * vol) + 64
    return MCSpec(int(d), int(Nmax), float(H), float(Lperp), float(sigma),
                  float(z_act), float(dmax), bool(slit), float(p_disp))


class MCState(NamedTuple):
    pos: jnp.ndarray       # (C, Nmax, d)
    alive: jnp.ndarray     # (C, Nmax) bool
    key: jnp.ndarray       # (C, 2) PRNG keys
    acc: jnp.ndarray       # (C, 3) accepted moves per type
    tot: jnp.ndarray       # (C, 3) attempted moves per type


def lattice_fill(spec: MCSpec, n0: int, seed: int = 0) -> np.ndarray:
    """(<= n0, d) non-overlapping cubic-lattice sites inside the box. Chains
    started empty must FILL grand-canonically one accepted insertion at a time
    — for dense states the fill time rivals the burn-in and biases everything
    sampled before it completes (reviewed finding). Seating ~90% of the target
    N on a lattice and melting it in the burn removes the transient."""
    a = 1.05 * spec.sigma
    zs = np.arange(0.5 * a, spec.H - 0.01, a)
    per = [np.arange(0.5 * a, spec.Lperp - 0.49 * a, a)
           for _ in range(spec.d - 1)]
    grids = np.meshgrid(*(per + [zs]), indexing="ij")
    sites = np.stack([g.ravel() for g in grids], axis=-1)
    rng = np.random.default_rng(seed)
    rng.shuffle(sites)
    return sites[:min(n0, len(sites))]


def init_state(spec: MCSpec, C: int, seed: int = 0, n0: int = 0) -> MCState:
    """``n0 > 0`` seats that many particles on a melt-in lattice (all chains
    share the start; different keys decorrelate them during burn-in)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), C)
    pos = jnp.zeros((C, spec.Nmax, spec.d))
    alive = jnp.zeros((C, spec.Nmax), dtype=bool)
    if n0 > 0:
        sites = lattice_fill(spec, n0, seed)
        pos = pos.at[:, :len(sites)].set(jnp.asarray(sites)[None])
        alive = alive.at[:, :len(sites)].set(True)
    return MCState(
        pos=pos,
        alive=alive,
        key=keys,
        acc=jnp.zeros((C, 3), dtype=jnp.int64),
        tot=jnp.zeros((C, 3), dtype=jnp.int64),
    )


# ---- per-chain kernels (vmapped over the batch by run()) ------------------ #

def _dist2(spec: MCSpec, a, b):
    """Squared centre distance with minimum image on transverse axes; the wall
    axis (last) is direct in a slit, periodic in bulk."""
    dr = a - b
    if spec.slit:
        per = dr[..., :-1]
        per = per - spec.Lperp * jnp.round(per / spec.Lperp)
        dz = dr[..., -1:]
        dr = jnp.concatenate([per, dz], axis=-1) if spec.d > 1 else dz
    else:
        L = jnp.array([spec.Lperp] * (spec.d - 1) + [spec.H])
        dr = dr - L * jnp.round(dr / L)
    return jnp.sum(dr * dr, axis=-1)


def _overlaps_any(spec: MCSpec, pos, alive, trial, skip_idx):
    """True if ``trial`` overlaps any alive particle (excluding skip_idx).
    Pass ``skip_idx = spec.Nmax`` to skip nobody — an out-of-bounds scatter is
    dropped by JAX, whereas -1 would silently mask the LAST capacity slot."""
    d2 = _dist2(spec, pos, trial[None, :])
    ok = (d2 >= spec.sigma ** 2) | ~alive
    ok = ok.at[skip_idx].set(True)
    return ~jnp.all(ok)


def _pick_alive(key, alive):
    """Uniform index among alive (argmax of uniforms masked to alive)."""
    u = jax.random.uniform(key, alive.shape)
    return jnp.argmax(jnp.where(alive, u, -1.0))


def _step_chain(spec: MCSpec, carry, _):
    pos, alive, key, acc, tot = carry
    key, k_type, k_pick, k_move, k_acc = jax.random.split(key, 5)
    N = jnp.sum(alive)
    V = spec.H * spec.Lperp ** (spec.d - 1)
    u_type = jax.random.uniform(k_type)
    p_ins = spec.p_disp + 0.5 * (1.0 - spec.p_disp)
    move = jnp.where(u_type < spec.p_disp, 0, jnp.where(u_type < p_ins, 1, 2))

    # -- displacement branch ------------------------------------------------ #
    idx = _pick_alive(k_pick, alive)
    delta = spec.dmax * jax.random.uniform(k_move, (spec.d,), minval=-1.0, maxval=1.0)
    trial = pos[idx] + delta
    if spec.slit:
        perp = jnp.mod(trial[:-1], spec.Lperp) if spec.d > 1 else trial[:0]
        zc = trial[-1:]
        trial_d = jnp.concatenate([perp, zc])
        in_wall = (zc[0] >= 0.0) & (zc[0] <= spec.H)
    else:
        L = jnp.array([spec.Lperp] * (spec.d - 1) + [spec.H])
        trial_d = jnp.mod(trial, L)
        in_wall = jnp.asarray(True)
    disp_ok = alive[idx] & in_wall & \
        ~_overlaps_any(spec, pos, alive, trial_d, idx)
    pos_disp = pos.at[idx].set(jnp.where(disp_ok, trial_d, pos[idx]))

    # -- insertion branch --------------------------------------------------- #
    slot = jnp.argmax(~alive)                       # first free slot
    has_slot = ~jnp.all(alive)
    if spec.slit:
        lo = jnp.array([0.0] * (spec.d - 1) + [0.0])
        hi = jnp.array([spec.Lperp] * (spec.d - 1) + [spec.H])
    else:
        lo = jnp.zeros(spec.d)
        hi = jnp.array([spec.Lperp] * (spec.d - 1) + [spec.H])
    new = jax.random.uniform(k_move, (spec.d,), minval=lo, maxval=hi)
    a_ins = spec.z_act * V / (N + 1.0)
    ins_ok = has_slot & (jax.random.uniform(k_acc) < jnp.minimum(1.0, a_ins)) & \
        ~_overlaps_any(spec, pos, alive, new, spec.Nmax)
    pos_ins = pos.at[slot].set(jnp.where(ins_ok, new, pos[slot]))
    alive_ins = alive.at[slot].set(jnp.where(ins_ok, True, alive[slot]))

    # -- deletion branch ---------------------------------------------------- #
    a_del = N / jnp.maximum(spec.z_act * V, 1e-300)
    del_ok = (N > 0) & (jax.random.uniform(k_acc) < jnp.minimum(1.0, a_del))
    alive_del = alive.at[idx].set(jnp.where(del_ok, False, alive[idx]))

    # -- select by move type ------------------------------------------------ #
    pos_out = jnp.where(move == 0, pos_disp, jnp.where(move == 1, pos_ins, pos))
    alive_out = jnp.where(move == 1, alive_ins,
                          jnp.where(move == 2, alive_del, alive))
    accepted = jnp.where(move == 0, disp_ok,
                         jnp.where(move == 1, ins_ok, del_ok))
    acc = acc.at[move].add(accepted.astype(jnp.int64))
    tot = tot.at[move].add(1)
    return (pos_out, alive_out, key, acc, tot), None


def _hist_chain(spec: MCSpec, pos, alive, nbins):
    """z-histogram of alive centres (the wall axis)."""
    zc = pos[:, -1]
    idx = jnp.clip((zc / spec.H * nbins).astype(jnp.int32), 0, nbins - 1)
    return jnp.zeros(nbins).at[idx].add(alive.astype(jnp.float64))


@partial(jax.jit, static_argnums=(0, 2, 3, 4))
def run(spec: MCSpec, state: MCState, nsteps: int, thin: int, nbins: int):
    """Advance all chains ``nsteps``; between thinning intervals accumulate the
    z-histogram and the N-series. Returns (state, hist (C, nbins), Ns (C, k))."""
    C = state.pos.shape[0]
    nchunks = nsteps // thin

    def chunk(carry, _):
        st = carry

        def inner(c, _):
            return jax.vmap(lambda p, a, k, ac, t: _step_chain(
                spec, (p, a, k, ac, t), None)[0])(*c), None

        st, _ = jax.lax.scan(inner, st, None, length=thin)
        pos, alive, key, acc, tot = st
        h = jax.vmap(lambda p, a: _hist_chain(spec, p, a, nbins))(pos, alive)
        n = jnp.sum(alive, axis=1)
        return st, (h, n)

    carry = (state.pos, state.alive, state.key, state.acc, state.tot)
    carry, (hists, Ns) = jax.lax.scan(chunk, carry, None, length=nchunks)
    hist = jnp.sum(hists, axis=0)                   # (C, nbins)
    state = MCState(*carry)
    return state, hist, jnp.swapaxes(Ns, 0, 1)      # Ns: (C, nchunks)


def density_profile(spec: MCSpec, hist: np.ndarray, nsamples: int):
    """Average over chains -> rho(z) on bin centres (per unit d-volume)."""
    nbins = hist.shape[-1]
    dz = spec.H / nbins
    area = spec.Lperp ** (spec.d - 1)
    rho = hist.sum(axis=0) / (hist.shape[0] * nsamples * area * dz)
    z = (np.arange(nbins) + 0.5) * dz
    return z, rho


def burn_and_sample(spec: MCSpec, C: int, seed: int, n_burn: int, n_run: int,
                    thin: int, nbins: int, n0: int = 0):
    """Convenience: burn-in, then sample. Returns (z, rho, Ns, acc_rates).
    ``n0``: lattice-prefill count (see ``lattice_fill``); keep the returned
    ``Ns`` (C, nchunks) series — its drift is the equilibration diagnostic."""
    st = init_state(spec, C, seed, n0=n0)
    st, _, _ = run(spec, st, n_burn, max(n_burn // 4, 1), nbins)
    st = st._replace(acc=jnp.zeros_like(st.acc), tot=jnp.zeros_like(st.tot))
    st, hist, Ns = run(spec, st, n_run, thin, nbins)
    z, rho = density_profile(spec, np.asarray(hist), n_run // thin)
    acc = np.asarray(st.acc).sum(0) / np.maximum(np.asarray(st.tot).sum(0), 1)
    return z, rho, np.asarray(Ns), acc
