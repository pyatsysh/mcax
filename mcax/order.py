"""Translational ordering: the structure factor, as a freezing alarm.

A grand-canonical chain pushed past freezing does not announce it. The density
comes out, the acceptances look ordinary, the error bars stay small, and the
profile is a perfectly plausible curve that happens to describe a crystal. For
hard spheres that is survivable, because the transition is strongly first order
and a compressed fluid sits metastable behind a nucleation barrier for a long
time. For PARALLEL superballs it is not: the cube end of the family freezes
near eta = 0.48 through a transition that is continuous or very nearly so, with
no barrier to sit behind, so an over-compressed chain simply orders. Training
data taken there is poisoned and nothing else in the output says so.

Hence a monitor that is watched, not assumed. The static structure factor

    S(k)  =  < |sum_j exp(i k . r_j)|^2 > / <N>

is the direct measure: it is O(1) at every k in a fluid, and a crystal puts
O(N) into the Bragg peaks of its reciprocal lattice. The discriminator is
therefore not S itself but S/<N>, which vanishes as 1/N in a fluid and tends to
a constant in a crystal, so it can be compared across box sizes and state
points without recalibration.

**Which k.** Only the periodic axes have reciprocal vectors at all, so a slit
contributes in-plane wavevectors and a hard box none whatever: `kvectors`
returns an empty set there and the monitor reports nothing rather than
something meaningless. Within the periodic subspace the vectors are the exact
reciprocal lattice of the simulation cell, 2 pi n_i / L_i for integer n, which
is the only choice for which a perfectly ordered configuration lands ON a
sampled k rather than between two of them.

**Aligned particles make this easier than it usually is.** The crystal a
parallel superball fluid forms is axis-aligned too, so its Bragg peaks sit on
the coordinate axes of reciprocal space and a modest ball of n reaches them.
Nothing here assumes that, but it is why a few hundred wavevectors suffice
where an orientationally disordered system would want thousands.
"""
import numpy as onp
import jax.numpy as np

from . import geometry


def kvectors(spec, nmax = 4, nfam = None, kmax = None):
    """(K, d) reciprocal vectors of the cell: a small ball plus the cubic
    families, over the PERIODIC axes only.

    n = 0 is dropped (S(0) is <N> by construction and says nothing) and each
    pair +-n is counted once, since S(-k) = S(k) for a real density.

    **The split is the whole design, and getting it wrong makes the monitor
    silent rather than wrong.** A crystal in a periodic cell of edge L and
    lattice constant a puts its first Bragg peak at n = L/a, which for the
    campaign's L = 8 and a near sigma is n = 7 or 8. A ball of radius 4 or 5
    does not reach it, so the alarm sees a fluid-looking S(k) at every
    wavevector it sampled and reports nothing at all. The fix is not simply a
    bigger ball: |n| <= 10 in three dimensions is two thousand vectors and
    doubles the cost of a run.

    What makes the cheap answer available is that these particles are ALIGNED,
    so the crystal they form is aligned too and its peaks lie on the cubic
    families {100}, {110}, {111} of the cell. So:

        ball     |n| <= nmax        the fluid peak and general low-order
                                    structure, isotropically
        families m * (100), m * (110), m * (111) and their permutations,
                 m <= nfam          the Bragg orders, along the only directions
                                    an axis-aligned crystal can put them

    `nfam` defaults to the cell edge in units of sigma plus two, which is the
    smallest choice that reaches the first Bragg order of any lattice denser
    than close-packed. `kmax` truncates in physical units on top of that.
    """
    per, flag = geometry.periods(spec)
    if not flag.any():
        return onp.zeros((0, spec.d))
    ax = onp.flatnonzero(flag)
    dper = onp.asarray(per)[ax]
    dd = len(ax)
    if nfam is None:
        nfam = int(onp.ceil(float(dper.max()) / spec.sigma)) + 2

    # the isotropic ball, in the periodic subspace
    rng = onp.arange(-nmax, nmax + 1)
    ball = onp.stack([g.ravel() for g in
                      onp.meshgrid(*([rng] * dd), indexing = "ij")], axis = -1)
    n2 = onp.sum(ball ** 2, axis = -1)
    ball = ball[(n2 > 0) & (n2 <= nmax ** 2)]

    # the cubic families: every sign and permutation pattern with entries in
    # {0, +-1}, scaled out to nfam. Generated rather than listed so that d = 2
    # gets {10} and {11} without a separate table.
    pat = onp.stack([g.ravel() for g in
                     onp.meshgrid(*([onp.array([-1, 0, 1])] * dd),
                                  indexing = "ij")], axis = -1)
    pat = pat[onp.sum(onp.abs(pat), axis = -1) > 0]
    fam = (pat[None, :, :] * onp.arange(1, nfam + 1)[:, None, None])
    fam = fam.reshape(-1, dd)

    grid = onp.unique(onp.concatenate([ball, fam]), axis = 0)
    # Keep one of each +-n pair: the first non-zero component positive.
    lead = grid[onp.arange(len(grid)), onp.argmax(grid != 0, axis = 1)]
    grid = grid[lead > 0]

    kd = 2.0 * onp.pi * grid / dper[None, :]
    if kmax is not None:
        kd = kd[onp.sqrt(onp.sum(kd ** 2, axis = -1)) <= kmax]
    k = onp.zeros((len(kd), spec.d))
    k[:, ax] = kd
    return k


def sk_chain(pos, alive, kvecs):
    """(K,) instantaneous |sum_j exp(i k . r_j)|^2 for one chain.

    NOT divided by N: the caller accumulates this and <N> separately and
    divides once at the end, because <|rho_k|^2>/<N> and <|rho_k|^2/N> are
    different averages and the first is the one S(k) means.

    Dead slots are masked out of the sums rather than out of the positions, so
    a stale coordinate left in a freed slot contributes exactly nothing instead
    of contributing a spurious plane wave.
    """
    phase = pos @ kvecs.T                                   # (Nmax, K)
    w = alive[:, None]
    c = np.sum(np.where(w, np.cos(phase), 0.0), axis = 0)
    s = np.sum(np.where(w, np.sin(phase), 0.0), axis = 0)
    return c ** 2 + s ** 2


def summarise(result, kvecs):
    """The freezing diagnostic as a dict, with a chain-to-chain error bar.

    `s_max_over_n` is the number to read. In a fluid every mode carries O(1)
    and this falls off as 1/<N>; in a crystal one mode carries O(N) and it
    settles at a constant. The onset is a rise over the fluid baseline by
    something like an order of magnitude, and `ORDER_TRIP` is where this
    library calls it, deliberately low: the cost of excluding a state that was
    still fluid is one wasted run, and the cost of keeping one that was not is
    a poisoned label nobody downstream can detect.
    """
    sk = onp.asarray(result.sk, dtype = float)              # (C, K), summed
    if sk.size == 0:
        return dict(s_max = float("nan"), s_max_over_n = float("nan"),
                    s_max_err = float("nan"), k_at_max = None, n_kvectors = 0,
                    ordered = False)
    n_per_chain = onp.asarray(result.Ns, dtype = float).mean(axis = 1)  # (C,)
    s = sk / result.nsamples / onp.maximum(n_per_chain, 1e-300)[:, None]
    mean_s = s.mean(axis = 0)
    j = int(onp.argmax(mean_s))
    n_mean = float(n_per_chain.mean())
    # The error bar is the chain scatter AT the chosen wavevector, not the
    # scatter of each chain's own maximum. The latter is biased upwards by its
    # own selection: every chain picks its luckiest k, so the spread of those
    # maxima is not the uncertainty of the number being reported.
    return dict(
        s_max = float(mean_s[j]),
        s_max_err = float(s[:, j].std(ddof = 1) / onp.sqrt(len(s)))
        if len(s) > 1 else float("nan"),
        s_max_over_n = float(mean_s[j] / max(n_mean, 1e-300)),
        k_at_max = [float(x) for x in onp.asarray(kvecs)[j]],
        n_kvectors = int(sk.shape[1]),
        ordered = bool(mean_s[j] / max(n_mean, 1e-300) > ORDER_TRIP
                       and mean_s[j] > ORDER_SMAX_MIN))


# Fraction of the particle number a single mode has to carry before the state
# is called ordered. Measured on configurations built by hand in a campaign-
# sized cell: a jittered crystal sits at 0.99, an ideal gas of 512 at 0.013.
# 0.10 puts the alarm well inside the transition rather than after it.
ORDER_TRIP = 0.10

# ...and an absolute floor on S itself, because the ratio alone is not safe in
# a small system. The fluid baseline is not zero but about ln(K)/<N>, K being
# the number of wavevectors scanned, so a slit holding eighty particles sits
# near 0.05 on the ratio for no reason but the maximum being taken over sixty
# modes. S_max in absolute terms tells the two apart with room to spare: a
# fluid never exceeds three or four at ANY wavevector, whatever its size, and
# a crystal is at O(N). Both conditions must hold.
ORDER_SMAX_MIN = 8.0
