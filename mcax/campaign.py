"""Reading a measured bulk ladder: the analysis layer of the superball campaign.

The superball campaign (`scripts/superball_campaign.py`) measures one bulk
mu-ladder per (d, p), a list of grand-canonical runs at increasing activity,
and everything downstream reads its activity or its pressure off that ladder.
No reference equation of state exists off p = 2, so the ladder IS the equation
of state, and the three questions asked of it are answered here:

    usable_ladder(bulk, d, p)        which rungs are equilibrium (mu, rho) pairs
    mu_of_eta(bulk, d, p)            a callable eta -> ln z, read off the ladder
    pressure_at(bulk, d, p, rho)     beta P at one density, by Gibbs-Duhem

plus `pressure_curve`, the integration `pressure_at` reads from. This lives in
the package rather than in the script because the consumer is another project:
`dimint-dft` trains on the campaign's data and reads the measured curves through
exactly these three functions, and a data consumer should not have to import a
fifteen hundred line campaign driver to get three accessors. The script
imports them back from here, so it and anything importing it see the same
names as before.

Everything here is NumPy on plain dicts. Nothing samples, nothing traces, and
the tests in `tests/test_campaign.py` run it on ladders built from the exact
equations of state, where the right answer is known to the last digit.

**The ladder row.** A rung is a dict written by the campaign's `sample`, and
the accessors read these keys of it (the data card in the consumer's
`data/superball/README.md` documents the full row; this is the part the
analysis rests on):

    d             fluid dimension, 2 or 3
    p             superball exponent as a float; `inf` for the cube
    eta_mean      measured packing fraction, n_mean v(p) / V
    rho_mean      measured number density, n_mean / V
    mu            ln z the rung was SET at: an exact input, not a measurement
    n_mean        <N> over chains and draws
    n_err         independent replica standard error of <N>
    drift_sigma   shift of <N> between the first and last third of the
                  sampling window, in units of n_err
    ordered       optional; True where the ordering monitor tripped and the
                  verdict survived confirmation (`eta_max_table` sets it)

`check_ladder` refuses a row missing any of the required keys by name, at the
top of `usable_ladder`, so a schema drift fails where it can be read rather
than as a KeyError deep inside a polynomial fit.

**Two things the accessors refuse to do, on purpose.** Neither extrapolates:
`mu_of_eta` raises above the top usable rung and below the bottom one, and
`pressure_at` raises off either end of the integrated curve, because a value
read off the end of a ladder is not a value at the density asked for and the
freezing guard's whole claim is that nothing is invented above the measured
range. And neither draws a chord: beta mu(eta) and beta P(rho) are convex, so
linear interpolation between rungs sits above both curves everywhere between
them, zero on the rungs and maximal mid-gap, which masquerades as
state-dependent physics. Both carry the smooth EXCESS part across by a local
quadratic and add the ideal part back analytically.
"""
from math import inf

import numpy as onp

from . import shapes
from .shapes import Superball

# Relative drift above which a ladder point is not used to set activities.
DRIFT_TOL = 0.02

# Smallest reduced compressibility Var(N)/<N> = S(0) = dln(rho)/d(beta mu) that
# a hard-particle fluid can have anywhere below freezing. Carnahan-Starling
# gives 0.068 at eta = 0.37 and does not fall below 0.025 until eta = 0.47, so
# this floor is thirty times clear of any physics and fires only on a chain that
# stopped moving in the wrong place. See `usable_ladder`.
STALL_CHI = 0.005

# How far a ladder point's measured mu_ex may sit ABOVE the value the three
# points below it predict, before the point is read as a chain that did not
# reach its density. One-sided on purpose: a stalled chain keeps the mu it was
# set at and reports a density short of it, which inflates mu_ex and nothing
# else. Measured across all twelve ladders, the equilibrated points scatter
# within +/-7% of the prediction and the two top points of every single ladder
# jump to between +14% and +44%. See `usable_ladder`.
LADDER_EXCESS = 0.10

# ...and an absolute floor on the same residual, because a purely relative
# band is a few-hundredths tolerance at the DILUTE end, where mu_ex is small,
# the three fit points sit in the other (larger) box, and the prediction is a
# noisy quadratic extrapolated to double its range. Measured on the 2026-08-07
# ladders: the three genuine stalls fired at residuals +1.07 to +1.78 against
# predictions of 7 to 8, while a fully healthy d3 p2.5 rung at eta = 0.08 was
# cut on a residual of +0.098 against 0.652, which then truncated the ladder
# to three dilute rungs and silently cost that shape its whole confined row.
# A stall cannot exist where insertions accept freely, so nothing real lives
# below this floor.
LADDER_EXCESS_ABS = 0.5

# The keys of a ladder row that the accessors read. See the module docstring.
LADDER_KEYS = ("d", "p", "eta_mean", "rho_mean", "mu", "n_mean", "n_err",
               "drift_sigma")


def pname(p):
    """The p that goes in a state tag. `inf` rather than a large number, so the
    benchmark shape is never confused with a member of the training grid."""
    return "inf" if p == inf else f"{p:g}"


def check_ladder(bulk):
    """Refuse a ladder whose rows do not carry the keys the analysis reads.

    Named after the row and the key, because the alternative is a bare KeyError
    from inside `usable_ladder` after the rows have been sorted and filtered,
    where the offending rung is no longer identifiable.
    """
    for i, r in enumerate(bulk):
        missing = [k for k in LADDER_KEYS if k not in r]
        if missing:
            tag = r.get("tag", f"row {i}") if isinstance(r, dict) else f"row {i}"
            raise KeyError(f"ladder {tag} lacks {missing}; a rung must carry "
                           f"{LADDER_KEYS}")


def usable_ladder(bulk, d, pexp):
    """The ladder points that are equilibrium (beta mu, rho) pairs, in density
    order. Two cuts, and the second makes the one the first cannot.

    **Relative drift** catches a chain still moving when the sampling window
    closed. It is the scale-free reading: the sigma drift tightens itself as
    statistics improve and would reject the best runs on the ladder.

    **The stall** catches a chain that has stopped moving in the WRONG PLACE,
    which the drift test is blind to by construction: a chain sitting against
    the equilibration wall has a flat <N> series and reports the SMALLEST drift
    on the ladder. Two consecutive points give the reduced compressibility

        S(0)  =  Var(N)/<N>  =  d ln(rho) / d(beta mu),

    which for a hard-particle fluid below freezing never falls under about
    0.025. On the campaign's own d = 3, p = 2 ladder the pair at
    eta 0.3696 -> 0.3710 gives 0.00086: two chains at the same density with
    chemical potentials four apart, which is not a fluid, it is one chain that
    could not fill and a second that could not either. That is exactly the pair
    the Gibbs-Duhem check against Carnahan-Starling reported at 26% and 97%
    while the drift cut passed both.

    The stalled point and everything above it goes, because a chain that could
    not reach eta = 0.40 did not reach 0.46 on the way past.

    S(0) alone only catches the pathological end of that. The sharper test is a
    FORWARD PREDICTION and it needs no reference equation of state, which is
    what makes it usable off p = 2: fit mu_ex(rho) quadratically on the three
    points below and see where the next one lands. A stalled chain keeps the mu
    it was set at while reporting a density short of it, so its mu_ex overshoots
    and nothing else in the run says so. Measured across all twelve of the
    campaign's ladders, the equilibrated points scatter within +/-7% of the
    prediction and the top two points of EVERY ladder jump to +14% ... +44%.
    Both cuts run, and the first to fire truncates.

    That the top two points of every ladder fail is not a coincidence and not
    freezing: the same d = 3, p = 2 state re-run with the burn multiplied by
    thirty-two climbs from eta 0.3692 to 0.3931 against an exact 0.4000, so it
    is the burn-in, and the campaign's `burn_for` is where it is fixed rather
    than here.
    """
    check_ladder(bulk)
    # `ordered` rungs are excluded outright: a crystal's (mu, rho) is a valid
    # measurement of the wrong phase, and before this cut the onset rung
    # itself could serve as the upper interpolation node for activities set
    # just below eta_max, and as an integration node in `pressure_curve`.
    sel = sorted((r for r in bulk if r["d"] == d and r["p"] == pexp
                  and not r.get("ordered")
                  and r["drift_sigma"] * r["n_err"] / max(r["n_mean"], 1e-12)
                  <= DRIFT_TOL),
                 key=lambda r: r["mu"])
    out = []
    for r in sel:
        if out:
            dmu = r["mu"] - out[-1]["mu"]
            chi = (onp.log(max(r["rho_mean"], 1e-300))
                   - onp.log(max(out[-1]["rho_mean"], 1e-300))) / max(dmu, 1e-12)
            if dmu > 1e-9 and chi < STALL_CHI:
                break
        if len(out) >= 3:
            rho = onp.array([q["rho_mean"] for q in out[-3:]])
            mux = onp.array([q["mu"] for q in out[-3:]]) - onp.log(rho)
            pred = onp.polyval(onp.polyfit(rho, mux, 2), r["rho_mean"])
            got = r["mu"] - onp.log(max(r["rho_mean"], 1e-300))
            if got - pred > max(LADDER_EXCESS * abs(pred), LADDER_EXCESS_ABS):
                break
        out.append(r)
    out.sort(key=lambda r: r["rho_mean"])
    keep, last = [], -onp.inf
    for r in out:                                  # strictly increasing in rho
        if r["rho_mean"] > last + 1e-12:
            keep.append(r)
            last = r["rho_mean"]
    return keep


def mu_of_eta(bulk, d, p):
    """A callable eta -> ln z, read off the measured ladder.

    **Not a chord through (eta, mu).** beta mu(eta) is convex, it rises with
    the pressure, so linear interpolation between rungs sits ABOVE the curve
    everywhere between them, and the ladder is coarsest exactly at the dense
    end. This is the same bias `pressure_at` was cured of, one function over,
    in the function that sets EVERY confined state's activity: measured on a
    synthetic ladder built from the exact equations of state, the chord set
    activities high by up to delta mu = +0.19 mid-gap, which is a reservoir
    density label wrong by +0.3% to +1.2%, zero on the rungs, maximal between
    them, so it masquerades as state-dependent physics. The cure is also the
    same one: split mu into the exact ideal part and a smooth excess, carry
    mu_ex across by the local quadratic, and add ln rho back analytically.

    Extrapolation past the measured range is still refused: nothing is
    invented above the ladder top.
    """
    sel = usable_ladder(bulk, d, p)
    if len(sel) < 3:
        raise ValueError(f"fewer than three usable ladder points for d={d} "
                         f"p={pname(p)} after the drift and stall cuts")
    e = onp.array([r["eta_mean"] for r in sel])
    rho = onp.array([r["rho_mean"] for r in sel])
    mux = onp.array([r["mu"] for r in sel]) - onp.log(rho)
    v = float(onp.mean(e / rho))            # particle volume, same on every row

    def f(eta):
        if eta > e[-1] or eta < e[0]:
            raise ValueError(f"eta {eta} outside the measured ladder "
                             f"[{e[0]:.4f}, {e[-1]:.4f}] for d={d} p={pname(p)}")
        rho_q = eta / v
        # A CENTRED window, four nodes where the ladder has them. The
        # off-centre three-node quadratic that `pressure_curve` uses is fine
        # under an integral, which smooths its segment error; read pointwise
        # at the low-density end it left a third of the chord bias in place.
        k = int(onp.searchsorted(rho, rho_q))
        lo = max(0, min(k - 2, len(rho) - 4))
        w = rho[lo:lo + 4]
        c = onp.polyfit(w, mux[lo:lo + len(w)], len(w) - 1)
        return float(onp.log(rho_q) + onp.polyval(c, rho_q))
    return f


def pressure_curve(bulk, d, pexp, dense=False):
    """(rho, beta P) along a measured ladder, by Gibbs-Duhem.

    beta P = integral rho d(beta mu) is the only route to a pressure here,
    because there is no reference equation of state off p = 2. Two details
    carry most of the accuracy:

    **Integrate the EXCESS, not the whole thing.** The obvious route,
    beta P = integral rho d(beta mu), integrates against a variable that is
    logarithmically singular as rho -> 0 and is changing fastest exactly where
    the ladder is coarsest. One integration by parts removes both problems:

        beta P  =  rho + rho mu_ex(rho) - integral_0^rho mu_ex drho'

    with mu_ex = beta mu - ln rho. The ideal-gas part is now exact and analytic,
    the remaining integrand VANISHES at the origin (so the anchor carries no
    error at all rather than an order-rho^2 one), and mu_ex is smooth and
    nearly linear at low density, where 2 B_2 rho is its exact slope. On a
    four-point ladder the difference between the two forms is 33% at the dense
    end against Carnahan-Starling; on the campaign's fourteen points it is
    below a per cent.

    The segment below the first measured point is done with the exact virial,
    integral_0^rho1 mu_ex drho = B_2 rho1^2, rather than by extrapolating the
    ladder into a region it does not cover.

    The ladder comes from `usable_ladder`, which cuts on relative drift and on
    the stall the drift test cannot see. Before that cut this integration read
    3.6%, 26% and 97% against Carnahan-Starling across the top three points of
    the d = 3, p = 2 ladder, and the comment here recorded those numbers as
    though the cut had removed them; it had not, because a stalled chain has
    the smallest drift on the ladder, not the largest.

    `dense = True` refines the quadrature onto a fine grid, carrying mu_ex
    across by a local quadratic. Nothing new is measured: it removes the
    trapezoid's discretisation, which is what `pressure_at` needs and what the
    node-by-node table in the campaign's `gibbs_duhem_against_reference` must
    not have.
    """
    sel = usable_ladder(bulk, d, pexp)
    if len(sel) < 2:
        raise ValueError(f"fewer than two usable ladder points for d={d} "
                         f"p={pname(pexp)}: there is no curve to integrate")
    rho = onp.array([r["rho_mean"] for r in sel])
    mu_ex = onp.array([r["mu"] for r in sel]) - onp.log(rho)
    if dense and len(rho) >= 3:
        rho, mu_ex = _mu_ex_dense(rho, mu_ex)
    b2 = shapes.b2(Superball(pexp), d, 1.0)
    f_ex = b2 * rho[0] ** 2 + onp.concatenate([[0.0], onp.cumsum(
        0.5 * (mu_ex[1:] + mu_ex[:-1]) * onp.diff(rho))])
    return rho, rho + rho * mu_ex - f_ex


def _mu_ex_dense(rho, mu_ex, n=4001):
    """(grid, mu_ex on it) by piecewise-QUADRATIC interpolation on three nodes.

    Linear would be defensible for mu_ex itself, which is smooth and nearly
    straight at low density. What it does not survive is what the caller does
    next: integrate it, then subtract the integral from something of the same
    size, where a chord's error has nothing to cancel against.
    """
    g = onp.linspace(rho[0], rho[-1], n)
    j = onp.clip(onp.searchsorted(rho, g) - 1, 0, len(rho) - 3)
    out = onp.empty_like(g)
    for k in range(len(rho) - 2):
        m = j == k
        if m.any():
            out[m] = onp.polyval(onp.polyfit(rho[k:k + 3], mu_ex[k:k + 3], 2),
                                 g[m])
    return g, out


def pressure_at(bulk, d, pexp, rho_q):
    """beta P at one density, read off the measured ladder.

    **Not `onp.interp` on the pressure**, which is what this was, and which put
    a bias into every A3 row. beta P(rho) is strongly convex, it rises like
    1/(1 - eta)^3, so a chord between two ladder points lies ABOVE the curve
    everywhere between them, and the ladder is coarsest exactly where the
    curvature is largest.

    Measured at p = 2, where the answer is known independently. The campaign's
    d = 3 ladder brackets the A3 slit state's reservoir density with eta = 0.12
    and eta = 0.18; the chord across them reads beta P = 0.5644 against
    Carnahan-Starling's 0.5454, high by 3.5%. In d = 2 it is 2.2% high against
    Henderson. That bias does not depend on the shape, so it entered all twelve
    A3 rows as a deviation the engine had not committed.

    Integrating on a dense grid instead, with mu_ex carried across by the local
    quadratic above, gives 0.5435 (-0.35%) and 0.7833 (-0.64%): the ladder's own
    statistics, and no longer a discretisation.
    """
    g, P = pressure_curve(bulk, d, pexp, dense=True)
    if not g[0] <= rho_q <= g[-1]:
        # `onp.interp` clamps to the endpoint silently, and `mu_of_eta`
        # refuses to extrapolate for exactly this reason; a pressure read off
        # the end of the ladder is not a pressure at rho_q.
        raise ValueError(f"rho {rho_q:.4f} outside the integrated ladder "
                         f"[{g[0]:.4f}, {g[-1]:.4f}] for d={d} p={pname(pexp)}")
    return float(onp.interp(rho_q, g, P))
