"""The superball campaign's analysis layer, on synthetic ladders.

`mcax.campaign` is where the measured numbers become the numbers a functional
is trained on, and for a while none of it was tested: the sampler had a hundred
tests and the arithmetic that reads it had none. That is the wrong way round,
because a sampler defect shows up against Tonks or Carnahan-Starling and an
analysis defect shows up as a plausible table.

Everything here runs on a ladder BUILT from the exact equations of state rather
than measured, so the right answer is known to the last digit and any deviation
is the analysis. No sampling, no JAX, milliseconds. The campaign driver
`scripts/superball_campaign.py` imports the accessors back from the package and
re-exports them, which is the path `dimint-dft` reads them through, so the
tests at the end are about the script rather than the analysis.
"""
import os
import sys

import numpy as onp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from mcax import campaign, eos, shapes                          # noqa: E402
import superball_campaign as camp                               # noqa: E402


def synthetic_ladder(d, etas, p=2.0):
    """Ladder rows carrying the EXACT (beta mu, rho) of a hard-sphere fluid.

    Same keys the sampler writes, so the analysis cannot tell the difference,
    and no drift or scatter anywhere: whatever these functions do to this is
    what they do to the physics.
    """
    rows = []
    for e in etas:
        rho = e / eos.B[d]
        rows.append(dict(d=d, p=p, eta_target=float(e), eta_mean=float(e),
                         rho_mean=float(rho), mu=float(eos.mu_of_rho(d, rho)),
                         n_mean=1000.0, n_err=0.1, drift_sigma=0.5, H=8.0))
    return rows


ETAS_3D = [0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.32]


def test_pressure_at_beats_a_chord_on_the_convex_pressure():
    """beta P(rho) rises like 1/(1-eta)^3, so a chord between ladder points sits
    ABOVE it, and the ladder is coarsest where the curvature is worst.

    This is the defect that put a bias into every A3 row: at the reservoir
    density of the campaign's A3 slit state the chord read 3.5% high against
    Carnahan-Starling at every exponent, and it was reported as a deviation of
    the engine from a sum rule the engine satisfies.
    """
    bulk = synthetic_ladder(3, ETAS_3D)
    rho_q = 0.15 / eos.B[3]                       # between the 0.12 and 0.18 rungs
    exact = float(eos.p_of_rho(3, rho_q))

    got = campaign.pressure_at(bulk, 3, 2.0, rho_q)
    assert abs(got - exact) / exact < 0.01

    # ... and the chord it replaced is several times worse, which is the whole
    # reason for the change and so is asserted rather than described.
    node_rho, node_P = campaign.pressure_curve(bulk, 3, 2.0)
    chord = float(onp.interp(rho_q, node_rho, node_P))
    assert abs(chord - exact) / exact > 3.0 * abs(got - exact) / exact


@pytest.mark.parametrize("d,etas", [
    (3, ETAS_3D),
    (2, [0.02, 0.05, 0.10, 0.18, 0.28, 0.38, 0.48]),
])
def test_the_gibbs_duhem_integration_reproduces_the_reference(d, etas):
    """On an exact ladder the integration must give back the equation of state
    it was integrated from, at every node. Anything else is the quadrature or
    the virial anchor below the first point."""
    bulk = synthetic_ladder(d, etas)
    rho, P = campaign.pressure_curve(bulk, d, 2.0)
    dev = onp.abs(P - eos.p_of_rho(d, rho)) / eos.p_of_rho(d, rho)
    assert dev.max() < 0.01


def test_a_stalled_ladder_point_is_dropped_and_a_dense_one_is_not():
    """The cut that the drift test cannot make.

    A chain that never reached its activity keeps the mu it was set at and
    reports a density short of it. Its <N> series is FLAT, so it has the
    smallest drift on the ladder and every drift threshold passes it, and the
    pressure integral then reads its inflated mu_ex directly — 26% and 97% off
    Carnahan-Starling on the campaign's own d = 3 ladder.

    Here the stall is manufactured: a genuine eta = 0.40 rung, and a second copy
    of it carrying the chemical potential of eta = 0.46. Both are dense, so a
    cut on density alone would have to lose the real one too.
    """
    bulk = synthetic_ladder(3, ETAS_3D + [0.40])
    assert len(campaign.usable_ladder(bulk, 3, 2.0)) == len(ETAS_3D) + 1

    stalled = dict(bulk[-1])
    stalled["mu"] = float(eos.mu_of_rho(3, 0.46 / eos.B[3]))
    stalled["rho_mean"] *= 1.004                  # crept, nowhere near arrived
    stalled["eta_mean"] = stalled["rho_mean"] * eos.B[3]
    stalled["drift_sigma"] = 0.1                  # ... and looks beautifully still

    kept = campaign.usable_ladder(bulk + [stalled], 3, 2.0)
    assert len(kept) == len(ETAS_3D) + 1
    assert max(r["mu"] for r in kept) == pytest.approx(bulk[-1]["mu"])


def test_the_forward_prediction_catches_a_stall_the_compressibility_floor_misses():
    """The two cuts are not the same cut, and the campaign needs both.

    The compressibility floor only fires when two points sit at essentially one
    density with chemical potentials far apart. A chain that fell short by a
    tenth rather than by everything still moves, so it clears the floor and is
    caught only by comparing it with the three points below.
    """
    bulk = synthetic_ladder(3, ETAS_3D + [0.40])
    short = dict(bulk[-1])
    short["mu"] = float(eos.mu_of_rho(3, 0.46 / eos.B[3]))
    short["rho_mean"] = 0.41 / eos.B[3]           # reached 0.41, was set for 0.46
    short["eta_mean"] = 0.41
    short["drift_sigma"] = 0.1
    a = short["mu"] - onp.log(short["rho_mean"])
    b = onp.log(short["rho_mean"]) - onp.log(bulk[-1]["rho_mean"])
    assert b / (short["mu"] - bulk[-1]["mu"]) > campaign.STALL_CHI   # floor passes it
    assert len(campaign.usable_ladder(bulk + [short], 3, 2.0)) == len(ETAS_3D) + 1


def test_mu_of_eta_refuses_to_invent_activities_above_the_usable_ladder():
    """The freezing guard's whole claim is that nothing is extrapolated. Once
    the stall cut has shortened a ladder, the refusal has to move down with
    it — otherwise a confined state reads its activity off a rung that was
    dropped for not being a state point."""
    bulk = synthetic_ladder(3, ETAS_3D + [0.40])
    stalled = dict(bulk[-1])
    stalled["mu"] = float(eos.mu_of_rho(3, 0.46 / eos.B[3]))
    stalled["rho_mean"] *= 1.004
    stalled["eta_mean"] = stalled["rho_mean"] * eos.B[3]
    f = campaign.mu_of_eta(bulk + [stalled], 3, 2.0)
    assert f(0.30) == pytest.approx(eos.mu_of_rho(3, 0.30 / eos.B[3]), rel=0.02)
    with pytest.raises(ValueError):
        f(0.42)


def test_burn_in_grows_with_the_system_rather_than_shrinking_per_particle():
    """A square-root burn hands the LARGEST states the FEWEST moves per particle,
    which is backwards for a transient that is per-particle. That is how a 587
    particle slit came to get thirteen displacement attempts each, kept the
    lattice it was seated on, and reported it as freezing."""
    camp.NBURN = 8000
    small, big = camp.burn_for(50.0), camp.burn_for(600.0)
    assert big / 600.0 >= small / 600.0
    assert camp.burn_for(600.0) / 600.0 == pytest.approx(camp.BURN_SWEEPS)
    assert camp.burn_for(1.0) == camp.NBURN         # floor for tiny cavities


def test_the_stall_cut_has_an_absolute_floor_at_the_dilute_end():
    """A purely relative excess band is a few hundredths of beta mu at the
    dilute end, where mu_ex is small and the prediction is an extrapolated
    quadratic through the other box's rungs. Measured on real ladders: true
    stalls fire at residuals +1.1 to +1.8, and a healthy d3 p2.5 rung was cut
    at +0.098 — costing that shape its entire confined row. The floor must
    swallow the latter and not the former."""
    # the measured false positive, reconstructed: a dilute rung whose mu_ex
    # sits 0.098 above a small prediction. The old relative-only band cut it
    # (0.098 > 0.10 x ~0.7) and the ladder died at eta = 0.04; the floor
    # keeps it. Perturbing the TOP rung isolates the floor: a wobbled rung
    # kept mid-ladder corrupts the fit windows above it, and rungs cut off a
    # corrupted window are the guard working, not the bug.
    rows = synthetic_ladder(3, [0.01, 0.02, 0.04, 0.08])
    rows[-1]["mu"] += 0.098
    kept = [r["eta_mean"] for r in campaign.usable_ladder(rows, 3, 2.0)]
    assert max(kept) == pytest.approx(0.08), \
        f"dilute wobble cut the ladder at {max(kept)}"
    # a genuine stall signature: residual of order one, cut as before
    rows = synthetic_ladder(3, ETAS_3D)
    rows[-1]["mu"] += 1.5
    kept = [r["eta_mean"] for r in campaign.usable_ladder(rows, 3, 2.0)]
    assert max(kept) == pytest.approx(0.25)


def test_mu_of_eta_does_not_sit_on_a_chord():
    """beta mu(eta) is convex, so a straight line between ladder rungs lies
    above the curve everywhere between them — the same bias `pressure_at` was
    cured of, in the function that sets EVERY confined state's activity. On
    the production rung spacing the chord overshoots by delta mu up to +0.19
    at the dense end, a reservoir label wrong by up to +1.2%, zero on rungs
    and maximal mid-gap. The quadratic-on-mu_ex route must beat it by an
    order of magnitude at every off-rung target the campaign uses."""
    rows = synthetic_ladder(3, ETAS_3D)
    f = campaign.mu_of_eta(rows, 3, 2.0)
    for eta in (0.15, 0.20, 0.30):                    # all between rungs
        exact = float(eos.mu_of_rho(3, eta / eos.B[3]))
        e = onp.array([r["eta_mean"] for r in rows])
        m = onp.array([r["mu"] for r in rows])
        chord = float(onp.interp(eta, e, m))
        assert abs(f(eta) - exact) < 0.1 * abs(chord - exact), \
            f"eta={eta}: quadratic {f(eta) - exact:+.4f} vs chord " \
            f"{chord - exact:+.4f}"
    # and on a rung both are exact, so nothing regressed there
    assert f(0.25) == pytest.approx(eos.mu_of_rho(3, 0.25 / eos.B[3]),
                                    abs=1e-6)


def test_the_prefill_lattice_saturates_and_the_dense_rungs_start_short():
    """A cubic lattice of pitch 1.05 in a periodic box of edge 8 holds
    floor(8/1.05)^3 = 343 sites and not one more, which is eta = 0.351 for
    spheres: the eta = 0.40 and 0.46 ladder rungs can NEVER be seated at 98%
    of target. They start 12% and 24% short and must close the gap one
    accepted insertion at a time, which is the initial condition behind the
    'equilibration wall' at the top of every d = 3 ladder. `lattice_fill`
    truncates silently by design (variance reduction must not crash a run),
    so the campaign records the seated count next to the requested one, and
    this test is what keeps that mechanism from being forgotten."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from mcax import make_spec, lattice_fill, geometry
    spec = make_spec(3, H=8.0, Lperp=8.0, z_act=1.0, geom="bulk",
                     shape=shapes.Superball(2.0))
    v = shapes.volume(shapes.Superball(2.0), 3)
    n_t = 0.46 / v * geometry.volume(spec)
    asked = int(0.98 * n_t)
    seated = len(lattice_fill(spec, asked))
    assert seated == 343                     # floor(8 / 1.05)^3, the hard cap
    assert seated < asked                    # the 0.46 rung starts 24% short
    # and a rung the lattice CAN seat is seated in full
    n_low = int(0.98 * 0.32 / v * geometry.volume(spec))
    assert len(lattice_fill(spec, n_low)) == n_low


def test_b2_is_the_exact_one_for_every_shape_in_the_grid():
    """The one closed-form equation-of-state coefficient available across the
    family, and the anchor the Gibbs-Duhem integral starts from below its first
    measured point. 2^(d-1) v, by the Minkowski argument in `mcax.shapes`."""
    for p in camp.P_GRID:
        for d in (2, 3):
            b = shapes.b2(shapes.Superball(p), d, 1.0)
            assert b == pytest.approx(2.0 ** (d - 1)
                                      * shapes.volume(shapes.Superball(p), d))


# --------------------------------------------------------------------------- #
#  The script, and the path dimint reads the accessors through                 #
# --------------------------------------------------------------------------- #

def test_the_script_re_exports_the_package_accessors():
    """`dimint-dft` reaches the accessors through the campaign script, so the
    names it imports there have to be the package's own objects and not a
    second copy that could drift."""
    for name in ("pressure_at", "mu_of_eta", "usable_ladder", "pressure_curve",
                 "pname", "DRIFT_TOL"):
        assert getattr(camp, name) is getattr(campaign, name), name


def test_a_ladder_row_missing_a_key_is_refused_by_name():
    """A schema drift in the rows has to fail where it can be read, naming the
    rung and the key, and not as a KeyError out of a polynomial fit after the
    rows have been sorted and filtered."""
    rows = synthetic_ladder(3, ETAS_3D)
    rows[3]["tag"] = "d3_bulk_p2_eta0.08"
    del rows[3]["drift_sigma"]
    with pytest.raises(KeyError, match="d3_bulk_p2_eta0.08.*drift_sigma"):
        campaign.usable_ladder(rows, 3, 2.0)
    # and every accessor sits on the same check
    with pytest.raises(KeyError):
        campaign.pressure_at(rows, 3, 2.0, 0.2)
    with pytest.raises(KeyError):
        campaign.mu_of_eta(rows, 3, 2.0)


def test_pressure_at_refuses_to_read_off_the_end_of_the_ladder():
    """`onp.interp` clamps silently, so the refusal is explicit: a pressure
    read off the end of the ladder is not a pressure at the density asked."""
    bulk = synthetic_ladder(3, ETAS_3D)
    top = max(r["rho_mean"] for r in bulk)
    with pytest.raises(ValueError):
        campaign.pressure_at(bulk, 3, 2.0, 1.01 * top)
    assert campaign.pressure_at(bulk, 3, 2.0, 0.99 * top) > 0.0


def test_pname_spells_the_cube_as_inf():
    """State tags carry the exponent, and the benchmark shape must never be
    confused with a member of the training grid."""
    assert campaign.pname(float("inf")) == "inf"
    assert campaign.pname(2.0) == "2"
    assert campaign.pname(2.5) == "2.5"
