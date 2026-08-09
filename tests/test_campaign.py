"""The superball campaign's analysis layer, on synthetic ladders.

`scripts/superball_campaign.py` is where the measured numbers become the numbers
a functional is trained on, and until now none of it was tested: the sampler had
a hundred tests and the arithmetic that reads it had none. That is the wrong way
round, because a sampler defect shows up against Tonks or Carnahan-Starling and
an analysis defect shows up as a plausible table.

Everything here runs on a ladder BUILT from the exact equations of state rather
than measured, so the right answer is known to the last digit and any deviation
is the analysis. No sampling, no JAX, milliseconds.
"""
import os
import sys

import numpy as onp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from mcax import eos, shapes                                    # noqa: E402
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

    got = camp.pressure_at(bulk, 3, 2.0, rho_q)
    assert abs(got - exact) / exact < 0.01

    # ... and the chord it replaced is several times worse, which is the whole
    # reason for the change and so is asserted rather than described.
    node_rho, node_P = camp.pressure_curve(bulk, 3, 2.0)
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
    rho, P = camp.pressure_curve(bulk, d, 2.0)
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
    assert len(camp.usable_ladder(bulk, 3, 2.0)) == len(ETAS_3D) + 1

    stalled = dict(bulk[-1])
    stalled["mu"] = float(eos.mu_of_rho(3, 0.46 / eos.B[3]))
    stalled["rho_mean"] *= 1.004                  # crept, nowhere near arrived
    stalled["eta_mean"] = stalled["rho_mean"] * eos.B[3]
    stalled["drift_sigma"] = 0.1                  # ... and looks beautifully still

    kept = camp.usable_ladder(bulk + [stalled], 3, 2.0)
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
    assert b / (short["mu"] - bulk[-1]["mu"]) > camp.STALL_CHI   # floor passes it
    assert len(camp.usable_ladder(bulk + [short], 3, 2.0)) == len(ETAS_3D) + 1


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
    f = camp.mu_of_eta(bulk + [stalled], 3, 2.0)
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


def test_b2_is_the_exact_one_for_every_shape_in_the_grid():
    """The one closed-form equation-of-state coefficient available across the
    family, and the anchor the Gibbs-Duhem integral starts from below its first
    measured point. 2^(d-1) v, by the Minkowski argument in `mcax.shapes`."""
    for p in camp.P_GRID:
        for d in (2, 3):
            b = shapes.b2(shapes.Superball(p), d, 1.0)
            assert b == pytest.approx(2.0 ** (d - 1)
                                      * shapes.volume(shapes.Superball(p), d))
