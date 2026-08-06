"""Full statistical validation against the reference equations of state.

All `slow`, and deselected by default (see addopts in pyproject) so that a bare
pytest stays a usable pre-commit gate. Run with

    pytest -m slow                    # everything
    pytest -m slow -k "rods"          # just the exact d = 1 cases

What each case is worth: the d = 1 cases are the razor, since Tonks/Percus is
exact, so a failure there is a bug and nothing else. The d = 2 and d = 3 cases
test against Henderson and Carnahan-Starling, which carry their own ~0.1%
error, so their tolerances are looser and a marginal failure is ambiguous
between our bug and the reference equation's own accuracy.

Tolerances are deliberately several sigma wide. A gate that false-fails on a
coin flip trains everyone to ignore it: the contact-theorem check in particular
amplifies bin noise through a quadratic extrapolation to sigma(contact) ~ 0.03,
so a nominally strict 5% gate on it was a 43% chance of a spurious failure.

Every case asserts split R-hat as well as the density. Recorded 2026-07-26,
because it caught something a density check alone cannot: at eta = 0.3 in 3D
the insertion acceptance is about 1%, so chains started EMPTY spend the whole
burn-in filling and never meet (R-hat 1.23 on the first GPU run, while the mean
density still looked right to 0.1%). The remedy is the lattice prefill, which
is exactly what it exists for. A mean can be right for the wrong reason; R-hat
is what notices.
"""
import numpy as onp
import pytest

from mcax import (make_spec, burn_and_sample, summary, eos, geometry,
                  observables, potentials, fields)

pytestmark = pytest.mark.slow


# (d, eta, chains, n_burn, n_run, rel tolerance, label)
#
# The 3-D case costs 10x the others and has earned it. N there decorrelates over
# roughly 70 draws, so 16 chains of 800 draws carried about seven independent
# draws each and reported R-hat 1.13. These settings were measured up until the
# diagnostic actually cleared: 32 chains, 2e6 steps, R-hat 1.014.
BULK_CASES = [
    (1, 0.5, 16, 100_000, 200_000, 0.02, "rods-exact-tonks"),
    (2, 0.4, 16, 100_000, 200_000, 0.02, "discs-henderson"),
    (3, 0.3, 32, 500_000, 2_000_000, 0.02, "spheres-carnahan-starling"),
]


@pytest.mark.parametrize("d,eta,C,n_burn,n_run,tol,label", BULK_CASES,
                         ids = [c[-1] for c in BULK_CASES])
def test_bulk_equation_of_state(d, eta, C, n_burn, n_run, tol, label):
    """Measured <rho> in a periodic box must match the EOS inversion rho(z)."""
    rho_target = eta / eos.B[d]
    spec = make_spec(d, H = 8.0, z_act = float(eos.z_of_rho(d, rho_target)),
                     Lperp = 8.0, slit = False)
    # Seat ~90% of the target N on a lattice and melt it in the burn. Filling
    # grand-canonically from empty at a 1% insertion acceptance takes longer
    # than the burn-in itself.
    V = spec.H * spec.Lperp ** (d - 1)
    n0 = int(0.9 * rho_target * V)
    res = burn_and_sample(spec, C = C, seed = d, n_burn = n_burn,
                          n_run = n_run, thin = 500, nbins = 80, n0 = n0)

    assert res.capacity_warning is None, res.capacity_warning
    rho = res.n_mean / V
    rel = abs(rho - rho_target) / rho_target

    s = summary(res.Ns, "N")
    assert s["split_rhat"] < 1.05, f"chains not mixed: R-hat {s['split_rhat']:.3f}"
    assert rel < tol, (f"{label}: rho {rho:.4f} vs EOS {rho_target:.4f} "
                       f"(rel {rel:.4f}), acc {onp.round(res.acc, 3)}")


def test_wall_contact_theorem_rods():
    """rho(0+) = beta P at a hard wall, against exact Tonks pressure.

    The sum rule holds literally here because mcax's slit excludes CENTRES at
    0 and H, the same convention as the DFT reference data.

    Both walls are equivalent, so the profile is mirror-averaged to halve the
    counting variance before a quadratic extrapolation of the first three bins
    down to z = 0+.
    """
    rho_b = 0.5
    P = eos.p_of_rho(1, rho_b)
    spec = make_spec(1, H = 8.0, z_act = float(eos.z_of_rho(1, rho_b)),
                     slit = True)
    res = burn_and_sample(spec, C = 48, seed = 7, n_burn = 100_000,
                          n_run = 600_000, thin = 500, nbins = 160)

    assert res.capacity_warning is None, res.capacity_warning
    s = summary(res.Ns, "N")
    assert s["split_rhat"] < 1.05, f"chains not mixed: R-hat {s['split_rhat']:.3f}"

    rho_sym = 0.5 * (res.rho + res.rho[::-1])
    contact = onp.polyval(onp.polyfit(res.z[:3], rho_sym[:3], 2), 0.0)
    rel = abs(contact - P) / P
    assert rel < 0.12, f"contact {contact:.4f} vs beta P {P:.4f} (rel {rel:.4f})"


def test_slit_interior_recovers_the_bulk_density_rods():
    """Far from both walls a wide slit must return to the bulk density.

    Independent of the contact check and much less noisy, because it averages
    the middle of the profile instead of extrapolating its edge.
    """
    rho_b = 0.4
    spec = make_spec(1, H = 20.0, z_act = float(eos.z_of_rho(1, rho_b)),
                     slit = True)
    res = burn_and_sample(spec, C = 32, seed = 5, n_burn = 100_000,
                          n_run = 400_000, thin = 500, nbins = 100)

    assert res.capacity_warning is None, res.capacity_warning
    mid = res.rho[len(res.rho) // 2 - 10:len(res.rho) // 2 + 10].mean()
    assert mid == pytest.approx(rho_b, rel = 0.05), f"interior {mid:.4f}"


# ---- attraction: the razor survives, because d = 1 is still exact --------- #

SW_CASES = [
    (0.5, 1.5, 0.30, "sw-shallow"),
    (1.0, 1.5, 0.30, "sw-standard"),
    (1.0, 1.5, 0.45, "sw-dense"),
    (1.5, 1.8, 0.25, "sw-deep-wide"),
]


@pytest.mark.parametrize("eps,lam,rho_target,label", SW_CASES,
                         ids = [c[-1] for c in SW_CASES])
def test_square_well_rods_against_takahashi(eps, lam, rho_target, label):
    """Measured <rho> against the EXACT 1-D square-well equation of state.

    This is the attractive counterpart of `test_bulk_equation_of_state` at
    d = 1, and it carries the same weight: Takahashi's nearest-neighbour
    solution is exact for lam <= 2, so a failure here is a bug in the energy
    bookkeeping and nothing else. No literature value, no error bar but ours.
    """
    z_act = eos.sw1d_z_of_rho(eps, lam, rho_target)
    spec = make_spec(1, H = 60.0, z_act = z_act, slit = False,
                     pair = potentials.SquareWell(eps, lam))
    n0 = int(0.9 * rho_target * spec.H)
    res = burn_and_sample(spec, C = 32, seed = 11, n_burn = 200_000,
                          n_run = 600_000, thin = 500, nbins = 40, n0 = n0)

    assert res.capacity_warning is None, res.capacity_warning
    s = summary(res.Ns, "N")
    assert s["split_rhat"] < 1.05, f"chains not mixed: R-hat {s['split_rhat']:.3f}"
    rho = res.n_mean / spec.H
    rel = abs(rho - rho_target) / rho_target
    assert rel < 0.02, (f"{label}: rho {rho:.4f} vs exact {rho_target:.4f} "
                        f"(rel {rel:.4f}), acc {onp.round(res.acc, 3)}")


def test_square_well_contact_theorem_rods():
    """rho(0+) = beta P against a hard wall, with attraction switched on.

    The sum rule constrains the WALL, not the fluid, so it survives an
    interparticle attraction untouched. That makes it an independent check on
    the attractive engine: the density comes from the profile's edge and the
    pressure from the exact solution, and nothing connects them but the physics
    being right.
    """
    eps, lam, rho_b = 1.0, 1.5, 0.35
    z_act = eos.sw1d_z_of_rho(eps, lam, rho_b)
    P = eos.sw1d_p_of_rho(eps, lam, rho_b)
    spec = make_spec(1, H = 10.0, z_act = z_act, slit = True,
                     pair = potentials.SquareWell(eps, lam))
    res = burn_and_sample(spec, C = 48, seed = 13, n_burn = 150_000,
                          n_run = 600_000, thin = 500, nbins = 200)

    assert res.capacity_warning is None, res.capacity_warning
    rho_sym = 0.5 * (res.rho + res.rho[::-1])
    contact = onp.polyval(onp.polyfit(res.z[:3], rho_sym[:3], 2), 0.0)
    rel = abs(contact - P) / P
    assert rel < 0.12, f"contact {contact:.4f} vs beta P {P:.4f} (rel {rel:.4f})"


# ---- the fluctuation identities, at precision ---------------------------- #

@pytest.mark.parametrize("d,eta", [(1, 0.4), (2, 0.35), (3, 0.25)])
def test_compressibility_against_the_equation_of_state(d, eta):
    """Var(N)/<N> = [d(beta P)/d(rho)]^{-1}, exact for rods.

    A variance is a much noisier estimator than a mean, so this needs the long
    runs; that is the whole reason it lives out here rather than in the fast
    tier, where the same check runs at 20%.
    """
    rho = eta / eos.B[d]
    L = {1: 60.0, 2: 14.0, 3: 8.0}[d]
    spec = make_spec(d, H = L, Lperp = L, z_act = float(eos.z_of_rho(d, rho)),
                     slit = False)
    V = geometry.volume(spec)
    res = burn_and_sample(spec, C = 32, seed = 17, n_burn = 200_000,
                          n_run = 800_000, thin = 500, nbins = 20,
                          n0 = int(0.9 * rho * V))

    assert res.capacity_warning is None, res.capacity_warning
    kappa, err = observables.compressibility(res.Ns)
    exact = float(eos.compressibility(d, rho))
    assert kappa == pytest.approx(exact, rel = 0.06), (
        f"d = {d}: measured {kappa:.4f} +/- {err:.4f} against {exact:.4f}")


def test_structure_factor_two_ways_rods():
    """S(0) from the pair correlations against S(0) from the number variance.

    In one dimension the shells tile the minimum-image cell, so with
    rmax = L/2 these are the same number algebraically and the tolerance can be
    tight. It is the check that caught the canonical g(r) normalisation.
    """
    rho = 0.4
    spec = make_spec(1, H = 40.0, z_act = float(eos.z_of_rho(1, rho)),
                     slit = False)
    res = burn_and_sample(spec, C = 32, seed = 19, n_burn = 100_000,
                          n_run = 400_000, thin = 500, nbins = 20,
                          nbins_g = 200, n0 = int(0.9 * rho * spec.H))

    s0_pairs = observables.structure_factor_zero(spec, res, rho)
    s0_number, _ = observables.compressibility(res.Ns)
    exact = float(eos.compressibility(1, rho))
    assert s0_number == pytest.approx(exact, rel = 0.06)
    assert s0_pairs == pytest.approx(s0_number, rel = 0.03), (
        f"S(0) from g(r) {s0_pairs:.4f} against Var(N)/<N> {s0_number:.4f}")


def test_local_response_integrates_to_the_susceptibility_rods():
    """int d rho(z)/d(beta mu) dV = Var(N), and Var(N) matches the exact
    compressibility. The first half is algebra and holds to roundoff; the
    second is physics and needs the statistics."""
    rho = 0.4
    spec = make_spec(1, H = 20.0, z_act = float(eos.z_of_rho(1, rho)),
                     slit = True)
    res = burn_and_sample(spec, C = 32, seed = 23, n_burn = 100_000,
                          n_run = 400_000, thin = 500, nbins = 40)
    z, chi, err = observables.response_profile(spec, res)
    cell = geometry.bin_volume(spec, len(z))
    assert float((chi * cell).sum()) == pytest.approx(
        float(res.Ns.var(axis = 1, ddof = 0).mean()), rel = 1e-9)


# ---- the curved pores ---------------------------------------------------- #

def test_disc_pore_interior_recovers_the_bulk_density():
    """Deep inside a wide circular pore the fluid must forget the wall.

    The radial bin volumes are what make this work: get them wrong and the
    interior tilts, which is exactly the failure a flat profile would hide.
    """
    eta, R = 0.3, 8.0
    rho_b = eta / eos.B[2]
    spec = make_spec(2, geom = "sphere", H = R,
                     z_act = float(eos.z_of_rho(2, rho_b)))
    res = burn_and_sample(spec, C = 32, seed = 29, n_burn = 200_000,
                          n_run = 600_000, thin = 500, nbins = 40,
                          n0 = int(0.9 * rho_b * geometry.volume(spec)))

    assert res.capacity_warning is None, res.capacity_warning
    # bins 2 to 20 of 40 span r/R in [0.05, 0.5], three diameters clear of the
    # wall and past the innermost shells where the volume is tiny and noisy.
    mid = res.rho[2:20].mean()
    assert mid == pytest.approx(rho_b, rel = 0.06), (
        f"interior {mid:.4f} against bulk {rho_b:.4f}")


def test_barometric_law_at_precision():
    """rho(z) = z exp(-g z) for an ideal gas under gravity, bin-averaged.

    Exact, so this is only limited by statistics and by the bin average, and it
    is the sharpest test the external-field path has.
    """
    g, H, nbins = 0.5, 10.0, 20
    spec = make_spec(1, H = H, z_act = 1.0, sigma = 1e-6, slit = True,
                     Nmax = 128, field = fields.Gravity(g))
    res = burn_and_sample(spec, C = 48, seed = 31, n_burn = 50_000,
                          n_run = 400_000, thin = 200, nbins = nbins)

    edges = onp.linspace(0.0, H, nbins + 1)
    want = spec.z_act * (onp.exp(-g * edges[:-1]) - onp.exp(-g * edges[1:])) \
        / (g * (edges[1] - edges[0]))
    rel = onp.abs(res.rho - want) / want
    assert rel.max() < 0.05, f"worst bin {rel.max():.4f}\n{res.rho}\n{want}"
