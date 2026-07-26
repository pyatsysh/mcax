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

from mcax import make_spec, burn_and_sample, summary, eos

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
