"""Fast engine tests: structural invariants, geometry, and exact limits.

These run on CPU in seconds. The heavy statistical validation against the
equations of state lives in ``test_validation.py`` behind the ``slow`` marker.

The most valuable test here is ``test_no_overlaps_survive_a_run``: for a hard
particle system, "no two live centres closer than sigma" is the whole of the
physics, and it is checked with an independent NumPy implementation of the
boundary conventions rather than by reusing the kernel's own distance code.
"""
import numpy as onp
import pytest

import jax.numpy as np

from mcax import (make_spec, init_state, run, burn_and_sample, lattice_fill,
                  density_profile, eos)
from conftest import alive_positions, min_pair_separation


DIMS = [1, 2, 3]

# The precision guard has to be tested in a fresh interpreter. `jax_enable_x64`
# is read once, at first array creation, and conftest sets it for the whole
# session, so there is no way to unset it in-process without poisoning every
# jit cache in the suite. A subprocess is also the honest test: it reproduces
# exactly what a user gets for forgetting the flag in their own script.
_NO_X64 = """
import sys
sys.path.insert(0, %r)
import mcax
try:
    mcax.require_float64()
except RuntimeError as e:
    print("RAISED")
    assert "jax_enable_x64" in str(e), "the message must name the flag"
    sys.exit(0)
sys.exit("guard did not fire in single precision")
"""


def _short(spec, C=4, seed=0, n_burn=2_000, n_run=4_000, thin=200, nbins=20,
           n0=0):
    return burn_and_sample(spec, C=C, seed=seed, n_burn=n_burn, n_run=n_run,
                           thin=thin, nbins=nbins, n0=n0)


# ---- the hard-core invariant ---------------------------------------------- #

@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("slit", [True, False])
def test_no_overlaps_survive_a_run(d, slit):
    """No two live centres closer than sigma, in any chain, after a real run."""
    spec = make_spec(d=d, H=6.0, z_act=3.0, Lperp=6.0, slit=slit)
    res = _short(spec, C=4)
    for c in range(4):
        pts = alive_positions(res, c)
        sep = min_pair_separation(spec, pts)
        assert sep >= spec.sigma - 1e-9, (
            f"overlap in chain {c}: min separation {sep:.6f} < sigma")


@pytest.mark.parametrize("d", DIMS)
def test_no_overlaps_at_high_activity(d):
    """Dense states are where a bookkeeping slip actually shows up."""
    rho = 0.35 / eos.B[d]
    spec = make_spec(d=d, H=5.0, z_act=float(eos.z_of_rho(d, rho)),
                     Lperp=5.0, slit=False)
    res = _short(spec, C=4, n_burn=4_000, n_run=4_000)
    pts = alive_positions(res, 0)
    assert min_pair_separation(spec, pts) >= spec.sigma - 1e-9


# ---- geometry ------------------------------------------------------------- #

@pytest.mark.parametrize("d", DIMS)
def test_slit_confines_centres_to_the_walls(d):
    """Hard walls act on CENTRES at 0 and H, the DFT reference convention."""
    spec = make_spec(d=d, H=4.0, z_act=5.0, Lperp=5.0, slit=True)
    res = _short(spec)
    for c in range(4):
        pts = alive_positions(res, c)
        if len(pts) == 0:
            continue
        zc = pts[:, -1]
        assert zc.min() >= 0.0 - 1e-12
        assert zc.max() <= spec.H + 1e-12


@pytest.mark.parametrize("d", [2, 3])
def test_transverse_axes_stay_inside_the_periodic_box(d):
    spec = make_spec(d=d, H=4.0, z_act=5.0, Lperp=5.0, slit=True)
    res = _short(spec)
    pts = alive_positions(res, 0)
    assert onp.all(pts[:, :-1] >= 0.0 - 1e-12)
    assert onp.all(pts[:, :-1] <= spec.Lperp + 1e-12)


@pytest.mark.parametrize("d", DIMS)
def test_bulk_profile_is_uniform(d):
    """In a periodic box nothing breaks translational symmetry along z."""
    spec = make_spec(d=d, H=8.0, z_act=2.0, Lperp=6.0, slit=False)
    res = _short(spec, C=8, n_run=20_000, thin=200, nbins=8)
    rel_spread = res.rho.std() / res.rho.mean()
    assert rel_spread < 0.10, f"bulk profile not flat: spread {rel_spread:.3f}"


def test_slit_profile_is_not_uniform():
    """Walls must actually structure the fluid. The contrast with the bulk
    test above is what shows the wall geometry is wired in at all."""
    rho = 0.5
    spec = make_spec(d=1, H=8.0, z_act=float(eos.z_of_rho(1, rho)), slit=True)
    res = _short(spec, C=16, n_run=40_000, thin=200, nbins=40)
    assert res.rho.std() / res.rho.mean() > 0.05


def test_slit_profile_is_symmetric_between_the_two_walls():
    """Both walls are identical, so rho(z) must equal rho(H - z)."""
    spec = make_spec(d=1, H=8.0, z_act=float(eos.z_of_rho(1, 0.4)), slit=True)
    res = _short(spec, C=32, n_run=60_000, thin=200, nbins=32)
    asym = onp.abs(res.rho - res.rho[::-1]).max() / res.rho.mean()
    assert asym < 0.25, f"wall asymmetry {asym:.3f} too large"


# ---- exact limits --------------------------------------------------------- #

@pytest.mark.parametrize("d", DIMS)
def test_ideal_gas_limit_recovers_rho_equals_z(d):
    """With a vanishing core the fluid is ideal and rho = z exactly.

    A strong test of the muVT acceptances alone: it removes the overlap test
    from the picture entirely, so any factor of V, N or N+1 out of place in the
    insertion/deletion ratios fails here with nothing else to blame.
    """
    z = 0.7
    # Cost per step is O(Nmax) whatever the occupancy, so size the capacity to
    # the dimension rather than using one oversized number for all three.
    L = {1: 12.0, 2: 5.0, 3: 3.5}[d]
    V = L ** d
    spec = make_spec(d=d, H=L, z_act=z, Lperp=L, sigma=1e-6, slit=False,
                     Nmax=int(3 * z * V) + 20)
    res = _short(spec, C=16, n_burn=6_000, n_run=24_000, thin=100)
    rho = res.n_mean / V
    assert rho == pytest.approx(z, rel=0.05), f"ideal gas: rho {rho} vs z {z}"
    assert res.capacity_warning is None


def test_zero_activity_empties_the_box():
    """z -> 0 means mu -> -inf: deletions always win."""
    spec = make_spec(d=1, H=10.0, z_act=1e-8, slit=False)
    res = _short(spec, C=4, n_burn=5_000, n_run=5_000)
    assert res.n_mean < 0.05
    assert res.n_hi <= 2


def test_low_density_tonks_activity_inversion():
    """d = 1 at low density against the exact Tonks inversion."""
    rho_target = 0.2
    spec = make_spec(d=1, H=20.0, z_act=float(eos.z_of_rho(1, rho_target)),
                     slit=False, Nmax=64)
    res = _short(spec, C=16, n_burn=20_000, n_run=60_000, thin=200)
    rho = res.n_mean / spec.H
    assert rho == pytest.approx(rho_target, rel=0.06)


# ---- capacity watchdog ---------------------------------------------------- #

def test_capacity_warning_silent_when_there_is_headroom():
    spec = make_spec(d=1, H=10.0, z_act=1.0, slit=False)
    res = _short(spec, C=4)
    assert res.saturation < 0.95
    assert res.capacity_warning is None


def test_capacity_warning_fires_when_the_box_fills():
    """Deliberately starve the chain of slots.

    Insertion into a full chain is rejected by ``has_slot``, which is NOT a
    Metropolis rejection: it truncates the ensemble at N = Nmax and breaks
    detailed balance. Nothing else in the output reveals it, so the watchdog
    must fire.
    """
    spec = make_spec(d=1, H=40.0, z_act=50.0, slit=False, Nmax=6)
    res = _short(spec, C=4, n_burn=2_000, n_run=2_000)
    assert res.n_hi == spec.Nmax
    assert res.saturation >= 1.0
    assert res.capacity_warning is not None
    assert "REACHED" in res.capacity_warning


def test_capacity_watchdog_records_the_burn_in_peak():
    """nhi is deliberately not reset between burn and sample: capacity
    pressure during burn-in invalidates the run just as thoroughly."""
    spec = make_spec(d=1, H=40.0, z_act=50.0, slit=False, Nmax=6)
    res = _short(spec, C=2, n_burn=2_000, n_run=200, thin=100)
    assert res.n_hi == 6


# ---- reproducibility ------------------------------------------------------ #

def test_same_seed_reproduces_exactly():
    spec = make_spec(d=2, H=5.0, z_act=2.0, Lperp=5.0, slit=True)
    a = _short(spec, seed=11)
    b = _short(spec, seed=11)
    onp.testing.assert_array_equal(a.Ns, b.Ns)
    onp.testing.assert_allclose(a.rho, b.rho, rtol=0, atol=0)


def test_different_seed_gives_a_different_realisation():
    spec = make_spec(d=2, H=5.0, z_act=2.0, Lperp=5.0, slit=True)
    a = _short(spec, seed=11)
    b = _short(spec, seed=12)
    assert not onp.array_equal(a.Ns, b.Ns)


def test_chains_are_not_identical_to_each_other():
    """Each chain gets its own split key; if the batch axis were broadcast by
    mistake every chain would be the same run and the error bars would lie."""
    spec = make_spec(d=1, H=10.0, z_act=1.0, slit=False)
    res = _short(spec, C=8, n_burn=2_000, n_run=4_000, thin=200)
    assert len({tuple(row) for row in res.Ns} ) > 1


# ---- lattice prefill ------------------------------------------------------ #

@pytest.mark.parametrize("d", DIMS)
def test_lattice_fill_sites_do_not_overlap(d):
    spec = make_spec(d=d, H=6.0, z_act=1.0, Lperp=6.0, slit=True)
    sites = lattice_fill(spec, 1000, seed=0)
    assert len(sites) > 0
    assert min_pair_separation(spec, sites) >= spec.sigma


@pytest.mark.parametrize("d", DIMS)
def test_lattice_fill_sites_are_inside_the_box(d):
    spec = make_spec(d=d, H=6.0, z_act=1.0, Lperp=6.0, slit=True)
    sites = lattice_fill(spec, 1000, seed=0)
    assert onp.all(sites[:, -1] >= 0.0) and onp.all(sites[:, -1] <= spec.H)
    if d > 1:
        assert onp.all(sites[:, :-1] >= 0.0)
        assert onp.all(sites[:, :-1] <= spec.Lperp)


def test_lattice_fill_respects_the_requested_count():
    spec = make_spec(d=3, H=8.0, z_act=1.0, Lperp=8.0, slit=True)
    assert len(lattice_fill(spec, 17, seed=0)) == 17


def test_prefilled_run_starts_populated_and_stays_legal():
    spec = make_spec(d=1, H=20.0, z_act=float(eos.z_of_rho(1, 0.5)),
                     slit=False, Nmax=64)
    res = _short(spec, C=4, n_burn=2_000, n_run=2_000, n0=8)
    assert res.n_hi >= 8
    assert min_pair_separation(spec, alive_positions(res, 0)) >= spec.sigma - 1e-9


def test_prefill_and_empty_start_agree_after_burn_in():
    """The prefill is a variance-reduction device, not a physics change: both
    starts must equilibrate to the same density."""
    spec = make_spec(d=1, H=20.0, z_act=float(eos.z_of_rho(1, 0.4)),
                     slit=False, Nmax=64)
    cold = _short(spec, C=16, seed=3, n_burn=30_000, n_run=40_000, thin=200)
    warm = _short(spec, C=16, seed=3, n_burn=30_000, n_run=40_000, thin=200,
                  n0=7)
    assert cold.n_mean == pytest.approx(warm.n_mean, rel=0.08)


# ---- plumbing ------------------------------------------------------------- #

def test_result_reports_acceptance_rates_as_probabilities():
    spec = make_spec(d=2, H=5.0, z_act=2.0, Lperp=5.0, slit=True)
    res = _short(spec)
    assert res.acc.shape == (3,)
    assert onp.all(res.acc >= 0.0) and onp.all(res.acc <= 1.0)


def test_insertion_and_deletion_acceptances_balance_at_equilibrium():
    """At equilibrium <N> is stationary, so accepted insertions must equal
    accepted deletions to within counting noise. The move mix gives them equal
    attempt rates, so their acceptance *rates* must match too."""
    spec = make_spec(d=1, H=20.0, z_act=float(eos.z_of_rho(1, 0.3)),
                     slit=False, Nmax=64)
    res = _short(spec, C=16, n_burn=20_000, n_run=40_000, thin=200)
    ins, dele = res.acc[1], res.acc[2]
    assert ins == pytest.approx(dele, rel=0.06), f"ins {ins} vs del {dele}"


def test_ns_series_has_chain_draw_layout():
    spec = make_spec(d=1, H=10.0, z_act=1.0, slit=False)
    res = burn_and_sample(spec, C=5, seed=0, n_burn=1_000, n_run=3_000,
                          thin=300, nbins=10)
    assert res.Ns.shape == (5, 10)          # (chains, n_run // thin)


def test_density_profile_integrates_to_the_mean_particle_number():
    """Internal consistency between the histogram route and the N counter."""
    spec = make_spec(d=2, H=6.0, z_act=2.0, Lperp=5.0, slit=True)
    res = _short(spec, C=8, n_run=20_000, thin=200, nbins=30)
    dz = spec.H / len(res.z)
    area = spec.Lperp ** (spec.d - 1)
    n_from_profile = res.rho.sum() * dz * area
    assert n_from_profile == pytest.approx(res.n_mean, rel=1e-6)


def test_state_can_be_continued():
    """The returned state is a real restart point, not a summary."""
    spec = make_spec(d=1, H=10.0, z_act=1.0, slit=False)
    res = _short(spec)
    st2, ac = run(spec, res.state, 1_000, 200, 10)
    assert ac.Ns.shape == (4, 5)
    assert onp.all(onp.asarray(st2.nhi) >= onp.asarray(res.state.nhi))


def test_make_spec_sizes_capacity_with_headroom():
    spec = make_spec(d=3, H=8.0, z_act=1.0, Lperp=8.0)
    vol = spec.H * spec.Lperp ** 2
    assert spec.Nmax > 0.75 / eos.B[3] * vol, "capacity below the dense limit"


def test_spec_is_hashable_so_jit_can_treat_it_as_static():
    """``run`` takes the spec as a static argument, which requires hashability.

    A stray array field would break jit in a way that only shows up at call
    time, so check the property directly rather than the types it happens to
    hold today.
    """
    spec = make_spec(d=1, H=10.0, z_act=1.0, slit=False)
    hash(spec)
    assert not any(isinstance(v, (onp.ndarray, np.ndarray)) for v in spec)


def test_equal_specs_hash_equal_so_jit_reuses_the_compilation():
    """Two specs built the same way must be interchangeable to the jit cache.

    This is why the fields in ``mcax.fields`` are NamedTuples: a closure would
    hash by identity, so a sweep that rebuilt the same wall at every state
    point would recompile the kernel at every state point.
    """
    from mcax import fields
    kw = dict(d=1, H=10.0, z_act=1.0, slit=True)
    assert make_spec(**kw) == make_spec(**kw)
    a = make_spec(field=fields.Gravity(2.0), **kw)
    b = make_spec(field=fields.Gravity(2.0), **kw)
    assert a == b and hash(a) == hash(b)
    assert a != make_spec(field=fields.Gravity(2.5), **kw)


@pytest.mark.parametrize("d", [0, 4, 2.5])
def test_make_spec_rejects_dimensions_with_no_reference_eos(d):
    """d = 4 would run happily and could never be checked against anything."""
    with pytest.raises(ValueError):
        make_spec(d=d, H=8.0, z_act=1.0)


# ---- the precision guard -------------------------------------------------- #

def test_float64_guard_passes_when_x64_is_on():
    """conftest enables it for the suite, so this is the happy path."""
    from mcax import require_float64
    require_float64()


def test_float64_guard_raises_in_a_fresh_single_precision_interpreter():
    """The failure this prevents is silent, which is why the guard exists.

    In float32 the overlap test `d2 >= sigma**2` admits marginal overlaps and
    nothing downstream shows it: the density is plausible and the diagnostics
    converge. Anything that lets a run start in single precision is therefore a
    correctness regression, not an ergonomics one.
    """
    import subprocess
    import sys
    import pathlib
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    out = subprocess.run([sys.executable, "-c", _NO_X64 % root],
                         capture_output=True, text=True, timeout=300,
                         env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "RAISED" in out.stdout
