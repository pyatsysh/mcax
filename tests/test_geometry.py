"""The confining geometries: bulk, slit, spherical and cylindrical pores, wedge.

The test that carries the most weight here is
``test_ideal_gas_fills_the_accessible_volume``. Insertions are proposed in the
BOUNDING box and rejected outside the region, so the acceptance ratio carries
V_box while the density is reported against the true accessible volume V. If
either of those two volumes is wrong, or if the rejection is not equivalent to a
hard wall, the measured density comes out wrong by exactly the ratio V/V_box:
pi/6 for a sphere, pi/4 for a cylinder, 1/2 for a wedge. Those are large, clean
factors, which makes the ideal gas a sharp instrument rather than a formality.

The closed-form volumes are checked against plain Monte Carlo integration of the
bounding box, so the geometry's own arithmetic is never the thing verifying it.
"""
import numpy as onp
import pytest

from mcax import make_spec, burn_and_sample, geometry, eos
from conftest import alive_positions, min_pair_separation, inside_region


CURVED = [
    ("sphere", 3, dict(H=5.0)),
    ("sphere", 2, dict(H=6.0)),
    ("sphere", 1, dict(H=8.0)),
    ("cylinder", 3, dict(H=8.0, Lperp=4.0)),
    ("cylinder", 2, dict(H=10.0, Lperp=3.0)),
    ("wedge", 3, dict(H=6.0, Lperp=5.0, psi=onp.pi / 6)),
    ("wedge", 2, dict(H=8.0, psi=onp.pi / 5)),
]
IDS = [f"{g}-d{d}" for g, d, _ in CURVED]


# ---- the static answers, against something independent -------------------- #

@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_volume_matches_monte_carlo_integration(geom, d, kw):
    """The closed-form accessible volume against a crude MC estimate.

    Sampling the bounding box and counting hits uses only `contains`, so an
    error in `volume` and an error in `contains` would have to agree to slip
    through, which they will not.
    """
    spec = make_spec(d=d, geom=geom, z_act=1.0, **kw)
    lo, hi = geometry.bbox(spec)
    rng = onp.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(400_000, d))
    frac = inside_region(spec, pts).mean()
    want = frac * float(onp.prod(hi - lo))
    assert geometry.volume(spec) == pytest.approx(want, rel=0.02)


@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_bin_volumes_sum_to_the_accessible_volume(geom, d, kw):
    """A profile normalised bin by bin must integrate to the whole pore."""
    spec = make_spec(d=d, geom=geom, z_act=1.0, **kw)
    assert geometry.bin_volume(spec, 64).sum() == pytest.approx(
        geometry.volume(spec), rel=1e-12)


def test_sphere_bin_volumes_are_shells_not_slabs():
    """Growing like r^{d-1} is the whole difference, so state it as a test."""
    spec = make_spec(d=3, geom="sphere", H=5.0, z_act=1.0)
    v = geometry.bin_volume(spec, 20)
    assert onp.all(onp.diff(v) > 0), "shell volumes must increase with radius"
    assert v[-1] / v[0] > 100


def test_cylinder_is_a_slit_when_it_is_two_dimensional():
    """A 2-D cylinder of radius R is a strip of width 2R, so its volume is
    2 R H. Worth pinning because the radial formula has to degrade correctly."""
    spec = make_spec(d=2, geom="cylinder", H=7.0, Lperp=3.0, z_act=1.0)
    assert geometry.volume(spec) == pytest.approx(2 * 3.0 * 7.0, rel=1e-12)


def test_wedge_volume_is_half_its_bounding_box():
    """The triangular cross-section is exactly half the rectangle around it,
    at any opening angle, which fixes the proposal efficiency at 1/2."""
    for psi in (onp.pi / 8, onp.pi / 6, onp.pi / 4):
        spec = make_spec(d=3, geom="wedge", H=6.0, Lperp=5.0, psi=psi,
                         z_act=1.0)
        assert geometry.volume(spec) == pytest.approx(
            0.5 * geometry.bbox_volume(spec), rel=1e-12)


# ---- refusals ------------------------------------------------------------- #

@pytest.mark.parametrize("geom", ["cylinder", "wedge"])
def test_geometries_that_need_a_second_axis_refuse_one_dimension(geom):
    with pytest.raises(ValueError, match="d >="):
        make_spec(d=1, geom=geom, H=5.0, z_act=1.0)


def test_unknown_geometry_rejected():
    with pytest.raises(ValueError, match="geom must be"):
        make_spec(d=3, geom="torus", H=5.0, z_act=1.0)


def test_geom_and_slit_together_rejected():
    """Two ways of saying the same thing, so refuse rather than pick one."""
    with pytest.raises(ValueError, match="not both"):
        make_spec(d=3, geom="bulk", slit=True, H=5.0, z_act=1.0)


def test_slit_property_still_reads_the_old_way():
    assert make_spec(d=1, H=5.0, z_act=1.0, slit=True).slit is True
    assert make_spec(d=1, H=5.0, z_act=1.0, slit=False).slit is False
    assert make_spec(d=1, H=5.0, z_act=1.0, geom="sphere").slit is False


def test_describe_names_the_parameters():
    spec = make_spec(d=3, geom="cylinder", H=10.0, Lperp=4.0, z_act=1.0)
    text = geometry.describe(spec)
    assert "radius 4.0" in text and "axial period 10.0" in text


# ---- the sampler in each geometry ----------------------------------------- #

@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_no_overlaps_and_nothing_escapes(geom, d, kw):
    """The two invariants that must hold in every geometry at once."""
    spec = make_spec(d=d, geom=geom, z_act=3.0, **kw)
    res = burn_and_sample(spec, C=4, seed=0, n_burn=4_000, n_run=8_000,
                          thin=400, nbins=16)
    for c in range(4):
        pts = alive_positions(res, c)
        assert min_pair_separation(spec, pts) >= spec.sigma - 1e-9
        assert onp.all(inside_region(spec, pts)), (
            f"{geom} d={d}: {(~inside_region(spec, pts)).sum()} centres escaped")


@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_ideal_gas_fills_the_accessible_volume(geom, d, kw):
    """<N> = z V with V the ACCESSIBLE volume, not the bounding box.

    This is what proves the bounding-box proposal is exact rather than merely
    plausible. Getting V and V_box the wrong way round would show up here as a
    factor of pi/6, pi/4 or 1/2 depending on the geometry, none of which hides
    inside a 6% tolerance.
    """
    spec = make_spec(d=d, geom=geom, z_act=0.5, sigma=1e-6, **kw)
    V = geometry.volume(spec)
    res = burn_and_sample(spec, C=16, seed=1, n_burn=8_000, n_run=40_000,
                          thin=100, nbins=8)
    rho = res.n_mean / V
    assert rho == pytest.approx(0.5, rel=0.06), (
        f"{geom} d={d}: rho {rho:.4f} against z 0.5, "
        f"V {V:.2f} of bbox {geometry.bbox_volume(spec):.2f}")
    assert res.capacity_warning is None


@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_ideal_gas_profile_is_flat_in_density(geom, d, kw):
    """An ideal gas has uniform density everywhere, including in a pore.

    So rho(r) must come out flat, and it only does if the bin volumes are the
    geometry's own. Using a constant slab in a spherical pore would give a
    profile falling like 1/r^2, which reads convincingly like a packing effect.
    """
    spec = make_spec(d=d, geom=geom, z_act=0.5, sigma=1e-6, **kw)
    res = burn_and_sample(spec, C=16, seed=2, n_burn=8_000, n_run=40_000,
                          thin=100, nbins=10)
    # The first bin of a curved pore holds a tiny volume and is noisy; the
    # wedge's first bin is a genuine near-zero sliver at the apex. Judge the
    # interior.
    rho = res.rho[2:]
    assert rho.std() / rho.mean() < 0.15, f"{geom} d={d}: profile {res.rho}"


@pytest.mark.parametrize("geom,d,kw", CURVED, ids=IDS)
def test_profile_integrates_to_the_mean_particle_number(geom, d, kw):
    """int rho dV = <N>, with dV the geometry's bin volumes."""
    spec = make_spec(d=d, geom=geom, z_act=2.0, **kw)
    res = burn_and_sample(spec, C=8, seed=3, n_burn=4_000, n_run=16_000,
                          thin=400, nbins=24)
    n_from_profile = float((res.rho * geometry.bin_volume(spec, 24)).sum())
    assert n_from_profile == pytest.approx(res.n_mean, rel=1e-9)


def test_dense_sphere_shows_wall_layering():
    """A pore is only worth having if it structures the fluid, so check that
    the density near the wall exceeds the density at the centre."""
    rho_b = 0.35 / eos.B[3]
    spec = make_spec(d=3, geom="sphere", H=4.0,
                     z_act=float(eos.z_of_rho(3, rho_b)))
    res = burn_and_sample(spec, C=16, seed=4, n_burn=40_000, n_run=120_000,
                          thin=300, nbins=20)
    inner = res.rho[:8].mean()
    contact = res.rho[-3:].mean()
    assert contact > 1.3 * inner, f"no layering: contact {contact} vs {inner}"


def test_wedge_apex_is_enriched_not_starved():
    """The apex fills UP, and that is the centre-exclusion convention showing.

    Walls act on centres, never on surfaces, so a centre is welcome wherever
    the opening is, however narrow. The line density stays finite as the
    opening closes while the area it occupies goes to zero, so rho = lambda / w
    grows like 1/z on the way in. Reading that as a packing effect would be the
    natural mistake, hence the test.
    """
    spec = make_spec(d=2, geom="wedge", H=10.0, psi=onp.pi / 8,
                     z_act=float(eos.z_of_rho(2, 0.3 / eos.B[2])))
    res = burn_and_sample(spec, C=16, seed=5, n_burn=30_000, n_run=90_000,
                          thin=300, nbins=20)
    assert res.rho[0] > 2.0 * res.rho[8:16].mean()
    assert onp.all(onp.diff(res.rho[:3]) < 0), "should fall away from the apex"


def test_wedge_apex_becomes_a_one_dimensional_hard_rod_fluid():
    """Near the apex the wedge is a channel narrower than sigma, and a channel
    narrower than sigma IS a Tonks gas: two centres in it are in contact when
    their separation along the axis reaches sigma.

    So the line density there must match the exact 1-D inversion evaluated at
    the LOCAL one-dimensional activity, which is the two-dimensional activity
    times the local opening width. That turns the apex into an exact target in
    a geometry that otherwise has none.

    It is an ASYMPTOTIC target and only the innermost bin is inside it. The
    reduction neglects the transverse freedom, which enters at order
    (w/sigma)^2, and the measured numbers say so plainly: at w = 0.21 sigma the
    line density is 0.264 against Tonks 0.263, and one bin further out at
    w = 0.62 sigma it is 0.519 against 0.423, off by 23%. Testing the second
    bin would be testing the approximation, not the sampler.
    """
    psi, nbins = onp.pi / 8, 20
    spec = make_spec(d=2, geom="wedge", H=10.0, psi=psi,
                     z_act=float(eos.z_of_rho(2, 0.3 / eos.B[2])))
    res = burn_and_sample(spec, C=16, seed=5, n_burn=30_000, n_run=90_000,
                          thin=300, nbins=nbins)
    w = 2.0 * res.z[0] * onp.tan(psi)               # opening at the bin centre
    assert w < 0.25, "the innermost bin has to be narrow for this to apply"
    lam = res.rho[0] * w                            # measured line density
    want = eos.rho_of_z(1, spec.z_act * w)          # Tonks at the local z
    assert lam == pytest.approx(want, rel=0.05), (
        f"width {w:.3f}, lambda {lam:.4f} against Tonks {want:.4f}")
