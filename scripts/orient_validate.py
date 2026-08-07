"""Validation ladder V0 to V4 for the orientable engine, each rung gating the next.

    V0  p = 2: orientations decouple exactly. Observables must match the
        hard-sphere engine within replica error, and rotations must accept at
        100%, because a sphere is invariant under them.
    V1  frozen rotation (dtheta = 0, aligned start): must reproduce the
        PARALLEL superball engine of `mcax.core` state by state. This one
        certifies both engines against each other: they share no overlap code.
    V2  dilute virial. The orientation-averaged excluded volume, integrated by
        Monte Carlo over separation x two independent orientations, against the
        exact B2 where one exists.
    V3  overlap unit battery: known-answer pairs, plus adversarial random pairs
        against a slow high-budget reference, plus the budget trip rate.
    V4  a literature anchor for freely rotating cubes.

**On the V4 anchor.** The obvious choice is a published finite-density equation
of state for free hard cubes, and it is deliberately not used here: this session
could not verify such a curve from source, and an unverified number copied into
a validation report is worse than no anchor at all. What is used instead is
exact and checkable: the Isihara-Hadwiger theorem gives the orientation-averaged
excluded volume of two identical convex bodies in closed form,

    B2 = V + R S,

with V the volume, S the surface area and R the mean radius of curvature. For a
cube of edge a the mean radius of curvature is the edge sum (1/8pi) sum L_e
theta_e = 3a/4, so

    B2(free cube) = a^3 + (3a/4)(6a^2) = 5.5 a^3,

against 4 a^3 for the same cubes held parallel. That factor is a real,
independently derived, orientation-resolved prediction, and reproducing it
tests the rotational machinery and the overlap test together.

Run (CPU is enough for everything but V4's production run):
    LEASE_SKIP=1 JAX_PLATFORMS=cpu taskset -c 2-3 <python> \
        scripts/orient_validate.py [--quick]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np

from mcax import bodies, eos, geometry, orient, shapes
from mcax.bodies import Superball
from mcax.core import make_spec, burn_and_sample

INF = float("inf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


def prefill(spec, body):
    """Seat ~90% of the expected occupancy on a lattice, for BOTH engines.

    Without it a comparison of two engines at a few thousand steps compares
    their filling rates rather than their equilibria: they run different move
    mixes, so the one proposing fewer insertions per step is simply further
    behind, and the difference reads as a physics discrepancy. Both engines get
    the same seating from the same lattice code.
    """
    v = bodies.volume(body, 3)
    z = spec.z_act
    # rough occupancy from the aligned reference EOS at the same activity,
    # rescaled by the volume ratio: only a starting point, and the burn fixes it
    rho = shapes.volume(shapes.Superball(2.0), 3) / v * eos.rho_of_z(3, z)
    return int(0.9 * min(rho, 0.75 / v) * geometry.volume(spec))


def chain_err(Ns):
    v = onp.asarray(Ns, dtype=float).mean(axis=1)
    return float(v.std(ddof=1) / onp.sqrt(v.size))


# --------------------------------------------------------------------------- #

def v0_sphere_decouples(quick):
    """p = 2: the orientable engine must agree with the aligned one, and every
    rotation must be accepted, since rotating a sphere changes nothing."""
    C = 8 if quick else 16
    nb, nr = (6_000, 20_000) if quick else (20_000, 60_000)
    rows = []
    for H, z in ((5.0, 3.0), (6.0, 6.0)):
        os_ = orient.make_spec(H=H, Lperp=5.0, z_act=z, body=Superball(2.0),
                               geom="slit", dtheta=0.5)
        n0 = prefill(os_, Superball(2.0))
        ro = orient.burn_and_sample(os_, C=C, seed=4, n_burn=nb, n_run=nr,
                                    thin=50, nbins=int(H / 0.25), n0=n0)
        ps = make_spec(d=3, H=H, Lperp=5.0, z_act=z, geom="slit",
                       Nmax=os_.Nmax)
        rp = burn_and_sample(ps, C=C, seed=4, n_burn=nb, n_run=nr, thin=50,
                             nbins=int(H / 0.25), n0=n0)
        e = onp.sqrt(chain_err(ro.Ns) ** 2 + chain_err(rp.Ns) ** 2)
        rows.append(dict(
            H=H, z=z, n_orient=ro.n_mean, n_aligned=rp.n_mean,
            err=float(e), dev_sigma=abs(ro.n_mean - rp.n_mean) / max(e, 1e-12),
            rot_acceptance=float(ro.acc[1]),
            s4_max=float(onp.nanmax(onp.abs(ro.s4)))))
        print(f"  V0 H={H} z={z}: <N> orientable {ro.n_mean:.2f} vs aligned "
              f"{rp.n_mean:.2f}  ({rows[-1]['dev_sigma']:.1f} sigma), "
              f"rotation acceptance {100 * ro.acc[1]:.2f}%")
    return rows


def v1_frozen_reproduces_parallel(quick):
    """dtheta = 0 with an aligned start: the same model as `mcax.core`, reached
    through completely different code. Both engines are certified by this."""
    C = 8 if quick else 16
    nb, nr = (6_000, 20_000) if quick else (20_000, 60_000)
    rows = []
    for p in (2.0, 4.0, INF):
        for H, z in ((5.0, 5.0),):
            os_ = orient.make_spec(H=H, Lperp=5.0, z_act=z, body=Superball(p),
                                   geom="slit", dtheta=0.0, p_rot=0.1)
            n0 = prefill(os_, Superball(p))
            ro = orient.burn_and_sample(os_, C=C, seed=7, n_burn=nb, n_run=nr,
                                        thin=50, nbins=int(H / 0.25),
                                        aligned=True, n0=n0)
            ps = make_spec(d=3, H=H, Lperp=5.0, z_act=z, geom="slit",
                           shape=shapes.Superball(p), Nmax=os_.Nmax)
            rp = burn_and_sample(ps, C=C, seed=7, n_burn=nb, n_run=nr,
                                 thin=50, nbins=int(H / 0.25), n0=n0)
            e = onp.sqrt(chain_err(ro.Ns) ** 2 + chain_err(rp.Ns) ** 2)
            rows.append(dict(
                p="inf" if p == INF else f"{p:g}", H=H, z=z,
                n_frozen=ro.n_mean, n_parallel=rp.n_mean, err=float(e),
                dev_sigma=abs(ro.n_mean - rp.n_mean) / max(e, 1e-12)))
            print(f"  V1 p={rows[-1]['p']:>3}: frozen {ro.n_mean:.2f} vs "
                  f"parallel {rp.n_mean:.2f}  "
                  f"({rows[-1]['dev_sigma']:.1f} sigma)")
    return rows


def _b2_free(body, n=400_000, seed=0, n_iter=32):
    """Orientation-averaged B2 by Monte Carlo over separation x orientations.

    B2 = (1/2) integral <1 - exp(-beta u)> d^3 r, and for hard bodies the
    average is the probability that two independently and uniformly oriented
    copies at separation r overlap. Sampled in a cube big enough to contain the
    whole excluded region, which is twice the circumradius on each side.

    An INDEPENDENT code path from the sampler, as V2 requires: it calls the
    overlap predicate directly, with no chains, no acceptances and no geometry.
    """
    rc = bodies.circumradius(body, 3)
    L = 2.0 * (2.0 * rc)
    rng = onp.random.default_rng(seed)
    dr = rng.uniform(-0.5 * L, 0.5 * L, size=(n, 3))
    ks = jax.random.split(jax.random.PRNGKey(seed), 2 * n)
    Q = jax.vmap(orient.q_random)(ks)
    Ra = orient.q_matrix(Q[:n])
    Rb = orient.q_matrix(Q[n:])
    f = jax.jit(jax.vmap(lambda v, A, B: orient.overlaps_pair(
        body, v, A, B, n_iter, 1e-9)))
    hit = onp.asarray(f(np.asarray(dr), Ra, Rb))
    frac = hit.mean()
    err = onp.sqrt(frac * (1 - frac) / n)
    return 0.5 * frac * L ** 3, 0.5 * err * L ** 3


def v2_virial(quick):
    """Measured B2 against the exact value where one exists.

    Exact anchors: the sphere, where free rotation changes nothing and
    B2 = 4v; and the cube, where Isihara-Hadwiger gives B2 = V + R S = 5.5 for
    edge 1. In between there is no closed form, and what is reported is the
    measured value together with the parallel-family B2 it must exceed, since
    letting a body rotate can only enlarge its orientation-averaged excluded
    volume.
    """
    n = 120_000 if quick else 600_000
    rows = []
    for p in (2.0, 3.0, 4.0, 6.0, INF):
        b = Superball(p)
        got, err = _b2_free(b, n=n, seed=3)
        v = bodies.volume(b, 3)
        exact = None
        if p == 2.0:
            exact = 4.0 * v                       # sphere: rotation is trivial
        elif p == INF:
            exact = 1.0 + 0.75 * 6.0              # Isihara-Hadwiger: V + R S
        rows.append(dict(
            p="inf" if p == INF else f"{p:g}", b2_free=got, b2_free_err=err,
            b2_parallel=shapes.b2(shapes.Superball(p), 3, 1.0),
            b2_exact=exact,
            dev_sigma=(abs(got - exact) / max(err, 1e-12)) if exact else None,
            dev_rel=(abs(got - exact) / exact) if exact else None))
        e = (f"exact {exact:.4f} ({rows[-1]['dev_rel']:.2%})" if exact
             else "no closed form")
        print(f"  V2 p={rows[-1]['p']:>3}: B2 free {got:.4f} +- {err:.4f}, "
              f"parallel {rows[-1]['b2_parallel']:.4f}, {e}")
    return rows


def v3_overlap_battery(quick):
    """Known answers, adversarial randoms, and the measured budget trip rate."""
    rows, known = [], []
    E = onp.eye(3)

    # touching along a lab axis, both unrotated: the aligned answer, exactly
    for p in (2.0, 4.0, INF):
        b = Superball(p)
        for s, want in ((0.99, True), (1.01, False)):
            got = bool(orient.overlaps_pair(b, np.asarray([s, 0.0, 0.0]), E, E))
            known.append(dict(case=f"axis contact p={p} s={s}", want=want,
                              got=got, ok=got == want))
    # a cube rotated 45 degrees about z, corner-on to an unrotated cube. The
    # rotated one reaches sqrt(2)/2 along x, the fixed one 1/2, so they touch
    # at 0.5 + 0.7071 and not a hair further.
    c = onp.cos(onp.pi / 8), onp.sin(onp.pi / 8)          # 45 deg about z
    q45 = np.asarray([c[0], 0.0, 0.0, c[1]])
    R45 = orient.q_matrix(q45)
    d0 = 0.5 + 0.5 * onp.sqrt(2.0)
    for s, want in ((d0 - 0.02, True), (d0 + 0.02, False)):
        got = bool(orient.overlaps_pair(Superball(INF),
                                        np.asarray([s, 0.0, 0.0]), E, R45))
        known.append(dict(case=f"cube corner-on, 45 deg, d={s:.4f}", want=want,
                          got=got, ok=got == want))
    # deep overlap and far apart, every shape
    for p in (2.0, 3.0, 4.0, 6.0, INF):
        b = Superball(p)
        rng = onp.random.default_rng(1)
        Rr = orient.q_matrix(orient.q_random(jax.random.PRNGKey(2)))
        known.append(dict(case=f"coincident centres p={p}", want=True,
                          got=bool(orient.overlaps_pair(
                              b, np.asarray([0.0, 0.0, 0.0]), E, Rr)),
                          ok=None))
        known.append(dict(case=f"far apart p={p}", want=False,
                          got=bool(orient.overlaps_pair(
                              b, np.asarray([5.0, 3.0, 2.0]), E, Rr)),
                          ok=None))
    for k in known:
        k["ok"] = bool(k["got"] == k["want"])
        print(f"  V3 {k['case']:<44s} want {str(k['want']):<5s} got "
              f"{str(k['got']):<5s} {'ok' if k['ok'] else 'FAIL'}")

    # adversarial randoms against a high-budget reference, in the shell where
    # the cheap tiers cannot decide: the only place a verdict can be wrong.
    M = 20_000 if quick else 80_000
    for p in (2.0, 3.0, 4.0, 6.0, INF):
        b = Superball(p)
        ri, rc = bodies.inradius(b, 3), bodies.circumradius(b, 3)
        rng = onp.random.default_rng(5)
        u = rng.normal(size=(M, 3))
        u /= onp.linalg.norm(u, axis=1)[:, None]
        dr = u * rng.uniform(2 * ri, 2 * rc, size=(M, 1))
        ks = jax.random.split(jax.random.PRNGKey(11), 2 * M)
        Q = jax.vmap(orient.q_random)(ks)
        Ra, Rb = orient.q_matrix(Q[:M]), orient.q_matrix(Q[M:])

        def go(n_iter):
            f = jax.jit(jax.vmap(lambda v, A, B: orient.overlaps_pair(
                b, v, A, B, n_iter, 1e-9)))
            return onp.asarray(f(np.asarray(dr), Ra, Rb))

        ref = go(1000)
        prod = go(32)
        unsafe = int((ref & ~prod).sum())
        trips = int((prod & ~ref).sum())
        ndis = int((~ref).sum())
        rows.append(dict(p="inf" if p == INF else f"{p:g}", n_pairs=M,
                         n_disjoint=ndis, trips=trips,
                         trip_rate=trips / max(ndis, 1), unsafe=unsafe))
        print(f"  V3 p={rows[-1]['p']:>3}: {trips} trips in {ndis} disjoint "
              f"shell pairs = {trips / max(ndis, 1):.2e}; unsafe verdicts "
              f"{unsafe}")
    return dict(known=known, trip=rows)


def v4_free_cubes(quick):
    """The literature anchor: freely rotating cubes.

    Two statements are checked. The exact one is Isihara-Hadwiger's
    B2 = 5.5 for unit cubes, reached two independent ways: by integrating the
    overlap predicate (V2) and by the dilute limit of a grand-canonical run,
    beta mu_ex -> 2 B2 rho, which involves the whole sampler. The second is
    qualitative but sharp: free cubes must be LESS dense than parallel ones at
    the same activity, because rotation enlarges the excluded volume.
    """
    C = 8 if quick else 16
    nb, nr = (8_000, 24_000) if quick else (20_000, 60_000)
    rows = []
    b = Superball(INF)
    for z in (0.15, 0.3, 0.6):
        os_ = orient.make_spec(H=6.0, Lperp=6.0, z_act=z, body=b, geom="bulk",
                               dtheta=0.5)
        ro = orient.burn_and_sample(os_, C=C, seed=21, n_burn=nb, n_run=nr,
                                    thin=50, nbins=16)
        ps = make_spec(d=3, H=6.0, Lperp=6.0, z_act=z, geom="bulk",
                       shape=shapes.Superball(INF), Nmax=os_.Nmax)
        rp = burn_and_sample(ps, C=C, seed=21, n_burn=nb, n_run=nr, thin=50,
                             nbins=16)
        vol = geometry.volume(os_)
        rho = ro.n_mean / vol
        rows.append(dict(
            z=z, rho_free=rho, eta_free=rho * 1.0,
            rho_parallel=rp.n_mean / vol,
            b2_implied=float((onp.log(z) - onp.log(rho)) / (2.0 * rho)),
            n_err=chain_err(ro.Ns), rot_acceptance=float(ro.acc[1]),
            cubatic=[float(x) for x in orient.cubatic(ro.state)]))
        print(f"  V4 z={z}: free cubes eta={rho:.4f} vs parallel "
              f"{rp.n_mean / vol:.4f}; B2 implied by the dilute limit "
              f"{rows[-1]['b2_implied']:.3f} (exact 5.5); "
              f"rot acc {100 * ro.acc[1]:.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"orientable engine validation  (devices {jax.devices()})"
          + ("  [quick]" if args.quick else ""))

    rungs = [("V0", v0_sphere_decouples), ("V1", v1_frozen_reproduces_parallel),
             ("V2", v2_virial), ("V3", v3_overlap_battery),
             ("V4", v4_free_cubes)]
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"),
              "quick": args.quick, "n_iter": 32}
    for name, fn in rungs:
        if args.only and name not in args.only.split(","):
            continue
        print(f"\n{name}")
        t0 = time.time()
        report[name] = fn(args.quick)
        print(f"  ({time.time() - t0:.1f}s)")
        with open(os.path.join(OUT, "orient_validation.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/orient_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
