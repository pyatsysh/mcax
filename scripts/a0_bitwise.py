"""Acceptance test A0: shapes cost the hard-sphere engine nothing, bit for bit.

`tests/test_shapes.py` checks that `shape = None` and `Superball(2.0)` are the
same trajectory, which is necessary but is a comparison of the new engine with
itself. This is the other half: it checks out the LAST COMMIT BEFORE shapes into
a throwaway git worktree, runs the same state points through both engines with
the same seeds, and compares the raw state arrays.

That distinction matters because the failure mode being excluded is a change of
arithmetic that both new paths share. Reordering the overlap reduction, folding
the minimum image differently, or computing the separation once instead of
twice would all pass a self-comparison and all move the last bits of a float,
which moves an acceptance, which moves a trajectory.

Run:
    LEASE_SKIP=1 JAX_PLATFORMS=cpu python scripts/a0_bitwise.py [<ref-commit>]
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as onp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same states in both engines. Chosen to exercise every branch the refactor
# touched: a periodic bulk box, a slit with one walled axis, a curved pore
# whose insertions are rejected on the region, and one case dense enough that
# the overlap test is doing real work rather than passing everything.
CASES = [
    dict(d=1, geom="bulk", H=30.0, Lperp=1.0, z_act=3.0, n0=10),
    dict(d=2, geom="slit", H=6.0, Lperp=8.0, z_act=4.0, n0=25),
    dict(d=3, geom="slit", H=6.0, Lperp=6.0, z_act=8.0, n0=60),
    dict(d=3, geom="sphere", H=3.0, Lperp=1.0, z_act=6.0, n0=30),
    dict(d=3, geom="bulk", H=6.0, Lperp=6.0, z_act=12.0, n0=90),
]

DRIVER = r'''
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, sys.argv[1])
import numpy as onp, jax
jax.config.update("jax_enable_x64", True)
from mcax import make_spec, burn_and_sample
CASES = json.loads(sys.argv[2])
USE_SHAPE = sys.argv[3] == "shape"
out = {}
for i, c in enumerate(CASES):
    n0 = c.pop("n0")
    if USE_SHAPE:
        from mcax.shapes import Superball
        c = dict(c, shape=Superball(2.0))
    spec = make_spec(**c)
    r = burn_and_sample(spec, C=4, seed=17 + i, n_burn=1500, n_run=6000,
                        thin=50, nbins=32, n0=n0)
    out[str(i)] = dict(pos=onp.asarray(r.state.pos).tolist(),
                       alive=onp.asarray(r.state.alive).tolist(),
                       Ns=onp.asarray(r.Ns).tolist(),
                       rho=onp.asarray(r.rho).tolist(),
                       acc=onp.asarray(r.state.acc).tolist(),
                       Nmax=int(spec.Nmax))
json.dump(out, open(sys.argv[4], "w"))
'''


def run_driver(tree, tag, path):
    import json
    drv = os.path.join(tempfile.gettempdir(), "a0_driver.py")
    with open(drv, "w") as fh:
        fh.write(DRIVER)
    env = dict(os.environ, LEASE_SKIP="1", JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, drv, tree, json.dumps(CASES), tag, path],
                   check=True, env=env)
    return json.load(open(path))


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    if ref is None:
        # The commit before this session: HEAD is still the pre-shapes tree
        # while the work is uncommitted, which is the usual case when this is
        # run as a gate.
        ref = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
    print(f"reference commit {ref[:9]}")

    tmp = tempfile.mkdtemp(prefix="mcax-a0-")
    tree = os.path.join(tmp, "ref")
    subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", tree,
                    ref], check=True, capture_output=True)
    try:
        old = run_driver(tree, "plain", os.path.join(tmp, "old.json"))
        new_none = run_driver(REPO, "plain", os.path.join(tmp, "new0.json"))
        new_p2 = run_driver(REPO, "shape", os.path.join(tmp, "new2.json"))
    finally:
        subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force",
                        tree], capture_output=True)

    bad = 0
    for i, c in enumerate(CASES):
        k = str(i)
        name = f"d{c['d']}_{c['geom']}_z{c['z_act']}"
        for label, cand in (("shape=None", new_none), ("Superball(2)", new_p2)):
            same = all(onp.array_equal(onp.asarray(old[k][f]),
                                       onp.asarray(cand[k][f]))
                       for f in ("pos", "alive", "Ns", "rho", "acc", "Nmax"))
            print(f"  {name:<22s} {label:<14s} "
                  f"{'IDENTICAL' if same else 'DIFFERS'}")
            bad += not same
    print("\nA0", "PASS" if not bad else f"FAIL ({bad} mismatches)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
