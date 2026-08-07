"""Grand-canonical Monte Carlo for hard convex bodies that ROTATE.

The aligned engine in `mcax.core` samples positions alone: every superball
there carries the same orientation and the configuration space is R^d per
particle. This one gives each particle a quaternion as well, so the space is
R^3 x SO(3) per particle, the moves gain a rotation, and grand-canonical
insertion has to draw an orientation from the Haar measure. The physics that
changes is the overlap test, and it changes completely: two rotated superballs
have no closed-form contact condition, so it has to be computed.

    state        positions (C, Nmax, 3) and unit quaternions (C, Nmax, 4)
    moves        translate | rotate | insert | delete
    overlap      three tiers, cheapest first (below)
    bodies       anything in `mcax.bodies` with a support map

**The overlap test, and which way it is allowed to be wrong.** Two convex
bodies A and B are disjoint exactly when some direction separates them, that is
when the support function of the Minkowski difference M = B - A is negative
somewhere:

    h_M(u) = u . (c_B - c_A) + h_K(Omega_B^T u) + h_K(Omega_A^T u),
    disjoint  <=>  h_M(u) < 0 for some u.

So a single direction with h_M(u) < 0 is a CERTIFICATE of no overlap, checkable
exactly and cheaply. There is no equally cheap certificate the other way, and
that asymmetry decides the design: this engine SEARCHES FOR SEPARATION, and if
it fails to find any within a fixed budget it declares overlap. The budget is
fixed because `jit` will not take a data-dependent while loop, and the guarantee
direction is chosen because rejecting a legal move costs efficiency while
accepting an illegal one destroys the ensemble.

That is a real approximation and it is worth being precise about what kind.
The verdict is a deterministic function of the pair configuration, so the
sampler is exact for a slightly modified model in which bodies are imperceptibly
fattened inside the shell where the search can run out of budget. It is not a
broken Markov chain: detailed balance is untouched, because the predicate does
not depend on the path taken to reach the configuration. The measured budget
trip rate is reported in `docs/orientable-overlap.md`.

**The search.** Two stages, both certificate-checked:

  1. Fifteen candidate axes: the three face normals of each body in the lab
     frame, and the nine cross products between them. This is the separating
     axis theorem's candidate set, which is COMPLETE for cubes, so p = infinity
     is exact at this stage and never reaches the second.
  2. Frank-Wolfe on min ||x||^2 over M, started at x = c_B - c_A, which is
     always a point of M. Each step takes u = -x/||x||, checks h_M(u) for a
     certificate, and moves x towards the support point along that direction by
     an exact line search. No simplex bookkeeping, no branches, one fixed-length
     scan.

**Central symmetry is load-bearing.** Every body in `mcax.bodies` satisfies
K = -K, which is what collapses the Minkowski support map to

    s_M(u) = (c_B - c_A) + Omega_B s_K(Omega_B^T u) + Omega_A s_K(Omega_A^T u)

with a plus rather than a minus in the last term. A body without it would need
the general form and a second support map.

**Walls still act on CENTRES**, exactly as in `mcax.core`, so a rotating
particle may lean out of the box. That is the same centre-exclusion convention
the DFT reference data uses, and keeping it is what makes the frozen-rotation
mode reproduce the aligned engine state by state.
"""
from functools import partial
from typing import NamedTuple

import numpy as onp
import jax
import jax.numpy as np

from . import bodies
from . import fields
from . import geometry
from .core import require_float64


# --------------------------------------------------------------------------- #
#  Quaternions                                                                 #
# --------------------------------------------------------------------------- #
#
# (w, x, y, z), unit norm, acting as body -> lab. Stored RAW: the superball's
# cubic symmetry means twenty-four quaternions describe the same physical
# particle, and folding them into a canonical representative would cost work
# every step to buy nothing the sampler needs. Symmetry is handled at observable
# time, by using invariants of the cubic group rather than the quaternion.

def q_matrix(q):
    """(3, 3) rotation matrix of a unit quaternion."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y)], axis = -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                  2 * (y * z - w * x)], axis = -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)], axis = -1)], axis = -2)


def q_mul(a, b):
    """Quaternion product, so that R(a b) = R(a) R(b)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis = -1)


def q_random(key, shape = ()):
    """Uniform on SO(3) by Shoemake's method.

    Three uniforms map to a point uniform on the 3-sphere, and uniform on S^3
    IS Haar on SO(3) under the double cover. This is the measure grand-canonical
    insertion has to draw from: the acceptance ratio below carries no
    orientational factor precisely because the proposal density and the
    reference measure are the same, and any other sampling of orientations
    (Euler angles uniform in their ranges, say) would silently reweight the
    ensemble towards the poles.
    """
    u = jax.random.uniform(key, tuple(shape) + (3,))
    r1, r2 = np.sqrt(1.0 - u[..., 0]), np.sqrt(u[..., 0])
    t1, t2 = 2.0 * np.pi * u[..., 1], 2.0 * np.pi * u[..., 2]
    return np.stack([r2 * np.cos(t2), r1 * np.sin(t1),
                     r1 * np.cos(t1), r2 * np.sin(t2)], axis = -1)


def q_perturb(key, q, dtheta):
    """A small symmetric rotation applied to `q`.

    Uniform axis on the sphere, angle uniform on [-dtheta, dtheta], composed on
    the LEFT. Symmetry of the proposal is what lets the acceptance omit a
    Hastings ratio, and it holds because the reverse move is the same rotation
    through -angle about the same axis and both are drawn with equal density.
    Composing on the left rather than the right makes the perturbation a lab
    frame rotation; either is symmetric, but mixing the two in one sampler is
    not, so it is written down.
    """
    ka, kt = jax.random.split(key)
    n = jax.random.normal(ka, (3,))
    n = n / np.maximum(np.linalg.norm(n), 1e-300)
    a = 0.5 * dtheta * jax.random.uniform(kt, minval = -1.0, maxval = 1.0)
    dq = np.concatenate([np.cos(a)[None], np.sin(a) * n])
    out = q_mul(dq, q)
    return out / np.maximum(np.linalg.norm(out), 1e-300)


# --------------------------------------------------------------------------- #
#  The overlap test                                                            #
# --------------------------------------------------------------------------- #

def _sat_axes(dr, Ra, Rb):
    """(32, 3) separating-axis candidates for one pair.

    The two bodies' face normals, every cross product between them, the centre
    separation, and the NEGATIVE of each. Complete for boxes, which is the
    whole reason it is worth its cost: at p = infinity a negative value here
    settles the question exactly and the Frank-Wolfe stage never contributes.
    For finite p it is not complete, but every entry is still a valid
    certificate and it is where the closest features of two nearly touching
    superballs usually are.

    **Both signs, and this is not padding.** h_M(u) < 0 says the origin lies
    strictly on one side of the hyperplane with normal u; separation along the
    axis in the other sense is h_M(-u) < 0, a different number. Testing only +u
    finds about half of all separations and misses the rest, which does not
    show up as a wrong answer (the missed ones fall through to Frank-Wolfe and
    are usually recovered) but as a budget-trip rate two orders of magnitude
    worse than it should be, cubes included, where the test ought to be exact.
    """
    A, B = Ra.T, Rb.T                                   # (3, 3) rows are axes
    cross = np.cross(A[:, None, :], B[None, :, :]).reshape(9, 3)
    n = np.linalg.norm(cross, axis = -1, keepdims = True)
    # Parallel axes give a zero cross product, which is not a direction. Sent
    # to a fixed unit vector instead of normalised: it then duplicates a
    # candidate already in the list rather than producing a nan that would
    # compare False and read as "no certificate here".
    cross = np.where(n > 1e-9, cross / np.maximum(n, 1e-300),
                     np.array([1.0, 0.0, 0.0]))
    sep = dr / np.maximum(np.linalg.norm(dr), 1e-300)
    half = np.concatenate([A, B, cross, sep[None, :]], axis = 0)
    return np.concatenate([half, -half], axis = 0)


def _h_minkowski(body, u, dr, Ra, Rb):
    """h_M(u) for the difference body: negative anywhere means disjoint."""
    return (np.sum(u * dr, axis = -1)
            + bodies.h(body, u @ Rb)                    # = h(Rb^T u)
            + bodies.h(body, u @ Ra))


def _s_minkowski(body, u, dr, Ra, Rb):
    """The support POINT of the difference body along u."""
    return (dr + bodies.support(body, u @ Rb) @ Rb.T
            + bodies.support(body, u @ Ra) @ Ra.T)


def overlaps_pair(body, dr, Ra, Rb, n_iter = 32, tol = 1e-9):
    """Do the two bodies overlap. `dr = c_B - c_A`, rotations body -> lab.

    Returns True unless a separating direction was FOUND, so every uncertain
    answer is an overlap. See the module docstring for why that is the safe
    direction and what it costs.
    """
    d = dr.shape[-1]
    r_in, r_out = bodies.inradius(body, d), bodies.circumradius(body, d)
    d2 = np.sum(dr * dr, axis = -1)
    sure_over = d2 < (2.0 * r_in) ** 2
    sure_free = d2 > (2.0 * r_out) ** 2

    axes = _sat_axes(dr, Ra, Rb)
    hv = _h_minkowski(body, axes, dr, Ra, Rb)
    sep = np.any(hv < -tol)
    # Frank-Wolfe starts from the support point along the most promising
    # candidate rather than from the centre separation. Both are points of M,
    # so both are legal starts, but the descent has a great deal less to do
    # from the one the axis search already narrowed down to.
    x0 = _s_minkowski(body, axes[np.argmin(hv)], dr, Ra, Rb)

    def step(carry, _):
        x, sep = carry
        nx = np.maximum(np.linalg.norm(x), 1e-300)
        u = -x / nx
        sep = sep | (_h_minkowski(body, u, dr, Ra, Rb) < -tol)
        s = _s_minkowski(body, u, dr, Ra, Rb)
        w = x - s
        # Exact line search along [x, s]: the minimiser of ||x - gam w||^2 in
        # gam, clipped to the segment. Closed form, so no inner loop and no
        # branch, which is what keeps the whole test one flat scan.
        gam = np.clip(np.sum(x * w) / np.maximum(np.sum(w * w), 1e-300),
                      0.0, 1.0)
        return (x - gam * w, sep), None

    (_, sep), _ = jax.lax.scan(step, (x0, sep), None, length = n_iter)
    return sure_over | (~sure_free & ~sep)


# --------------------------------------------------------------------------- #
#  Spec and state                                                              #
# --------------------------------------------------------------------------- #

class OSpec(NamedTuple):
    d: int             # 3 only: SO(3) is what the quaternions describe
    Nmax: int
    H: float
    Lperp: float
    z_act: float
    dmax: float        # translation half-width
    dtheta: float      # rotation half-angle window; 0.0 freezes orientations
    geom: str
    p_disp: float      # move mix: translate
    p_rot: float       # ... rotate; the rest splits insert/delete
    psi: float
    body: object       # a `mcax.bodies` body
    field: object      # beta V_ext(r), or None
    n_iter: int        # Frank-Wolfe budget per pair
    tol: float
    n_near: int        # neighbours sent to the expensive test; see _any_overlap

    @property
    def sigma(self):
        """The axial width, so that `mcax.geometry` can size a lattice."""
        return self.body.sigma

    @property
    def shape(self):
        """`mcax.geometry` asks specs for their shape when deciding whether a
        cavity is zero-dimensional. A rotating body sweeps its circumsphere, so
        the honest answer for that question is the sphere it fits inside."""
        return None


def make_spec(H, z_act, body = None, Lperp = 10.0, Nmax = None, dmax = 0.15,
              dtheta = 0.35, geom = "slit", p_disp = 0.3, p_rot = 0.3,
              psi = onp.pi / 6.0, field = None, n_iter = 32, tol = 1e-9,
              n_near = 24):
    """Static description of an orientable state point. Three dimensions only.

    `dtheta = 0.0` freezes every orientation where it started, which with an
    aligned initial state reproduces the parallel engine of `mcax.core` and is
    validation rung V1.
    """
    body = bodies.Superball(2.0) if body is None else body
    bodies.check(body)
    geometry.check(geom, 3)
    if not 0.0 <= p_disp + p_rot < 1.0:
        raise ValueError(f"p_disp + p_rot must lie in [0, 1), got "
                         f"{p_disp + p_rot}")
    spec = OSpec(3, 0, float(H), float(Lperp), float(z_act), float(dmax),
                 float(dtheta), str(geom), float(p_disp), float(p_rot),
                 float(psi), body, field, int(n_iter), float(tol),
                 int(n_near))
    if Nmax is None:
        Nmax = int(0.84 * geometry.volume(spec) / bodies.volume(body, 3)) + 64
    return spec._replace(Nmax = int(Nmax))


class OState(NamedTuple):
    pos: np.ndarray        # (C, Nmax, 3)
    quat: np.ndarray       # (C, Nmax, 4) unit quaternions
    alive: np.ndarray      # (C, Nmax)
    key: np.ndarray
    acc: np.ndarray        # (C, 4) accepted: translate, rotate, insert, delete
    tot: np.ndarray        # (C, 4) attempted
    nhi: np.ndarray        # (C,) capacity watchdog


def init_state(spec, C, seed = 0, n0 = 0, aligned = False):
    """`n0` seats that many particles on a melt-in cubic lattice.

    `aligned = True` gives every particle the identity orientation, which with
    `dtheta = 0` is the frozen-rotation mode that has to reproduce the aligned
    engine. Otherwise orientations start Haar-uniform, as they must for the
    burn-in not to have to unwind a correlated start.
    """
    require_float64()
    from .core import lattice_fill
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    keys = jax.random.split(k0, C)
    pos = np.zeros((C, spec.Nmax, 3))
    alive = np.zeros((C, spec.Nmax), dtype = bool)
    if aligned:
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (C, spec.Nmax, 1))
    else:
        quat = q_random(k1, (C, spec.Nmax))
    if n0 > 0:
        sites = onp.stack([lattice_fill(spec, n0, seed + c) for c in range(C)])
        pos = pos.at[:, :sites.shape[1]].set(np.asarray(sites))
        alive = alive.at[:, :sites.shape[1]].set(True)
    return OState(pos = pos, quat = quat, alive = alive, key = keys,
                  acc = np.zeros((C, 4), dtype = np.int64),
                  tot = np.zeros((C, 4), dtype = np.int64),
                  nhi = np.sum(alive, axis = 1).astype(np.int64))


# --------------------------------------------------------------------------- #
#  The kernel                                                                  #
# --------------------------------------------------------------------------- #

def _any_overlap(spec, pos, quat, alive, trial_r, trial_q, skip_idx):
    """Does `trial` overlap any live particle other than `skip_idx`.

    **Only the nearest `n_near` neighbours reach the expensive test.** Sending
    every capacity slot through it, as the aligned engine sends every slot
    through its p-norm, is unaffordable here: the certificate search costs some
    sixty support evaluations per pair, so a chain with four hundred slots pays
    twenty-five thousand of them per trial move and a validation run that
    should take a minute takes hours. A data-dependent skip cannot be expressed
    under `vmap`, but a fixed-size GATHER can, and `top_k` is exactly that.

    The gather is made safe rather than assumed safe. Taking the `n_near`
    closest live particles could in principle miss an overlapping one, so the
    (n_near + 1)-th distance is checked against twice the circumradius: if even
    that one is close enough to have mattered, the gather might have dropped a
    genuine overlap and the move is rejected. That is the same guarantee
    direction as the budget itself, and it makes `n_near` a performance knob
    that cannot change the physics, only the acceptance.

    The certain-overlap tier is applied to ALL slots, not just the gathered
    ones: it is a single squared distance and skipping it would be a false
    economy that also lost a guarantee.
    """
    dr = geometry.min_image(spec, pos - trial_r[None, :])
    d2 = np.sum(dr * dr, axis = -1)                     # (Nmax,)
    live = alive.at[skip_idx].set(False)
    r_in = bodies.inradius(spec.body, 3)
    r_out = bodies.circumradius(spec.body, 3)
    sure = np.any(live & (d2 < (2.0 * r_in) ** 2))

    d2m = np.where(live, d2, np.inf)
    k = min(spec.n_near + 1, spec.Nmax)
    near = jax.lax.top_k(-d2m, k)[1]                    # (k,) closest first
    idx = near[:spec.n_near]
    # The one just outside the gather. If it is still within reach, the gather
    # was too small for this configuration and the answer is "overlap".
    truncated = d2m[near[-1]] < (2.0 * r_out) ** 2

    Ra = q_matrix(trial_q)
    Rb = q_matrix(quat[idx])
    hit = jax.vmap(lambda v, R: overlaps_pair(
        spec.body, v, Ra, R, spec.n_iter, spec.tol))(dr[idx], Rb)
    return sure | truncated | np.any(np.take(live, idx) & hit)


def _step_chain(spec, carry, _):
    pos, quat, alive, key, acc, tot, nhi = carry
    key, k_type, k_pick, k_move, k_rot, k_q, k_acc = jax.random.split(key, 7)
    N = np.sum(alive)
    V = geometry.bbox_volume(spec)
    lo, hi = (np.asarray(x) for x in geometry.bbox(spec))
    u_type = jax.random.uniform(k_type)
    p_r = spec.p_disp + spec.p_rot
    p_i = p_r + 0.5 * (1.0 - p_r)
    move = np.where(u_type < spec.p_disp, 0,
                    np.where(u_type < p_r, 1, np.where(u_type < p_i, 2, 3)))

    idx = _pick_alive(k_pick, alive)

    # -- translation -------------------------------------------------------- #
    delta = spec.dmax * jax.random.uniform(k_move, (3,), minval = -1.0,
                                           maxval = 1.0)
    r_new = geometry.wrap(spec, pos[idx] + delta)
    v_old = fields.evaluate(spec.field, pos[idx])
    v_new = fields.evaluate(spec.field, r_new)
    over_t = _any_overlap(spec, pos, quat, alive, r_new, quat[idx], idx)
    t_ok = alive[idx] & geometry.contains(spec, r_new) & ~over_t & \
        (jax.random.uniform(k_acc) < np.exp(-(v_new - v_old)))
    pos_t = pos.at[idx].set(np.where(t_ok, r_new, pos[idx]))

    # -- rotation ----------------------------------------------------------- #
    # The position does not move, so neither the field nor the walls enter:
    # this is a pure hard-body accept/reject, and at dtheta = 0 the proposal is
    # the identity and it always accepts.
    q_new = q_perturb(k_rot, quat[idx], spec.dtheta)
    over_r = _any_overlap(spec, pos, quat, alive, pos[idx], q_new, idx)
    r_ok = alive[idx] & ~over_r
    quat_r = quat.at[idx].set(np.where(r_ok, q_new, quat[idx]))

    # -- insertion ---------------------------------------------------------- #
    slot = np.argmax(~alive)
    has_slot = ~np.all(alive)
    r_ins = jax.random.uniform(k_move, (3,), minval = lo, maxval = hi)
    q_ins = q_random(k_q)
    over_i = _any_overlap(spec, pos, quat, alive, r_ins, q_ins, spec.Nmax)
    # No orientational factor: the proposal draws from the Haar measure the
    # ensemble is defined against, so it cancels exactly, and deletion picks
    # uniformly among particles which is the same measure read backwards.
    a_ins = spec.z_act * V / (N + 1.0) * \
        np.exp(-fields.evaluate(spec.field, r_ins))
    i_ok = has_slot & geometry.contains(spec, r_ins) & ~over_i & \
        (jax.random.uniform(k_acc) < np.minimum(1.0, a_ins))
    pos_i = pos.at[slot].set(np.where(i_ok, r_ins, pos[slot]))
    quat_i = quat.at[slot].set(np.where(i_ok, q_ins, quat[slot]))
    alive_i = alive.at[slot].set(np.where(i_ok, True, alive[slot]))

    # -- deletion ----------------------------------------------------------- #
    a_del = N / np.maximum(spec.z_act * V, 1e-300) * np.exp(v_old)
    d_ok = (N > 0) & (jax.random.uniform(k_acc) < np.minimum(1.0, a_del))
    alive_d = alive.at[idx].set(np.where(d_ok, False, alive[idx]))

    # -- select ------------------------------------------------------------- #
    pos_out = np.where(move == 0, pos_t, np.where(move == 2, pos_i, pos))
    quat_out = np.where(move == 1, quat_r, np.where(move == 2, quat_i, quat))
    alive_out = np.where(move == 2, alive_i,
                         np.where(move == 3, alive_d, alive))
    accepted = np.where(move == 0, t_ok,
                        np.where(move == 1, r_ok,
                                 np.where(move == 2, i_ok, d_ok)))
    acc = acc.at[move].add(accepted.astype(np.int64))
    tot = tot.at[move].add(1)
    nhi = np.maximum(nhi, np.sum(alive_out))
    return (pos_out, quat_out, alive_out, key, acc, tot, nhi), None


def _pick_alive(key, alive):
    u = jax.random.uniform(key, alive.shape)
    return np.argmax(np.where(alive, u, -1.0))


def _hist_chain(spec, pos, alive, nbins):
    zc = geometry.profile_coord(spec, pos)
    i = np.clip((zc / geometry.extent(spec) * nbins).astype(np.int32),
                0, nbins - 1)
    return np.zeros(nbins).at[i].add(alive.astype(np.float64))


def _cubic_chain(spec, pos, quat, alive, nbins):
    """(S4, S6) histograms binned along the profile coordinate.

    The superball has the full cubic symmetry O_h, so the orientational
    distribution is a function on SO(3)/O and a raw histogram of quaternions
    would be twenty-four-fold redundant and unreadable. What is wanted instead
    are invariants of the group, and against a WALL the lab z-axis picks the
    frame: write c = Omega^T zhat for the wall normal expressed in the body
    frame, so that sum c_a^2 = 1, and the first two non-trivial cubic
    invariants are

        K4 = c1^4 + c2^4 + c3^4          isotropic average 3/5
        K6 = c1^2 c2^2 c3^2              isotropic average 1/105

    Normalised here so that both vanish for an isotropic distribution:

        S4 = (5/2)(K4 - 3/5)   = +1 with a body AXIS along the normal,
                                 -2/3 with a body DIAGONAL along it
        S6 = (105 K6 - 1)/2

    S4 is the one to read. It is the l = 4 cubic harmonic and the lowest order
    at which a cubic body can express itself at all: everything below l = 4 is
    identically zero by symmetry, which is why a nematic order parameter finds
    nothing here however ordered the fluid is.
    """
    c = np.einsum("nij,j->ni", q_matrix(quat), np.array([0.0, 0.0, 1.0]))
    c2 = c * c
    K4 = np.sum(c2 * c2, axis = -1)
    K6 = c2[:, 0] * c2[:, 1] * c2[:, 2]
    S4 = 2.5 * (K4 - 0.6)
    S6 = 0.5 * (105.0 * K6 - 1.0)
    zc = geometry.profile_coord(spec, pos)
    i = np.clip((zc / geometry.extent(spec) * nbins).astype(np.int32),
                0, nbins - 1)
    w = alive.astype(np.float64)
    return (np.zeros(nbins).at[i].add(w * S4),
            np.zeros(nbins).at[i].add(w * S6))


class OAccum(NamedTuple):
    hist: np.ndarray       # (C, nbins)
    Ns: np.ndarray         # (C, nchunks)
    s4: np.ndarray         # (C, nbins) sum of S4 over particles and draws
    s6: np.ndarray         # (C, nbins)


@partial(jax.jit, static_argnums = (0, 2, 3, 4))
def run(spec, state, nsteps, thin, nbins):
    """Advance every chain `nsteps`, accumulating at each thinning boundary."""
    require_float64()

    def chunk(carry, _):
        def inner(c, _):
            return jax.vmap(lambda *a: _step_chain(spec, a, None)[0])(*c), None
        st, _ = jax.lax.scan(inner, carry, None, length = thin)
        pos, quat, alive, key, acc, tot, nhi = st
        hh = jax.vmap(lambda p, a: _hist_chain(spec, p, a, nbins))(pos, alive)
        s4, s6 = jax.vmap(lambda p, q, a: _cubic_chain(
            spec, p, q, a, nbins))(pos, quat, alive)
        return st, (hh, np.sum(alive, axis = 1), s4, s6)

    carry = (state.pos, state.quat, state.alive, state.key, state.acc,
             state.tot, state.nhi)
    carry, (hists, Ns, s4, s6) = jax.lax.scan(chunk, carry, None,
                                              length = nsteps // thin)
    return OState(*carry), OAccum(hist = np.sum(hists, axis = 0),
                                  Ns = np.swapaxes(Ns, 0, 1),
                                  s4 = np.sum(s4, axis = 0),
                                  s6 = np.sum(s6, axis = 0))


class OResult(NamedTuple):
    z: onp.ndarray
    rho: onp.ndarray
    s4: onp.ndarray        # (nbins,) <S4>(z), per particle
    s6: onp.ndarray
    Ns: onp.ndarray
    acc: onp.ndarray       # (4,) translate, rotate, insert, delete
    n_mean: float
    n_hi: int
    capacity: int
    saturation: float
    state: OState
    hist: onp.ndarray
    nsamples: int

    @property
    def capacity_warning(self):
        if self.saturation >= 1.0:
            return (f"capacity REACHED: some chain held all {self.capacity} "
                    f"slots, so the ensemble is truncated at N = Nmax and "
                    f"detailed balance is broken. Re-run with a larger Nmax.")
        if self.saturation >= 0.95:
            return (f"capacity {100 * self.saturation:.1f}% used "
                    f"({self.n_hi} of {self.capacity}). Raise Nmax.")
        return None


def burn_and_sample(spec, C, seed, n_burn, n_run, thin, nbins, n0 = 0,
                    aligned = False):
    """Burn in, then sample. The orientable counterpart of
    `mcax.core.burn_and_sample`, with the same (chain, draw) output layout."""
    st = init_state(spec, C, seed, n0 = n0, aligned = aligned)
    st, _ = run(spec, st, n_burn, max(n_burn // 4, 1), nbins)
    st = st._replace(acc = np.zeros_like(st.acc), tot = np.zeros_like(st.tot))
    st, ac = run(spec, st, n_run, thin, nbins)
    ns = n_run // thin
    hist = onp.asarray(ac.hist)
    dz = geometry.extent(spec) / nbins
    cell = geometry.bin_volume(spec, nbins)
    rho = hist.sum(axis = 0) / (hist.shape[0] * ns * cell)
    # The cubic moments are PER PARTICLE, so they divide by the occupancy of
    # each bin and not by its volume. A bin nobody ever visited is nan rather
    # than zero: zero is a real and different statement (isotropic there) and
    # the two must not be confused in a profile that gets plotted.
    occ = hist.sum(axis = 0)
    with onp.errstate(invalid = "ignore", divide = "ignore"):
        s4 = onp.where(occ > 0, onp.asarray(ac.s4).sum(axis = 0) / occ, onp.nan)
        s6 = onp.where(occ > 0, onp.asarray(ac.s6).sum(axis = 0) / occ, onp.nan)
    Ns = onp.asarray(ac.Ns)
    n_hi = int(onp.max(onp.asarray(st.nhi)))
    return OResult(
        z = (onp.arange(nbins) + 0.5) * dz, rho = rho, s4 = s4, s6 = s6,
        Ns = Ns,
        acc = onp.asarray(st.acc).sum(0) / onp.maximum(
            onp.asarray(st.tot).sum(0), 1),
        n_mean = float(Ns.mean()), n_hi = n_hi, capacity = spec.Nmax,
        saturation = n_hi / spec.Nmax, state = st, hist = hist, nsamples = ns)


# --------------------------------------------------------------------------- #
#  Ordering monitors                                                           #
# --------------------------------------------------------------------------- #

def cubatic(state, chain = None):
    """Global cubatic order parameter <S4> about each lab axis.

    Returns (3,) — one per axis — because a cubatic phase picks a frame and
    only aligns with the lab if something (a wall, the periodic cell) has
    already broken the symmetry. In a confined run read the wall-normal entry;
    in bulk read the largest, and treat a large value with the caution due to a
    maximum taken over three correlated numbers.

    This is the monitor the plastic crystal needs. Near p = 2 the free superball
    fluid freezes into a ROTATOR phase, positions ordered and orientations
    still free, so a translational monitor alone reports order where there is
    orientational disorder, and this one reports the reverse. Both are needed
    and neither implies the other.
    """
    R = onp.asarray(q_matrix(state.quat))               # (C, Nmax, 3, 3)
    alive = onp.asarray(state.alive)[..., None]         # (C, Nmax, 1)
    c2 = R ** 2                                          # rows: lab axes in body
    K4 = onp.sum(c2 ** 2, axis = -1)                     # (C, Nmax, 3)
    S4 = 2.5 * (K4 - 0.6)
    n = onp.maximum(alive.sum(axis = (0, 1)), 1)
    return (S4 * alive).sum(axis = (0, 1)) / n


def steinhardt(state, rcut, l = 6, chain = 0, max_bonds = 4000, seed = 0):
    """Global Steinhardt Q_l of the bond network, for one chain.

    Computed through the addition theorem rather than through spherical
    harmonics:

        Q_l^2 = (4 pi / (2l+1)) sum_m |<Y_lm>|^2 = < P_l(cos theta_ij) >

    averaged over ORDERED PAIRS of bonds, because summing Y_lm Y*_lm over m
    collapses to a Legendre polynomial of the angle between the two bonds. That
    removes every spherical harmonic from the implementation and leaves one
    polynomial, which is both shorter and harder to get wrong.

    Reference values: about 0 for a liquid (falling as 1/sqrt(N_bonds)), 0.575
    for fcc, 0.511 for bcc and 0.354 for simple cubic at l = 6.
    """
    pos = onp.asarray(state.pos[chain])[onp.asarray(state.alive[chain])]
    n = len(pos)
    if n < 2:
        return float("nan")
    dr = pos[:, None, :] - pos[None, :, :]
    d = onp.sqrt(onp.sum(dr ** 2, axis = -1))
    i, j = onp.where((d < rcut) & (d > 0))
    if len(i) < 2:
        return float("nan")
    b = dr[i, j] / d[i, j][:, None]
    if len(b) > max_bonds:                  # the pair sum is quadratic in bonds
        b = b[onp.random.default_rng(seed).choice(len(b), max_bonds, False)]
    x = onp.clip(b @ b.T, -1.0, 1.0)
    P = {4: lambda t: (35 * t ** 4 - 30 * t ** 2 + 3) / 8.0,
         6: lambda t: (231 * t ** 6 - 315 * t ** 4 + 105 * t ** 2 - 5) / 16.0}[l]
    return float(onp.sqrt(max(P(x).mean(), 0.0)))
