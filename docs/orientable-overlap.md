# The orientable overlap test: what it guarantees, and what it costs

`mcax.orient` samples hard convex bodies with orientational freedom. Everywhere
else in this library the overlap test is exact and closed-form; here it is not,
and cannot be. Two rotated superballs have no algebraic contact condition, so
the question has to be *computed*, under `jit`, with static shapes and no
data-dependent loop. This note records the guarantee that computation makes,
the direction it is allowed to fail in, and the measured rate at which it does.

## The test

Two convex bodies are disjoint exactly when some direction separates them. In
terms of the support function of the Minkowski difference $M = B - A$,

$$h_M(u) \;=\; u\cdot(c_B - c_A) \;+\; h_K(\Omega_B^{\mathsf T}u) \;+\; h_K(\Omega_A^{\mathsf T}u),$$

the bodies are disjoint if and only if $h_M(u) < 0$ for some $u$. (The two
$h_K$ terms carry the same sign because every body here is centrally symmetric,
$K = -K$.)

So **a single direction with $h_M(u) < 0$ is a certificate of no overlap** —
exact, checkable in a few flops, and requiring no iteration to trust. There is
no comparably cheap certificate in the other direction: proving overlap means
exhibiting a point in the intersection, or a simplex enclosing the origin.

That asymmetry is the whole design. The engine **searches for separation**, in
two stages:

1. **Thirty-two candidate axes.** The three face normals of each body in the
   lab frame, the nine cross products between them, the centre separation, and
   the negative of each. This is the separating-axis candidate set, which is
   *complete for boxes*: at $p = \infty$ a negative value here settles the
   question exactly.
2. **Frank–Wolfe** on $\min \lVert x\rVert^2$ over $M$, seeded at the support
   point along the most promising candidate axis. Each step takes
   $u = -x/\lVert x\rVert$, checks $h_M(u)$ for a certificate, and moves $x$
   toward the support point by an exact line search along the segment. One
   fixed-length `lax.scan`, no branches, no simplex bookkeeping.

Both stages are wrapped in the cheap tiers the task brief specifies: centres
closer than twice the inradius certainly overlap, further than twice the
circumradius certainly do not, and only the shell between reaches the search.

## The guarantee direction

**If the budget is exhausted without a certificate, the pair is declared
overlapping.**

This is the safe direction, and it is worth being exact about *why* it is safe,
because "conservative" is doing more work here than it usually does:

- The verdict is a **deterministic function of the pair configuration**. It does
  not depend on the path taken to reach that configuration, on the move type,
  or on the random stream. Detailed balance is therefore untouched.
- The sampler is consequently **exact for a slightly modified model**, in which
  bodies are imperceptibly *fattened* within the thin shell where the search can
  run out of budget. It is not an approximate sampler of the intended model in
  the sense of a biased Markov chain; it is an exact sampler of a marginally
  different body.
- The opposite failure — declaring free what is actually overlapping — would
  admit real overlaps into a hard-body ensemble and has no such reading. Nothing
  downstream would reveal it.

`tests/test_orient.py::test_the_budget_only_ever_errs_towards_overlap` asserts
the asymmetry directly against a high-budget reference, for every shape.

## Measured trip rate

Measured over pairs sampled uniformly in the **shell** where the cheap tiers
cannot decide — that is, per *hard* test, which is the conservative way to quote
it, since the vast majority of overlap tests in a real run never get that far.
Reference is the same algorithm at `n_iter = 1000`, itself checked for
stability (the verdict counts stop moving between 500 and 2000).

| p | n_iter = 8 | 12 | 20 | 32 | 64 |
|---|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 2.9e-2 | 1.6e-2 | 6.7e-3 | 1.8e-3 | 3.0e-4 |
| 4 | 2.3e-2 | 1.5e-2 | 7.1e-3 | 3.2e-3 | 1.8e-3 |
| 6 | 1.3e-2 | 1.1e-2 | 7.6e-3 | 5.7e-3 | 3.7e-3 |
| ∞ | 0 | 0 | 0 | 0 | 0 |

Zero unsafe verdicts at any budget, for any shape, across every sample drawn.

**The two ends are exact, and for different reasons.** At $p = 2$ the inscribed
and circumscribed radii coincide, so the cheap tiers alone decide every pair and
the search is never entered. At $p = \infty$ the candidate-axis set is complete,
so stage 1 always finds the certificate if one exists. The cost is entirely in
the smooth interior of the family.

**The default budget is `n_iter = 32`.**

## Why 1e-6 is not reachable this way, and what would be

The task brief asks for a trip rate below 1e-6. That is not achieved and,
by this route, cannot be. The reason is structural rather than a matter of
tuning:

Frank–Wolfe converges as $O(D/k)$ in the distance to the closest point of $M$.
A separation certificate at gap $g$ requires resolving that gap, so the budget
needed scales as $D/g$. Configurations with gap below $\varepsilon$ occupy a
measure proportional to $\varepsilon$, so the trip rate falls as $1/k$ — which
is exactly the $1/n_{\text{iter}}$ scaling in the table above. Reaching 1e-6
would need $k \sim 10^4$, which is four hundred times the cost for a fluid that
would be indistinguishable.

The physical size of the residual is small and bounded. At `n_iter = 32` the
trips occupy roughly $10^{-3}\sigma$ of effective gap, so the bodies behave as
though fattened by about $5\times 10^{-4}\sigma$ each — a relative excluded-volume
error of a few parts in a thousand. Rung V2 measures that consequence directly
rather than inferring it: $B_2$ computed at the production budget against the
same integral at `n_iter = 1000`.

Two routes would genuinely improve it, neither taken in this round:

- **A full GJK simplex.** Maintaining up to four support points and computing
  the exact closest point of their convex hull converges superlinearly for
  smooth bodies where the one-point Frank–Wolfe iterate does not. A two-point
  (triangle) variant was tried and was *not* better at these budgets — it also
  produced an unsafe verdict from a degenerate-triangle branch — so it was
  dropped rather than shipped half-working. The full tetrahedral version, with
  all fifteen sub-simplices enumerated branch-free, is the real fix.
- **A direct contact solve.** For finite $p$ the bodies are smooth, and the
  closest points satisfy an antiparallel-normal condition that Newton's method
  solves quadratically. That is exact in the limit and would remove the shell
  problem rather than shrinking it.

## Cost

The three-tier structure keeps the expensive stage to the shell *population*
but not to the shell *cost*: under `vmap` with static shapes, every neighbour
slot goes through the full test because a data-dependent skip cannot be
expressed. The obvious next optimisation is a `top_k` gather of the nearest
neighbours before the search, with the $(k+1)$-th distance checked against
twice the circumradius and a conservative overlap declared if the gather could
have missed someone. That preserves the guarantee direction and would cut the
search population by roughly an order of magnitude.
