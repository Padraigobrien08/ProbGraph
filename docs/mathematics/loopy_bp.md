# P13 — Loopy belief propagation

> Run the sum-product messages of P12 on the **factor graph** itself, instead of a clique tree.
> If the factor graph has no cycles, this is exact. If it has cycles, messages travel round the
> loops and evidence is counted more than once. The iteration may converge to wrong beliefs,
> converge slowly, or never converge. Damping changes the dynamics but not the fixed points.
> The exact junction tree (P12) is the oracle that measures all of this.

Prerequisites: P10 (Markov networks), P12 (messages are partial eliminations), P9 (log space).

---

## 1. The factor graph and its messages

A Markov network's **factor graph** is bipartite. It has one node per variable, one node per
factor with two or more variables, and an edge $x$–$f$ when $x\in\mathrm{scope}(f)$.
Single-variable factors are absorbed into their variable as a **local potential** $u_x$
(the product of all of them). Constant factors, with empty scope, are ignored, since they only
rescale.

At iteration $t$, every message is recomputed from the previous iteration's messages (a
**synchronous**, or flooding, schedule):
$$\mu^{(t)}_{x\to f}(x)\propto u_x(x)\prod_{g\ni x,\,g\ne f}\nu^{(t-1)}_{g\to x}(x),\qquad
\nu^{(t)}_{f\to x}(x)\propto\sum_{\mathrm{scope}(f)\setminus x}f\prod_{y\in\mathrm{scope}(f)\setminus x}\mu^{(t)}_{y\to f}(y).$$
Each message is normalised to sum to 1. This is harmless, because beliefs are normalised anyway,
and it keeps the numbers bounded. The initial messages are uniform. Beliefs are
$b_x\propto u_x\prod_{f\ni x}\nu_{f\to x}$. Everything is computed in log space.

## 2. On a tree: exact, in at most diameter-many iterations

**Theorem 1.** If the factor graph is a tree (more generally, a forest), then after at most $D$
iterations, where $D$ is its diameter, the messages stop changing, and $b_x=P(x)$ (or $P(x\mid e)$)
exactly.

*Proof sketch.* A message $\nu_{f\to x}$ depends only on the subtree behind it. By induction on
the height $h$ of that subtree, from iteration $h$ onwards the message equals the partial
elimination of Lemma 2 in P12: the product of all potentials behind it with everything except
$x$ summed out. The initial values are forgotten once every leaf's message has propagated
through. When all messages are exact, $b_x$ is the exact marginal by Theorem 3 of P12. The tree
version is a clique tree whose clusters are the factor scopes and single variables. $\square$

**Consequence.** The fixed point on a tree is unique, whatever the initial messages. The tests
check this with random initial messages (`run(seed=...)`).

## 3. On graphs with cycles

On a cycle, the assumption behind Lemma 2 of P12 fails: the two sides of an edge are no longer
separate. A message eventually returns to where it started, carrying that node's own evidence
back to it as if it were independent new evidence. Three behaviours occur, and the tests
demonstrate each one with the exact junction tree as the oracle:

| Behaviour | Example in the tests |
|---|---|
| Converges, but to badly wrong beliefs | the misconception 4-cycle (F2): about 210 undamped iterations, then $P(A{=}1)\approx0.43$ against an exact $\approx0.18$ (maximum error about 0.25) |
| Converges very slowly | a 4-cycle with strong couplings and conflicting evidence at opposite corners: about 4,000 undamped iterations, 62 damped |
| **Never converges** (a period-2 oscillation) | a 3×3 grid with strong mixed couplings: undamped beliefs flip between exactly 0 and 1 on every iteration indefinitely; damping 0.5 converges in 52 iterations, but to beliefs that are **confidently wrong** (maximum error 0.446) |

The last row matters most. **Convergence of loopy BP is not evidence of accuracy.** A damped
run can report convergence with a fixed point far from the truth.

For positive potentials, fixed points always exist (by Brouwer's theorem, since messages live in
a product of simplices). Yedidia, Freeman & Weiss (2001) showed they are exactly the stationary
points of the Bethe free energy. This is stated with citations; the library does not rely on it.

## 4. Damping

With damping $\alpha\in[0,1)$, the factor-to-variable messages are updated as the convex
combination, in **probability** space,
$$\nu^{(t)}\leftarrow(1-\alpha)\,\nu^{(t)}_{\text{new}}+\alpha\,\nu^{(t-1)}.$$
In log space this is a log-sum-exp of two terms.

**Proposition 2 (damping keeps the fixed points).** Let $F$ be the undamped update, and $G_\alpha$
the damped one, with $G_\alpha(\nu)=(1-\alpha)F(\nu)+\alpha\nu$. For $\alpha<1$, $G_\alpha(\nu)=\nu$
if and only if $F(\nu)=\nu$.

*Proof.* $G_\alpha(\nu)-\nu=(1-\alpha)(F(\nu)-\nu)$, and $1-\alpha\neq0$. $\square$

So damping can only change **whether, and how fast**, a fixed point is reached, never **which
beliefs** a fixed point represents. In particular, on trees damping still gives exact answers.

## 5. Convergence criterion (B4)

`run` reports `converged=True` only if the largest absolute change in any normalised
factor-to-variable message, measured in probability space, is below `tolerance` (default
$10^{-10}$). `result.residual` is that largest change at the final iteration, whether or not the
run converged. A run that stops at `max_iterations` reports `converged=False`. Its beliefs are
whatever the last iteration produced, and they must not be trusted.
