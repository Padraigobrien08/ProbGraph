# P24 — Gibbs sampling

> To sample from a joint distribution that is too large to tabulate, change one variable at a
> time: draw it from its distribution given all the others. That **full conditional** involves
> only the factors that mention the variable, that is, its **Markov blanket**, so each step costs a
> few table lookups whatever the size of the model. P23 (M6.2) proves that the resulting chain has
> the target as its stationary distribution. This note derives the local computation, fixes the
> conventions, and shows how determinism breaks the method.

Prerequisites: P1 (factorisation), P7 (d-separation), P10 (Markov networks), P18 (Markov chains).

---

## 1. The target

The target is the posterior $\pi(x)=P(x\mid e)$ over the unobserved variables $x$, given evidence $e$.
Both model classes write the unnormalised joint as a product of factors:
$$\tilde\pi(x)=\prod_k\phi_k(x_{S_k},e),$$
with one factor per CPD for a Bayesian network (P5), or the potentials of a Markov network (P10). Each
is reduced by the evidence. The normaliser $P(e)$, or $Z\cdot P(e)$, never needs to be computed.

## 2. The full conditional is local (G1)

**Proposition 1.** For an unobserved variable $X$,
$$P(X=x\mid x_{-X},e)=\frac{\prod_{k:\,X\in S_k}\phi_k(x,x_{-X})}{\sum_{x'}\prod_{k:\,X\in S_k}\phi_k(x',x_{-X})}.$$

*Proof.* $P(X=x\mid x_{-X},e)=\tilde\pi(x,x_{-X})/\sum_{x'}\tilde\pi(x',x_{-X})$. Every factor that does not
mention $X$ has the same value in the numerator and in every term of the denominator, so it cancels.
$\square$

The variables that appear in the remaining factors, other than $X$, form the **Markov blanket**
$\mathrm{MB}(X)$:

- **Markov network:** the neighbours of $X$ in the graph.
- **Bayesian network:** the factors containing $X$ are its own CPD and its children's CPDs, so
  $$P(X\mid x_{-X},e)\propto P(X\mid\mathrm{pa}(X))\prod_{C\in\mathrm{ch}(X)}P(C\mid\mathrm{pa}(C)),$$
  and $\mathrm{MB}(X)$ consists of the parents, the children, and the children's other parents.

**Corollary.** $X$ is independent of all other variables given $\mathrm{MB}(X)$. In a Bayesian network,
$\mathrm{MB}(X)$ d-separates $X$ from the rest (P7). Every trail from $X$ leaves through a parent (a chain
or fork, blocked), through a child to that child's descendants (a chain, blocked at the child), or
through a child to a co-parent (the child is an observed collider, so the trail continues, but it is
blocked at the co-parent).

The library works with log factors (P9). The conditional is a softmax of a sum of a few table entries,
so even a full conditional built from extreme CPDs cannot underflow to $0/0$.

## 3. One step, one sweep

Given the current state $x$:

- **Systematic scan.** Update each unobserved variable once, in the model's declaration order, each
  draw using the latest values of the others. One recorded sample = one sweep.
- **Random scan.** Pick one unobserved variable uniformly at random and update it. One recorded
  sample = one such step.

So `run(n)` applies the chosen kernel $n$ times after `burn_in` applications, and keeps every
`thin`-th state. In P23 these are a composition of kernels and a mixture of kernels, respectively.
Both leave $\pi$ invariant.

Each draw uses one uniform $u$ and the inverse CDF of the conditional, as in P3. The random scan uses
one more uniform to pick the variable. The random stream then depends only on the number of kernel
applications, so a run with `burn_in` $b$ and `thin` $k$ reproduces exactly the states $b+k,\,b+2k,\ldots$
of a plain run with the same seed (G4).

## 4. Initialisation

The chain must start in a state with $\tilde\pi(x)>0$, or every conditional may be undefined. For a
Bayesian network, the sampler draws a state by forward sampling with the evidence clamped (P3, as in
likelihood weighting, P8). That state has $P(x)>0$ for the sampled part, and is accepted if
$\tilde\pi(x)>0$. For a Markov network, it draws each variable uniformly. Either way it retries up to
1,000 times (spec ⚑4), and raises `ZeroProbabilityEvidenceError` if it never succeeds. An explicit
`initial` state is checked the same way.

## 5. Determinism breaks Gibbs (G5, fixture F3)

If $Y=X$ with probability 1, then from $(0,0)$ the conditional of $X$ given $Y=0$ is a point mass at 0,
and the conditional of $Y$ given $X=0$ is a point mass at 0: the chain never leaves $(0,0)$. The same
holds for $(1,1)$. The chain is **reducible**. Its stationary distributions include $\pi$, but also the
point masses, and a single run converges to whichever class it starts in. Its estimate of
$P(X{=}1)$ is then 0 or 1, never the true $\frac12$, and it looks perfectly converged. P25 shows how
several chains expose this, and P27 shows how blocking cures it.

## 6. How the tests check this independently

- **Proposition 1:** `full_conditional` against the conditional computed from the full joint table, on
  random Bayesian and Markov networks with evidence.
- **Locality:** changing any variable outside the Markov blanket leaves the conditional unchanged, and the
  blanket d-separates $X$ from the rest (M2's `d_separated`).
- **Streams:** reproducibility; `burn_in` and `thin` equal slicing a longer run; evidence never changes.
- **F3:** the chain does not move; impossible evidence and impossible initial states raise.
