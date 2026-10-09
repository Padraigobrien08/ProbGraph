# P10 — Markov networks

> A **Markov network** is a product of nonnegative factors, normalised by the **partition
> function** $Z$. Its graph connects variables that share a factor. Separation in that graph
> implies conditional independence: this is M2's Lemma 3, reused unchanged. Converting a Bayesian
> network to a Markov network (moralisation) keeps the distribution but **loses** the
> independences that depend on colliders. In the other direction, some undirected independence
> structures, such as the 4-cycle, cannot be captured by any DAG.

Prerequisites: P2 (joint normalisation), P5 (factors), P7 (d-separation, Lemma 3 and Theorem 2),
P9 (log space), and `elimination_orders.md` (moral and interaction graphs).

---

## 1. The Gibbs distribution

Given factors $\phi_1,\ldots,\phi_m$ with scopes $C_1,\ldots,C_m$ over variables $V$:
$$\tilde P(x)=\prod_{k=1}^m\phi_k(x_{C_k}),\qquad Z=\sum_{x\in\mathcal X_V}\tilde P(x),\qquad P(x)=\tilde P(x)/Z.$$

- $P$ is a probability distribution exactly when $0<Z<\infty$. Finite inputs always give a
  finite $Z$. If every assignment has $\tilde P=0$, then $Z=0$ and the model is rejected (M2).
- A variable that appears in no factor is given the all-ones factor, so it is uniform and
  independent of everything else. This keeps the scope of every query well defined.
- The **graph** $H$ is the interaction graph (`elimination_orders.md` §1): $u$–$v$ is an edge
  exactly when $u$ and $v$ share a factor. By construction, every scope $C_k$ is a clique of $H$.

**Computing $Z$.** $Z$ is the sum over *every* variable, so it is variable elimination with the
query empty (P6). In probability space, $Z$ overflows easily: 200 factors whose entries are
around 100 give $Z\approx10^{400}$. So it is computed in log space (P9): $\log Z$ is the
log-sum-exp of the eliminated product. `partition_function()` raises rather than returning
`inf` when $Z$ is beyond float64.

**Queries.** $P(Q\mid e)\propto\sum_{\text{rest}}\prod_k\phi_k[e]$, which is variable elimination
on the reduced factors (P6), normalised. Also $P(e)=Z(e)/Z$, where $Z(e)$ is the same sum
restricted to $x_E=e$. In log space, $\log P(e)=\log Z(e)-\log Z$.

## 2. The global Markov property

**Theorem 1.** If $Z$ separates $X$ from $Y$ in $H$, then $X\perp Y\mid Z$ under $P$.

*Proof.* $P(x)=\frac1Z\prod_k\phi_k(x_{C_k})$, where each $C_k$ is a clique of $H$. Absorb $1/Z$
into $\phi_1$. This is exactly the hypothesis of M2's Lemma 3 (P7 §6), whose conclusion is the
statement. $\square$

**The converse, for positive distributions (Hammersley–Clifford).** If $P(x)>0$ for every $x$,
and $P$ satisfies the global Markov property for $H$, then $P$ factorises over the cliques of
$H$ (Hammersley & Clifford, 1971; Besag, 1974). *Sketch:* write
$\log P(x)=\sum_{S\subseteq V}\psi_S(x_S)$ by Möbius inversion around a reference assignment.
Every $\psi_S$ whose $S$ contains two non-adjacent vertices vanishes, by their conditional
independence given the rest. So only cliques remain. Positivity is essential: Moussouris (1974)
gives a distribution with zeros that satisfies the Markov property but does not factorise. This
is stated with a citation, not proved here, and the library does not rely on it.

## 3. From a Bayesian network: moralisation

`BayesianNetwork.to_markov_network()` uses the CPD factors unchanged.

**Proposition 2.** The resulting Markov network has:
1. graph $H=\mathcal M(G)$, the moral graph;
2. $Z=1$;
3. exactly the joint distribution of the Bayesian network.

*Proof.* (1) is Proposition 1 of `elimination_orders.md`. (2) and (3): $\tilde P$ is the product
of the CPDs, which is the joint $P$ by P1, and $\sum_xP(x)=1$ by P2. $\square$

**Proposition 3 (moralisation never invents an independence).** If $Z$ separates $X$ from $Y$ in
$\mathcal M(G)$, then $X\perp_G Y\mid Z$.

*Proof.* Let $A=\mathrm{An}^*(X\cup Y\cup Z)$. Every edge of $\mathcal M(G[A])$ is an edge of
$\mathcal M(G)$, because a marriage inside $G[A]$ is a marriage in $G$. So any $X$–$Y$ path avoiding $Z$ in
$\mathcal M(G[A])$ would also be one in $\mathcal M(G)$. There is none, so $Z$ separates $X$ and $Y$
in $\mathcal M(G[A])$, which is d-separation by Theorem 2 of P7. $\square$

**It can lose independences.** In $A\to T\leftarrow R$, $A\perp_G R$ (the collider is
unobserved), but $A$ and $R$ are married in $\mathcal M(G)$, so they are **not** separated. The
Markov network represents the same distribution, but its graph no longer *shows* $A\perp R$. In
the language of independence maps: $\mathcal M(G)$ is an I-map of $P$, but not a perfect map.

## 4. What only undirected graphs can say

Take the 4-cycle $A$–$B$–$C$–$D$–$A$. Its separations include $A\perp C\mid\{B,D\}$ and
$B\perp D\mid\{A,C\}$.

**Proposition 4.** No DAG on $\{A,B,C,D\}$ has exactly the same set of d-separations as the
4-cycle has separations (Koller & Friedman, §3.4).

*Argument.* The cycle's separations say every pair of non-adjacent nodes is separated only by
the other two. Take a DAG with exactly these independences. Its skeleton must be the 4-cycle,
because adjacent nodes are never separated and non-adjacent ones are. An acyclic orientation of
a 4-cycle has at least one collider. At a collider $u\to w\leftarrow v$, $u$ and $v$ are not
d-separated by $\{w,\ldots\}$, which contradicts $u\perp v\mid\{w,\text{other}\}$. The tests also
confirm the proposition **exhaustively**: none of the 543 labelled DAGs on four nodes matches the
cycle on every one of the $6\times4=24$ (pair, conditioning set) queries.

Directed and undirected graphs are therefore incomparable languages. Each can say things the
other cannot: colliders on one side, cycles of independences on the other.

## 5. Numerical policy

Everything is computed in log space (spec ⚑2): $\log Z$, $\log P(e)$, and queries. Factors may be
given as `DiscreteFactor` or `LogFactor`. A potential like $e^{1000}$ is only representable as a
`LogFactor`. `probability(x)` returns $e^{\log\tilde P(x)-\log Z}$, which may underflow to 0 for
very improbable assignments. `log_probability(x)` does not.
