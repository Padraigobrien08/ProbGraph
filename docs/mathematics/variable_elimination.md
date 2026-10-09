# P6 — Variable elimination

> **Theorem.** Let $\Phi_0=\{\phi_i[e]\}$ be the CPD factors of a Bayesian network, reduced by
> the evidence $E=e$. Let $Z=(z_1,\ldots,z_m)$ be any ordering of the variables outside
> $Q\cup E$. Eliminate them one at a time: multiply the factors that mention $z_k$, sum $z_k$ out,
> and put the result back. Then the product $\tau$ of the factors that remain satisfies
> $$\tau=P(Q,e),\qquad \tau.\mathrm{total}()=P(e),\qquad \tau/P(e)=P(Q\mid e).$$
> The answer does not depend on the order. The cost does.

Prerequisites: P1 (factorisation), P2 C-a (barren leaves), and P5 laws F5–F8 and F10
([`factor_algebra.md`](factor_algebra.md)).

---

## 1. The target expression

By P1 the joint distribution is $\prod_i\phi_i$. Use the indicator view of reduction and F8 (P5 §3–§4):
$$P(Q,e)=\sum_{z\in\mathcal X_Z}\ \prod_{\phi\in\Phi_0}\phi.$$

Evaluating this directly builds a table over every unobserved variable. That is brute-force
enumeration, with cost $\prod_{v\notin E}|\mathcal X_v|$. Variable elimination evaluates the same
expression by pushing each sum as far inward as F7 allows.

## 2. One elimination step

**Lemma (elimination step).** For a multiset of factors $\Phi$ and a variable $z$, split
$\Phi=\Phi_z\cup\Phi_{\bar z}$ by whether a factor's scope contains $z$. Then
$$\sum_z\prod_{\phi\in\Phi}\phi=\Big(\prod_{\phi\in\Phi_{\bar z}}\phi\Big)\cdot\underbrace{\sum_z\prod_{\phi\in\Phi_z}\phi}_{\tau_z}.$$

*Proof.* Regroup the product into $(\prod\Phi_{\bar z})(\prod\Phi_z)$, which F5 allows. The first
group does not mention $z$, so F7 moves the sum past it. $\square$

The algorithm replaces $\Phi$ with $\Phi'=\Phi_{\bar z}\cup\{\tau_z\}$, and by the lemma
$\prod\Phi'=\sum_z\prod\Phi$.

Note that $\Phi_{\bar z}$ must be **kept**. Dropping the factors that do not mention $z$ is a
typical bug. The tests break the code this way on purpose to check it is caught.

## 3. Correctness

**Invariant.** After eliminating $z_1,\ldots,z_k$:
$$\prod\Phi_k=\sum_{z_1,\ldots,z_k}\prod\Phi_0.$$

*Induction.* For $k=0$ there is nothing to prove. For the step, apply the lemma to $\Phi_k$ with
$z_{k+1}$:
$$\prod\Phi_{k+1}=\sum_{z_{k+1}}\prod\Phi_k=\sum_{z_{k+1}}\sum_{z_1..z_k}\prod\Phi_0=\sum_{z_1..z_{k+1}}\prod\Phi_0,$$
using F6 to merge the sums. $\square$

At $k=m$, every variable outside $Q\cup E$ has been summed out, so $\tau=\prod\Phi_m=P(Q,e)$.
Its scope is exactly $Q$. Variables in $E$ left every scope at the reduction step, and every other
non-query variable was eliminated.

- $\tau.\mathrm{total}()=\sum_qP(q,e)=P(e)$.
- If $P(e)>0$, then $\tau/P(e)=P(Q\mid e)$ by the definition of conditional probability.
- If $P(e)=0$, the posterior is undefined. The implementation raises
  `ZeroProbabilityEvidenceError` (V5) and never divides $0/0$.

**Order independence (V2).** The invariant holds for *every* ordering, and the final value is the
same sum, $\sum_Z\prod\Phi_0$, whichever order is used. So all orders give identical results in exact
arithmetic, and floating-point results differ only by rounding (F6). The tests compare every
ordering against enumeration with `atol=1e-12`.

**Corollary (normalising an intermediate factor is invisible to the posterior).** Scaling any
intermediate $\tau_z$ by a constant $c>0$ scales $\tau$ by $c$. Normalising at the end removes
$c$ again. So a bug that normalises intermediate factors still produces correct posteriors and
**only** corrupts $P(e)$. This is why `probability_of_evidence` has its own tests.

## 4. Cost

Eliminating $z$ multiplies the factors in $\Phi_z$ into a table $\psi_z$ over
$S_z=\bigcup_{\phi\in\Phi_z}\mathrm{scope}(\phi)$, then sums out $z$. The work is proportional to
$|\mathcal X_{S_z}|=\prod_{v\in S_z}|\mathcal X_v|$, so the total is
$$\mathrm{cost}(Z)=\sum_{k=1}^{m}\ \prod_{v\in S_{z_k}}|\mathcal X_v|.$$

The *width* of the order is $w(Z)=\max_k|S_{z_k}|-1$, the largest scope of any $\tau_{z_k}$. With
$n$ variables of cardinality at most $d$, the cost is $O(n\,d^{\,w(Z)+1})$. It is exponential in the
**width**, not in $n$.

**Example: naive Bayes.** Take a class $C$ and binary features $F_1,\ldots,F_n$, with
$C\to F_i$. Query $P(F_1)$ with no evidence, so the variables to eliminate are $C$ and
$F_2,\ldots,F_n$.

| Order | Largest table $\psi_z$ | Total cost |
|---|---|---|
| $F_2,\ldots,F_n$, then $C$ | $\{F_i,C\}$: 4 cells | $4n$ |
| $C$ first, then $F_2,\ldots,F_n$ | $\{C,F_1,\ldots,F_n\}$: $2^{n+1}$ cells | $\Theta(2^n)$ |

In the first order, each $\sum_{f_i}P(f_i\mid C)=\mathbb 1_C$ costs 4. The last step,
$\sum_cP(c)P(F_1\mid c)$, also costs 4. In the second order, eliminating $C$ first multiplies
*every* factor together, because they all mention $C$.

Both orders return the same $P(C)$. The good order's cost is linear in $n$ and the bad order's is
exponential, from the same algorithm. Choosing good orders is task M2.5 (heuristics and induced
width). Finding the *optimal* order is NP-hard in general (Arnborg, Corneil & Proskurowski, 1987),
so heuristics are the practical answer.

## 5. Barren nodes and ancestral pruning

Let $z$ be a leaf outside $Q\cup E$. Its only factor is its own CPD, because a leaf is nobody's
parent. So $\Phi_z=\{\phi_z\}$, and by F10
$$\tau_z=\sum_z\phi_z=\mathbb 1_{\mathrm{pa}(z)}.$$
Multiplying by an all-ones factor changes nothing, so such a **barren** leaf can be deleted
before inference. Deleting it can make its parents barren leaves in turn. The following
proposition says exactly where repeated deletion stops.

Write $\mathrm{An}^*(S)=S\cup\bigcup_{s\in S}\mathrm{anc}(s)$ for the **ancestral closure** of $S$
(`DAG.ancestral_set`).

**Proposition 5 (what pruning removes).** Repeatedly deleting leaves outside $Q\cup E$
removes **exactly** $V\setminus\mathrm{An}^*(Q\cup E)$.

*Proof.* Let $K=\mathrm{An}^*(Q\cup E)$ and $D=V\setminus K$.

- *Nothing in $K$ is ever removed.* A vertex of $Q\cup E$ is never eligible for removal. Any
  other $v\in K$ has a directed path to some $q\in Q\cup E$. Every vertex on that path is in $K$.
  By induction on the number of deletions, the path survives, so $v$ always has a child and is
  never a leaf.
- *Everything in $D$ is removed.* $D$ is closed under taking children: if $v\in D$ had a child in
  $K$, then $v$ would be an ancestor of $Q\cup E$, so $v\in K$. While $D$ still has remaining
  vertices, the remaining part of $D$ is a non-empty DAG, so it has a sink $v$. All of $v$'s
  children lie in $D$, and none remain, so $v$ is a leaf. It is outside $Q\cup E\subseteq K$, so
  it is barren and gets deleted. $\square$

**Corollary (V7).** $P(Q,e)$, and hence $P(Q\mid e)$ and $P(e)$, can be computed from the
CPDs of $K$ alone:
$$P(Q,e)=\sum_{K\setminus(Q\cup E)}\ \prod_{i\in K}\phi_i[e].$$
This is P2's corollary C-a for the ancestral set $K$, followed by summing out its non-query,
non-evidence variables. All parents of a vertex in $K$ are themselves in $K$, so every
factor kept is complete.

**Pruning never makes a fixed order more expensive.** Let $H$ be the interaction graph before
pruning and $H'$ the one after. $H'$ is a subgraph of $H$: it has fewer vertices, and it may
also lack marriages whose only shared child was pruned. Eliminate both with the same order,
skipping pruned vertices in $H'$. Then **every step's scope in $H'$ is a subset of the same
variable's scope in $H$**.

*Proof.* The **fill-path lemma** (Rose, Tarjan & Lueker, 1976) says that $u$ and $v$ are
adjacent when the first of them is eliminated exactly when $H$ contains a $u$–$v$ path whose
interior vertices are all eliminated before both $u$ and $v$. Any such path in $H'$ is also a
path in $H$, and its vertices are eliminated in the same relative order. $\square$

The tests check this subset relation on measured traces.

With heuristic orders, pruning changes the graph, so the chosen order can change too. The
answer is the same either way (V2). For example, in naive Bayes with query $F_1$, every
$F_{i\ge2}$ is pruned. What is left is the two factors $P(C)$ and $P(F_1\mid C)$, with one
elimination step of 4 cells, compared with 40 cells in total for min-fill without pruning.

## 6. What this does not cover

- **Many queries.** Variable elimination recomputes everything for each query. Junction trees
  (M3) cache the intermediate messages.
- **Precision.** All computation is in probability space. Products of many small numbers can
  underflow, which would show up as $P(e)=0$. Log-space factors are deferred (spec §9 ⚑7).
- **Optimal orders.** NP-hard to find, as noted in §4. Greedy heuristics (M2.5,
  [`elimination_orders.md`](elimination_orders.md)) are used instead.
- **Irrelevant evidence.** Observed variables that are d-separated from the query can be
  dropped as well. That requires d-separation, so it comes in M2.9.
