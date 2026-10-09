# P12 — Sum-product message passing on a clique tree (Shafer–Shenoy)

> Calibrating a clique tree costs two messages per edge, one sweep in and one sweep out. After
> that, **every** clique holds the exact unnormalised marginal of its variables, so every
> single-variable marginal is a small sum away. The proof is variable elimination (P6) again,
> organised by the tree. The running intersection property guarantees that no variable is ever
> summed out too early.

Prerequisites: P5 (factor laws, especially distributivity F7), P6 (variable elimination), P9 (log
space), P10 (Gibbs distributions), P11 (clique trees and RIP).

---

## 1. Setup

- A model whose unnormalised product is $\tilde P(x)=\prod_k\phi_k(x_{C(\phi_k)})$. For a Bayesian
  network, $\tilde P=P$ and $Z=1$. For a Markov network, $\tilde P=ZP$.
- A clique tree $T$ over clusters $C_1,\ldots,C_m$, with RIP, built from the model's interaction
  graph (P11 §6).
- **Initial potentials.** Each factor is assigned to one cluster that contains its scope (J6).
  Then $\psi_i=\mathbb 1_{C_i}\prod_{\phi\text{ assigned to }i}\phi$. The all-ones factor
  $\mathbb 1_{C_i}$ makes the scope of $\psi_i$ exactly $C_i$, which matters for clusters that
  received no factor. Since every factor is assigned exactly once,
  $$\prod_{i=1}^m\psi_i=\tilde P.$$

## 2. Messages and beliefs

For neighbours $i$ and $j$ with separator $S_{ij}=C_i\cap C_j$:
$$\delta_{i\to j}=\sum_{C_i\setminus S_{ij}}\ \psi_i\prod_{k\in N(i)\setminus\{j\}}\delta_{k\to i},\qquad
\beta_i=\psi_i\prod_{k\in N(i)}\delta_{k\to i}.$$
A message from $i$ to $j$ needs all of $i$'s other incoming messages first. On a tree, this is
always satisfiable. An **inward sweep** (leaves to a root) followed by an **outward sweep** (root
to leaves) computes all $2(m-1)$ messages, each exactly once (**C1**).

## 3. Correctness: beliefs are marginals

Removing the edge $i$–$j$ splits $T$ in two. Let $T_{i\to j}$ be the side containing $i$, and let
$W_{i\to j}$ be the variables that occur in $T_{i\to j}$ but are **not** in $S_{ij}$.

**Lemma 1 (RIP separates the two sides).** A variable occurring on both sides of edge $i$–$j$
lies in $S_{ij}$. So the two sides share no variable outside the separator.

*Proof.* If $x$ occurs in a cluster on each side, then by RIP $x$ occurs in every cluster on the
path between them, and that path uses the edge $i$–$j$. So $x\in C_i\cap C_j=S_{ij}$. $\square$

**Lemma 2 (what a message is).**
$$\delta_{i\to j}=\sum_{W_{i\to j}}\ \prod_{k\in T_{i\to j}}\psi_k.$$
That is, a message is the product of all potentials on its side, with every variable not in
the separator summed out.

*Proof.* By induction on the size of $T_{i\to j}$. Let $k_1,\ldots,k_r$ be $i$'s neighbours other
than $j$. Their subtrees $T_{k_t\to i}$ partition $T_{i\to j}\setminus\{i\}$. By the induction
hypothesis, $\delta_{k_t\to i}$ is the product over $T_{k_t\to i}$ with $W_{k_t\to i}$ summed out.

By Lemma 1, applied at each edge $k_t$–$i$, the sets $W_{k_t\to i}$ are pairwise disjoint. They
are also disjoint from $C_i$, and from the factors outside $T_{k_t\to i}$. So distributivity (F7)
lets all those inner sums be pulled outside the product with $\psi_i$:
$$\psi_i\prod_t\delta_{k_t\to i}=\sum_{\bigcup_tW_{k_t\to i}}\ \prod_{k\in T_{i\to j}}\psi_k.$$
Summing out $C_i\setminus S_{ij}$ as well gives exactly
$W_{i\to j}=(C_i\setminus S_{ij})\cup\bigcup_tW_{k_t\to i}$. A variable in some $W_{k_t\to i}$ is
not in $C_i$, so not in $S_{ij}$. Conversely, every variable of $T_{i\to j}$ outside $S_{ij}$
lies in $C_i$ or in one of the subtrees. $\square$

**Theorem 3 (beliefs are marginals).**
$$\beta_i=\sum_{V\setminus C_i}\tilde P\quad\text{for every }i.$$

*Proof.* The subtrees $T_{k\to i}$ for $k\in N(i)$ partition the clusters other than $C_i$. By
Lemma 1, a variable outside $C_i$ lies in exactly one $W_{k\to i}$: it occurs on one side only,
otherwise it would be in $C_i$. Apply Lemma 2 to each incoming message and pull the
(disjoint) sums out, by F7:
$$\beta_i=\psi_i\prod_{k}\delta_{k\to i}=\sum_{V\setminus C_i}\ \prod_{\text{all }k}\psi_k=\sum_{V\setminus C_i}\tilde P.\qquad\square$$

**Where RIP is used, and why it is necessary.** Lemma 1 is the only place, and it is used twice:
once to keep different children's summed variables apart, and once to stop a variable from being
summed out on one side while still needed on the other. Without RIP, a variable $x$ could sit in
two clusters that are joined only through clusters without $x$. Each side would then sum $x$
out independently, as if they were two different variables. The tests demonstrate this by
breaking RIP on purpose.

## 4. Consequences

- **C2 (calibration).** For neighbours $i$ and $j$,
  $\sum_{C_i\setminus S_{ij}}\beta_i=\sum_{V\setminus S_{ij}}\tilde P=\sum_{C_j\setminus S_{ij}}\beta_j$.
- **C3.** $\beta_i/Z$ is the joint marginal $P(C_i)$.
- **C4.** $\sum_{C_i}\beta_i=Z$ for **every** $i$. One calibration gives $\log Z$, and every
  clique agrees on it.
- **C5.** For any variable $x$, take any cluster containing it:
  $P(x)=\sum_{C_i\setminus\{x\}}\beta_i/Z$.
- **C6 (schedule independence).** Lemma 2 gives each message in closed form, in terms of the
  potentials alone. So the root, the order within a sweep, and even the choice of clique tree
  cannot change any belief or marginal, beyond floating-point rounding.

## 5. Cost

There are $2(m-1)$ messages. Computing $\delta_{i\to j}$ touches at most $|\mathcal X_{C_i}|$ cells
per incoming factor, so the total is $O(m\cdot\deg\cdot d^{\,w+1})$, where $w$ is the tree width. Variable
elimination needs about $O(n\,d^{\,w+1})$ **per query**. For all $n$ single-variable marginals,
message passing is therefore cheaper by a factor of about $n/\deg$. M3.7 measures this (C7).

## 6. Log space

Everything above uses only products and sums, so it holds unchanged in log space (P9 §5).
Products are $+$, sums are log-sum-exp, $\mathbb 1$ is the zero log factor, and $\log Z$ is the
log-sum-exp of any belief. The implementation always works in log space (spec ⚑2).

## 7. Scope

`JunctionTree.query` answers joint queries over variables that lie **together in one clique**,
by marginalising that clique's belief (C3). A query spanning several cliques would need
additional message passing. Spec ⚑6 decides to raise an error pointing to variable elimination
instead.

## 8. Evidence

Reduce every factor by the evidence before building the tree (P5's indicator view):
$\phi_k\mapsto\phi_k[e]$. Then $\prod_i\psi_i=\tilde P[e]$, and Theorem 3, applied to the reduced
factors, gives
$$\beta_i=\sum_{V\setminus(C_i\cup E)}\tilde P[e]\;\propto\;P(C_i\mid e).$$

- **The observed variables leave the graph.** Reduction removes them from every scope, so the
  interaction graph of the reduced factors is the original graph with the observed vertices
  deleted (`elimination_orders.md` §1). The tree is built on that smaller graph, which gives
  smaller cliques and cheaper messages. A factor reduced to a scalar is assigned to any clique,
  where it only rescales the beliefs. If every variable is observed, there is no tree, and
  $Z(e)$ is the product of the scalars.
- **$P(e)$ comes for free.** Every clique has the same total,
  $Z(e)=\sum_{x:x_E=e}\tilde P(x)$, by C4 applied to the reduced factors. So
  $\log P(e)=\log Z(e)-\log Z$, where $\log Z$ is the evidence-free partition function: 0 for a
  Bayesian network, `MarkovNetwork.log_partition_function()` otherwise.
- **Zero detection is exact (C8).** In log space, $\log Z(e)=-\infty$ exactly when the evidence
  is impossible (P9 §8). The tree can still be built, and `log_probability_of_evidence` returns
  $-\infty$, but every posterior query raises `ZeroProbabilityEvidenceError`. Tiny but possible
  evidence, such as fixture F3 with $\log P(e)\approx-784.9$, is handled exactly.
- **Observed variables in queries.** `marginal(x)` for an observed $x$ returns the point mass
  at its observed value, which is the true $P(x\mid e)$. A joint `query` that includes an observed
  variable is rejected, as in `VariableElimination`.

## 9. Measuring cost (C7)

Both engines are charged in the same currency: **table cells**.

- **Variable elimination:** `query_trace(...).total_cost`, the sum of the sizes of the product
  tables $\psi_z$ (M2).
- **Junction tree:** `calibration_cost()`. Each message $\delta_{i\to j}$ is charged
  $|\mathcal X_{C_i}|$, the size of the product table it sums. Each belief is charged
  $|\mathcal X_{C_i}|$ again.

**What was measured, and what that means.** The comparison depends on how much variable
elimination can **prune** (P6 §5). P(X) needs only $\mathrm{An}^*(\{X\}\cup E)$.

| Situation | Measured (sparse random DAGs, 25–30 binary variables) |
|---|---|
| Evidence at the leaves | calibration is **4–10× cheaper** than pruned VE (e.g. 240 vs 1,662 cells). The ancestral sets cover most of the network, so every query pays nearly full price. |
| No evidence, VE **without** pruning | calibration is about **10× cheaper** |
| No evidence, VE **with** pruning | pruned VE is often **cheaper** (212–1,652 vs 386–2,476 cells). Each query keeps only $X$'s few ancestors, while calibration pays for the whole tree. |
| A chain of $n$ variables, no evidence | calibration $\approx12n$ cells against $\approx2n^2$: cheaper for $n\gtrsim10$. The later variables have long ancestral chains. |

So calibration is the right tool when **many marginals are wanted given evidence**: diagnosis
from observed symptoms, or the E-step of learning with missing data. For a few prior marginals
of a sparse model, pruned variable elimination can be cheaper. The first draft of this section,
and spec v1.2's C7, claimed an unconditional advantage. Measuring showed that to be false
without evidence, and both are corrected. The tests pin down both sides, including a case where
variable elimination wins.
