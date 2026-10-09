# P22 — Dynamic Bayesian networks

> An HMM has one hidden variable per step. A **dynamic Bayesian network** (DBN) lets each time slice
> be a whole Bayesian network, with edges inside a slice and edges from one slice to the next. It is
> specified once, as a *two-slice temporal Bayesian network* (2-TBN), and unrolled for any length.
> The unrolled network is an ordinary M1 network, so every exact algorithm applies. But the cost of
> exact inference is governed by the **interface** between slices, and that can be large even when
> the slices themselves are small and only loosely connected.

Prerequisites: P1 (factorisation), P7 (d-separation), P11–P12 (junction trees), P18 (HMMs).

---

## 1. The 2-TBN

Fix a set $V$ of **template variables**: the variables of one slice. A 2-TBN consists of:

1. an **initial network** $B_1$: a Bayesian network over $V$, for slice 1;
2. a **transition model**: for each $X\in V$, one CPD $P(X_t\mid\mathrm{Pa}(X_t))$, where every parent is
   either a template variable of the same slice $t$ or a copy $Y_{t-1}$ of a template variable in the
   previous slice. The library writes $Y_{t-1}$ as `previous(Y)`, named `Y[t-1]`.

The same transition CPDs are used for every $t\ge2$. The model is homogeneous, as in P18. The
**intra-slice graph** (edges between template variables of the same slice) must be acyclic.

**Example: the HMM.** $V=\{X,Y\}$. $B_1$ is $X\to Y$ with $P(X_1)=\pi$. The transition model is
$P(X_t\mid X_{t-1})=A$ and $P(Y_t\mid X_t)=B$. Unrolling gives exactly `HiddenMarkovModel.to_bayesian_network`
(N1).

## 2. Unrolling

`unroll(T)` creates the variables $X_t$ (named `X_1`, …, `X_T`) and gives slice 1 the CPDs of $B_1$,
and each slice $t\ge2$ the transition CPDs with `Y` read as $Y_t$ and `Y[t-1]` as $Y_{t-1}$.

**Proposition 1.** The unrolled graph is acyclic, so `unroll(T)` is a Bayesian network, and its joint
distribution is
$$P(x_{1:T})=\prod_{X\in V}P_{B_1}(x_1\mid\mathrm{pa}_1)\;\prod_{t=2}^T\prod_{X\in V}P(x_t\mid\mathrm{pa}_t).$$

*Proof.* Order the unrolled variables slice by slice, and within each slice by a topological order of
the intra-slice graph (and of $B_1$ for slice 1). Every edge goes either within a slice, which respects
this order, or from slice $t-1$ to slice $t$, which also does. So there is no cycle. The joint is then
P1's factorisation. $\square$

## 3. The interface separates past and future

The **interface** $I$ is the set of template variables that have a child in the next slice: the $Y$
with some transition CPD having `Y[t-1]` as a parent.

**Proposition 2.** In the unrolled network, for every $t<T$, the interface of slice $t$, $I_t$,
d-separates the earlier variables (slices $1..t$, apart from $I_t$) from the later ones (slices
$t+1..T$).

*Proof.* Take any trail from an earlier variable to a later one. Inter-slice edges join consecutive
slices and point forward, so the trail must use, at some point, an edge $u\to w$ with $u$ in slice $t$
and $w$ in slice $t+1$. Take the first such edge. Then $u\in I_t$, since it has a child in the next
slice. At $u$ the trail continues into $w$ along an outgoing edge, so $u$ is not a collider on the
trail: it is a chain or a fork. A chain or fork node in the conditioning set blocks the trail (P7).
$\square$

So $P(I_t\mid\text{evidence up to }t)$, the **belief state**, carries everything the past says about the
future. For an HMM, $I=\{X\}$, and the belief state is P19's filtered belief.

## 4. Exact inference, and why its cost depends on the interface

The library does exact DBN inference by unrolling and calling `JunctionTree` or `VariableElimination`
(N2). Eliminating slice by slice, the variables that remain linked to the future are those of the
interface. So for a fixed DBN, the largest clique stops growing after a few slices: the cost is
linear in $T$, with a constant set by the interface.

**Entanglement.** The interface can be large even when it looks harmless. A *factorial HMM* has $n$
hidden chains $X^{(1)},\ldots,X^{(n)}$, each evolving independently,
$P(X^{(k)}_t\mid X^{(k)}_{t-1})$, and one observation $Y_t$ with all $n$ chains as parents. A priori the
chains are independent. But $Y_t$ is a common child of all of them, so observing it couples them, by
explaining away (P1). After a few observations, the belief state $P(X^{(1)}_t,\ldots,X^{(n)}_t\mid y_{1:t})$
does not factorise at all. The moral graph links all $n$ chains at every step, and the junction tree
needs cliques with all of them: the cost is exponential in $n$, whatever $T$ is.

This is why approximate DBN inference (Boyen–Koller, particle filtering) exists. It is out of scope
here (spec §2), but the tests measure the exact cost:

- the largest clique of the unrolled factorial HMM is the same for $T=6,10,16$ (linear in $T$);
- it grows with the number of chains $n$;
- the filtered joint of two chains is far from the product of its marginals once $Y$ is observed,
  and is exactly that product when nothing is observed.

## 5. How the tests check this independently

- **N1:** from an HMM, `unroll(T)` has the same variables, edges and CPD tables as
  `HiddenMarkovModel.to_bayesian_network(T)`.
- **Proposition 1:** for random 2-TBNs, the unrolled joint equals the product formula, assignment by
  assignment.
- **Proposition 2:** M2's `d_separated` on random 2-TBNs, for every $t$.
- **N2:** junction-tree marginals with random evidence match brute force, and the clique sizes behave
  as §4 says.
- **Validation:** every malformed 2-TBN is rejected with a clear message.
