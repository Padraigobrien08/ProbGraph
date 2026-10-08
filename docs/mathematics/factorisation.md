# P1 — DAG factorisation

> **Claim.** If a joint distribution $p$ satisfies the conditional independences
> asserted by a DAG $G$, then
> $$p(x_1,\ldots,x_d)=\prod_{i=1}^{d}p(x_i\mid \mathrm{pa}(x_i)).$$

This note first works through the smallest interesting case, the collider
$A\rightarrow C\leftarrow B$, and then proves the general result. The general
proof shows why `DAG` must be able to produce a **topological ordering**, and
why it must reject **cycles**.

---

## 1. Ingredients

**Chain rule.** For *any* joint distribution and *any* ordering of its variables,
$$p(x_1,\ldots,x_d)=\prod_{i=1}^{d}p(x_i\mid x_1,\ldots,x_{i-1}).$$
This follows from repeatedly applying $p(a,b)=p(a)\,p(b\mid a)$. It assumes nothing
about the distribution, so on its own it saves nothing: the last factor still
conditions on all $d-1$ other variables.

**What a DAG asserts (local Markov property).** Every variable is conditionally
independent of its non-descendants, given its parents:
$$X_i \;\perp\; \mathrm{nd}(X_i)\setminus\mathrm{pa}(X_i)\;\big|\;\mathrm{pa}(X_i).$$

**Conditional independence as a rewrite rule.** If $X\perp Y\mid Z$ then
$p(x\mid y,z)=p(x\mid z)$. The factorisation comes from using this rule to delete
variables from the conditioning sets of the chain rule.

---

## 2. Worked example: $A\rightarrow C\leftarrow B$

| Node | Parents | Descendants | Non-descendants (excluding itself) |
|------|---------|-------------|------------------------------|
| $A$  | $\varnothing$ | $\{C\}$ | $\{B\}$ |
| $B$  | $\varnothing$ | $\{C\}$ | $\{A\}$ |
| $C$  | $\{A,B\}$ | $\varnothing$ | $\{A,B\}$ |

**Step 1: pick an ordering in which parents come first.** $(A,B,C)$ works.
($(B,A,C)$ works too. Topological orderings are not unique.)

**Step 2: apply the chain rule in that order.**
$$p(A,B,C)=p(A)\;p(B\mid A)\;p(C\mid A,B).$$

**Step 3: simplify each factor using the local Markov property.**

- $p(A)$ has nothing to simplify.
- $p(B\mid A)$: $B$ has no parents and $A$ is a non-descendant of $B$, so
  $B\perp A$ and $p(B\mid A)=p(B)$.
- $p(C\mid A,B)$: the conditioning set is exactly $\mathrm{pa}(C)$, so nothing is removed.

$$\boxed{p(A,B,C)=p(A)\,p(B)\,p(C\mid A,B)}$$

### What the collider does and does not imply

*Marginal independence.* Sum out $C$:
$$p(A,B)=\sum_c p(A)\,p(B)\,p(c\mid A,B)=p(A)\,p(B)\underbrace{\sum_c p(c\mid A,B)}_{=1}=p(A)\,p(B).$$
So $A\perp B$. This step uses the CPD normalisation invariant C2, and it is the
same move as the general normalisation proof P2.

*Conditional dependence ("explaining away").* Condition on $C$:
$$p(A,B\mid C)=\frac{p(A)\,p(B)\,p(C\mid A,B)}{p(C)}.$$
Because $p(C\mid A,B)$ couples $A$ and $B$, this does **not** factorise into
$f(A)\,g(B)$ in general, so $A\not\perp B\mid C$.

*Numerical check with the canonical fixture* ($R$ = Rain, $A$ = Accident, $T$ = Traffic, with
$R\rightarrow T\leftarrow A$):

| Quantity | Value | Reading |
|---|---|---|
| $P(A=1)$ | 0.1000 | prior |
| $P(A=1\mid R=1)$ | 0.1000 | rain alone says nothing about accidents ($A\perp R$) |
| $P(T=1)$ | 0.3565 | $=0.063+0.049+0.216+0.0285$ |
| $P(A=1\mid T=1)$ | 0.2174 | $=0.0775/0.3565$; traffic makes an accident more likely |
| $P(A=1\mid T=1,R=1)$ | 0.1166 | rain explains the traffic, so the accident becomes less likely again |
| $P(A=1\mid T=1,R=0)$ | 0.4375 | no rain, so the accident must explain the traffic |

These values become oracle tests in Milestone 1 tasks 6–7.

---

## 3. General proof

Let $\pi=(x_{\pi_1},\ldots,x_{\pi_d})$ be a **topological ordering** of $G$: for
every edge $u\rightarrow v$, $u$ appears before $v$ (invariant G2). Relabel so
that $\pi$ is $(x_1,\ldots,x_d)$.

1. Chain rule in this order:
   $p(x)=\prod_i p(x_i\mid x_1,\ldots,x_{i-1})$.
2. **Lemma.** In a topological ordering, every predecessor of $x_i$ is a
   non-descendant of $x_i$.
   *Proof.* If $x_j$ with $j<i$ were a descendant of $x_i$, there would be a path
   $x_i\rightarrow\cdots\rightarrow x_j$. By G2, positions strictly increase along
   every edge of that path, so $i<j$. Contradiction. $\square$
3. Every parent of $x_i$ comes before it (G2), so
   $\mathrm{pa}(x_i)\subseteq\{x_1,\ldots,x_{i-1}\}\subseteq\mathrm{nd}(x_i)$.
4. Local Markov: $x_i\perp \{x_1,\ldots,x_{i-1}\}\setminus\mathrm{pa}(x_i)\mid \mathrm{pa}(x_i)$, hence
   $p(x_i\mid x_1,\ldots,x_{i-1})=p(x_i\mid\mathrm{pa}(x_i))$.
5. Substituting into (1) gives the factorisation. $\blacksquare$

### Where the code comes in

- The proof **needs** a topological ordering. One exists exactly when $G$ is
  acyclic (see [`dag_algorithms.md`](dag_algorithms.md)). With a cycle
  $x\rightarrow y\rightarrow x$ no ordering puts every parent first, so the
  chain-rule argument has nothing to start from. Invariant G1 is therefore a
  precondition of the mathematics, not a convenience.
- `DAG.topological_sort()` produces $\pi$. The same ordering is the visiting
  order for ancestral sampling (P3) and the elimination order, reversed, for
  the normalisation proof (P2).
- The result does **not** depend on *which* topological ordering is chosen. So
  tests must check that the ordering is valid, not that it equals one
  particular ordering.
