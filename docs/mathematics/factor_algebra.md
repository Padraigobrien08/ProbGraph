# P5 — Factor algebra

> A **factor** is a nonnegative function of a set of named variables. Inference needs only four
> operations on factors: product, sum-out, reduce and normalise. All of the algebraic laws come
> from pointwise arithmetic on real numbers. The law that matters most is **distributivity**: a
> sum can be moved past any factor that does not mention the summed variable. Variable
> elimination (P6) is that law applied repeatedly.

---

## 1. Definitions

Let $S$ be a finite set of variables, the **scope**. $\mathcal X_S=\prod_{V\in S}\mathcal X_V$ is
the set of joint assignments to $S$. For $x\in\mathcal X_S$ and $T\subseteq S$, $x|_T$ is the
restriction of $x$ to $T$.

- A **factor** is a function $\phi:\mathcal X_S\to\mathbb R_{\ge0}$, written $\phi(x_S)$.
- **Empty scope.** $\mathcal X_\varnothing=\{()\}$ has exactly one element, the empty assignment.
  A factor on it is just a nonnegative number (a *scalar factor*). The **unit** factor $\mathbb 1$
  is the scalar 1.

The four operations:

| Operation | Definition | Scope of result |
|---|---|---|
| Product | $(\phi\cdot\psi)(x)=\phi(x\vert_S)\,\psi(x\vert_T)$ | $S\cup T$ |
| Sum-out (marginalise) | $\big(\sum_Y\phi\big)(x)=\sum_{y\in\mathcal X_Y}\phi(x,y)$ for $Y\subseteq S$ | $S\setminus Y$ |
| Reduce by evidence $E=e$ | $\phi[e](x)=\phi\big(x,\ e\vert_{S\cap E}\big)$ | $S\setminus E$ |
| Normalise | $\phi/Z$, where $Z=\sum_{x\in\mathcal X_S}\phi(x)$ | $S$ |

Notes:

- Reduction looks only at the evidence variables that are **in** the scope. Evidence on other
  variables leaves $\phi$ unchanged. That is what lets variable elimination apply the same
  evidence to every factor.
- Normalising is undefined when $Z=0$. The implementation raises an error and never divides by
  zero (F9).
- Summing out *every* variable gives the scalar $Z$. So `total()` is
  `marginalise(scope).value({})`.

---

## 2. A factor is a function, and the array is only one encoding of it

The implementation stores $\phi$ as an array with one axis per variable, in some order
$(V_1,\ldots,V_k)$:
$$\texttt{values}[i_1,\ldots,i_k]=\phi\big(V_1{=}(\mathcal X_{V_1})_{i_1},\ldots,V_k{=}(\mathcal X_{V_k})_{i_k}\big).$$

Any permutation of the axes encodes **the same function**. So the axis order must not be visible
in any result: values are looked up by name, and comparisons align axes by name first (**F4**).
This is the factor version of M1's named CPD lookup (C5). It also rules out the most likely bug,
which is multiplying two arrays position by position when their axes refer to different variables.

### Product by broadcasting (the implementation identity)

The **cylindrical extension** of $\phi$ from $S$ to $U\supseteq S$ is
$\bar\phi(x_U)=\phi(x_U|_S)$: the same function, constant along every new variable. Then
$$\phi\cdot\psi=\bar\phi\cdot\bar\psi\quad\text{pointwise on }\mathcal X_{S\cup T}.$$

In NumPy, the extension is exactly **transpose to the union order, then insert size-1 axes**. The
pointwise product is then `*` with broadcasting. Each step is pure bookkeeping, so the product is
correct exactly when the alignment is correct. The tests check this against the definition cell
by cell.

---

## 3. Laws

Each law below is an identity between functions, so proving it means checking it at an arbitrary
point $x$.

**F5: commutative monoid.** $\phi\psi=\psi\phi$, $(\phi\psi)\chi=\phi(\psi\chi)$, and
$\mathbb 1\phi=\phi$. At each point these reduce to commutativity, associativity and the identity
element of real multiplication.

**F6: sums commute.** $\sum_X\sum_Y\phi=\sum_Y\sum_X\phi=\sum_{X\cup Y}\phi$. These are finite
sums of the same terms in a different order. Exact equality holds in real arithmetic. In floating
point the results agree to rounding error, so the tests use a tolerance.

**F7: distributivity.** If $X\notin\mathrm{scope}(\phi)$, then
$$\sum_X\phi\,\psi=\phi\sum_X\psi.$$
*Proof.* Fix $x$ on $(S\cup T)\setminus\{X\}$. The value $\phi(x|_S)$ does not depend on the
summation index, so it factors out of the sum:
$\sum_{v}\phi(x|_S)\,\psi(x|_{T\setminus X},v)=\phi(x|_S)\sum_v\psi(x|_{T\setminus X},v)$. $\square$

*Why it matters.* The left side builds a table over $S\cup T$ and then sums. The right side sums a
table over $T$ only, then multiplies. Suppose $\phi$ is over $m$ binary variables, $\psi$ is over
$\{X\}\cup T'$, and the two scopes are disjoint. Then the left side touches $2^{m+|T'|+1}$ cells, while the right side needs only
$2^{|T'|+1}+2^{m+|T'|}$. Variable elimination is the systematic use of this saving.

*F7 needs the side condition.* If $X\in\mathrm{scope}(\phi)$, then $\sum_X\phi\psi\ne\phi\sum_X\psi$
in general. Take $\phi(x)=\psi(x)=x$ for a binary $X$: the left side is
$0\cdot0+1\cdot1=1$, a scalar, while $\phi\cdot\sum_X\psi$ still has $X$ in its scope. The tests include this
counterexample.

**F8: reduction commutes with the other operations.**

- $(\phi\psi)[e]=\phi[e]\,\psi[e]$: evaluate both sides at a point; the evidence enters each
  factor through the same restriction.
- $\big(\sum_Y\phi\big)[e]=\sum_Y\big(\phi[e]\big)$ when $Y\cap E=\varnothing$: fixing $E$ and
  summing over $Y$ touch disjoint sets of coordinates.

**Reduction as a product (the indicator view).** Let $\delta_e(x_E)=1$ if $x_E=e$ and 0
otherwise. Then
$$\phi[e]=\sum_{E\cap S}\phi\cdot\delta_{e|_{S\cap E}}.$$
So reduction is not a new primitive: it is "multiply by an indicator, then sum out". This is
why evidence fits into variable elimination without special cases. P6 uses it to show that
$P(Q,e)=\sum_{\text{rest}}\prod_i\phi_i[e]$. It is also a property test.

**F9.** $\sum_x\phi(x)/Z=Z/Z=1$, and $Z=0$ is rejected.

**F10: a CPD is a factor whose child sums to 1.** For $\phi_{X\mid U}=\texttt{from\_cpd}$,
$\sum_X\phi_{X\mid U}=\mathbb 1_U$ (the all-ones factor over $U$). This is C2 restated. It is the
step used in P2's leaf elimination, and the step that makes barren nodes vanish in P6.

---

## 4. Connecting back to Milestone 1

By P1, the joint distribution is the product of the CPD factors:
$P=\prod_i\texttt{from\_cpd}(\mathrm{CPD}_i)$. Every query is then a short expression in the
algebra:
$$P(Q\mid e)=\texttt{normalise}\Big(\sum_{\text{all but }Q}\Big(\prod_i\phi_i\Big)[e]\Big)
=\texttt{normalise}\Big(\sum_{\text{all but }Q\cup E}\ \prod_i\phi_i[e]\Big),$$
where the second form uses F8. Read left to right, the expression is brute-force enumeration: it
builds the whole joint first. Using F7 to move sums inward, past factors that do not mention the
summed variable, turns it into variable elimination.

---

## 5. Numerical policy

- **Validation (F1, F2):** every factor, including one produced by an operation, has finite,
  nonnegative entries and a shape that matches its scope.
- **Conservation:** products and sums of nonnegative numbers stay nonnegative, so operations
  cannot violate F1 except by **overflow** to $+\infty$. The finiteness check runs on every result,
  so overflow raises an error instead of propagating `inf`. (Factors derived from CPDs have entries
  at most 1, so this cannot happen in Bayesian-network inference without evidence weights.)
- **Underflow** of very small products to 0 is not detected. It is a precision limit, not a
  validity violation, and log-space factors are deferred (M2 §9 ⚑7).
- **Comparison:** factors are compared with `allclose` after aligning by name, with
  `atol=1e-12` by default. Summing in a different order (F6) can change the last few bits.
