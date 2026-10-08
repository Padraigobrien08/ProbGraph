# Conditional probability tables

> A CPD $p(X\mid U_1,\ldots,U_k)$ is not one distribution. It is a **family of
> distributions over $X$, one for each joint configuration of the parents.**
> `TabularCPD` stores that family as an array with $1+k$ axes.

---

## 1. From definition to array

Let $X$ have domain $\mathcal X$, and let each parent $U_j$ have domain
$\mathcal U_j$. A parent configuration is a tuple
$u=(u_1,\ldots,u_k)\in\mathcal U_1\times\cdots\times\mathcal U_k$. For each $u$
there are $|\mathcal X|$ numbers $p(x\mid u)$. Altogether that makes
$|\mathcal X|\cdot\prod_j|\mathcal U_j|$ numbers. These fit naturally into an
array with axes in the order $(X,U_1,\ldots,U_k)$:

$$\texttt{values}[i_X,\,i_1,\ldots,i_k]=p\big(X=\mathcal X_{i_X}\;\big|\;U_1=(\mathcal U_1)_{i_1},\ldots,U_k=(\mathcal U_k)_{i_k}\big).$$

State labels are turned into array indices by `DiscreteVariable.index_of`. This
is why each domain's order must be fixed.

**Axis convention (used throughout the library):**
$$\texttt{values.shape}=(|X|,|U_1|,\ldots,|U_k|).$$
The child is axis 0, and the parents follow on axes $1..k$ in the order they
were declared.

### Example

$X\in\{x_0,x_1\}$, $A\in\{a_0,a_1,a_2\}$, $B\in\{b_0,b_1\}$, so the shape is
$(2,3,2)$. Fixing a parent configuration such as $(a_2,b_0)$ selects the
**fibre** `values[:, 2, 0]`. That fibre is a vector of length 2, and it is the
distribution $p(X\mid A=a_2,B=b_0)$.

---

## 2. Validity: every fibre lies in the probability simplex

A vector $q\in\mathbb R^{n}$ is a probability distribution exactly when it lies
in the simplex
$$\Delta^{n-1}=\Big\{q\in\mathbb R^n : q_i\ge0,\ \textstyle\sum_i q_i=1\Big\}.$$

So `values` is a valid CPD **iff every fibre along axis 0 lies in
$\Delta^{|X|-1}$**. In terms of the invariants:

| Invariant | Statement |
|---|---|
| C3 shape | the array has exactly the shape $(|X|,|U_1|,\ldots,|U_k|)$ |
| C4 finite | every entry is a real number, with no NaN or $\pm\infty$ |
| C1 nonnegative | $p(x\mid u)\ge 0$ for all $x,u$ |
| C2 normalised | $\sum_x p(x\mid u)=1$ for all $u$, i.e. `values.sum(axis=0) == 1` everywhere |

C2 says to **sum over axis 0 only**. A common mistake is to require the *whole
table* to sum to 1. That would describe a *joint* distribution over $(X,U)$, and
its total is $1$ only when there are no parents. For $p(T\mid R,A)$, which has 4
parent configurations, the whole table sums to $4$.

C1 and C2 together already give the upper bound $p(x\mid u)\le 1$, so there is
no separate check for it.

C4 must be checked before C1 and C2. `NaN >= 0` evaluates to `False`, but
`NaN` would also propagate silently through sums. An explicit finiteness check
gives a clear error message instead of a confusing one.

---

## 3. Numerical tolerance

Floating-point values like `0.1` are not exactly representable, so
`0.1 + 0.2 + 0.7` may differ from `1.0` by a few units of machine epsilon
($\varepsilon\approx2.2\times10^{-16}$). Summing $n$ terms gives an error of
roughly $n\varepsilon$ in the worst case.

The policy is $|\sum_x p(x\mid u)-1|\le 10^{-10}$, with **absolute tolerance only**:

- It is about six orders of magnitude looser than rounding error for any
  realistic $|X|$, so correctly entered tables are never rejected.
- It is about nine orders of magnitude tighter than typical data-entry errors
  (a column summing to 0.9), so mistakes are never accepted.
- Relative tolerance is meaningless here, because the target is the constant 1.

Tables that fail are **rejected, never silently renormalised.** Dividing by the
column sum would hide the bug that produced the bad table, and it would change
probabilities the user explicitly wrote down.

---

## 4. Named lookup (C5, C6)

`probability(state, given)` takes the parent assignment as a mapping from names
to states, not as a positional tuple. The order of parent axes is a storage
detail, and named lookup keeps it out of the interface: `{"B": "b0", "A": "a2"}`
and `{"A": "a2", "B": "b0"}` select the same fibre.

The mapping must assign **exactly** the parent set:

- A **missing** parent leaves $u$ undetermined. The query has no single answer.
- An **extra** name is rejected, not ignored. Silently ignoring it would hide
  mistakes such as conditioning on a variable that is not actually a parent,
  which would be a different model.

---

## 5. Preview of P4: counting free parameters

Each fibre has $|X|$ entries, but normalisation fixes the last one. So a CPD has
$$(|X|-1)\prod_{j=1}^{k}|U_j|$$
free parameters. For the Rain/Accident/Traffic fixture:

| Representation | Free parameters |
|---|---|
| Full joint table over $(R,A,T)$ | $2^3-1=7$ |
| $p(R)+p(A)+p(T\mid R,A)$ | $1+1+4=6$ |

The single saved parameter is exactly the assertion $R\perp A$. P4 makes this
general: a node with $k$ binary parents costs $2^k$ parameters, compared with
the $2^d-1$ needed for an unrestricted joint.
