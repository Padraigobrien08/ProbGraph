# P14 — Data, likelihood and maximum likelihood

> A dataset is a list of joint observations, some of them with values missing. For a Bayesian
> network on complete data, the likelihood depends on the data only through **counts**, one
> table per CPD family. Maximising it splits into one small, independent problem per CPD
> column, and each has a ratio of counts as its answer. This note has two parts:
>
> - **Part 1 (M4.1):** the data model and its sufficient statistics.
> - **Part 2 (M4.2):** the likelihood and its maximiser.

Prerequisites: P1 (factorisation).

---

# Part 1 — Data and sufficient statistics

## 1. The data model

A **dataset** over variables $V=(X_1,\ldots,X_d)$ is a sequence of rows $x^{(1)},\ldots,x^{(N)}$. Each
row assigns every variable either a state of its domain or **missing**. Rows are treated as
**independent and identically distributed (i.i.d.)** draws from the model. The order of the rows
carries no information.

- **Missing is not a state.** A missing value means "this variable was not observed". It is not
  an extra state of the variable, and it is represented by `None` and nothing else (D2). An
  empty string, or a label not in the domain, is invalid input, not "missing".
- **Every row names every variable** (D1). A row that silently omits a variable is rejected,
  because "forgotten" and "missing" must not be confused. A missing value has to be written as
  `None`.
- Internally, a dataset is an $N\times d$ integer array of state indices, with $-1$ for missing.
  It is read-only after construction (D3).

## 2. Counts are sufficient statistics

For a set of variables $S$ and a configuration $s$ of them, define
$$N(s)=\#\{m:\ x^{(m)}_S=s\},$$
counting only the rows in which **every** variable of $S$ is observed. `Dataset.counts(S)` returns
these numbers as a factor over $S$. Its total is the number of rows in which all of $S$ is observed
(D4). For $S=\varnothing$ it is the scalar $N$.

**Lemma 1 (sufficiency, for complete data).** For a Bayesian network,
$$\log P(D\mid\theta)=\sum_{m=1}^N\log P(x^{(m)}\mid\theta)=\sum_{i=1}^d\ \sum_{x_i,u_i}N(x_i,u_i)\,\log\theta_{x_i\mid u_i},$$
where $u_i$ ranges over the configurations of $\mathrm{pa}(X_i)$. So the family counts
$N(X_i,\mathrm{pa}_i)$ are **sufficient**: two datasets with the same family counts have the same
likelihood for every $\theta$.

*Proof.* By P1, $\log P(x^{(m)})=\sum_i\log\theta_{x^{(m)}_i\mid u^{(m)}_i}$. Summing over rows and
grouping equal terms gives each $\log\theta_{x\mid u}$ exactly $N(x,u)$ times. $\square$

This is why learning needs only `counts`, one call per family. It is also why the rows' order is
irrelevant: counts are invariant to permuting the rows.

## 3. Missing completely at random (for experiments)

To study learning with missing data, `Dataset.with_missing(fraction, seed)` hides each observed
cell **independently** with probability `fraction`, using a dedicated random generator. The
choice of which cells are hidden does not depend on any value, observed or not. This is
**missing completely at random (MCAR)**, the simplest case of the MAR assumption used by EM (P16).
Cells that were already missing stay missing, so masking only ever removes information.

**Checking the mask rate.** Each hidden cell is a Bernoulli trial. Over $k$ observed cells, the
number hidden is $\mathrm{Binomial}(k,\text{fraction})$. M1's Bernstein tolerance therefore bounds
the observed rate, and the tests use exactly that bound.

*(Part 2, the likelihood and its maximiser, is added in M4.2.)*
