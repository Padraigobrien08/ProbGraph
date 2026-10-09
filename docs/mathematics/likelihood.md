# P14 — Data, likelihood and maximum likelihood

> A dataset is a list of joint observations, some of them with values missing. For a Bayesian
> network on complete data, the likelihood depends on the data only through **counts**, one
> table per CPD family. Maximising it splits into one small, independent problem per CPD
> column, and each has a ratio of counts as its answer. This note has two parts:
>
> - **Part 1 (M4.1):** the data model and its sufficient statistics.
> - **Part 2 (M4.2):** the likelihood, its maximiser, and what changes when values are missing.

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

# Part 2 — Likelihood and maximum likelihood

## 4. The complete-data log-likelihood decomposes

By Lemma 1,
$$\log L(\theta)=\sum_{i=1}^d\ \sum_{u\in\mathcal X_{\mathrm{pa}_i}}\ \underbrace{\sum_{x\in\mathcal X_i}N(x,u)\log\theta_{x\mid u}}_{\ell_{i,u}(\theta_{\cdot\mid u})}.$$
Each term $\ell_{i,u}$ involves only **one CPD column**, $\theta_{\cdot\mid u}$ of variable $X_i$.
Distinct columns are free to vary independently: the parameter space is a product of simplices,
one per column (P4). So maximising $\log L$ is the same as maximising each $\ell_{i,u}$ on its
own simplex. That is a family of small problems, not one large coupled one.

Conventions: $0\log0=0$, so a configuration that was never observed contributes nothing. If
$N(x,u)>0$ but $\theta_{x\mid u}=0$, then $\log L=-\infty$: the model calls an observed row
impossible.

## 5. The maximiser of one column

**Gibbs' inequality.** For distributions $p,q$ on the same finite set,
$\sum_xp_x\log q_x\le\sum_xp_x\log p_x$, with equality if and only if $q=p$ on the support of $p$.

*Proof.* $\sum_xp_x\log(q_x/p_x)\le\log\sum_{x:p_x>0}q_x\le\log1=0$, by Jensen's inequality for the
concave $\log$. Equality in Jensen forces $q_x/p_x$ to be constant on the support of $p$, and
equality in the second step forces $q$ to put no mass outside it. Together, $q=p$. $\square$

**Theorem 2 (the MLE).** If $N(u)=\sum_xN(x,u)>0$, then $\ell_{i,u}$ has the unique maximiser
$$\boxed{\hat\theta_{x\mid u}=\frac{N(x,u)}{N(u)}}.$$

*Proof.* Write $p_x=N(x,u)/N(u)$, a distribution. Then $\ell_{i,u}(\theta)=N(u)\sum_xp_x\log\theta_x$.
By Gibbs' inequality, this is at most $N(u)\sum_xp_x\log p_x$, with equality exactly at $\theta=p$. $\square$

**Optimality as a test (L3).** Any valid change to a column, such as mixing it with another
distribution, $\theta'=(1-\varepsilon)\hat\theta+\varepsilon q$, cannot increase $\log L$. It
strictly decreases it unless $\theta'=\hat\theta$ on the support of the counts. The tests apply
many such perturbations.

## 6. Unseen parent configurations

If $N(u)=0$, the column $\theta_{\cdot\mid u}$ does not appear in $\log L$ at all. **Every**
distribution maximises it, so the MLE is undefined there. This is not a numerical problem; it is
a lack of information. `maximum_likelihood` therefore **raises**, naming the configurations
(spec ⚑2). `unseen="uniform"` is an explicit opt-in, and a Dirichlet prior (P15) is the
principled fix, because it gives every column a posterior even with no data.

## 7. With missing values: the observed-data likelihood

When rows have missing values, each row contributes the probability of what *was* observed,
summing over the completions of what was not:
$$\log L_{\text{obs}}(\theta)=\sum_m\log\sum_{z}P(x^{(m)}_{\text{obs}},z\mid\theta)=\sum_m\log P(x^{(m)}_{\text{obs}}\mid\theta).$$
Each term is a $\log P(e)$, computed by M3's log-space variable elimination. Rows with identical
observed values share one computation, weighted by their multiplicity. A row the model deems
impossible gives $-\infty$.

The sum over completions sits **inside** the logarithm, so the decomposition of §4 breaks, and
there is no closed-form maximiser. EM (P16) maximises $\log L_{\text{obs}}$ by repeatedly
maximising the complete-data form instead. `maximum_likelihood` refuses incomplete data, and
points to EM (L6).

## 8. Consistency, and how close is close (L5)

In any one row, the child's value **given only its parents' values** has distribution
$\theta_{\cdot\mid u}$. That is what the CPD means. (It is *not* independent of the rest of the row:
the child is correlated with its own descendants, but the argument conditions on the parents
only.) Rows are independent. So, **conditional on which rows have parent configuration $u$**, the
child's values in those rows are $N(u)$ i.i.d. draws from $\theta_{\cdot\mid u}$. Hence
$N(x,u)\mid N(u)\sim\mathrm{Binomial}(N(u),\theta_{x\mid u})$, and:

- **Consistency.** If $P(u)>0$, then $N(u)\to\infty$ almost surely, and the strong law gives
  $\hat\theta_{x\mid u}\to\theta_{x\mid u}$.
- **A finite-sample bound.** M1's Bernstein tolerance, with $n=N(u)$ and $p=\theta_{x\mid u}$,
  bounds $|\hat\theta_{x\mid u}-\theta_{x\mid u}|$ with failure probability at most $2e^{-Z^2/2}$
  per entry. The rarest configuration dominates the error, exactly as fixture F2 shows. M4.3
  tests this on sampled data.
