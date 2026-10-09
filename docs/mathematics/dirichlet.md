# P15 — Bayesian estimation with Dirichlet priors

> Put an independent **Dirichlet** prior on every CPD column. Because the likelihood already
> splits into one term per column (P14 §4), the posterior is again a product of independent
> Dirichlets, with the **counts added to the pseudocounts**. Point estimates follow in closed
> form. The posterior mean is a weighted average of the prior mean and the MLE, and it is
> defined even for parent configurations that never occur in the data. This note has two parts:
>
> - **Part 1 (M4.4):** priors, conjugacy, and point estimates.
> - **Part 2 (M4.5):** the marginal likelihood, which is the Bayesian score of a structure.

Prerequisites: P14 (likelihood, decomposition, MLE).

---

# Part 1 — Priors and posteriors

## 1. The Dirichlet distribution

For a distribution $\theta=(\theta_1,\ldots,\theta_K)$ on the simplex and pseudocounts
$\alpha_k>0$, with total $\alpha_0=\sum_k\alpha_k$:
$$\mathrm{Dir}(\theta\mid\alpha)=\frac{1}{B(\alpha)}\prod_{k=1}^K\theta_k^{\alpha_k-1},\qquad B(\alpha)=\frac{\prod_k\Gamma(\alpha_k)}{\Gamma(\alpha_0)}.$$

| Quantity | Value | Defined when |
|---|---|---|
| Mean | $\alpha_k/\alpha_0$ | always ($\alpha_k>0$) |
| Mode | $(\alpha_k-1)/(\alpha_0-K)$ | every $\alpha_k\ge1$ and $\alpha_0>K$ |

For $K=2$ this is the Beta distribution. $\alpha=(1,\ldots,1)$ is uniform on the simplex.

## 2. Priors over a whole network

A prior for a Bayesian network gives every CPD column $\theta_{\cdot\mid u}$ of every variable
$X_i$ its own Dirichlet, $\mathrm{Dir}(\alpha_{\cdot\mid u})$, and makes all the columns
**independent**. This is *global* independence (across variables) together with *local*
independence (across parent configurations of one variable). Three ways to choose the
pseudocounts:

| Prior | $\alpha_{x\mid u}$ | Comment |
|---|---|---|
| `uniform(alpha)` | $\alpha$ for every cell | $\alpha=1$ is the K2 / Laplace prior |
| `bdeu(ess)` | $\dfrac{ess}{\lvert\mathcal X_i\rvert\cdot q_i}$, with $q_i$ the number of parent configurations | the family's pseudocounts total exactly $ess$, the **equivalent sample size**, whatever the structure |
| `explicit({...})` | given, per variable | full control; shape $(\lvert X\rvert,\lvert U_1\rvert,\ldots)$ with parents in declaration order |

BDeu is the only one of the three whose total prior weight per family does not grow with the
number of parent configurations. P15 Part 2 shows why that matters (score equivalence).

## 3. Conjugacy

**Theorem 1.** With complete data and a prior as in §2, the posterior over the parameters is
$$p(\theta\mid D)=\prod_{i}\prod_{u}\mathrm{Dir}\big(\theta_{\cdot\mid u}\;\big|\;\alpha_{\cdot\mid u}+N(\cdot,u)\big),$$
again independent across columns.

*Proof.* By Bayes' rule, the posterior is proportional to likelihood × prior. By P14 §4, the
likelihood is $\prod_{i,u}\prod_x\theta_{x\mid u}^{N(x,u)}$. The prior is
$\prod_{i,u}\prod_x\theta_{x\mid u}^{\alpha_{x\mid u}-1}$, up to a constant. Multiply column by column:
$\prod_x\theta_{x\mid u}^{N(x,u)+\alpha_{x\mid u}-1}$, which is the unnormalised
$\mathrm{Dir}(\alpha+N)$. A product of functions of separate columns is an independent product. $\square$

So learning is **adding counts to pseudocounts**. That is also why pseudocounts are called "prior
counts": $\alpha_{x\mid u}$ behaves like imaginary observations seen before the data.

## 4. Point estimates

Write $\alpha_{\cdot\mid u}=\sum_x\alpha_{x\mid u}$ for the column total.

**Posterior mean (the default, spec ⚑4).**
$$\tilde\theta_{x\mid u}=\frac{N(x,u)+\alpha_{x\mid u}}{N(u)+\alpha_{\cdot\mid u}}
=\lambda_u\,\underbrace{\frac{N(x,u)}{N(u)}}_{\text{MLE}}+(1-\lambda_u)\,\underbrace{\frac{\alpha_{x\mid u}}{\alpha_{\cdot\mid u}}}_{\text{prior mean}},\qquad
\lambda_u=\frac{N(u)}{N(u)+\alpha_{\cdot\mid u}}.$$
This is a convex combination that shrinks the MLE towards the prior mean, by less and less as
$N(u)$ grows. When $N(u)=0$, $\lambda_u=0$ and the estimate **is** the prior mean. So unseen parent
configurations need no special case, which is the principled fix promised in P14 §6.

**Posterior mode (MAP).** $\hat\theta^{\text{MAP}}_{x\mid u}=\dfrac{N(x,u)+\alpha_{x\mid u}-1}{N(u)+\alpha_{\cdot\mid u}-K}$,
which needs every posterior pseudocount $\ge1$ and a positive denominator. The library requires
every **prior** $\alpha\ge1$, which guarantees the first condition. The denominator is then
positive unless $N(u)=0$ and every $\alpha_{x\mid u}=1$. That posterior is flat, every point is a
mode, and the column is undetermined exactly as for the MLE, so it raises. With $\alpha\equiv1$
(K2), MAP **equals** the MLE.

**Limits.**

- As $\alpha\to0$ (with $N(u)>0$), $\lambda_u\to1$ and the posterior mean tends to the MLE (B3).
- As $N(u)\to\infty$, $\lambda_u\to1$ too: with enough data the prior washes out, at rate
  $|\tilde\theta-\hat\theta|\le\alpha_{\cdot\mid u}/(N(u)+\alpha_{\cdot\mid u})$.

## 5. Posterior predictive

By parameter independence, the probability of a new complete row given the data is
$$P(x^{\text{new}}\mid D)=\prod_i\mathbb E\big[\theta_{x_i^{\text{new}}\mid u_i^{\text{new}}}\mid D\big]=\prod_i\tilde\theta_{x_i^{\text{new}}\mid u_i^{\text{new}}}.$$
The row uses one column per variable, and those columns are independent a posteriori, so the
expectation of the product is the product of expectations. The posterior-mean network is
therefore the right network for **predicting** the next row. Part 2 chains these predictions to
obtain the marginal likelihood.

## 6. How the tests check this independently

- **Quadrature.** For two- and three-state columns, the posterior mean and mode are computed by
  numerically integrating, or maximising, likelihood × prior on a fine grid. That uses no
  conjugacy at all.
- **Tallies.** Random networks and data, checked against direct counts plus pseudocounts.
- **The convex combination, and both limits.**
- **BDeu's equivalent sample size:** each family's pseudocounts must total $ess$.
