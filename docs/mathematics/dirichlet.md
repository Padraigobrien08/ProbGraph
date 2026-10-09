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

---

# Part 2 — The marginal likelihood

## 7. Definition

The **marginal likelihood** of a structure $G$ under a prior $p(\theta\mid G)$ is the probability
of the data with the parameters integrated out:
$$P(D\mid G)=\int P(D\mid\theta,G)\,p(\theta\mid G)\,d\theta.$$
Here $D$ is the *ordered* sequence of $N$ rows, as in P14. A multiset of rows would add a
multinomial coefficient, which is the same for every structure and so changes no comparison.
Unlike $\max_\theta P(D\mid\theta)$, which can only grow as edges are added, $P(D\mid G)$ averages
over the parameters. A structure with more parameters spreads its prior over more ways of
explaining the data, and it pays for that unless the data need it. This is the Bayesian score of a
structure (Bayesian Occam's razor).

## 8. The closed form

**Theorem 2 (B4).** With complete data and a prior as in §2,
$$P(D\mid G)=\prod_i\prod_u\frac{B(\alpha_{\cdot\mid u}+N(\cdot,u))}{B(\alpha_{\cdot\mid u})}
=\prod_i\prod_u\frac{\Gamma(\alpha_{\cdot\mid u})}{\Gamma(\alpha_{\cdot\mid u}+N(u))}\prod_x\frac{\Gamma(\alpha_{x\mid u}+N(x,u))}{\Gamma(\alpha_{x\mid u})}.$$

*Proof.* The likelihood and the prior both factorise over columns (P14 §4 and §2), so the
integral over all columns is a product of one integral per column:
$$\int\prod_x\theta_x^{N(x,u)}\cdot\frac{1}{B(\alpha)}\prod_x\theta_x^{\alpha_x-1}\,d\theta
=\frac{1}{B(\alpha)}\int\prod_x\theta_x^{\alpha_x+N(x,u)-1}\,d\theta=\frac{B(\alpha+N)}{B(\alpha)},$$
because the last integrand is an unnormalised $\mathrm{Dir}(\alpha+N)$, whose integral is its
normalising constant by §1. Expanding $B$ gives the gamma form. $\square$

So the score is a **ratio of normalising constants**, and it is a sum of one term per family
in log space. A column with $N(u)=0$ contributes exactly $\log1=0$. The library computes it with
`math.lgamma`, so a large $N$ cannot overflow.

## 9. The same number by sequential prediction

**Theorem 3.** For any ordering of the rows,
$$P(D\mid G)=\prod_{m=1}^{N}P\big(x^{(m)}\mid x^{(1)},\ldots,x^{(m-1)},G\big),$$
where each factor is the posterior predictive of §5 after the first $m-1$ rows.

*Proof.* The first equality is the chain rule of probability, which needs nothing about Dirichlets.
To see that it agrees with Theorem 2, follow one column $u$ of one variable. By §5, the factor
for row $m$ contributes $\big(N_{<m}(x,u)+\alpha_{x\mid u}\big)/\big(N_{<m}(u)+\alpha_{\cdot\mid u}\big)$
when row $m$ has parent configuration $u$ and value $x$, where $N_{<m}$ counts the earlier rows.
Over the whole sequence, the numerators for value $x$ run through
$\alpha_{x\mid u},\,\alpha_{x\mid u}+1,\ldots,\alpha_{x\mid u}+N(x,u)-1$, whose product is the
rising factorial $\Gamma(\alpha_{x\mid u}+N(x,u))/\Gamma(\alpha_{x\mid u})$. The denominators
likewise give $\Gamma(\alpha_{\cdot\mid u}+N(u))/\Gamma(\alpha_{\cdot\mid u})$. This is Theorem
2's factor for that column. $\square$

Each factor depends on the order, but their product does not: the rows are **exchangeable** under
the prior. The tests compute the product of predictives directly, with `bayesian_estimate`
refitted on each prefix and several random orderings. That shares no code with the gamma formula.

## 10. A distribution over datasets

Since $P(D\mid G)$ is a probability of the data, summing it over every possible sequence of $N$
rows gives exactly 1. For a few small cases the tests enumerate every dataset and check this. It
catches any wrong constant, for example a stray multinomial coefficient.

## 11. Score equivalence (B5)

Markov-equivalent structures (M2: same skeleton and v-structures) describe exactly the same set
of distributions. So a score should not prefer either of them, unless the prior does so on purpose.

**Theorem 4 (two nodes).** With BDeu, $P(D\mid R\to T)=P(D\mid T\to R)$.

*Proof.* Write $a=\lvert R\rvert$, $b=\lvert T\rvert$, and $e$ for the equivalent sample size. For
$R\to T$, the root's cells get $e/a$ and the child's cells get $e/(ab)$. So each child column
totals $e/a$, the **same** as the root's pseudocount for that value of $R$. Theorem 2 gives
$$\underbrace{\frac{\Gamma(e)}{\Gamma(e+N)}\prod_r\frac{\Gamma(e/a+N_r)}{\Gamma(e/a)}}_{R}\;\cdot\;
\underbrace{\prod_r\frac{\Gamma(e/a)}{\Gamma(e/a+N_r)}\prod_t\frac{\Gamma(e/ab+N_{rt})}{\Gamma(e/ab)}}_{T\mid R}
=\frac{\Gamma(e)}{\Gamma(e+N)}\prod_{r,t}\frac{\Gamma(e/ab+N_{rt})}{\Gamma(e/ab)}.$$
The $\Gamma(e/a+N_r)$ factors cancel. The result is symmetric in $R$ and $T$, so $T\to R$ gives
the same value. $\square$

The cancellation needed **each child column total to equal its parent's pseudocount**. K2 breaks
this: the root's cells are 1, but each child column totals $b$. So K2 is not score-equivalent,
and fixture F5 shows the two directions scoring differently. (On F1 they tie for a different
reason: $R$ and $T$ have the same margins there, so F1 cannot test B5.)

**General statement (Heckerman, Geiger and Chickering, 1995).** If the pseudocounts are
$\alpha_{x\mid u}=e\cdot P'(X_i=x,\mathrm{Pa}_i=u)$ for a single joint distribution $P'$ (the
**BDe** prior), then Markov-equivalent structures get equal marginal likelihoods. BDeu is the case
of uniform $P'$. The proof generalises Theorem 4: equivalent DAGs are linked by a sequence of
*covered* edge reversals (Chickering, 1995), and the cancellation above happens at each one. The
tests check it on random DAGs, reversing covered edges for BDeu and for BDe priors built from a
random $P'$ with `DirichletPrior.explicit`.

## 12. How the tests check this independently

- **Fixture F5**, exactly as in the spec, for both priors and both directions; also F1.
- **Sequential prediction** (Theorem 3), over random networks, priors and row orderings.
- **Quadrature**, for a single two-state column, of $\int P(D\mid\theta)\,p(\theta)\,d\theta$.
- **Summing to one** over every dataset of a small size (§10).
- **Covered-edge reversals** with BDeu and with BDe; K2 differs on F5, and a v-structure
  (not equivalent) differs even under BDeu.
