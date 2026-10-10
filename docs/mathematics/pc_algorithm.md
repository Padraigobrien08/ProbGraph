# P30 — Independence tests and the PC algorithm

> Constraint-based structure learning asks the data a series of questions of the form "is $X$
> independent of $Y$ given $Z$?", and reads a CPDAG off the answers, using P28's two lemmas. This
> note has two parts:
>
> - **Part 1 (M7.4):** the G² test: what it measures, why it is approximately $\chi^2$, and how its tail
>   probability is computed exactly.
> - **Part 2 (M7.5):** the PC algorithm, which is correct with a perfect oracle.

Prerequisites: P14 (likelihood, Wilks), P17 (BIC), P28 (separating sets, Meek's rules).

---

# Part 1 — The G² test

## 1. The statistic

For discrete $X$, $Y$ and a set $Z$ (possibly empty), let $N(x,y,z)$ be the counts in rows where all of them
are observed, and write $N(x,z)$, $N(y,z)$ and $N(z)$ for the margins. Under the hypothesis
$H_0:X\perp Y\mid Z$, the expected count is $E(x,y,z)=N(x,z)N(y,z)/N(z)$. The **G² statistic** is
$$G^2=2\sum_{x,y,z:\,N>0}N(x,y,z)\log\frac{N(x,y,z)\,N(z)}{N(x,z)\,N(y,z)}.$$

**Proposition 1.** $G^2=2N\,\hat I(X;Y\mid Z)$, where $\hat I$ is the conditional mutual information, in nats, of
the empirical distribution. It is also the likelihood-ratio statistic: twice the difference in maximised
log-likelihood between the saturated model of $(X,Y,Z)$ and the model in which $X$ and $Y$ are independent
given $Z$.

*Proof.* Divide each count by $N$ to get the empirical probabilities $\hat p$. The summand becomes
$N\hat p(x,y,z)\log\frac{\hat p(x,y,z)\hat p(z)}{\hat p(x,z)\hat p(y,z)}$, which sums to $N\hat I(X;Y\mid Z)$. For the second form,
the MLE of the full model has $\hat p(x,y,z)$, and the MLE under $H_0$ is $\hat p(x\mid z)\hat p(y\mid z)\hat p(z)$ (P14,
with the factorisation $Z\to X$, $Z\to Y$). The log-likelihood ratio is $\sum N(x,y,z)\log$ of their ratio, which is
the same summand. $\square$

Since mutual information is non-negative, $G^2\ge0$, with equality exactly when the empirical table factorises.
It is also the change in log-likelihood behind adding the edge set $X\to Y$ in a score (P17): the G² test and
BIC weigh the same quantity against different thresholds.

## 2. Its distribution, and the degrees of freedom

By Wilks' theorem (P14 §10), under $H_0$ and as $N\to\infty$, $G^2\to\chi^2_d$, where $d$ is the difference in
free parameters between the two models:
$$d=\sum_z(|X|-1)(|Y|-1).$$

**Adjusting for sparse tables** (spec ⚑5). A stratum $z$ with no data contributes no information, and a value
of $x$ that never occurs in stratum $z$ has no parameter to estimate there. So the library counts, for each
stratum with $N(z)>0$, the numbers $r_z$ and $c_z$ of non-empty $x$ and $y$ margins, and uses
$$d=\sum_{z:\,N(z)>0}(r_z-1)(c_z-1).$$
Without this, sparse tables give a $d$ that is too large, and $p$-values that are too large: the test would be
too reluctant to reject. If $d=0$ there is nothing to test, and the library returns $p=1$.

## 3. Exact chi-square tails for integer degrees of freedom (T1)

The $p$-value is $Q_d(G^2)=P(\chi^2_d>G^2)=\Gamma(d/2,\,G^2/2)/\Gamma(d/2)$, an upper regularised incomplete gamma
function. For integer $d$ it has closed forms (spec ⚑6). Write $y=x/2$.

- **Even $d=2m$:** $Q=e^{-y}\sum_{j=0}^{m-1}\frac{y^j}{j!}$. This is the probability that a Poisson($y$) variable is
  below $m$, by repeated integration by parts.
- **Odd $d=2m+1$:** $Q=\operatorname{erfc}(\sqrt y)+e^{-y}\sum_{j=0}^{m-1}\frac{y^{j+1/2}}{\Gamma(j+\frac32)}$. Start from
  $Q_1=\operatorname{erfc}(\sqrt y)$, and use the recurrence $Q_{d+2}(x)=Q_d(x)+\frac{e^{-y}y^{d/2}}{\Gamma(d/2+1)}$, which is
  integration by parts once more.

Each term is computed as $\exp(-y+j\log y-\log\Gamma(\cdot))$ and the terms are added in log space, so large $d$ and
large $y$ do not overflow. For $\sqrt y>25$, $\log\operatorname{erfc}$ uses its asymptotic expansion,
$\log\operatorname{erfc}(z)\approx-z^2-\log(z\sqrt\pi)+\log\big(1-\frac1{2z^2}+\frac3{4z^4}-\frac{15}{8z^6}\big)$, whose error is
far below double precision there. The tests check these against simulated chi-square samples as well as
the standard critical values.

## 4. Calibration and its limits (T3)

Under $H_0$ the $p$-values are approximately uniform, so a test at level $\alpha$ rejects true independences
about a fraction $\alpha$ of the time. The tests check this on data sampled from a model in which $X\perp Y\mid Z$
holds by construction. Two cautions carry over to PC:

- **Small samples.** The $\chi^2$ approximation needs reasonably large expected counts. With many conditioning
  variables the strata become sparse, and the test loses power long before it loses validity.
- **Multiple testing.** PC runs many tests, each at level $\alpha$, and makes no correction. The errors compound.
  Part 2 measures the consequence.

## 5. How the tests check Part 1

- **Fixture F3:** $G^2=20.929926$, with $d=1$ and $p=4.763938\times10^{-6}$; Pearson's $\chi^2=20$ for comparison.
- **Proposition 1:** $G^2=2N\hat I$, with conditional mutual information computed from the table directly.
- **Tails:** the closed forms for small $d$; the 5% critical values (F4); the recurrence; the far tail against
  $\operatorname{erfc}$; and simulated chi-square samples for $d$ up to 50.
- **Degrees of freedom:** hand tables with empty strata and empty margins.
- **Calibration and power:** about 5% rejections under true conditional independence, and near-certain rejection
  under clear dependence.
