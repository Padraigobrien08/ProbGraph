# P23 — Markov chain Monte Carlo: foundations

> MCMC replaces "draw independent samples from $\pi$" (impossible when $\pi$ is a huge table known
> only up to a constant) with "run a Markov chain whose stationary distribution is $\pi$". This note
> proves that the Gibbs kernels of P24 and the Metropolis–Hastings kernel of M6.3 leave $\pi$
> invariant, states the ergodic theorem and CLT that turn a chain into estimates, and derives the
> **exact asymptotic variance** from the transition matrix. On small models the tests build that
> matrix and so know, before sampling, how accurate an estimate must be.

Prerequisites: P18 (Markov chains, stationarity, closed classes), P24 (Gibbs conditionals).

---

## 1. Kernels on a finite state space

Let $\mathcal X$ be the set of assignments of the unobserved variables, and $\pi$ the target. A **kernel**
$K$ is a stochastic matrix: $K(x,x')$ is the probability of moving from $x$ to $x'$. As in P18, $\pi$ is
**stationary** if $\pi K=\pi$.

**Lemma 1 (closure).** If $K_1,\ldots,K_d$ all leave $\pi$ invariant, then so does their product
$K_1K_2\cdots K_d$ (apply them in turn: a *systematic scan*) and every mixture $\sum_jw_jK_j$ with
$w_j\ge0$, $\sum w_j=1$ (pick one at random: a *random scan*).

*Proof.* $\pi K_1K_2=\pi K_2=\pi$, and so on. $\pi\sum_jw_jK_j=\sum_jw_j\pi=\pi$. $\square$

**Detailed balance.** $K$ is **reversible** with respect to $\pi$ if $\pi(x)K(x,x')=\pi(x')K(x',x)$ for all
$x,x'$. Summing over $x$ gives $\sum_x\pi(x)K(x,x')=\pi(x')\sum_xK(x',x)=\pi(x')$: reversibility implies
stationarity. The converse fails, and systematic scans are typically not reversible, which Lemma 1 does
not need.

## 2. The single-site Gibbs kernel

$K_j$ redraws variable $j$ from its full conditional (P24):
$K_j(x,x')=\pi(x'_j\mid x_{-j})$ if $x'_{-j}=x_{-j}$, and 0 otherwise.

**Proposition 2.** $K_j$ is reversible with respect to $\pi$, hence leaves it invariant.

*Proof.* If $x'_{-j}\ne x_{-j}$ both sides of detailed balance are 0. Otherwise write $r=x_{-j}=x'_{-j}$:
$$\pi(x)K_j(x,x')=\pi(r)\,\pi(x_j\mid r)\,\pi(x'_j\mid r)=\pi(x')K_j(x',x).\qquad\square$$

By Lemma 1, the systematic sweep $K_1\cdots K_d$ and the random scan $\frac1d\sum_jK_j$ leave $\pi$
invariant. The random scan is also reversible, as a mixture of reversible kernels. The sweep is not, in
general.

**Restricting to the support.** From a state with $\pi(x)>0$, Gibbs only moves to states with
$\pi(x')>0$ (the conditional of an impossible value is 0). So the chain lives on
$\mathcal X_+=\{x:\pi(x)>0\}$, and the tests build $K$ on $\mathcal X_+$ only.

## 3. Metropolis–Hastings (used in M6.3)

Given a proposal $q(x,x')$, propose $x'$ and accept with probability
$$\alpha(x,x')=\min\Big(1,\frac{\pi(x')q(x',x)}{\pi(x)q(x,x')}\Big),$$
otherwise stay at $x$. For $x'\ne x$, $\pi(x)q(x,x')\alpha(x,x')=\min\big(\pi(x)q(x,x'),\pi(x')q(x',x)\big)$, which
is symmetric in $x$ and $x'$. That is detailed balance, so $\pi$ is stationary. Only the **ratio**
$\pi(x')/\pi(x)$ enters, so the normaliser is never needed, and for a single-site change it involves only
the Markov blanket (P24). Gibbs is the special case $q=K_j$: then $\alpha\equiv1$.

## 4. From a chain to estimates (stated)

If the chain is **irreducible** on $\mathcal X_+$ (one communicating class: every state reachable from
every other) then $\pi$ is its only stationary distribution (P18 Theorem 1), and:

- **Ergodic theorem.** For any function $f$ and any start, $\hat\mu_n=\frac1n\sum_{i=1}^nf(X_i)\to\mathbb E_\pi f$
  with probability 1.
- **Central limit theorem.** If moreover the chain is aperiodic (or the estimator averages over a
  period), $\sqrt n(\hat\mu_n-\mathbb E_\pi f)\to\mathcal N(0,\sigma^2_{\text{asym}})$, with
  $$\sigma^2_{\text{asym}}=\mathrm{Var}_\pi(f)\Big(1+2\sum_{k\ge1}\rho_k\Big),\qquad\rho_k=\mathrm{Corr}_\pi\big(f(X_0),f(X_k)\big).$$
  The factor $\tau=1+2\sum_k\rho_k$ is the **integrated autocorrelation time**: $n$ correlated samples
  are worth about $n/\tau$ independent ones.

Irreducibility is not automatic: F3 (a deterministic copy) gives a chain with two closed classes, where
the ergodic theorem converges to the wrong answer.

## 5. The exact asymptotic variance

Let $\bar f=f-\mathbb E_\pi f$ (so $\pi\bar f=0$) and let $\Pi$ be the matrix whose every row is $\pi$.

**Proposition 3.** For an irreducible aperiodic chain,
$$\sigma^2_{\text{asym}}=2\,\langle\bar f,Z\bar f\rangle_\pi-\langle\bar f,\bar f\rangle_\pi,\qquad Z=(I-K+\Pi)^{-1},$$
where $\langle g,h\rangle_\pi=\sum_x\pi(x)g(x)h(x)$.

*Proof.* Under stationarity, $\mathrm{Cov}(f(X_0),f(X_k))=\langle\bar f,K^k\bar f\rangle_\pi$, so
$\sigma^2_{\text{asym}}=\langle\bar f,\bar f\rangle_\pi+2\sum_{k\ge1}\langle\bar f,K^k\bar f\rangle_\pi=2\langle\bar f,S\bar f\rangle_\pi-\langle\bar f,\bar f\rangle_\pi$
with $S\bar f=\sum_{k\ge0}K^k\bar f$. Since $K\Pi=\Pi K=\Pi^2=\Pi$, we have $(K-\Pi)^k=K^k-\Pi$ for $k\ge1$, and
$\Pi\bar f=0$, so $K^k\bar f=(K-\Pi)^k\bar f$. For an irreducible aperiodic chain every eigenvalue of
$K-\Pi$ has modulus below 1 (P18 §6), so $\sum_{k\ge0}(K-\Pi)^k=(I-K+\Pi)^{-1}=Z$ converges. $\square$

The tests compute $Z$ with a linear solve, from a kernel built by enumerating states and using conditionals
from the full joint table. Nothing there comes from the sampler. The tolerance for a chain of $n$
recorded samples is then $Z_{\text{crit}}\sqrt{\sigma^2_{\text{asym}}/n}$ with $Z_{\text{crit}}=5$.

## 6. Two-state reductions (F1, F2)

For two variables $X\to Y$, a systematic sweep (update $X$, then $Y$) produces a new $Y$ that depends
on the old state only through the old $Y$. So the $Y$ values alone form a two-state Markov chain $K_Y$,
whose autocorrelation is $\rho_k=\lambda_2^k$, with $\lambda_2=\operatorname{tr}K_Y-1$. Hence
$\tau=\frac{1+\lambda_2}{1-\lambda_2}$. For F1, $\lambda_2=1029/2419$. For F2, $\lambda_2=(1-2\varepsilon)^2$:
the flip probability in each direction of the $Y$-chain is $2\varepsilon(1-\varepsilon)$, so
$\lambda_2=1-4\varepsilon(1-\varepsilon)=(1-2\varepsilon)^2$. The second eigenvalue of the full 4×4 sweep
kernel is the same $\lambda_2$.

## 7. Single-site Metropolis–Hastings, and Peskun's ordering (H1–H3)

The library's MH sampler picks a variable (randomly by default) and proposes one of its **other**
states uniformly: $q(x,x')=\frac1{|X_j|-1}$ for the $|X_j|-1$ neighbours $x'$ that differ only in $X_j$. The
proposal is symmetric, so the acceptance probability is $\min\big(1,\pi(x')/\pi(x)\big)$, and the ratio
involves only the factors that mention $X_j$: the same local scores as Gibbs. By §3 each site kernel
is reversible, so the random scan is reversible and the systematic scan is stationary (Lemma 1).

**Peskun's theorem (stated).** If $K_1$ and $K_2$ are both reversible with respect to $\pi$ and
$K_1(x,x')\ge K_2(x,x')$ for every $x\ne x'$, then $\sigma^2_{\text{asym}}(f,K_1)\le\sigma^2_{\text{asym}}(f,K_2)$ for
every $f$. Moving away from the current state more often can only help.

**For binary variables, MH dominates Gibbs.** From $x$, the only move at site $j$ is the flip to $x'$.
Gibbs makes it with probability $\frac{\pi(x')}{\pi(x)+\pi(x')}$, and MH with $\min\big(1,\frac{\pi(x')}{\pi(x)}\big)$, which is
at least as large: if $\pi(x')\ge\pi(x)$ it is 1, and otherwise $\frac{\pi(x')}{\pi(x)}\ge\frac{\pi(x')}{\pi(x)+\pi(x')}$.
Both random-scan kernels are reversible, so Peskun applies: on binary models, random-scan MH is never
worse than random-scan Gibbs, for any function. On F1 (fixture F4) the asymptotic variance of
$\mathbb 1[Y{=}1]$ drops from 2.158305 to 1.644986. With more than two states the dominance can fail,
because MH proposes uniformly while Gibbs moves towards likely states, so there is no general ordering.

## 8. How the tests check this independently

- **Stationarity:** $\pi K=\pi$ for systematic and random scans on random Bayesian and Markov networks
  with evidence; exactly, with fractions, on F1. The random scan also satisfies detailed balance.
- **F1–F3:** the second eigenvalues $1029/2419$ and $(1-2\varepsilon)^2$; F3's kernel is the identity
  on its support.
- **CLT tolerances:** Gibbs estimates within $5\sqrt{\sigma^2_{\text{asym}}/n}$ of the exact posterior on
  many models, seeds and scans.
- **Calibration:** across 200 replicate chains, the variance of $\hat\mu_n$ matches $\sigma^2_{\text{asym}}/n$,
  which is $\tau$ times the naive $\mathrm{Var}_\pi(f)/n$.
- **MH (M6.3):** exact MH kernels satisfy detailed balance on random models; estimates fall within exact
  CLT bounds; the acceptance rate matches its exact expectation; F4's variances, and Peskun's ordering
  on random binary models for every indicator and random functions.
