# P8 — Sampling-based inference: rejection sampling and likelihood weighting

> Ancestral sampling (P3) draws from $P(x)$. To answer $P(Q\mid e)$ with samples, there are two
> classical approaches. **Rejection sampling** throws away samples that disagree with the
> evidence. What survives is an exact sample from $P(\cdot\mid e)$, but the yield is only
> $P(e)$. **Likelihood weighting** never throws a sample away: it fixes the evidence variables at
> their observed values and corrects the bias with importance weights. Both are checked against
> exact variable elimination (P6), which serves as their oracle.

Prerequisites: P1 (factorisation), P3 (ancestral sampling), and the M1 Bernstein bound
([`ancestral_sampling.md`](ancestral_sampling.md) §4).

---

## 1. Rejection sampling

Draw $X^{(1)},\ldots,X^{(n)}$ i.i.d. from $P$ by ancestral sampling, and keep the samples with
$X_E^{(s)}=e$.

**Proposition 1 (A1).**

1. Each sample is accepted independently with probability $P(e)$. So the number accepted,
   $A$, follows $\mathrm{Binomial}(n,P(e))$, and $A/n$ is an unbiased estimate of $P(e)$.
2. Conditional on being accepted, a sample has distribution exactly $P(\cdot\mid e)$.

*Proof.* (1) The samples are i.i.d., and acceptance is the event $X_E=e$. (2) For an assignment $x$,
$P(X=x\mid X_E=e)=P(x)\,\mathbb 1[x_E=e]/P(e)$, which is the definition of $P(x\mid e)$. $\square$

So, given $A=a$, the accepted samples are $a$ i.i.d. exact draws from the posterior. The
estimate $\hat P(Q=q\mid e)$ is a binomial proportion **out of $a$**, and M1's tolerance
applies with $N=a$.

**The curse of rare evidence.** To accept $a$ samples you need about $a/P(e)$ draws. In the
fixture, $P(L{=}1,U{=}1)\approx0.142$ costs about 7 draws per accepted sample. Evidence with
$P(e)=10^{-6}$ would cost a million. If no sample is accepted, the estimate is undefined. The
implementation raises `InsufficientSamplesError` rather than returning NaN.

## 2. Likelihood weighting as importance sampling

Sample the network in topological order, but **clamp** each evidence variable at its observed
value instead of drawing it. Give the sample the weight
$$w(x)=\prod_{i\in E}p(e_i\mid\mathrm{pa}_i(x)).$$

**Proposition 2 (A2).** Let $q$ be the distribution of the clamped samples $x_{\bar E}$. Then
$w(x)=P(x_{\bar E},e)/q(x_{\bar E})$, $w\in[0,1]$, and $\mathbb E_q[w]=P(e)$. So the mean weight
$\hat P(e)=\frac1n\sum_sw^{(s)}$ is an **unbiased** estimate of $P(e)$.

*Proof.* Clamping draws each non-evidence variable from its own CPD, given parent values that
are either sampled or clamped. So $q(x_{\bar E})=\prod_{i\notin E}p(x_i\mid\mathrm{pa}_i)$, with
$x_E=e$ substituted into the parent values. By P1, with $x_E=e$,
$P(x_{\bar E},e)=\prod_{i\notin E}p(x_i\mid\mathrm{pa}_i)\prod_{i\in E}p(e_i\mid\mathrm{pa}_i)=q(x_{\bar E})\,w(x)$.
Each factor of $w$ is a probability, so $w\in[0,1]$. Finally,
$\mathbb E_q[w]=\sum_{x_{\bar E}}q\,w=\sum_{x_{\bar E}}P(x_{\bar E},e)=P(e)$. $\square$

Wherever $P(x_{\bar E},e)>0$, $q>0$ as well, so the importance weights are well defined.

**Bounded weights reuse M1's bound.** Because $0\le w\le1$, $w^2\le w$, and so
$$\mathrm{Var}(w)=\mathbb E[w^2]-P(e)^2\le P(e)-P(e)^2=P(e)(1-P(e)).$$
This is exactly the variance bound of a Bernoulli$(P(e))$ variable. Bernstein's inequality (M1
§4) therefore gives, with no change,
$$|\hat P(e)-P(e)|\le Z\sqrt{P(e)(1-P(e))/n}+Z^2/(3n)$$
with failure probability at most $2e^{-Z^2/2}$. The same argument applies to the numerator
$\hat N_q=\frac1n\sum_sw^{(s)}\mathbb 1[Q^{(s)}=q]$, whose mean is $P(q,e)$.

## 3. The posterior estimate is a ratio, so it is biased but consistent (A3)

The likelihood-weighting posterior is $\hat R_q=\hat N_q/\hat P(e)$, the ratio of two unbiased
estimates. A ratio of unbiased estimates is **not** unbiased in general.

**An exact demonstration.** With $n=1$, the weight cancels: $\hat R_q=\mathbb 1[Q^{(1)}=q]$. So
$\mathbb E[\hat R_q]=q(Q=q)$, the probability under the *proposal*, not the posterior. In the
fixture, take $Q=A$ and $e=\{L{=}1\}$. $A$ is a root, and clamping $L$ does not change how $A$ is
sampled, so $\mathbb E[\hat R]=P(A{=}1)=0.1$, while $P(A{=}1\mid L{=}1)=65/371\approx0.175$. The
tests check this bias numerically.

**Consistency.** By the strong law of large numbers, $\hat N_q\to P(q,e)$ and
$\hat P(e)\to P(e)>0$ almost surely. Division is continuous wherever the denominator is
non-zero, so $\hat R_q\to P(q\mid e)$.

**A finite-$n$ guarantee.** Suppose both Bernstein bounds hold: $|\hat N-N|\le t_N$ and
$|\hat P(e)-P(e)|\le t_D<P(e)$. Then
$$\frac{N-t_N}{P(e)+t_D}\ \le\ \hat R\ \le\ \frac{N+t_N}{P(e)-t_D}.$$
By the union bound, this fails with probability at most $4e^{-Z^2/2}$. The tests use exactly
this interval, with $N=P(q,e)$ and $P(e)$ computed exactly by variable elimination.

## 4. Effective sample size

The weights measure how well the proposal matches the posterior. **Kish's effective sample
size** is
$$\mathrm{ESS}=\frac{(\sum_sw^{(s)})^2}{\sum_s(w^{(s)})^2}\in[1,n].$$

- **Evidence only on root variables:** each $w=\prod_{i\in E}p(e_i)$ is the same for every
  sample, so ESS $=n$. The clamped samples are then exact posterior draws, because roots have no
  parents to be influenced by.
- **Evidence downstream of the variables it depends on** (for example $L$ observed, with
  $A$ and $R$ above it): the weights vary with the sampled ancestors, and the ESS drops. Clamping
  a child does not change how its ancestors are sampled. They are drawn from their priors and
  only re-weighted afterwards.

For rejection sampling, the effective sample size is simply the number of accepted samples.

## 5. What the tests check

| Claim | Test |
|---|---|
| A1: acceptance rate $\approx P(e)$; accepted samples are posterior draws | Bernstein bounds with $N=n$ for $P(e)$, and $N=a$ for each posterior cell |
| A2: $\hat P(e)$ for likelihood weighting | Bernstein bound with $p=P(e)$ (the bounded-weight argument) |
| A3: likelihood-weighting posterior | the ratio interval of §3, with exact $N$ and $P(e)$ from variable elimination |
| A3: bias at $n=1$ | the mean of many single-sample estimates $\approx0.1\ne0.175$ |
| §4: ESS | evidence only on roots gives ESS $=n$ exactly |
| no accepted samples / zero total weight | `InsufficientSamplesError` |
