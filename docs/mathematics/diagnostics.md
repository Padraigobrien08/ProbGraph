# P25 — MCMC diagnostics

> An MCMC estimate is only as good as the chain behind it. Diagnostics answer two questions from the
> samples alone. **How many independent samples is this chain worth?** That is the effective sample
> size, which comes from the autocorrelation. **Has the chain forgotten where it started?** That is
> $\hat R$, which compares several chains. Both can be fooled, and this note says exactly when.

Prerequisites: P18 (Markov chains), P23 (ergodic theorem, CLT, asymptotic variance).

---

## 1. Autocorrelation

For a scalar series $f_1,\ldots,f_n$ (say, an indicator of a state) with mean $\bar f$, the library uses
$$\hat\gamma_k=\frac1n\sum_{i=1}^{n-k}(f_i-\bar f)(f_{i+k}-\bar f),\qquad\hat\rho_k=\hat\gamma_k/\hat\gamma_0.$$
Dividing by $n$ rather than $n-k$ makes the estimated autocovariance sequence positive semi-definite,
which keeps every estimate below sane. It is computed with the FFT in $O(n\log n)$, and the tests check
it against the direct sum. A **constant** series has $\hat\gamma_0=0$: its autocorrelation, $\tau$ and ESS
are undefined, and the library returns `nan` rather than a misleading number.

## 2. Integrated autocorrelation time and ESS

By P23 §4, $n$ correlated samples estimate a mean with variance $\approx\tau\,\mathrm{Var}(f)/n$, where
$\tau=1+2\sum_{k\ge1}\rho_k$. So the **effective sample size** is $\mathrm{ESS}=n/\tau$, and the **Monte
Carlo standard error** is
$$\mathrm{MCSE}=\sqrt{\hat\gamma_0\,\hat\tau/n}=\sqrt{\hat\gamma_0/\mathrm{ESS}}.$$

Summing all the estimated $\hat\rho_k$ does not work: the noise in the large-$k$ terms is as big as the
signal. **Geyer's initial monotone sequence estimator** (spec ⚑5) truncates in a principled way. Let
$\Gamma_m=\rho_{2m}+\rho_{2m+1}$. For a reversible chain these pair sums are positive, decreasing and
convex (Geyer, 1992). So the estimator:

1. computes $\hat\Gamma_m$ for $m=0,1,\ldots$ and stops before the first $\hat\Gamma_m\le0$;
2. replaces each $\hat\Gamma_m$ by $\min(\hat\Gamma_m,\hat\Gamma_{m-1})$, making the sequence monotone;
3. returns $\hat\tau=-1+2\sum_m\hat\Gamma_m$. With $\Gamma_0=1+\rho_1$, this equals $1+2\sum_k\hat\rho_k$
   over the kept lags.

It needs no window to tune. It also handles negative autocorrelation correctly: a chain that tends to
alternate has $\tau<1$, so it is *more* efficient than independent sampling.

## 3. The two-state chain, exactly (D1, fixture F6)

Take a two-state chain that stays put with probability $p$ in either state. Its second eigenvalue is
$\lambda=2p-1$. Since any function of the state is affine in the indicator of state 1, its
autocorrelation is exactly $\rho_k=\lambda^k$. Then
$$\tau=1+2\sum_{k\ge1}\lambda^k=\frac{1+\lambda}{1-\lambda}=\frac p{1-p},\qquad\mathrm{ESS}=n\frac{1-p}p.$$
For $p=0.9$, $\tau=9$. For $p=0.25$, $\lambda=-\frac12$ and $\tau=\frac13$. The F1 and F2 Gibbs chains reduce
to exactly such chains in $Y$ (P23 §6), with $\tau=1724/695$ and $\tau=\frac{1+(1-2\varepsilon)^2}{1-(1-2\varepsilon)^2}$.

## 4. $\hat R$: did the chains forget their starts?

Run $c$ chains of length $2n$ from **overdispersed** starts, and split each in half, giving $m=2c$ sequences
of length $n$. With sequence means $\bar f_j$, overall mean $\bar f$, and within-sequence variances $s_j^2$:
$$B=\frac n{m-1}\sum_j(\bar f_j-\bar f)^2,\qquad W=\frac1m\sum_js_j^2,\qquad
\widehat{\mathrm{var}}^+=\frac{n-1}nW+\frac1nB,\qquad\hat R=\sqrt{\widehat{\mathrm{var}}^+/W}.$$
If every sequence has reached stationarity, $B$ and $W$ estimate the same variance and $\hat R\to1$. If
the chains still reflect their starts, $B$ is inflated and $\hat R>1$. The usual threshold is 1.01–1.1.
Splitting each chain also catches a single chain that drifts between its first and second halves.

**Edge cases.** If $W=0$ and $B>0$, the sequences are each constant but disagree: $\hat R=\infty$ (F3). If
$W=B=0$, every sample is identical, nothing can be concluded, and $\hat R$ is `nan`.

## 5. What diagnostics cannot see (D4)

- **One chain cannot detect a mode it rarely visits.** In F2 with $\varepsilon=0.005$ ($\tau\approx100$), a
  Gibbs chain of 500 sweeps flips between the modes only a few times. Over 40 replicates, about half
  of single chains pass split-$\hat R<1.1$, while their estimate of $P(Y{=}1)=\frac12$ is off by about 0.2.
  Four chains started in *different* modes give $\hat R>1.1$ in about 90% of replicates. With
  $\varepsilon=10^{-4}$ the chains do not flip at all, and the situation is that of F3.
- **Determinism (F3).** Each chain is constant, so its ESS is undefined, and several chains give
  $\hat R=\infty$. A diagnostic can only say that the chains disagree. The cure is a different sampler
  (P27), not more samples.
- **Passing is necessary, not sufficient.** All chains can agree and still all miss a region that none
  of the starts reaches. Overdispersed starts make that unlikely, not impossible.

## 6. How the tests check this independently

- **Autocorrelation** against the direct $O(n^2)$ sum.
- **Two-state chains** simulated directly: $\hat\rho_k\approx\lambda^k$, $\hat\tau\approx p/(1-p)$, including $p<\frac12$,
  where $\tau<1$.
- **Gibbs on F1 and F2:** $\hat\tau$ matches $1724/695$ and $(1+\lambda_2)/(1-\lambda_2)$.
- **Independent draws:** $\hat\tau\approx1$, ESS $\approx n$.
- **Calibration:** over 400 replicate chains, $|\hat\mu-\mu|\le2\,\mathrm{MCSE}$ about 95% of the time.
- **$\hat R$:** about 1 for mixing chains; above 1.1 for F2 from opposite modes while a single chain looks
  fine; $\infty$ for F3.
