# P9 — Log-space numerics

> Store $\ell=\log\phi$ instead of $\phi$. Products become sums, and sums become
> **log-sum-exp**. With the max-shift trick, log-sum-exp never overflows, never underflows to a
> wrong zero, and never produces NaN. The logarithm maps the whole factor algebra (P5) exactly
> onto this representation, so every law carries over without a new proof.

---

## 1. Why: two failures of probability space

| Failure | Example | What float64 does |
|---|---|---|
| **Underflow** | $P(e)$ for 1,100 observations (M3 fixture F3): $\log P(e)\approx-784.9$, so $P(e)\approx10^{-341}$ | rounds to `0.0`. M2's variable elimination then **raises `ZeroProbabilityEvidenceError` for possible evidence** |
| **Overflow** | a Markov network with 200 factors of size about $100$: $Z\approx100^{200}=10^{400}$ | rounds to `inf`. `DiscreteFactor` correctly refuses it, so inference stops |

float64 represents positive numbers from about $10^{-324}$ (the smallest subnormal) up to about
$1.8\times10^{308}$. Their logarithms, however, are ordinary numbers between about $-745$ and
$709$. Every quantity above has a perfectly representable logarithm.

## 2. The representation

A **log factor** over scope $S$ is $\ell:\mathcal X_S\to[-\infty,\infty)$, standing for
$\phi=e^{\ell}$.

- $-\infty$ is allowed and means $\phi=0$, a **structural zero** (an impossible configuration).
- $+\infty$ and NaN are **not** allowed (invariant L1). They do not correspond to any valid
  nonnegative real number.

## 3. The operations

| Probability space (P5) | Log space |
|---|---|
| $(\phi\cdot\psi)(x)=\phi(x_S)\,\psi(x_T)$ | $(\ell\oplus_\times m)(x)=\ell(x_S)+m(x_T)$ |
| $\sum_Y\phi$ | $\mathrm{LSE}_Y\,\ell=\log\sum_ye^{\ell(\cdot,y)}$ |
| $\phi[e]$ (reduce) | $\ell[e]$: the same slicing |
| $\phi/Z$ (normalise) | $\ell-\mathrm{LSE}_{\text{all}}\,\ell$ |

**Addition never produces NaN.** NaN would need $(+\infty)+(-\infty)$, and $+\infty$ is
excluded (L1). Moreover $-\infty+a=-\infty$ for every $a$, which is exactly "zero times anything is
zero". The only new risk is that two huge finite log values could sum past the float range, to
$+\infty$. That is checked on every result and raised as an error, just as `DiscreteFactor`
treats overflow (P5 §5).

## 4. Log-sum-exp without overflow (the max-shift lemma)

**Lemma 1.** For $a_1,\ldots,a_n\in[-\infty,\infty)$, not all $-\infty$, let $m=\max_ia_i$ (finite).
Then
$$\log\sum_{i=1}^ne^{a_i}=m+\log s,\qquad s=\sum_{i=1}^ne^{a_i-m}\in[1,n].$$

*Proof.* Factor $e^m$ out of the sum: $\sum_ie^{a_i}=e^m\sum_ie^{a_i-m}$. Each exponent $a_i-m$ is
$\le0$, so each term is in $[0,1]$, with $e^{-\infty}=0$. The term at the maximum is exactly
$e^0=1$. So $1\le s\le n$. $\square$

**What this buys numerically.** No term overflows, because each is at most 1. The sum $s$ is at
least 1, so $\log s\in[0,\log n]$ is computed without underflow or cancellation. Terms that
underflow to 0 are each less than $2^{-1074}$ relative to a sum of at least 1, so dropping them
changes the result by at most about $n\cdot2^{-1074}$, far below rounding. Each step is a
well-conditioned floating-point operation, so the absolute error in the result is $O(n\,u)$,
with $u=2^{-53}$ (Blanchard, Higham & Higham, 2021, give the precise bound).

**The all-$-\infty$ slice.** If every $a_i=-\infty$, then $m=-\infty$, and $a_i-m$ would be
$(-\infty)-(-\infty)=\mathrm{NaN}$. The correct answer is $\log0=-\infty$. The implementation
substitutes $m=0$ in that case, so the terms become $e^{-\infty}=0$, $s=0$ and $\log s=-\infty$,
without any NaN (L4). A dedicated test pins this down.

**Normalising.** $\ell-\mathrm{LSE}(\ell)$ requires $\mathrm{LSE}(\ell)>-\infty$. If every entry is
$-\infty$, the factor has total 0 and normalising raises `NormalisationError`, the same contract as
P5's F9.

## 5. The laws transfer: exp is an isomorphism

Define $a\oplus b=\log(e^a+e^b)$ on $[-\infty,\infty)$, with $-\infty$ as its identity. Then
$$\exp(a+b)=e^a e^b,\qquad \exp(a\oplus b)=e^a+e^b,\qquad \exp(-\infty)=0,\qquad\exp(0)=1.$$
So $\exp$ is a bijection from $([-\infty,\infty),\oplus,+)$ onto $([0,\infty),+,\times)$ that
preserves both operations: a **semiring isomorphism**. Every identity built only from
$+$ and $\times$ (products, sums over variables, distributivity) holds in one structure exactly
when its image holds in the other. In particular, P5's laws F5–F8 transfer to log factors
unchanged:

- products commute and associate, with the log-unit $\ell\equiv0$ (that is, $\phi\equiv1$) as identity;
- log-sum-exps over different variables commute;
- distributivity: if $x\notin\mathrm{scope}(\ell)$ then $\mathrm{LSE}_x(\ell+m)=\ell+\mathrm{LSE}_x m$.

The tests check this in two ways. First, *through* the isomorphism: random `DiscreteFactor`s are
converted, every operation is applied on both sides, and the results are compared. Second,
*directly*, on log factors with magnitudes around $\pm500$, where the probability-space
counterpart cannot even be represented.

## 6. Comparing log factors

An absolute error $\delta$ in $\ell$ is a **relative** error of about $\delta$ in $\phi$, because
$e^{\ell+\delta}=e^\ell(1+\delta+O(\delta^2))$. That is the right notion for quantities spanning
hundreds of orders of magnitude. So `LogFactor.allclose` compares finite entries with an
absolute tolerance in log space (default $10^{-12}$), and requires $-\infty$ entries to coincide
**exactly**. A structural zero is a different thing from a very small number.

## 7. Conversions

- `LogFactor.from_factor(φ)` uses $\log0=-\infty$, with the divide-by-zero warning suppressed
  deliberately. It is exact up to rounding.
- `to_factor()` computes $e^\ell$. Entries below about $-745$ **underflow to 0**. That is
  documented, and it is the reason to stay in log space for as long as possible. Entries above
  about $709$ overflow, and `DiscreteFactor` rejects them with a `ValidationError` rather than
  holding `inf`.
