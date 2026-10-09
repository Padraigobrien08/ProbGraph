# P19 — Forward–backward

> On a chain, inference needs only two sweeps. The **forward** sweep carries the belief about the
> current hidden state given everything seen so far (filtering). The **backward** sweep carries the
> evidence still to come. Their product gives every smoothed marginal. Both sweeps are
> variable elimination along the chain, and together they are Shafer–Shenoy message passing (P12)
> on the chain's clique tree. Normalising at every step keeps the numbers near 1 however long the
> sequence, and the normalisers multiply to the likelihood.
>
> - **Part 1 (M5.2):** the forward sweep: filtering and the likelihood.
> - **Part 2 (M5.3):** the backward sweep: smoothing, pairwise posteriors and prediction.

Prerequisites: P6 (variable elimination), P9 (log space), P12 (message passing), P18 (HMMs).

---

# Part 1 — Filtering and the likelihood

## 1. Notation

Observations $y_{1:T}$, with some possibly missing. The **evidence vector** of step $t$ is
$$e_t(i)=\begin{cases}B_{i,y_t}&\text{if }y_t\text{ is observed},\\1&\text{if }y_t\text{ is missing.}\end{cases}$$
A missing $Y_t$ is summed out: $\sum_mB_{im}=1$, so it contributes a factor 1. This is exactly
what the unrolled network gives with $Y_t$ unobserved, because $Y_t$ is then a barren leaf (P6).
Write $u\circ v$ for the elementwise product of two vectors.

## 2. The forward recursion

Define $\alpha_t(i)=P(y_{1:t},X_t=i)$, the joint probability of the evidence so far and the current
state. (For a missing $y_s$, read $y_{1:t}$ as the observed values only.)

**Proposition 1.** $\alpha_1=\pi\circ e_1$, and $\alpha_t=(\alpha_{t-1}A)\circ e_t$ for $t\ge2$.
Hence $P(y_{1:T})=\sum_i\alpha_T(i)$.

*Proof.* $\alpha_1(i)=P(X_1=i)P(y_1\mid X_1=i)$. For $t\ge2$, sum over the previous state, and use the
Markov property and local emissions (P18 §3):
$$\alpha_t(j)=\sum_iP(y_{1:t-1},X_{t-1}=i)\,P(X_t=j\mid X_{t-1}=i)\,P(y_t\mid X_t=j)
=\Big(\sum_i\alpha_{t-1}(i)A_{ij}\Big)e_t(j).\qquad\square$$

This is variable elimination on the unrolled network in the order $X_1,X_2,\ldots$: eliminating
$X_{t-1}$ multiplies the incoming message $\alpha_{t-1}$ by the factor $A$ and sums $X_{t-1}$ out.
Each step costs $O(K^2)$, so the whole sweep costs $O(TK^2)$. Enumeration would cost $O(K^T)$.

## 3. Normalising: filtering, and a likelihood that cannot underflow

$\alpha_t$ is a probability of $t$ observations, so it shrinks geometrically: in F4,
$P(y_{1:5000})\approx e^{-2070}$, far below the smallest positive float64 ($\approx e^{-745}$). The
unnormalised recursion then fails in one of two ways. Either it returns exactly 0, or, worse, it
gets **stuck at the smallest subnormal** $5\times10^{-324}$: for 5,000 umbrellas each step multiplies
by about $0.63$, and $0.63\times5\times10^{-324}$ rounds back *up* to $5\times10^{-324}$. It then
reports $\log P\approx-744.4$ for a true $-2069.6$, wrong by over 1,300 nats, with no error.

Instead, carry the **filtered** belief $f_t=P(X_t\mid y_{1:t})=\alpha_t/\sum_i\alpha_t(i)$.

**Proposition 2.** Let $f_0A$ stand for $\pi$. For $t\ge1$, let
$$a_t=(f_{t-1}A)\circ e_t,\qquad c_t=\sum_ia_t(i),\qquad f_t=a_t/c_t.$$
Then $f_t=P(X_t\mid y_{1:t})$, $c_t=P(y_t\mid y_{1:t-1})$, and
$$\log P(y_{1:T})=\sum_{t=1}^T\log c_t.$$

*Proof.* By induction, $\alpha_t=\big(\prod_{s\le t}c_s\big)f_t$. That holds for $t=1$ by definition.
If it holds for $t-1$, then by Proposition 1,
$\alpha_t=(\alpha_{t-1}A)\circ e_t=\big(\prod_{s<t}c_s\big)a_t=\big(\prod_{s\le t}c_s\big)f_t$.
Summing over states gives $P(y_{1:t})=\prod_{s\le t}c_s$, so $f_t=\alpha_t/P(y_{1:t})$ is the filtered
posterior, and $c_t=P(y_{1:t})/P(y_{1:t-1})$. $\square$

Each $f_t$ sums to 1, so it cannot underflow as a whole, and each $c_t$ is a one-step predictive
probability, typically of order 1. The tiny joint probability lives only in the *sum of logs*.

**Prediction is part of the step.** $f_{t-1}A=P(X_t\mid y_{1:t-1})$ is the one-step prediction
(P18 §4), and multiplying by $e_t$ and normalising is Bayes' rule. Filtering alternates
*predict* and *update*.

## 4. Exact zeros and very small emissions (F6)

If $c_t=0$, then $P(y_{1:t})=0$: the sequence is impossible. Then `log_likelihood` is $-\infty$, and
the beliefs, which would be conditional on an impossible event, raise
`ZeroProbabilityEvidenceError`.

One step can still underflow when the sequence is possible: an emission probability can be
subnormal (say $10^{-320}$), and subnormal numbers carry only a few significant digits. Two
measures keep the arithmetic exact up to rounding:

1. **Scale the evidence vector by its maximum.** Write $e_t=m_t\,\tilde e_t$ with
   $m_t=\max_ie_t(i)$, so $\max_i\tilde e_t(i)=1$. Then
   $c_t=m_t\sum_i(f_{t-1}A)_i\tilde e_t(i)$, and $\log c_t=\log m_t+\log\tilde c_t$. The factor
   $m_t$ cancels from $f_t$, and its logarithm is computed directly, never from a subnormal product.
2. **A log-space fallback.** If $\tilde c_t$ is still below $10^{-280}$, the step is redone with
   log-sum-exp (P9): $\log\tilde c_t=\operatorname{lse}_i\big(\log(f_{t-1}A)_i+\log\tilde e_t(i)\big)$,
   which is $-\infty$ exactly when every term is.

So $c_t=0$ is reported only when the sequence is truly impossible.

## 5. How the tests check this independently

- **Fixture F1 exactly:** $9/11$, $621/703$, and $\log P(y)$ for the 5-day sequence.
- **Brute force:** filtered beliefs and $\log P$ by enumerating all $K^t$ paths, for random models and
  sequences with missing values; and **`JunctionTree` / `VariableElimination`** on the unrolled network.
- **Step likelihoods:** $c_t=P(y_{1:t})/P(y_{1:t-1})$.
- **F4:** 5,000 steps against exact rational arithmetic, while the unnormalised recursion returns 0.
- **Extreme parameters:** 40 random models with entries spread down to $e^{-760}$, against log-space
  VE. Three more, found by searching 3,000 models, have a step whose scaled total is subnormal. Without
  the log-space fallback, one of them filters to the wrong state entirely (an error of 1.0); with it,
  every belief agrees with VE to $10^{-13}$.
- **F6:** impossible sequences give $-\infty$ and raise on access; missing everything gives
  $\log P=0$ and $f_t=\pi A^{t-1}$.
