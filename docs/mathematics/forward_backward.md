# P19 — Forward–backward

> On a chain, inference needs only two sweeps. The **forward** sweep carries the belief about the
> current hidden state given everything seen so far (filtering). The **backward** sweep carries the
> evidence still to come. Their product gives every smoothed marginal. Both sweeps are
> variable elimination along the chain, and together they are Shafer–Shenoy message passing (P12)
> on the chain's clique tree. Normalising at every step keeps the numbers near 1 however long the
> sequence, and the normalisers multiply to the likelihood.
>
> - **Part 1 (M5.2):** the forward sweep: filtering and the likelihood.
> - **Part 2 (M5.3):** the backward sweep: smoothing, pairwise posteriors and prediction, and the
>   equivalence with Shafer–Shenoy.

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

## 4. Carrying the recursion in log space, and exact zeros (F6)

Normalising keeps $f_t$ summing to 1, but that alone does not make linear arithmetic safe. Each
entry of $a_t=(f_{t-1}A)\circ e_t$ is a product, and when the model has very small transition or
emission probabilities, a product can underflow even though the state it belongs to has a
representable, and later decisive, probability. Two failures were found by searching 3,000 random
models whose entries reach down to $e^{-760}$:

- A step whose total $\sum_ia_t(i)$ is subnormal ($10^{-308}$ to $10^{-314}$). There a float carries
  only a few significant digits, and one filtered belief came out on the wrong state: an error of 1.0.
  Scaling each $e_t$ by its maximum (an earlier design) did not prevent this.
- A filtered entry of $7.6\times10^{-124}$, which is representable, that a product flushed to 0. Later
  evidence makes that state nearly certain: its smoothed probability is about 1. The information
  lost going forward cannot be recovered going backward.

So the library carries the **same normalised recursion** of Proposition 2, but as logarithms (P9):
$$\log\big(f_{t-1}A\big)_j=\operatorname{lse}_i\big(\log f_{t-1}(i)+\log A_{ij}\big),\qquad
\log c_t=\operatorname{lse}_j\big(\log(f_{t-1}A)_j+\log e_t(j)\big),$$
$$\log f_t=\log(f_{t-1}A)+\log e_t-\log c_t.$$
Log-sum-exp with the max shift never overflows, and it underflows only for terms that are
negligible next to the largest one. Every probability that a float can represent survives as its
logarithm. The cost is still $O(TK^2)$; only the constant grows.

**Exact zeros.** $\log c_t=-\infty$ exactly when every term is $-\infty$, that is, when
$P(y_{1:t})=0$. The sequence is impossible, `log_likelihood` is $-\infty$, and the beliefs, which
would be conditional on an impossible event, raise `ZeroProbabilityEvidenceError`. A missing
observation contributes $\log c_t=0$ exactly.

## 5. How the tests check this independently

- **Fixture F1 exactly:** $9/11$, $621/703$, and $\log P(y)$ for the 5-day sequence.
- **Brute force:** filtered beliefs and $\log P$ by enumerating all $K^t$ paths, for random models and
  sequences with missing values; and **`JunctionTree` / `VariableElimination`** on the unrolled network.
- **Step likelihoods:** $c_t=P(y_{1:t})/P(y_{1:t-1})$.
- **F4:** 5,000 steps against exact rational arithmetic, while the unnormalised recursion returns 0.
- **Extreme parameters:** 40 random models with entries spread down to $e^{-760}$, against log-space
  VE, and the cases found by search (§4), where linear arithmetic is wrong by up to 1.0. Every belief
  agrees with VE to $10^{-12}$ or better.
- **F6:** impossible sequences give $-\infty$ and raise on access; missing everything gives
  $\log P=0$ and $f_t=\pi A^{t-1}$.

---

# Part 2 — Smoothing, pairwise posteriors and prediction

## 6. The backward recursion

Define $\beta_t(i)=P(y_{t+1:T}\mid X_t=i)$, the probability of the evidence still to come, given the
current state. By convention $\beta_T=\mathbf 1$.

**Proposition 3.** $\beta_t=A\,(e_{t+1}\circ\beta_{t+1})$, as a column vector.

*Proof.* Sum over the next state. Given $X_{t+1}$, the observation $y_{t+1}$ and the later evidence
$y_{t+2:T}$ are independent of $X_t$ and of each other (P18 §3). So
$$\beta_t(i)=\sum_jP(X_{t+1}=j\mid X_t=i)\,P(y_{t+1}\mid X_{t+1}=j)\,P(y_{t+2:T}\mid X_{t+1}=j)
=\sum_jA_{ij}\,e_{t+1}(j)\,\beta_{t+1}(j).\qquad\square$$

## 7. Smoothing and pairwise posteriors

**Proposition 4.** For every $t$,
$$P(X_t=i\mid y_{1:T})\propto f_t(i)\,\beta_t(i),$$
$$P(X_t=i,X_{t+1}=j\mid y_{1:T})\propto f_t(i)\,A_{ij}\,e_{t+1}(j)\,\beta_{t+1}(j),$$
where each right side is normalised over its arguments: over $i$, and over $(i,j)$.

*Proof.* By the Markov property, the past evidence and the future evidence are independent
given $X_t$, so $P(X_t=i,y_{1:T})=\alpha_t(i)\beta_t(i)$, and $\alpha_t\propto f_t$. Likewise
$P(X_t=i,X_{t+1}=j,y_{1:T})=\alpha_t(i)A_{ij}e_{t+1}(j)\beta_{t+1}(j)$, because given $X_{t+1}$ the
future after $t+1$ is independent of everything before. Dividing by $P(y_{1:T})$ gives the
posteriors, and $\alpha_t$ differs from $f_t$ by a constant that the normalisation removes. $\square$

At $t=T$, $\beta_T=\mathbf1$, so the last smoothed belief **is** the last filtered belief.

**Consistency (F3).** Summing the pairwise posterior over $j$ gives the smoothed belief at $t$,
and summing over $i$ gives the smoothed belief at $t+1$. Both are marginals of one joint
distribution.

## 8. Normalising the backward sweep

Proposition 4 uses $\beta_t$ only **up to a positive constant per step**: any constant cancels in the
normalisation. So the library propagates $\log b_t=\log\beta_t-\lambda_t$, with $\lambda_t$ chosen so that
$\max_i\log b_t(i)=0$, by log-sum-exp (P9):
$$\log b_t(i)=\operatorname{lse}_j\big(\log A_{ij}+\log e_{t+1}(j)+\log b_{t+1}(j)\big)-\lambda_t.$$
Unlike the filtered beliefs, which are probabilities summing to 1, the entries of $\beta_t$ can differ
by any factor at all: a state from which the remaining evidence is nearly impossible has a tiny
$\beta_t(i)$. Log space keeps every such ratio, where a linear message scaled to maximum 1 would flush
the small entries to zero. The recursion uses none of the forward pass's constants $c_t$, so the two
sweeps cannot share an error.

The smoothed and pairwise posteriors are then normalised exponentials (softmax) of
$\log f_t+\log b_t$ and of $\log f_t(i)+\log A_{ij}+\log e_{t+1}(j)+\log b_{t+1}(j)$. Every entry of
$\log b_t$ is finite where it matters: for a possible sequence,
$\sum_i\alpha_t(i)\beta_t(i)=P(y_{1:T})>0$.

## 9. Equivalence with Shafer–Shenoy (P12)

Unroll the HMM and reduce by the observed $Y_t$. The cliques of the chain's junction tree are
$C_t=\{X_t,X_{t+1}\}$, joined in a path $C_1-C_2-\cdots-C_{T-1}$ with separators $\{X_{t+1}\}$. Give
$C_t$ the potential $\psi_t(i,j)=A_{ij}\,e_{t+1}(j)$, and $C_1$ the extra factor $\pi\circ e_1$.
Then:

- The message from $C_{t-1}$ to $C_t$ is $\sum_{x_{t-1}}(\ldots)=\alpha_t$: the **forward** variable.
- The message from $C_{t+1}$ to $C_t$ is $\sum_{x_{t+2}}\psi_{t+1}\cdot(\ldots)=\beta_{t+1}$: the
  **backward** variable.
- The belief of $C_t$ is $\alpha_t(i)A_{ij}e_{t+1}(j)\beta_{t+1}(j)$, which is Proposition 4's pairwise table.

So forward–backward is Shafer–Shenoy on this tree, with the messages normalised. The tests compare
it with `JunctionTree` on the unrolled network. Calibration costs $2(T-2)$ messages of $O(K^2)$
each, which is the same $O(TK^2)$ as above.

## 10. Prediction (F5)

$P(X_{T+k}\mid y_{1:T})=f_TA^k$: start from the last filtered belief and run the chain forward
with no evidence (P18 §4), one step at a time. For an irreducible aperiodic chain, this converges
geometrically to the stationary distribution (P18 §6). The filtered belief is forgotten at the
rate of the second eigenvalue. For the umbrella world after $u_1,u_2$:
$$P(\text{rain}_{2+k}\mid u_1,u_2)=\tfrac12+0.4^k\big(\tfrac{621}{703}-\tfrac12\big).$$

## 11. How the tests check Part 2

- **Fixture F1:** $P(\text{rain}_1\mid u_1,u_2)=621/703$, and the five-day smoothed beliefs, as exact
  fractions.
- **Brute force and the junction tree:** smoothed and pairwise posteriors on random models with
  missing values; pairwise against `JunctionTree.query([X_t, X_{t+1}])`.
- **Consistency:** the marginals of the pairwise posteriors agree with the smoothed beliefs, and the
  last smoothed belief equals the last filtered one.
- **Prediction:** the exact $0.4^k$ formula, agreement with the unrolled network, and convergence to
  `stationary_distribution()`.
- **Extreme parameters:** smoothing against log-space VE, including cases found by search where the
  backward fallback is needed.
