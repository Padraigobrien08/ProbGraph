# P26 — Particle filtering

> Exact filtering carries the belief state $P(I_t\mid e_{1:t})$ over the whole interface (P22), which is
> exponential in the number of entangled chains. A particle filter carries $N$ **samples** of the slice
> instead, each with a weight. It predicts by sampling each particle forward through the transition
> model, corrects by weighting with the evidence, and resamples to keep the weights balanced. It is
> M2's likelihood weighting (P8), run along time with a reset. Its estimate of $P(e_{1:T})$ is
> **exactly unbiased** for any $N$, a fact the tests check against the umbrella world's exact values.

Prerequisites: P3 (ancestral sampling), P8 (likelihood weighting), P19 (filtering), P22 (DBNs).

---

## 1. The bootstrap filter

A particle is an assignment $x^{(i)}_t$ of every template variable of slice $t$. Write $E_t$ for the
variables observed in slice $t$, with values $e_t$. At each step:

1. **Propagate.** For each particle, sample the unobserved variables of slice $t$ in intra-slice
   topological order: from $B_1$ for $t=1$, or from the transition CPDs given $x^{(i)}_{t-1}$ for $t\ge2$
   (P22). Observed variables are clamped to $e_t$ (P8).
2. **Weight.** The incremental weight is the probability of the clamped values,
   $w^{(i)}_t=\prod_{X\in E_t}P\big(e_{t,X}\mid\mathrm{pa}(X)^{(i)}\big)$.
3. **Estimate.** With $\bar W^{(i)}_{t-1}$ the normalised weights carried from the previous step ($1/N$
   after resampling), set
   $$\hat c_t=\sum_i\bar W^{(i)}_{t-1}w^{(i)}_t,\qquad\bar W^{(i)}_t=\bar W^{(i)}_{t-1}w^{(i)}_t/\hat c_t.$$
   Filtered estimates are the weighted particle frequencies $\sum_i\bar W^{(i)}_t\mathbb 1[x^{(i)}_t=x]$.
4. **Resample** (spec ⚑7). Draw $N$ particles with probabilities $\bar W_t$, and reset the weights to $1/N$.

The likelihood estimate is $\hat P(e_{1:T})=\prod_t\hat c_t$, and the library reports
$\log\hat P=\sum_t\log\hat c_t$.

## 2. The likelihood estimator is unbiased (P1)

**Theorem 1.** For any $N\ge1$, and any resampling scheme in which each particle's expected number of
offspring is $N\bar W^{(i)}$ (including none at all),
$$\mathbb E\big[\hat P(e_{1:T})\big]=P(e_{1:T}).$$

*Proof.* Let $\mathcal F_{t-1}$ be everything generated up to the end of step $t-1$. Define the weighted
particle measure $\eta_{t-1}=\sum_i\bar W^{(i)}_{t-1}\delta_{x^{(i)}_{t-1}}$, and for any function $g$ of the
previous slice write $(Qg)(x_{t-1})=\mathbb E\big[g(x_t)\,w_t(x_t)\mid x_{t-1}\big]$, the expectation under one
propagation step of the incremental weight times $g$. Two facts hold:

- **Propagation and weighting.** Given $\mathcal F_{t-1}$, each particle propagates independently, so
  $\mathbb E\big[\hat c_t\,\eta_t(g)\mid\mathcal F_{t-1}\big]=\eta_{t-1}(Qg)$, and in particular
  $\mathbb E[\hat c_t\mid\mathcal F_{t-1}]=\eta_{t-1}(Q1)$.
- **Resampling.** It changes the particles but, given the weighted measure before it, preserves its
  expectation, because each particle's expected offspring count is $N\bar W^{(i)}$. So it does not
  change any conditional expectation of the form above.

Now let $\gamma_t(g)=\mathbb E\big[g(X_t)\mathbb 1\{e_{1:t}\}\big]$, the unnormalised filter, so that
$\gamma_t(1)=P(e_{1:t})$ and $\gamma_t(g)=\gamma_{t-1}(Qg)$ by the definition of $Q$. We show by induction that
$\mathbb E\big[(\prod_{s\le t}\hat c_s)\,\eta_t(g)\big]=\gamma_t(g)$ for every $g$. For $t=1$ this is the
likelihood-weighting identity (P8). For the step,
$$\mathbb E\Big[\prod_{s\le t}\hat c_s\,\eta_t(g)\Big]=\mathbb E\Big[\prod_{s<t}\hat c_s\;\mathbb E\big[\hat c_t\eta_t(g)\mid\mathcal F_{t-1}\big]\Big]
=\mathbb E\Big[\prod_{s<t}\hat c_s\;\eta_{t-1}(Qg)\Big]=\gamma_{t-1}(Qg)=\gamma_t(g).$$
Take $g=1$ at $t=T$. $\square$

**But $\log\hat P$ is biased low (P2).** Since $\log$ is concave, Jensen's inequality gives
$\mathbb E[\log\hat P]\le\log\mathbb E[\hat P]=\log P$. If $\hat P$ is roughly lognormal with relative variance
$v/N$, the gap is about $\frac{v}{2N}$, so it shrinks like $1/N$. The tests check the sign and the
$1/N$ rate.

## 3. Weight degeneracy and resampling (P4, P5)

Without resampling, $\bar W_t$ is the product of $t$ incremental weights, normalised. Its spread grows
with $t$, and the **effective sample size** $\mathrm{ESS}_t=1/\sum_i(\bar W^{(i)}_t)^2$ collapses towards 1:
one particle carries all the weight. Resampling discards the low-weight particles and duplicates the
high-weight ones, keeping the ESS of the incremental weights bounded below.

Two unbiased schemes:

- **Multinomial.** $N$ independent draws with probabilities $\bar W$. Offspring counts are multinomial,
  with variance $N\bar W^{(i)}(1-\bar W^{(i)})$.
- **Systematic.** One uniform $u\in[0,1/N)$, and points $u+k/N$ for $k=0,\ldots,N-1$, each choosing the
  particle whose cumulative-weight interval contains it. Particle $i$ owns an interval of length
  $\bar W^{(i)}$, which contains either $\lfloor N\bar W^{(i)}\rfloor$ or $\lceil N\bar W^{(i)}\rceil$ points, and the
  expected number is exactly $N\bar W^{(i)}$. Its offspring variance is therefore at most $\frac14$ per particle,
  never more than multinomial's. This is the default.
- **Adaptive.** Systematic, but only when the ESS falls below $N/2$. Between resamplings, the weights carry
  over, and $\hat c_t$ uses them (§1, step 3).

Theorem 1 covers all of these.

## 4. Zero weight

If every incremental weight is 0, then $\hat c_t=0$, so $\hat P=0$ and $\log\hat P=-\infty$. That is a legitimate
value of an unbiased estimator: a small $N$ sometimes misses everything consistent with the evidence. The
filtered estimates after that point are undefined, and reading them raises
`ZeroProbabilityEvidenceError`, suggesting more particles.

## 5. Accuracy and cost (P3, P6)

**Accuracy.** For a fixed slice $t$, the weighted estimate $\sum_i\bar W^{(i)}_t\,g(x^{(i)}_t)$ of
$\mathbb E[g(X_t)\mid e_{1:t}]$ has error of order $1/\sqrt N$. A particle-filter central limit theorem holds
(Del Moral; stated), with an asymptotic variance that accumulates the resampling noise of earlier steps.
Halving the error needs four times the particles. The tests check that rate against forward–backward.

**Cost.** One step costs $O(N\cdot|V|)$ CPD lookups for $|V|$ template variables. That is linear in the
number of chains of a factorial HMM, whereas exact filtering needs a junction-tree clique over the whole
interface, with $\prod_k|X^{(k)}|$ entries: $2^{n}$ for $n$ binary chains (P22 §4). Because each particle
is a **joint** sample of the slice, the weighted particles represent the entangled belief state,
correlations included (`filtered_joint`), and not just its marginals.

## 6. How the tests check this independently

- **Unbiasedness (Theorem 1):** over thousands of seeds, the mean of $\hat P$ matches $703/2000$ and
  $68607401/2\cdot10^9$, within a CLT bound. This holds for every resampling scheme and for $N=1$.
- **Jensen:** the mean of $\log\hat P$ lies below $\log P$, and the gap shrinks about fourfold when $N$
  quadruples.
- **Degeneracy:** without resampling the final ESS collapses; with it the ESS stays high.
- **Resampling:** systematic offspring counts are always $\lfloor N\bar W\rfloor$ or $\lceil N\bar W\rceil$; both schemes
  have mean offspring $N\bar W$.
- **M6.6:** filtered estimates converge to forward–backward at rate $1/\sqrt N$, and the cost on M5's
  factorial HMM stays linear in the number of chains.
