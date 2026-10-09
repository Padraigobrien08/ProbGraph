# P21 — Baum–Welch

> Baum–Welch is EM (P16) for an HMM. What makes it more than M4's `ExpectationMaximisation` on the
> unrolled network is **tying**: every time step uses the same $\pi$, $A$ and $B$, so the counts of
> all the copies are pooled. The E-step needs only the smoothed and pairwise posteriors from
> forward–backward (P19). The M-step is a ratio of expected counts, and each iteration does not
> decrease the likelihood.

Prerequisites: P14 (MLE), P15 (Dirichlet), P16 (EM), P18–P19 (HMMs, forward–backward).

---

## 1. The complete-data likelihood pools across time

Take sequences $s=1,\ldots,S$, with hidden paths $x^{(s)}$ and observations $y^{(s)}$. If the paths
were known, the log-likelihood would be
$$\sum_s\Big[\log\pi_{x^{(s)}_1}+\sum_{t\ge2}\log A_{x^{(s)}_{t-1}x^{(s)}_t}+\sum_{t:\,y^{(s)}_t\text{ observed}}\log B_{x^{(s)}_ty^{(s)}_t}\Big]
=\sum_iN_1(i)\log\pi_i+\sum_{i,j}N(i\to j)\log A_{ij}+\sum_{i,m}N(i,m)\log B_{im},$$
where $N_1(i)$ counts sequences that start in $i$, $N(i\to j)$ counts transitions over all
sequences and times, and $N(i,m)$ counts emissions. Because the parameters are **tied**, the counts
from different times add up before the estimate is formed. Each row is then a separate multinomial
problem, and by P14 (Gibbs' inequality) the MLE is the ratio of counts:
$$\hat\pi_i=\frac{N_1(i)}{S},\qquad\hat A_{ij}=\frac{N(i\to j)}{\sum_kN(i\to k)},\qquad\hat B_{im}=\frac{N(i,m)}{\sum_kN(i,k)}.$$
With the hidden states labelled, this is the whole of learning (`supervised_estimate`). It is also
M4's MLE for a two-node network $X_{t-1}\to X_t$ fitted to the pooled transition pairs, which the
tests use as an oracle.

## 2. The E-step: expected counts from forward–backward

When the paths are hidden, EM replaces each count by its expectation under the current model,
given each sequence's observations (P16 §4). By linearity of expectation:
$$\bar N_1(i)=\sum_s\gamma^{(s)}_1(i),\qquad
\bar N(i\to j)=\sum_s\sum_{t=1}^{T_s-1}\xi^{(s)}_t(i,j),\qquad
\bar N(i,m)=\sum_s\sum_{t:\,y^{(s)}_t=m}\gamma^{(s)}_t(i),$$
with $\gamma_t=P(X_t\mid y)$ and $\xi_t=P(X_t,X_{t+1}\mid y)$ from P19. A step with a missing
observation contributes to the transition counts but to no emission count. That is what summing
$Y_t$ out means.

**Two valid E-steps for a missing observation.** M4's generic EM on the unrolled network makes a
different, equally valid choice: it treats a missing $Y_t$ as missing *data*, part of the complete
data, and fills in the expected count $\gamma_t(i)B_{im}$ for every $m$. Both choices give an EM
algorithm (P16 applies to any split into observed and unobserved data), so both are monotone. They
also have the same fixed points. M4's M-step gives
$B'_{im}\propto\bar N_{\text{obs}}(i,m)+g_iB_{im}$, with $g_i$ the summed $\gamma_t(i)$ of the missing
steps, and $B'=B$ reduces to $B_{im}\propto\bar N_{\text{obs}}(i,m)$, which is Baum–Welch's own
update. But their iterates differ: the filled-in counts pull $B$ back towards its current value, so
that version takes smaller steps. The library sums missing observations out.

**The tying, as an oracle (W1).** On the unrolled network, M4's `expected_counts` returns one table
per family: $X_1$, each $(X_t,X_{t-1})$ and each $(Y_t,X_t)$. Summing the transition tables over $t$
must give exactly $\bar N(i\to j)$. Summing the emission tables must give $\bar N(i,m)$ plus M4's
filled-in counts $\sum_{t:\,y_t\text{ missing}}\gamma_t(i)B_{im}$. M4 computes those tables with a
junction tree per distinct row, which is entirely separate code.

## 3. The M-step, and monotonicity (W3)

The M-step is §1's ratio applied to the expected counts. By P16 Theorem 1 the observed-data
log-likelihood $\ell=\sum_s\log P(y^{(s)})$ never decreases: the argument only needs the M-step to
maximise the expected complete-data log-likelihood, and with tying it does, since that expectation
is §1's expression with $\bar N$ in place of $N$.

**Pseudocounts.** With a pseudocount $\alpha>0$, the M-step is the posterior mean under a
symmetric Dirichlet on every row: $(\bar N+\alpha)/(\text{row total}+K\alpha)$, or with $M$ in place
of $K$ for emission rows. As in P16 §7, this is the MAP under pseudocounts $\alpha+1$, so the quantity
that never decreases is
$$\ell(\theta)+\alpha\Big(\sum_i\log\pi_i+\sum_{i,j}\log A_{ij}+\sum_{i,m}\log B_{im}\Big),$$
recorded as `log_objective`. Without a pseudocount it is $\ell$ itself.

**A row with no expected count.** If $\sum_j\bar N(i\to j)=0$, row $i$ of $A$ is undetermined, exactly
like an unseen parent configuration in P14 §6. This happens when no sequence can be in state $i$ at
a time that has a successor. Without a pseudocount the M-step raises, naming the row.

## 4. Label switching and symmetric fixed points

**Label switching.** Permuting the hidden states (with $\pi$, the rows and columns of $A$, and the
rows of $B$ permuted to match) gives the same distribution over observations. So every optimum
comes with $K!$ relabelled copies, and parameters can be compared with the truth only up to a
permutation (W5).

**A symmetric fixed point (W6).** Suppose that all rows of $B$ are equal, to $b$, and that $\pi$ is
stationary for $A$: $\pi A=\pi$.

**Theorem 1.** Then one iteration gives $\pi'=\pi$, $A'=A$, and every row of $B'$ equal to the
empirical frequency of the observed symbols. That point is a fixed point of Baum–Welch.

*Proof.* With equal emission rows, $P(y\mid x)=\prod_tb_{y_t}$ does not depend on the path, so the
posterior over paths is the prior. Then $\gamma_t=\pi A^{t-1}=\pi$ (stationarity) and
$\xi_t(i,j)=\pi_iA_{ij}$. Hence $\pi'=\gamma_1=\pi$,
$$A'_{ij}=\frac{\sum_t\pi_iA_{ij}}{\sum_t\pi_i}=A_{ij},\qquad
B'_{im}=\frac{\sum_s\sum_{t:\,y_t=m}\pi_i}{\sum_s\sum_{t:\,y_t\text{ observed}}\pi_i}=\frac{\#\{y=m\}}{\#\{\text{observed}\}}.$$
The new emission rows are again equal and $\pi$ is still stationary, so the next iteration
reproduces the same point. $\square$

Without stationarity this fails: $\gamma_t=\pi A^{t-1}$ changes with $t$, so state $i$'s emission row
weights the early and late observations differently from state $j$'s, and the rows separate.

**A weakly repelling saddle (observed).** The symmetric point is a stationary point of $\ell$
(P16 §6), but on the two-state "sea" data of the tests it is a saddle, not a maximum: random starts
reach $\ell=-1584.18$, while the symmetric point has $\ell=-1656.26$. It repels only weakly, though.
Nudging one emission row by $10^{-3}$, the first step pulls the rows almost back together, and $\ell$
then creeps up by about $10^{-5}$ per iteration: $-1655.95$ after 400 iterations and $-1655.64$
after 2,000. Somewhere between iterations 2,000 and 2,500 the states finally separate, and $\ell$
jumps to the same $-1584.18$. A tolerance-based stop ($10^{-6}$) halts on the plateau long before
that and reports convergence, so **"converged" does not mean "optimal"**. Random initialisation
avoids the plateau, but not every bad point: with three hidden states, one of four random starts
was observed converging to a local optimum 168 nats below the others. Several starts, keeping the
best, are the practical remedy.

## 5. How the tests check this independently

- **Fixture F3:** one iteration from the umbrella model on $(u,u,\neg u,u,u)$ gives the exact fractions
  of the spec, and $\ell$ rises from $-3.372502$ to $-2.458385$.
- **Expected counts (W1)** against brute force over all paths, and against M4's `expected_counts`
  on the unrolled network, summed over the tied families, with several sequences and missing values.
- **Monotonicity (W3)** on every run, with and without pseudocounts.
- **Supervised counts (W4)** against M4's `maximum_likelihood` on pooled pairs.
- **M5.7:** recovery up to relabelling (W5) and Theorem 1 (W6).
