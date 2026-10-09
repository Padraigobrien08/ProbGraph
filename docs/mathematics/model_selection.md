# P17 — Model selection: BIC and the Bayesian score

> Adding an edge can never lower the maximised likelihood, so the likelihood alone always prefers
> the complete graph. A structure score must charge for parameters. **BIC** charges
> $\frac d2\log N$, and that is no arbitrary choice: it is what the Bayesian score (P15 Part 2)
> becomes for large $N$. Both scores split into one term per family, which makes local search
> cheap. Both give Markov-equivalent structures the same score. With enough data, both pick the
> true structure's equivalence class.

Prerequisites: P14 (likelihood, Wilks), P15 Part 2 (marginal likelihood, score equivalence).

---

## 1. Definition (S1)

For a structure $G$ with $d$ free parameters (M1's P4: $d=\sum_i(|X_i|-1)\,q_i$, with $q_i$ the
number of parent configurations) and complete data of $N$ rows,
$$\mathrm{BIC}(G)=\log L(\hat\theta_G)-\frac d2\log N,$$
where $\hat\theta_G$ is the MLE (P14). Higher is better. A parent configuration that never occurs
contributes nothing to $\log L$, whatever its column holds, so it needs no special case here. The
library's `bic(structure, data)` fits $\hat\theta_G$ itself; any CPDs on `structure` are ignored.

## 2. Where the penalty comes from (Laplace, sketch)

Write $\ell(\theta)=\log P(D\mid\theta,G)$. The marginal likelihood is
$P(D\mid G)=\int e^{\ell(\theta)}p(\theta)\,d\theta$. For large $N$, $e^{\ell}$ is sharply peaked at
$\hat\theta$, so expand $\ell$ to second order there:
$\ell(\theta)\approx\ell(\hat\theta)-\frac12(\theta-\hat\theta)^\top H(\theta-\hat\theta)$, where
$H=-\nabla^2\ell(\hat\theta)$. The integral is then Gaussian:
$$\log P(D\mid G)\approx\ell(\hat\theta)+\log p(\hat\theta)+\frac d2\log2\pi-\frac12\log\det H.$$
The rows are independent, so $H\approx N\,I(\hat\theta)$, with $I$ the Fisher information of one
row, and $\log\det H=d\log N+\log\det I$. Every term except $\ell(\hat\theta)$ and
$-\frac d2\log N$ stays bounded as $N$ grows. Hence
$$\log P(D\mid G)=\mathrm{BIC}(G)+O(1).$$
The M4.5 tests check this: on the late network, the difference stays at about 4.1 while
$\frac d2\log N$ grows by 23.

## 3. Decomposability (S2)

The MLE log-likelihood is $\log L(\hat\theta)=\sum_i\sum_{x,u}N(x,u)\log\frac{N(x,u)}{N(u)}$ (P14), and
$d$ is a sum over variables. So
$$\mathrm{BIC}(G)=\sum_i\underbrace{\Big[\sum_{x,u}N(x,u)\log\frac{N(x,u)}{N(u)}-\frac{(|X_i|-1)\,q_i}{2}\log N\Big]}_{\text{depends only on }X_i\text{ and }\mathrm{Pa}_i}.$$
The Bayesian score is a sum of one term per family by P15 Theorem 2. Adding, removing or reversing
an edge therefore changes only the terms of the children whose parent sets change: one family for
an addition or removal, and two for a reversal. Search methods rely on this, because each move
costs one or two family recomputations rather than a full rescoring. `family_scores` returns the
terms.

## 4. Score equivalence (S4)

**Theorem 1.** If $G'$ is obtained from $G$ by reversing a covered edge $X\to Y$ (that is,
$\mathrm{Pa}(Y)=\mathrm{Pa}(X)\cup\{X\}$), then $\mathrm{BIC}(G')=\mathrm{BIC}(G)$.

*Proof.* Write $P=\mathrm{Pa}(X)$ and $q$ for its number of configurations. Only the families of $X$
and $Y$ change.

*Parameter count.* Before the reversal it is $(|X|-1)q+(|Y|-1)q|X|=q(|X||Y|-1)$. After the reversal,
$X$'s parents are $P\cup\{Y\}$ and $Y$'s are $P$, so it is $(|Y|-1)q+(|X|-1)q|Y|=q(|X||Y|-1)$, the
same.

*Likelihood.* The family term $\sum N(x,u)\log\frac{N(x,u)}{N(u)}$ equals $-N\,\hat H(X_i\mid\mathrm{Pa}_i)$,
where $\hat H$ is the conditional entropy under the empirical distribution. The chain rule
gives
$$\hat H(X\mid P)+\hat H(Y\mid P,X)=\hat H(X,Y\mid P)=\hat H(Y\mid P)+\hat H(X\mid P,Y).$$
The left side is the reversed pair's term before the reversal, and the right side is the term
after it. $\square$

Every two Markov-equivalent DAGs are joined by a sequence of covered-edge reversals (Chickering,
1995), so BIC is score-equivalent. The parameter counts of equivalent structures are always equal,
so the spec's condition "with equal parameter counts" holds automatically. BDeu is score-equivalent
by P15 §11, and K2 is not.

## 5. Consistency (S3, stated)

Let the data come from a distribution $P^*$ that is faithful to a DAG $G^*$. Compare $G^*$ with a
candidate $G$.

- **$G$ cannot represent $P^*$** (it lacks a needed dependence). Then
  $\frac1N\big(\log L(\hat\theta_{G^*})-\log L(\hat\theta_G)\big)\to\mathrm{KL}\big(P^*\,\|\,P_G^{\text{proj}}\big)>0$,
  using the projection of P14 §9. The likelihood gap grows **linearly** in $N$, while the penalty
  difference grows only like $\log N$. So $G^*$ wins eventually.
- **$G$ can represent $P^*$ with more parameters** (a strict superset of $G^*$'s equivalence class).
  Then $2\big(\log L(\hat\theta_G)-\log L(\hat\theta_{G^*})\big)\to\chi^2_{d_G-d_{G^*}}$ (Wilks, P14
  §10), which is $O(1)$, while the extra penalty $\frac{d_G-d_{G^*}}2\log N\to\infty$. So $G^*$ wins
  again.

The probability that BIC, or the Bayesian score by §2, ranks $G^*$'s equivalence class first among
a fixed candidate set therefore tends to 1. Within the class, the scores tie exactly (§4). At
small $N$ the penalty dominates and the scores prefer sparser structures than the truth, which is
honest underfitting rather than a bug.

## 6. How the tests check this independently

- **F1:** BIC is −15.136702 for $R\to T$ and −15.762818 for independence.
- **S1:** BIC equals `log_likelihood(maximum_likelihood(...))` minus $\frac d2\log N$, with $d$
  counted independently.
- **S2:** the family terms sum to the total, and a random edge addition changes exactly one family's
  term; this holds for BIC and BDeu.
- **S4:** covered-edge reversals on random DAGs leave BIC unchanged.
- **S3:** with 50,000 rows from the late network, the truth (or an equivalent structure) ranks first
  among subsets, supersets and equivalent variants, under both scores. The supersets fall short by
  about $\frac{\Delta d}2\log N$, as §5 predicts.
