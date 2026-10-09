# P4 — Parameter reduction

> **Theorem.** A Bayesian network over $G$ with cardinalities $k_i=|X_i|$ has
> $$\#\theta(G)=\sum_{i=1}^{d}(k_i-1)\prod_{j\in\mathrm{pa}(i)}k_j$$
> free parameters. An unrestricted joint distribution has $\prod_i k_i-1$.
> The two counts are equal when $G$ is complete. Each missing edge is an
> independence assumption, and it reduces the count.

---

## 1. Counting

**Unrestricted joint.** There are $\prod_ik_i$ cells, which must be
nonnegative and sum to 1. That is a point of the simplex
$\Delta^{\prod k_i-1}$, so there are $\prod_ik_i-1$ free numbers.

**One CPD.** $p(X_i\mid\mathrm{pa}_i)$ has one column for each parent
configuration, so $\prod_{j\in\mathrm{pa}(i)}k_j$ columns. Each column is a point
of $\Delta^{k_i-1}$ (C1, C2), so it has $k_i-1$ free numbers. The columns are
unconstrained by each other. This gives `TabularCPD.n_free_parameters`.

**The network.** The CPDs are chosen independently, so their counts add. This
gives `BayesianNetwork.n_free_parameters`. It depends only on the structure,
not on the CPD values.

## 2. A complete DAG loses nothing (telescoping)

Order the nodes topologically. In a complete DAG, every earlier node is a
parent. Let $K_i=\prod_{j\le i}k_j$, with $K_0=1$. Then
$$\#\theta=\sum_{i=1}^{d}(k_i-1)K_{i-1}=\sum_{i=1}^{d}(K_i-K_{i-1})=K_d-K_0=\prod_ik_i-1.$$
This matches the chain rule: a complete DAG asserts no conditional
independences, so it can represent every joint distribution.

Removing an edge $j\to i$ divides node $i$'s column count by $k_j$. So each
missing edge strictly reduces $\#\theta$, provided $k_j\ge2$ and $k_i\ge2$.

## 3. Examples (binary variables)

| Model | $\#\theta(G)$ | Full joint $2^d-1$ |
|---|---|---|
| Rain/Accident/Traffic, $R\rightarrow T\leftarrow A$ | $1+1+4=6$ | $7$ |
| Chain $X_1\to\cdots\to X_d$ | $1+2(d-1)=2d-1$ | $2^d-1$ |
| Naive Bayes: class $C$, $n$ features | $1+2n$ | $2^{n+1}-1$ |
| Every node has at most $m$ parents | $\le d\,2^m$ | $2^d-1$ |

The last row is the point of the whole construction. With bounded in-degree,
the cost grows **linearly** in $d$ instead of exponentially. In the fixture, the
single saved parameter is exactly the one missing edge between $R$ and $A$,
which is the assumption $R\perp A$.

## 4. The count is the true dimension

Could some of the $\#\theta$ numbers be redundant? For example, could two
different sets of CPDs produce the same joint distribution? For strictly
positive CPDs, no:

- **Injectivity.** P2 corollary C-b says that if $P>0$, then
  $P(x_i\mid\mathrm{pa}_i)=p(x_i\mid\mathrm{pa}_i)$. The CPDs can be read back
  from the joint, so the map $\theta\mapsto P_\theta$ is one-to-one on the
  positive region.
- **Dimension.** $\theta\mapsto P_\theta$ is a polynomial map from an open set
  of $\mathbb R^{\#\theta}$ into the joint simplex. If its Jacobian has full
  rank $\#\theta$, its image is locally a $\#\theta$-dimensional surface, and
  no parameter is redundant.

The family of networks over $G$ is therefore a $\#\theta(G)$-dimensional
subset of the $(\prod k_i-1)$-dimensional simplex. When $G$ is not complete,
it is a proper, lower-dimensional subset.

*Caveat.* If some parent configuration $u$ has probability 0, the column
$p(\cdot\mid u)$ never contributes to any joint value, so it cannot be
recovered. Injectivity, and the parameter count as a dimension, hold only on
the positive region.

## 5. Numerical experiment

`tests/test_parameter_reduction.py` makes §4 concrete:

1. Parameterise each CPD column by its first $k-1$ entries, with the last entry
   set to $1-\sum$. Draw an interior point $\theta$ from a Dirichlet
   distribution.
2. Compute the Jacobian $\partial P_\theta(x)/\partial\theta$ by central
   differences. *Each* $P_\theta(x)$ takes exactly one entry from each CPD, so
   it is affine in each individual coordinate. Central differences are then
   exact up to floating-point rounding.
3. Check that the numerical rank is $\#\theta(G)$, and that it is below
   $\prod k_i-1$ when edges are missing.

The experiment runs on the fixture, a chain, naive Bayes, a complete DAG with
mixed cardinalities, and random DAGs.
