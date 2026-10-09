# P18 — Markov chains and hidden Markov models

> A hidden Markov model is a Bayesian network that repeats: a hidden state follows a Markov chain,
> and each step emits an observation. Unrolled for $T$ steps it is an ordinary M1 network, so its
> factorisation, independences and sampling are the ones already proved. What is new is the
> long-run behaviour of the chain. A finite chain always has a stationary distribution, and it has
> exactly one when the chain has exactly one closed class. If the chain is also aperiodic,
> predictions forget their starting point geometrically fast.

Prerequisites: P1 (factorisation), P3 (ancestral sampling), P4 (parameter counting), P7 (d-separation).

---

## 1. The model

A homogeneous HMM with $K$ hidden states and $M$ observation symbols has three parameters:

| Parameter | Shape | Meaning | Constraint |
|---|---|---|---|
| $\pi$ (`initial`) | $K$ | $\pi_i=P(X_1=i)$ | on the simplex |
| $A$ (`transition`) | $K\times K$ | $A_{ij}=P(X_{t+1}=j\mid X_t=i)$ | every **row** on the simplex |
| $B$ (`emission`) | $K\times M$ | $B_{im}=P(Y_t=m\mid X_t=i)$ | every **row** on the simplex |

*Homogeneous* means that $A$ and $B$ do not depend on $t$. The rows of $A$ and $B$ are
distributions, so $A$ is a **stochastic matrix**. Probability vectors are row vectors, and one step
of the chain is $p\mapsto pA$.

## 2. The unrolled network (H2)

For a length $T$, `to_bayesian_network(T)` builds the DAG with edges $X_t\to X_{t+1}$ and
$X_t\to Y_t$. Its CPDs are $P(X_1)=\pi$, $P(X_{t+1}\mid X_t)=A_{X_t,\cdot}$ and
$P(Y_t\mid X_t)=B_{X_t,\cdot}$. As a TabularCPD has the child on axis 0, the tables are $A^\top$ and
$B^\top$. By P1, its joint distribution is
$$P(x_{1:T},y_{1:T})=\pi_{x_1}B_{x_1y_1}\prod_{t=2}^{T}A_{x_{t-1}x_t}B_{x_ty_t}.$$
This *is* the HMM's definition, so unrolling is exact. Every M1–M4 tool then applies to an HMM,
and the tests use this network as the oracle for the specialised algorithms.

The unrolled network has $T$ copies of each CPD but only one set of numbers. The parameters are
**tied**, so the model has
$$d=(K-1)+K(K-1)+K(M-1)$$
free parameters, whatever $T$ is, rather than the unrolled network's $n_\text{free}$ (P4).

## 3. Conditional independences

Reading d-separation (P7) off the chain gives the two facts every HMM algorithm uses:

1. **Markov property.** Given $X_t$, the future $(X_{t+1:T},Y_{t+1:T})$ is independent of the past
   $(X_{1:t-1},Y_{1:t})$. Every trail from past to future passes through $X_t$ as a chain or a fork,
   and conditioning on $X_t$ blocks it.
2. **Emissions are local.** Given $X_t$, $Y_t$ is independent of every other variable, because its
   only neighbour is $X_t$.

The observations by themselves are **not** a Markov chain: $Y_{t+1}$ depends on all of
$Y_{1:t}$ through the unobserved $X_t$. That is why the model is useful. A small hidden state can
carry arbitrarily long memory about the observations.

## 4. The marginal chain and prediction

Without evidence, the marginal of the hidden state evolves linearly:
$$p_{t+1}(j)=\sum_ip_t(i)A_{ij},\qquad\text{so}\qquad p_t=\pi A^{t-1}.$$
This is the sum rule plus the Markov property. Prediction *with* evidence (M5.3) is the same
recursion started from a filtered belief instead of $\pi$.

## 5. Stationary distributions (H4)

A distribution $\mu$ is **stationary** if $\mu A=\mu$: starting from $\mu$, the chain stays at $\mu$.

**Communicating classes.** State $j$ is *reachable* from $i$ if $(A^n)_{ij}>0$ for some $n\ge0$. States
that are reachable from each other form a *communicating class*. A class is **closed** if no
transition leaves it. Every finite chain has at least one closed class: follow transitions out of
classes until none is possible, which must happen because the classes form a finite DAG.

**Theorem 1.** A finite chain has a stationary distribution. It is unique if and only if the chain
has exactly one closed class $C$. Then the stationary distribution is zero outside $C$ and
positive on $C$.

*Proof.*

*Existence.* Each closed class $C$ is a chain in its own right, with matrix $A_{CC}$, which is
stochastic and irreducible. By the Perron–Frobenius theorem (stated, not proved here), an
irreducible stochastic matrix has eigenvalue 1 with a one-dimensional left eigenspace, spanned by a
vector $\mu_C>0$. Normalise it and extend it by zeros: since $C$ is closed, $\mu_CA=\mu_C$ on the
whole chain.

*Not unique with two closed classes.* With two closed classes $C\ne C'$, both $\mu_C$ and $\mu_{C'}$
are stationary, and so is every mixture of them.

*Unique with one closed class.* Let $\mu$ be stationary, and let $S$ be the set of states outside
$C$. Every state of $S$ is transient: from it, some path reaches $C$ (the closed class is reached
by following transitions out of non-closed classes), and $C$ is never left. So there are $n$ and
$\varepsilon>0$ such that from every state of $S$ the chain enters $C$ within $n$ steps with
probability at least $\varepsilon$. Stationarity gives $\mu=\mu A^{kn}$ for every $k$, and the mass
that $\mu A^{kn}$ puts on $S$ is at most $(1-\varepsilon)^k\mu(S)\to0$. Hence $\mu(S)=0$. On $C$,
$\mu$ is then a stationary distribution of the irreducible chain $A_{CC}$, which is unique by
Perron–Frobenius. $\square$

**Computation.** The library finds the closed classes from the reachability relation of the
graph $\{(i,j):A_{ij}>0\}$, so this depends on exact zeros, not on a tolerance. With exactly one
closed class $C$ it solves $\mu_C(A_{CC}-I)=0$ with $\sum_{i\in C}\mu_i=1$, replacing one equation
by the normalisation. That system is nonsingular because the eigenvalue 1 is simple. With two or
more closed classes, `stationary_distribution` raises and names them.

## 6. Convergence

**Two states, exactly.** Write $A=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix}$ with $a+b>0$. The
stationary distribution is $\mu=\big(\frac b{a+b},\frac a{a+b}\big)$. Since the entries of
$p_t-\mu$ sum to zero, write $p_t-\mu=\delta_t\,(1,-1)$. Then $(1,-1)A=(1-a-b)(1,-1)$, and so
$$p_{t+k}-\mu=(1-a-b)^k\,(p_t-\mu).$$
The gap shrinks geometrically at rate $\lvert1-a-b\rvert$, which is the second eigenvalue of $A$.
For the umbrella world, $a=b=0.3$ and the rate is $0.4$. If $a=b=1$ the rate is 1: the chain
alternates deterministically and never converges, although its stationary distribution
$(\frac12,\frac12)$ is unique.

**In general (stated).** If the chain has one closed class and that class is **aperiodic** (the
gcd of its return times is 1), every other eigenvalue of $A$ has modulus less than 1, and
$\lVert pA^k-\mu\rVert\le C\rho^k$ for some $C$ and $\rho<1$, whatever $p$ is. Periodicity is the
only obstacle, as the alternating chain shows.

## 7. Sampling (H3)

`sample(length, seed)` is ancestral sampling (P3) on the unrolled network, in the order
$X_1,Y_1,X_2,Y_2,\ldots$: each value is drawn by inverse CDF from its row of $\pi$, $A$ or $B$, given
the values already drawn. It draws a $(T,2)$ block of uniforms, with row $t$ used for $X_t$ and $Y_t$.
The uniforms come in row-major order, so a shorter sequence with the same seed is a prefix of a
longer one.

## 8. How the tests check this independently

- **H1:** each malformed input is rejected with a message naming the parameter and row.
- **H2:** the unrolled joint is compared with the formula above for every assignment of small models,
  and it sums to 1.
- **H3:** frequencies of $X_1$, of transitions $(X_t,X_{t+1})$ and of emissions $(X_t,Y_t)$ in long
  samples are within M1's Bernstein bounds; seeds reproduce; prefixes agree.
- **H4:** $\mu A=\mu$ on random chains; the transient states get zero mass; two closed classes raise;
  the two-state formula; the alternating chain is unique but does not converge; $pA^k$ approaches
  $\mu$ at the rate of §6.
