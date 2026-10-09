# P20 — Max-product: the most probable explanation and Viterbi

> Variable elimination never used anything special about addition. It used only that $\times$
> distributes over $+$. Maximisation has the same property over nonnegative numbers:
> $\max_x\,a\,f(x)=a\max_xf(x)$ when $a\ge0$. So replacing every sum by a max turns P6's algorithm
> into one that computes $\max_xP(x,e)$, the probability of the **most probable explanation**
> (MPE). Recording where each max was attained, and reading the records backwards, recovers the
> maximiser itself. On an HMM this is the Viterbi algorithm.
>
> - **Part 1 (M5.4):** semirings, max-product variable elimination, traceback, and what does not
>   carry over from sum-product.
> - **Part 2 (M5.5):** Viterbi, and why decoding each step separately is not MAP.

Prerequisites: P5 (factor algebra), P6 (variable elimination), P9 (log space).

---

# Part 1 — Max-product variable elimination

## 1. The problem

Given evidence $e$ on some variables, the MPE is an assignment $x^*$ to **all** the others that
maximises the joint probability:
$$x^*\in\arg\max_xP(x,e),\qquad\text{equivalently}\qquad\arg\max_xP(x\mid e).$$
Dividing by $P(e)$ does not change the maximiser. The library returns $x^*$ and
$\log P(x^*,e)$. Brute force costs $\prod_i|X_i|$ evaluations.

The MPE is not the same as maximising each variable's marginal separately, and it is not the
same as maximising over a subset of the variables (marginal MAP). Both points are shown in §6.

## 2. Semirings

A **commutative semiring** $(S,\oplus,\otimes)$ has two commutative, associative operations, with
$\otimes$ distributing over $\oplus$: $a\otimes(b\oplus c)=(a\otimes b)\oplus(a\otimes c)$. Four
instances matter here:

| Semiring | $\oplus$ | $\otimes$ | Computes |
|---|---|---|---|
| sum-product on $[0,\infty)$ | $+$ | $\times$ | marginals, $P(e)$ (P6) |
| log-sum-exp on $[-\infty,\infty)$ | lse | $+$ | the same, in log space (P9) |
| max-product on $[0,\infty)$ | $\max$ | $\times$ | $\max_xP(x,e)$ |
| max-sum on $[-\infty,\infty)$ | $\max$ | $+$ | $\log\max_xP(x,e)$ |

Distributivity for max-product needs $a\ge0$: $a\max(b,c)=\max(ab,ac)$. Every factor of a Bayesian
network is nonnegative. For max-sum it holds for every $a$: $a+\max(b,c)=\max(a+b,a+c)$. The
logarithm is increasing, so max-sum is max-product transported by $\log$, exactly as log-sum-exp is
sum-product transported (P9).

## 3. Max-product variable elimination is exact (V1)

P6's proof that eliminating $z$ is correct is one line: the factors that do not mention $z$ move
out of the sum because $\otimes$ distributes over $\oplus$:
$$\bigoplus_z\Big(\bigotimes_{\phi\not\ni z}\phi\Big)\otimes\Big(\bigotimes_{\phi\ni z}\phi\Big)
=\Big(\bigotimes_{\phi\not\ni z}\phi\Big)\otimes\Big(\bigoplus_z\bigotimes_{\phi\ni z}\phi\Big).$$
It uses nothing else about $\oplus$ or $\otimes$. With $\oplus=\max$, the same algorithm in any
elimination order therefore computes
$$\max_{x}\prod_\phi\phi(x)=\max_xP(x,e),$$
where the factors are the CPDs reduced by the evidence. The cost is the same as sum-product: the
size of the largest intermediate table, which depends only on the graph and the order (P6).

The library works in max-sum (log space), so a long chain of small probabilities cannot underflow.
A value of $-\infty$ means that $P(e)=0$: every assignment is impossible.

## 4. Traceback: recovering the maximiser

Eliminating $z$ forms $\psi_z=\bigotimes_{\phi\ni z}\phi$, a table over $z$ and the set $R_z$ of
other variables in those factors, and replaces it by the message $m_z(r)=\max_z\psi_z(z,r)$. Record
the table
$$\delta_z(r)=\text{the first }z\text{ (in state order) with }\psi_z(z,r)=m_z(r).$$

**Proposition 1.** Process the eliminated variables in **reverse** order and set
$z^*=\delta_z(r^*)$, where $r^*$ is the already-chosen value of $R_z$. Then $x^*$ attains the maximum.

*Proof.* When $z$ is processed, every variable of $R_z$ has already been chosen: they were all still
present when $z$ was eliminated, so they were eliminated later and are processed earlier in the
reverse pass. (Observed variables are not in any scope, since the factors are reduced.)

Let $\Phi$ be the set of factors present just before eliminating $z$, and suppose the variables
eliminated after $z$ are fixed at their chosen values. We show by induction over the reverse pass
that the product of the factors in $\Phi$, evaluated at $x^*$, equals the final maximum $M$. Before
the first reverse step, $\Phi$ is the final set of messages. All their variables have been
eliminated, so they are constants whose product is $M$. Now take one step back, to just before $z$
was eliminated. The set changes by replacing $m_z$ with $\psi_z$'s factors, and
$$\psi_z(z^*,r^*)=\max_z\psi_z(z,r^*)=m_z(r^*)$$
by the choice of $z^*$. So the product is unchanged. At the start, $\Phi$ is the set of reduced
CPDs, so $P(x^*,e)=M=\max_xP(x,e)$. $\square$

**Ties (V5).** Taking the first maximising state is a deterministic rule. Every maximiser is equally
correct; the tests compare the value always, and the assignment only when the maximiser is unique.

## 5. What does not carry over: barren pruning

For sum-product, a leaf that is neither queried nor observed can be deleted, because
$\sum_xP(x\mid u)=1$ for every $u$ (P6, Proposition 5). For max-product the corresponding quantity is
$$\max_xP(x\mid u),$$
which depends on $u$. A barren leaf therefore still pulls the maximiser towards parent values under
which the leaf is predictable. MPE uses **every** CPD, and the library does not prune.

**Example.** $X\in\{0,1\}$ with $P(X{=}1)=0.6$. $Y\in\{0,1,2\}$: if $X=0$ then $Y=0$ surely; if
$X=1$, $Y$ is uniform. The joint has $P(0,0)=0.4$ and $P(1,y)=0.2$ for each $y$. With no evidence
$Y$ is barren, and pruning it would maximise $P(X)$ and choose $X=1$. The true MPE is $(X,Y)=(0,0)$,
with probability $0.4$.

## 6. MPE versus the individually most probable values

The same example shows that the MPE's value of $X$ (0) is **not** the most probable value of $X$
alone (1, with probability 0.6). Maximising the marginal of $X$ is *marginal MAP* over $\{X\}$: a
max over $X$ of a **sum** over $Y$. The two operations do not commute: max and sum cannot be
reordered. That is why marginal MAP needs constrained elimination orders and is much harder
(spec ⚑6). Part 2 shows a sharper version on an HMM, where decoding each step separately gives a
path of probability zero.

## 7. How the tests check this independently

- **Fixture F5** (the late network): the MPE is all "no" with probability 0.45927; given Late = yes
  it is (Rain, no Accident, Traffic, Umbrella), with probability 0.11016.
- **Brute force:** on random networks with random evidence, the value equals
  $\max_xP(x,e)$ over the full joint table, the returned assignment attains it, and it is the
  brute-force maximiser whenever that is unique.
- **Order independence:** every heuristic and random explicit orders give the same value.
- **The barren example and the marginal example** of §5–6.
- **Ties:** in a model where every assignment is equally likely, the answer is all first states.
- **Underflow:** with 1,100 observations ($P(x,e)$ far below float64's range), the value matches
  $\max_c\big(\log P(e)+\log P(c\mid e)\big)$ from log-space VE.
