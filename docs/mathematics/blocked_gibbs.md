# P27 — Blocked Gibbs sampling

> Single-site Gibbs fails when variables are tightly coupled. In F2 every move must go through an
> improbable intermediate state, and in F3 that state is impossible. The cure is to update the coupled
> variables **together**: draw the whole block from its joint conditional. That conditional is again
> local, and it is exact. The chain stays stationary for the same reason as before, and on F2 and F3
> it mixes perfectly.

Prerequisites: P23 (kernels, stationarity), P24 (local conditionals), P25 (diagnostics).

---

## 1. The block conditional (B1)

Let $B$ be a set of unobserved variables, and $x_{-B}$ the current values of all the others. As in P24
Proposition 1, factors that do not mention $B$ cancel:
$$\pi(x_B\mid x_{-B})=\frac{\prod_{k:\,S_k\cap B\ne\emptyset}\phi_k(x_B,x_{-B})}{\sum_{x'_B}\prod_{k:\,S_k\cap B\ne\emptyset}\phi_k(x'_B,x_{-B})}.$$
The library reduces each such factor by the fixed values of its non-block variables, adds the reduced
log tables over the block's joint state space (broadcasting each over the block variables it lacks), and
draws one joint state from the softmax. The cost per block update is $\prod_{X\in B}|X|$ table cells, so
blocks should be small: it is exactly the cost of one table in variable elimination.

## 2. Stationarity

The block kernel $K_B(x,x')=\pi(x'_B\mid x_{-B})$ when $x'_{-B}=x_{-B}$ (and 0 otherwise) satisfies detailed
balance by P23 Proposition 2's argument, with the block in place of the single variable:
$\pi(x)K_B(x,x')=\pi(x_{-B})\,\pi(x_B\mid x_{-B})\,\pi(x'_B\mid x_{-B})$, which is symmetric. A sweep over
blocks and singletons is a composition of such kernels, so $\pi$ is stationary (P23 Lemma 1). Blocks must
be disjoint. Every unobserved variable not in a named block is updated on its own.

## 3. What blocking buys (B2)

**F2 (near-deterministic coupling).** With the block $\{X,Y\}$ covering every unobserved variable, one
update draws $(X,Y)$ exactly from $\pi$, whatever the current state. The kernel has every row equal to
$\pi$, so its second eigenvalue is 0: $\tau=1$, and each sample is independent. Single-site Gibbs had
$\tau\approx1/(2\varepsilon)$: about 100 for $\varepsilon=0.005$, and 500 for $\varepsilon=0.001$.

**F3 (determinism).** The block conditional of $(X,Y)$ puts mass $\frac12$ on each of $(0,0)$ and
$(1,1)$, so the chain moves between the two classes that trapped single-site Gibbs. The chain is
irreducible again, and its estimate of $P(X{=}1)$ converges to $\frac12$.

**In general**, blocking strongly dependent variables together removes the slow directions of the
kernel. The trade-off is the block's table size. That is why the choice of blocks is the modeller's
(spec ⚑9), and the diagnostics of P25 are how to tell that it is needed.

## 4. How the tests check this independently

- **Block conditionals** against the joint table, for random blocks of random Bayesian and Markov
  networks with evidence.
- **Exact block kernels:** stationarity and detailed balance; estimates within exact CLT bounds.
- **F2:** with the block, the kernel's second eigenvalue is 0, the empirical $\tau\approx1$, and four chains from
  opposite modes give $\hat R<1.01$, where single-site Gibbs gives $\hat R>1.1$.
- **F3:** the blocked chain visits both states, and its estimate is within the i.i.d. CLT bound of $\frac12$.
