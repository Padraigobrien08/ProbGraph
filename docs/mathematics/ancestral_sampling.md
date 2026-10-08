# P3 — Ancestral sampling

> **Theorem.** Visit the nodes of a Bayesian network in topological order. At
> each node, draw $X_i\sim p(\cdot\mid\mathrm{pa}(x_i))$, using the parent values
> already drawn and a source of randomness that has not been used before. The
> resulting vector $(X_1,\ldots,X_d)$ has distribution exactly
> $P(x)=\prod_i p(x_i\mid\mathrm{pa}(x_i))$.

The proof has two parts. The first is how to draw one categorical value from a
uniform random number. The second is why chaining such draws in topological
order produces the joint distribution.

---

## 1. Lemma: categorical draws by inverse CDF

Let $q=(q_0,\ldots,q_{K-1})$ be a point in the simplex. Define the cumulative
sums $F_k=\sum_{j\le k}q_j$, with $F_{-1}=0$ and $F_{K-1}=1$. Let
$U\sim\mathrm{Uniform}[0,1)$, and let
$$\kappa(U)=\min\{k : U<F_k\}=\#\{k : F_k\le U\}.$$

Then $P(\kappa(U)=k)=q_k$.

*Proof.* $\kappa(U)=k$ exactly when $F_{k-1}\le U<F_k$. That interval has
length $F_k-F_{k-1}=q_k$, and a uniform variable lands in an interval with
probability equal to the interval's length. $\square$

The intervals $[F_{k-1},F_k)$ are **half-open** and **tile** $[0,1)$. This
settles the edge cases exactly:

- A state with $q_k=0$ has an empty interval, so it is **never** drawn. This
  holds wherever the state sits in the domain: first, last or in between.
- A $U$ that lands exactly on a boundary $F_k$ goes to the *next* state, $k+1$.
- Floating point: a CPD column may sum to $1\pm10^{-10}$ (C2). If $F_{K-1}$
  were slightly below 1, a $U\in[F_{K-1},1)$ would fall off the end. The
  implementation therefore divides each cumulative column by its own total.
  Then $F_{K-1}=x/x=1.0$ exactly in IEEE arithmetic, so $\kappa(U)\le K-1$
  for every $U<1$. Dividing by the total changes each probability by at most
  about $10^{-10}$, and it maps zero-mass states to zero-mass states.

Library samplers such as `Generator.choice` do this internally. Here it is
written out explicitly, so the code follows the lemma line by line.

---

## 2. Proof of the theorem

Fix a topological ordering $x_1,\ldots,x_d$, and let $U_1,\ldots,U_d$ be
independent $\mathrm{Uniform}[0,1)$ variables. The algorithm sets
$$X_i=\kappa_i(U_i),\qquad\text{where $\kappa_i$ uses the CDF of }p(\cdot\mid\mathrm{pa}_i=\text{the already-drawn }X_{\mathrm{pa}_i}).$$

**Claim.** For every $i$,
$P(X_1=x_1,\ldots,X_i=x_i)=\prod_{j\le i}p(x_j\mid\mathrm{pa}_j)$.

*Induction on $i$.* For $i=0$, both sides equal 1. For the step:
$$P(X_{\le i}=x_{\le i})=P(X_{<i}=x_{<i})\cdot P\big(X_i=x_i\;\big|\;X_{<i}=x_{<i}\big).$$

- $X_{<i}$ depends only on $U_1,\ldots,U_{i-1}$. Since $U_i$ is independent of
  them (**fresh randomness**), conditioning on $X_{<i}=x_{<i}$ does not change
  the distribution of $U_i$.
- Given $X_{<i}=x_{<i}$, the parent values are fixed: by **G2**,
  $\mathrm{pa}_i\subseteq\{x_1,\ldots,x_{i-1}\}$. So the algorithm applies the
  lemma to the fixed vector $p(\cdot\mid\mathrm{pa}_i)$, and the second factor
  equals $p(x_i\mid\mathrm{pa}_i)$.
- By the induction hypothesis, the first factor is
  $\prod_{j<i}p(x_j\mid\mathrm{pa}_j)$. $\blacksquare$

Taking $i=d$ gives the theorem. As a by-product, every prefix
$(X_1,\ldots,X_i)$ has the marginal distribution $P(x_1,\ldots,x_i)$, which
matches P2's corollary C-a.

### Where each invariant is used

| Requirement | Used for | If it is violated |
|---|---|---|
| G2: parents come first (**S2**) | the parent values are fixed when $X_i$ is drawn | the draw reads parent values that have not been set yet |
| fresh $U_i$ for each node | conditioning on the past leaves $U_i$ uniform | reusing one $U$ for several nodes makes them correlated |
| fresh $U$'s for each sample (**S5**) | the $N$ samples are i.i.d. | carried-over values make consecutive samples correlated |
| C1, C2 | the lemma applies | |

### The samples are independent

Sample $s$ uses only its own block $(U_{s,1},\ldots,U_{s,d})$. These blocks
are disjoint and independent, so the joint samples are i.i.d. draws from $P$.
The implementation draws the whole $N\times d$ array of uniforms up front, and
row $s$ is the block for sample $s$.

---

## 3. Reproducibility (S3)

The uniforms come from `numpy.random.default_rng(seed)` (PCG64), drawn as
`rng.random((n, d))` in row-major order. Consequences:

- The same seed, model and topological order give the same samples.
- **Stream consistency.** `sample(a)` followed by `sample(b)` uses the same
  uniforms, in the same order, as one call to `sample(a+b)`. So the samples
  match. Splitting a run into batches does not change the result.
- Bit-for-bit reproducibility holds only within a NumPy environment. NumPy
  does not promise identical streams across all versions, which is why
  spec S3 includes that qualifier.

---

## 4. Statistical testing (S4)

The proof is what establishes correctness. Tests can only check that the
implementation is *consistent* with it. The precise question is: how large an
error would a test detect?

For an event $E$ with $P(E)=p$, the empirical frequency $\hat p$ over $N$
i.i.d. samples satisfies $N\hat p\sim\mathrm{Binomial}(N,p)$, so
$$\mathbb E[\hat p]=p,\qquad \mathrm{SE}(\hat p)=\sqrt{p(1-p)/N},$$
and $\hat p$ is approximately normal for large $N$ (by the central limit theorem).

**False alarms: a bound that does not rely on the CLT.** The normal
approximation is poor when $Np$ is small. A random network can easily contain
cells with expected count below 1. The tests therefore use **Bernstein's
inequality**, which holds for any $p$ and $N$. Write $S=N\hat p$ and
$\sigma^2=Np(1-p)$. Each summand differs from its mean by at most 1, so
$$P\big(|S-Np|\ge t\big)\le2\exp\!\Big(-\frac{t^2}{2(\sigma^2+t/3)}\Big).$$
Setting the exponent to $-Z^2/2$ and solving the quadratic in $t$ gives
$t=Z^2/6+\sqrt{Z^4/36+Z^2\sigma^2}\le Z\sigma+Z^2/3$. Hence
$$\boxed{\;\big|\hat p-p\big|\le Z\,\mathrm{SE}+\frac{Z^2}{3N}\;}$$
fails for a *correct* sampler with probability at most
$2e^{-Z^2/2}\approx7.5\times10^{-6}$ when $Z=5$. The extra $Z^2/(3N)$ is the
documented minimum tolerance. At $N=2\times10^5$ it is $4\times10^{-5}$,
negligible next to the SE of any ordinary event, and it covers the rare cells
where the CLT breaks down. Events with $p\in\{0,1\}$ get **zero** tolerance:
an impossible event must never be sampled.

Over $m$ checks, the union bound gives a false-alarm probability of at most
$m\times7.5\times10^{-6}$. That is a few in a thousand for the roughly 600
cell checks in the random-network test. All test seeds are fixed, so every
test run is deterministic. The bound says that the chosen seeds were not
specially favourable.

**Power.** To reliably detect a true bias of $\delta$, $\delta$ must be well
above the $5\,\mathrm{SE}$ tolerance. Requiring $\delta\ge(5+2)\,\mathrm{SE}$ gives
$$N\gtrsim\frac{49\,p(1-p)}{\delta^2}.$$
For $p=0.3$ and $\delta=0.01$, that is $N\approx1.03\times10^5$. The fixture
tests use $N=2\times10^5$, so a one-percentage-point error is caught. Smaller
errors may not be.

**What statistical tests cannot see.**

- Biases much smaller than $\mathrm{SE}$, such as the $\le10^{-10}$ change from
  the renormalisation in §1.
- Errors confined to parent configurations that have low probability: their
  conditional counts are small, so their SE is large.

Two kinds of test therefore complement the frequency tests:

- **deterministic tests**, using zero-mass states, CPDs that copy their parent
  exactly, and hand-picked $U$ values at interval boundaries;
- the **proof above**.

**Conditional frequencies.** For a conditional probability $P(E\mid C)$, the
effective sample size is the number of samples in which $C$ occurs, $N_C$, so
$\mathrm{SE}=\sqrt{p(1-p)/N_C}$. Explaining away shows up directly in the
samples: among samples with $T=1,R=1$ (about 24% of $N$), the fraction with
$A=1$ is about $0.0285/0.2445\approx0.117$.
