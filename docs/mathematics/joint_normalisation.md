# P2 — Joint normalisation

> **Theorem.** Let $G$ be a DAG on $x_1,\ldots,x_d$. Give each node a CPD
> $p(x_i\mid\mathrm{pa}(x_i))$ that satisfies C1 and C2, with parents exactly
> $\mathrm{pa}_G(x_i)$. Then
> $$P(x)\;:=\;\prod_{i=1}^{d}p(x_i\mid\mathrm{pa}(x_i))$$
> is a probability distribution: $P(x)\ge0$ and $\sum_x P(x)=1$.

Here $P$ is a **definition**. P1 went in one direction: from a distribution to a
product of conditionals. P2 goes the other way: from the product of local
tables to a distribution. That direction is what `BayesianNetwork` relies on.
The user supplies only local tables, and the product is claimed to be a valid
joint distribution.

---

## 1. Why it is not obvious

Every factor sums to 1 over its *own* variable. But the full sum runs over all
variables at once, and each variable appears in several factors: in its own,
and in the factors of each of its children. Locally normalised factors do **not**
multiply to a normalised product in general:

$$\sum_{a,c}\underbrace{p(a\mid c)}_{\text{normalised in }a}\;\underbrace{p(c\mid a)}_{\text{normalised in }c},
\qquad p(a\mid c)=p(c\mid a)=\begin{cases}0.9 & a=c\\0.1&a\ne c\end{cases}$$
$$=2(0.9)^2+2(0.1)^2=1.64\ne1.$$

That product corresponds to the "graph" $A\rightleftarrows C$, which has a cycle.
The proof below uses acyclicity in exactly one place.

---

## 2. Proof (eliminate leaves in reverse topological order)

Induction on $d$. For $d=0$ the product is empty, so $P=1$.

For $d\ge1$, choose a topological ordering and let $x_d$ be its **last** node.

1. **$x_d$ is a leaf.** Any child of $x_d$ would have to come after it in the
   ordering (G2), and nothing does. So $x_d$ is nobody's parent, and it appears
   only in its own factor.
   *(This is the step that needs acyclicity. In $A\rightleftarrows C$ every
   node is someone's parent, so no variable can be summed out on its own.)*
2. Its parents come earlier in the ordering: $\mathrm{pa}(x_d)\subseteq\{x_1,\ldots,x_{d-1}\}$.
3. Factor the innermost sum:
   $$\sum_{x_1,\ldots,x_d}\prod_{i=1}^{d}p(x_i\mid\mathrm{pa}_i)
   =\sum_{x_1,\ldots,x_{d-1}}\Big[\prod_{i=1}^{d-1}p(x_i\mid\mathrm{pa}_i)\Big]\underbrace{\sum_{x_d}p(x_d\mid\mathrm{pa}_d)}_{=1\ \text{by C2}}.$$
4. The bracket is the product for the graph $G-x_d$, with its CPDs unchanged:
   none of them mentions $x_d$. $G-x_d$ is still a DAG, so by the induction
   hypothesis its sum is 1. $\blacksquare$

Nonnegativity is immediate: every factor is $\ge0$ by C1. Normalisation gives
$P(x)\le1$ for free.

### Where each invariant is used

| Step | Invariant |
|---|---|
| $x_d$ appears in one factor only | G1 / G2: acyclicity and topological order |
| inner sum equals 1 | C2: every column of the CPD is normalised |
| $P\ge0$ | C1 |
| factors match the graph's structure | **B2: CPD parents $=$ graph parents** |

B2 matters because, if a CPD conditions on a variable that is not a graph
parent, the graph no longer describes which factors mention which variables.
A CPD for $A$ conditioned on its own child $C$ recreates the cyclic example in
§1. `BayesianNetwork.add_cpd` therefore rejects any CPD whose parent set
differs from the graph's.

---

## 3. Corollaries (these become tests)

The same elimination argument, stopped part-way, proves more than normalisation.

**C-a. Removing a leaf.** Summing out a leaf gives exactly the network on
$G-x_d$. More generally, for any *ancestral* set $S$ (one closed under taking
parents), such as any prefix $x_1,\ldots,x_i$ of a topological ordering:
$$P(x_S)=\prod_{j\in S}p(x_j\mid\mathrm{pa}_j).$$

**C-b. The CPDs are the joint's own conditionals.** Take the prefix
$x_1,\ldots,x_i$ and divide C-a for $i$ by C-a for $i-1$:
$$P(x_i\mid x_1,\ldots,x_{i-1})=p(x_i\mid\mathrm{pa}_i)\qquad\text{whenever }P(x_1,\ldots,x_{i-1})>0.$$
The conditional given *all* predecessors depends only on the parents. Averaging
over the non-parent predecessors therefore gives
$P(x_i\mid\mathrm{pa}_i)=p(x_i\mid\mathrm{pa}_i)$. So computing the conditional
from the joint returns the table that was put in.

**C-c. The graph's independences hold.** The display in C-b is the *ordered
Markov property*. P1 showed that it gives the factorisation. C-b shows the
factorisation gives it back. So P1 and P2 together say that **a distribution
factorises over $G$ exactly when it satisfies $G$'s ordered Markov property.**
The local Markov property $x_i\perp\mathrm{nd}(x_i)\setminus\mathrm{pa}(x_i)\mid\mathrm{pa}(x_i)$
also follows. To see this, put all of $\mathrm{nd}(x_i)$ before $x_i$; that is
always possible because the non-descendants of $x_i$ form an ancestral set.

`tests/test_bayesian_network.py` checks C-b and C-c numerically on random
networks. It enumerates the joint, recomputes each $P(x_i\mid\mathrm{pa}_i)$
from it, and checks the local Markov property as a tensor identity. An
implementation that multiplied the wrong entries would still produce
nonnegative numbers, and could even sum to 1, but it would fail these checks.

---

## 4. Fixture check

For $R\rightarrow T\leftarrow A$, eliminate $T$, then $A$, then $R$:
$$\sum_{r,a,t}P(r)P(a)P(t\mid r,a)=\sum_{r,a}P(r)P(a)\cdot1=\sum_rP(r)\cdot1=1.$$
Numerically, over $(r,a,t)$ in binary order $000,001,\ldots,111$:
$0.567+0.063+0.021+0.049+0.054+0.216+0.0015+0.0285=1$.
