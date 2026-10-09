# P7 — d-separation

> **Question.** Given a DAG $G$ and disjoint sets of variables $X$, $Y$ and $Z$, does
> $X\perp Y\mid Z$ hold in **every** distribution that factorises over $G$? d-separation answers
> this from the graph alone. Bayes ball decides it in time linear in the size of the graph.

This note covers the definition, the three local structures, and the correctness of Bayes ball
(§1–§4), then the moralised-ancestral criterion, soundness, generic completeness and the
graphoid axioms (§5–§8).

---

## 1. Definitions

A **trail** is a sequence of *distinct* nodes $v_0,\ldots,v_k$ in which consecutive nodes are
joined by an edge in either direction. An interior node $v_i$ ($0<i<k$) is a **collider** on
the trail if both trail edges point into it: $v_{i-1}\to v_i\leftarrow v_{i+1}$. Otherwise it
is a **non-collider**.

Given $Z$, a trail is **active** if every interior node satisfies the following:

| Interior node $v_i$ | Active when |
|---|---|
| non-collider (chain $\to v\to$, or fork $\leftarrow v\to$) | $v_i\notin Z$ |
| collider ($\to v\leftarrow$) | $v_i\in Z$, **or** some descendant of $v_i$ is in $Z$ |

The collider condition is the same as $v_i\in\mathrm{An}^*(Z)$, the ancestral closure of $Z$
(`DAG.ancestral_set`, P6 §5).

$X$ and $Y$ are **d-separated** by $Z$, written $X\perp_G Y\mid Z$, if there is no active trail
from any $x\in X$ to any $y\in Y$.

## 2. The three local structures

Every interior node of a trail is one of these three:

| Structure | Unobserved middle | Observed middle | Fixture example |
|---|---|---|---|
| Chain $A\to T\to L$ | dependent | **independent** | $A\perp L\mid T$ |
| Fork $U\leftarrow R\to T$ | dependent | **independent** | $U\perp L\mid R$ |
| Collider $R\to T\leftarrow A$ | **independent** | dependent | $A\perp R$, but $A\not\perp R\mid T$ |

A collider is also opened by observing one of its **descendants**. In the fixture,
$A\not\perp R\mid L$, because $L$ is a child of the collider $T$. All nine facts in the M2 spec §4
are checked twice in the tests: structurally with d-separation, and numerically from the joint
distribution (M2.2).

## 3. Walks and trails

Bayes ball searches over **walks**, which may revisit nodes, because tracking visited nodes per
trail would be exponential. Call a walk **d-connecting** if each interior *occurrence* satisfies
the table in §1. That is, each position on the walk is judged by the two edges it uses at that
point.

**Lemma 1.** If there is a d-connecting walk from $x$ to $y$ (with $x\ne y$ and $x,y\notin Z$),
then there is an active trail from $x$ to $y$.

*Proof.* Take a d-connecting walk $w=v_0,\ldots,v_k$ of **minimal length**, and suppose some
node repeats: $v_i=v_j$ with $i<j$. Shortcut it: $w'=v_0,\ldots,v_i,v_{j+1},\ldots,v_k$. Every
occurrence in $w'$ except position $i$ keeps both of its edges. So if the new triple
$(v_{i-1},v_i,v_{j+1})$ satisfied the table, $w'$ would be a shorter d-connecting walk,
contradicting minimality. (If $i=0$ or $j=k$, the repeated node is an endpoint and there is
no triple to check, so we get the contradiction immediately.) So that triple fails, which can
happen in only two ways:

- **(a) It is a non-collider and $v_i\in Z$.** In $w$, the occurrences at $i$ and $j$ are in $Z$,
  so both must be colliders. That means $v_{i-1}\to v_i$ and $v_{j+1}\to v_j$. But then the new
  triple $v_{i-1}\to v_i\leftarrow v_{j+1}$ *is* a collider. Contradiction.
- **(b) It is a collider and $v_i\notin\mathrm{An}^*(Z)$.** So $v_{i-1}\to v_i$ and
  $v_{j+1}\to v_j$. In $w$, neither occurrence can be a collider, because
  $v_i\notin\mathrm{An}^*(Z)$. At position $i$ the walk arrives along $v_{i-1}\to v_i$, so it must
  leave along an out-edge, $v_i\to v_{i+1}$. At position $j$ the walk leaves along
  $v_{j+1}\to v_j$, so it must have arrived along an out-edge, $v_j\to v_{j-1}$. So the loop
  $v_i,\ldots,v_j$ leaves $v_i$ going *down*, and its last step, from $v_{j-1}$ back to $v_j$,
  goes *up*. Somewhere in between, the loop switches from going down to going up. The first
  such switch is a collider $c$, and the part of the loop before it is a directed path
  $v_i\to\cdots\to c$. Because $w$ is d-connecting, $c\in\mathrm{An}^*(Z)$. Since $v_i$ is an
  ancestor of $c$, also $v_i\in\mathrm{An}^*(Z)$. Contradiction.

So the shortest d-connecting walk repeats no node. It is therefore a trail, and an active
one. $\square$

The converse is immediate, because an active trail is a d-connecting walk. So
**$y$ is d-connected to $x$ given $Z$ exactly when a d-connecting walk from $x$ to $y$ exists.**

## 4. Bayes ball (Koller & Friedman, Alg. 3.1)

The search state is a pair $(v,d)$: the current node $v$, and $d$, the direction of the edge used
to arrive. $d=\uparrow$ means it arrived from a child, travelling against the edge direction.
$d=\downarrow$ means it arrived from a parent. Let $A=\mathrm{An}^*(Z)$.

| At $(v,d)$ | Continue to parents $(p,\uparrow)$ | Continue to children $(c,\downarrow)$ |
|---|---|---|
| $d=\uparrow$ (from a child) | if $v\notin Z$ (chain: $c'\to v\to p$ reversed) | if $v\notin Z$ (fork) |
| $d=\downarrow$ (from a parent) | if $v\in A$ (collider) | if $v\notin Z$ (chain) |

Start from $(x,\uparrow)$ for each $x\in X$; with $x\notin Z$, this allows every first edge. A node
$v\notin Z$ that is visited in any state is **d-connected** to $X$.

**Correctness.** Each row of the table allows exactly the continuations that satisfy §1 for the
triple (previous node, $v$, next node). So the states reachable from $(x,\uparrow)$ are exactly
the ends of d-connecting walks from $x$. The rules depend only on $(v,d)$, not on how the walk
got there, so visiting each state once loses nothing. By Lemma 1, a reachable $v\notin Z$ is
connected to $x$ by an active trail, and conversely.

**Remark (the descendant rule comes free with walks).** Replacing $A=\mathrm{An}^*(Z)$ by $Z$
in the collider row gives **the same** reachable set. Suppose a collider $c\notin Z$ has a
descendant in $Z$. Take a shortest directed path $c\to\cdots\to z$ with $z\in Z$; its
intermediate nodes are not in $Z$. A walk can go down that path (chain nodes, unobserved), turn
at $z$ (a collider that is in $Z$), come back up (chain nodes again), and leave $c$ upwards
(a chain at $c$, which is allowed because $c\notin Z$). Every step is legal under the
"collider $\in Z$" rule. So the two versions accept exactly the same walks' endpoints.

The implementation keeps $A$, as Koller & Friedman do, so that each step's legality matches
the trail definition in §1 directly. During testing this showed up as an *equivalent mutant*:
replacing $A$ with $Z$ passes every test, including the exhaustive check over all 543 labelled
four-node DAGs. That is what this remark predicts, not a gap in the tests.

**Complexity.** Computing $A$ takes $O(|V|+|E|)$. There are $2|V|$ states, each visited at most
once, and each scans its parents and children once. Total: **$O(|V|+|E|)$**. Enumerating trails
directly can take exponential time. The tests do enumerate them, on small graphs, as an oracle
that implements §1 literally.

## 5. The moralised-ancestral criterion

Let $A=\mathrm{An}^*(X\cup Y\cup Z)$, and let $\mathcal M_A=\mathcal M(G[A])$ be the moral graph
of the subgraph of $G$ induced on $A$ (`moral_graph(dag, restrict_to=A)`;
[`elimination_orders.md`](elimination_orders.md) §1 explains why this is not the same as
restricting $\mathcal M(G)$ to $A$).

**Theorem 2 (Lauritzen, Dawid, Larsen & Leimer, 1990).** $X\perp_G Y\mid Z$ exactly when $Z$
separates $X$ from $Y$ in $\mathcal M_A$: every path from $X$ to $Y$ in $\mathcal M_A$ passes
through $Z$.

*Proof that an active trail gives a path avoiding $Z$.* Let $\pi$ be an active trail from $x$ to $y$.

1. *$\pi$ lies in $A$.* A collider on $\pi$ is in $\mathrm{An}^*(Z)\subseteq A$. A non-collider
   has an out-edge along $\pi$. Following the trail in that direction, it keeps going down
   until it reaches a collider or an endpoint. So the non-collider is an ancestor of something
   in $A$, and is therefore in $A$.
2. *Skip the colliders.* For each collider $p\to c\leftarrow q$ on $\pi$, its trail neighbours
   $p$ and $q$ are parents of $c\in A$, so they are married in $\mathcal M_A$. Replace
   $p,c,q$ by the edge $p$–$q$. Neither $p$ nor $q$ is a collider (each has an out-edge to $c$).
   What remains is a path in $\mathcal M_A$ whose interior nodes are all non-colliders of an
   active trail, so none of them is in $Z$. $\square$

*Proof that a path avoiding $Z$ gives an active trail.* Let $x=u_0,\ldots,u_m=y$ be a path in
$\mathcal M_A$ with no interior node in $Z$. We build a d-connecting walk in $G$ and then apply
Lemma 1.

1. *Expand the path into a walk.* Keep every edge of $G$ as it is. Replace each marriage edge
   $u$–$v$ by $u\to c\leftarrow v$, where $c\in A$ is a common child. Call the result $w$.
   Every non-collider of $w$ is some $u_i$, so it is not in $Z$. A collider $c$ of $w$ is open
   when $c\in\mathrm{An}^*(Z)$. Call the colliders that are not open **bad**. A bad collider is
   in $A\setminus\mathrm{An}^*(Z)$, so it is an ancestor of $x$ or of $y$, and no node on a
   directed path from it is in $Z$ (otherwise it would be in $\mathrm{An}^*(Z)$).
2. *Reroute around bad colliders.* Let $w_1$ be the bad collider closest to $y$ among those
   that are ancestors of $x$. Replace the part of $w$ from $x$ to $w_1$ by a directed path
   $w_1\to\cdots\to x$, traversed upwards from $x$. Its interior nodes are unobserved chain
   nodes, and $w_1$ becomes a non-collider that is not in $Z$. Symmetrically, let $w_2$ be the
   first bad collider after $w_1$, which must be an ancestor of $y$. Replace the part from
   $w_2$ to $y$ by a directed path to $y$. Any bad collider before $w_1$ has been cut off.
   Any bad collider between $w_1$ and $w_2$ would be an ancestor of $y$ before $w_2$, which is
   impossible. Any bad collider after $w_2$ has been cut off. (If a rerouting path passes
   through the other endpoint, truncate the walk there. What remains is a d-connecting walk
   already.)
3. The result is a d-connecting walk from $x$ to $y$. By Lemma 1, there is an active trail. $\square$

Bayes ball (§4) and Theorem 2 are two different algorithms for the same relation. The tests
check that they agree on every query of every DAG with up to 5 nodes, and on random larger ones.

## 6. Soundness

**Lemma 3 (separation implies independence).** Let $H$ be an undirected graph, and let
$P(x_V)=\prod_C\psi_C(x_C)$ where every $C$ is a clique of $H$. If $Z$ separates $X$ from $Y$ in
$H$, then $X\perp Y\mid Z$ under $P$.

*Proof.* Let $X'$ be the set of nodes reachable from $X$ in $H-Z$, and let
$W=V\setminus(X'\cup Z)$, so that $Y\subseteq W$. No clique meets both $X'$ and $W$: an edge
between a node of $X'$ and a node of $W$ would make that node of $W$ reachable. So each
$\psi_C$ depends only on $(x_{X'},z)$ or only on $(x_W,z)$. Grouping the factors gives
$P=f(x_{X'},z)\,g(x_W,z)$. Hence
$P(x_{X'},x_W\mid z)=\frac{f}{\sum f}\cdot\frac{g}{\sum g}$, which is $X'\perp W\mid Z$.
Marginalising (decomposition) gives $X\perp Y\mid Z$. $\square$

**Theorem 4 (soundness).** If $X\perp_G Y\mid Z$, then $X\perp Y\mid Z$ in **every** $P$ that
factorises over $G$.

*Proof.* By P2's corollary C-a, the marginal on the ancestral set is
$P(x_A)=\prod_{i\in A}p(x_i\mid\mathrm{pa}_i)$. Each scope $\{i\}\cup\mathrm{pa}_i$ is a clique of
$\mathcal M_A$, because the parents are married. By Theorem 2, $Z$ separates $X$ from $Y$ in
$\mathcal M_A$, so Lemma 3 applies to $P(x_A)$. Since $X\cup Y\cup Z\subseteq A$, that is the
statement. $\square$

## 7. Completeness, but only for generic parameters

The converse cannot hold for *every* $P$. **Counterexample (XOR).** Take fair coins $X$ and $Y$
with $X\to W\leftarrow Y$ and $W=X\oplus Y$. Then $X\perp W$ numerically: $W$ is a fair coin
whatever $X$ is. Yet $X$ and $W$ are adjacent, so they are not d-separated. Such a $P$ is called
**unfaithful** to $G$. The tests reproduce this exactly.

**Theorem 5 (Meek, 1995; Geiger & Pearl, 1990).** For every DAG, the CPD parameters that give
an unfaithful $P$ form a set of Lebesgue measure zero.

*Sketch.* Fix a triple $(X,Y,Z)$ that is not d-separated. "$X\perp Y\mid Z$" is a set of
polynomial equations in the CPD entries. The polynomial is not identically zero: CPDs that
almost copy values along an active trail make it non-zero. The zero set of a non-zero
polynomial has measure zero, and a finite union of such sets (over all triples) still does. $\square$

In tests: with Dirichlet-random CPDs (cardinalities ≥ 2), every d-connected triple shows
measurable dependence, and every d-separated triple shows independence up to rounding.
Cardinality-1 variables are excluded, because a constant is independent of everything.

## 8. Graphoid axioms

d-separation satisfies the **semi-graphoid** axioms. It also satisfies **intersection** and
**composition**, which probabilistic independence does not satisfy in general:

| Axiom | Statement | Holds for $\perp$ in general? |
|---|---|---|
| symmetry | $X\perp Y\mid Z\Rightarrow Y\perp X\mid Z$ | yes |
| decomposition | $X\perp YW\mid Z\Rightarrow X\perp Y\mid Z$ | yes |
| weak union | $X\perp YW\mid Z\Rightarrow X\perp Y\mid ZW$ | yes |
| contraction | $X\perp Y\mid Z\ \wedge\ X\perp W\mid ZY\Rightarrow X\perp YW\mid Z$ | yes |
| intersection | $X\perp Y\mid ZW\ \wedge\ X\perp W\mid ZY\Rightarrow X\perp YW\mid Z$ | positive $P$ only |
| composition | $X\perp Y\mid Z\ \wedge\ X\perp W\mid Z\Rightarrow X\perp YW\mid Z$ | **no**: XOR again, with $X\perp Y$, $X\perp W$, but $X\not\perp YW$ |

For d-separation, composition is immediate from the definition: "no active trail from $X$ to
$Y\cup W$" is a statement about each target separately. The tests check every row on random
DAGs, and use the XOR network to show that composition fails for probabilities.
