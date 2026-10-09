# Elimination orders: the graph view of variable elimination

> Every intermediate table that variable elimination builds can be predicted from an undirected
> graph, without touching a single number. Eliminating a variable from the factors is the same
> as eliminating a vertex from that graph: delete it, and connect all of its neighbours. So the
> cost of an order is a graph property. Cheap orders are found with greedy graph heuristics,
> which are provably optimal on trees and on chordal graphs.

Prerequisites: P6 ([`variable_elimination.md`](variable_elimination.md)).

---

## 1. The interaction graph

For a set of factors $\Phi$, the **interaction graph** $H(\Phi)$ has one vertex per variable, and
an edge $\{u,v\}$ whenever $u$ and $v$ appear together in some factor's scope.

**Proposition 1 (CPD factors give the moral graph).** For the CPD factors of a Bayesian network
on $G$, $H(\Phi)=\mathcal M(G)$, the **moral graph**. That is the graph obtained by adding an
edge between every pair of parents that share a child (marrying them) and then dropping the
edge directions.

*Proof.* The scope of $\phi_i$ is $\{X_i\}\cup\mathrm{pa}(X_i)$. The pairs inside it are of two
kinds: child–parent pairs, which are exactly the edges of $G$ without their directions, and
parent–parent pairs, which are exactly the marriages. Every edge of $\mathcal M(G)$ arises this
way, and nothing else does. $\square$

**With evidence.** Reduction deletes the observed variables from every scope (P5) and keeps
the other pairs. So $H(\Phi[e])$ is the subgraph of $\mathcal M(G)$ induced on the unobserved
variables: `moral_graph(dag).subgraph(unobserved)`.

**Two different restrictions.** Restricting the moral graph, $\mathcal M(G)[S]$, is **not** the
same as moralising the restricted graph, $\mathcal M(G[S])$. This is true even when $S$ is
ancestral. In $A\to X\leftarrow B$ with $S=\{A,B\}$, the first keeps the marriage $A$–$B$, and
the second has no edges, because $X$ is gone. Evidence on $X$ produces the first. The
d-separation criterion (M2.8) needs the second, $\mathcal M(G[\mathrm{An}(X\cup Y\cup Z)])$, so
`moral_graph(dag, restrict_to=S)` means $\mathcal M(G[S])$. The tests pin down this example.

## 2. Elimination is vertex elimination

**Eliminating a vertex** $z$ from a graph means connecting every pair of its neighbours (these
new edges are **fill edges**) and then deleting $z$.

**Lemma 2.** Let $H=H(\Phi)$ and let $\Phi'$ be the result of eliminating $z$ (P6 §2). Then:

1. the table built at this step, $\psi_z=\prod\Phi_z$, has scope $S_z=\{z\}\cup N_H(z)$;
2. $H(\Phi')$ is $H$ with $z$ eliminated.

*Proof.*
1. $S_z$ is the union of the scopes that contain $z$. By definition of $H$, that is $z$ together
   with everything that shares a factor with $z$: that is, $z$ and its neighbours.
2. $\Phi'$ keeps every factor of $\Phi_{\bar z}$, which preserves every edge not touching $z$. It
   drops the factors that contain $z$. Any pair $u,v\ne z$ that shared one of those factors is
   inside $N_H(z)$, and it reappears in the new factor $\tau_z$, whose scope is
   $N_H(z)$. So $H(\Phi')$ has every old edge not touching $z$, plus a clique on $N_H(z)$, and
   nothing else. $\square$

**Consequence (V8).** Applying the lemma at each step, the whole sequence of scopes
$S_{z_1},S_{z_2},\ldots$ is determined by the graph $H$ and the order. It needs no numbers. So
the trace from running elimination numerically must equal the trace from simulating it on the
graph. `simulate_elimination` does the simulation, `VariableElimination.elimination_cost`
records a real run, and the tests check they agree.

**Width.** The **induced width** of an order is $w(Z)=\max_k|N_{H_k}(z_k)|=\max_k|S_{z_k}|-1$.
Its minimum over all orders that eliminate every vertex is the **treewidth** of $H$. Variable
elimination costs $O(n\,d^{\,w+1})$ (P6 §4), so a good order is one with small width, and
small total size $\sum_k\prod_{v\in S_{z_k}}|\mathcal X_v|$.

## 3. Greedy heuristics

At each step, the greedy algorithm eliminates the remaining eligible vertex with the smallest
score in the current graph. Ties go to the vertex that was inserted earliest, so results are
deterministic (spec §9 ⚑9).

| Heuristic | Score of $z$ | Intuition |
|---|---|---|
| `min_neighbours` | $\lvert N(z)\rvert$ | smallest new factor, counted in variables |
| `min_weight` | $\prod_{v\in N(z)}\lvert\mathcal X_v\rvert$ | smallest new factor, counted in table cells (Koller & Friedman §9.4.3.2) |
| `min_fill` | the number of non-adjacent pairs in $N(z)$ | adds the fewest edges, so it keeps future factors small |

Finding an order of minimum width is NP-hard (Arnborg, Corneil & Proskurowski, 1987). The
greedy heuristics are fast, often good, and **not optimal in general**. They are optimal on two
important classes of graphs.

**Proposition 3 (forests).** On a forest, `min_neighbours` and `min_fill` both produce width $\le1$.

*Proof.* A non-empty forest has a vertex of degree $\le1$. Eliminating such a vertex adds no
fill edge, and leaves a forest. `min_neighbours` always picks some vertex of degree $\le 1$. For
`min_fill`: a vertex of degree $\ge2$ has two neighbours that are not adjacent (a forest has no
triangles), so its fill is $\ge1$. Every degree-$\le1$ vertex has fill 0. So `min_fill` also
picks a degree-$\le1$ vertex. Each step therefore has $|N(z)|\le1$. $\square$

**Proposition 4 (chordal graphs).** A graph is **chordal** if every cycle of length $\ge4$ has a
chord. On a chordal graph, `min_fill` adds **no** fill edges, and its width is $\omega-1$, where
$\omega$ is the size of the largest clique. This is optimal.

*Proof.* Every chordal graph has a **simplicial** vertex, one whose neighbours form a clique
(Dirac, 1961). A vertex has fill 0 exactly when it is simplicial, so `min_fill` always picks
one. Eliminating a simplicial vertex adds nothing and leaves an induced subgraph, which is still
chordal. So the induction continues with zero fill. Each $\{z\}\cup N(z)$ is a clique of the
original graph, so the width is at most $\omega-1$. And *every* order has width at least
$\omega-1$: consider the first vertex of a maximum clique to be eliminated; the rest of that
clique are all still its neighbours. $\square$

These two propositions become exact tests: random forests and random chordal graphs, with the
heuristic's width compared to the known optimum. On general small graphs the tests check
heuristic width $\ge$ optimal width (computed by brute force over all orders). That is a
consistency check, not an optimality claim.

### 3.1 Updating scores after an elimination

Recomputing every score at every step costs $O\big(\sum_v \deg(v)^2\big)$ per step for
`min_fill`. On a star whose hub has $d$ leaves, that is $\Theta(d^2)$ per step and
$\Theta(d^3)$ overall, about 20 s for $d=1100$. Only a few scores change at each step, and they
change by amounts that can be computed locally. Write $\delta(v)=|N(v)|$, $w(v)$ and $f(v)$ for
the three scores, and $C(a,b)=N(a)\cap N(b)$.

**Lemma 5 (which scores change).** Let $H'$ be $H$ with $z$ eliminated, and $N=N_H(z)$.

1. If $v\notin N\cup\{z\}$, then $N_{H'}(v)=N_H(v)$. So $\delta(v)$ and $w(v)$ are unchanged,
   and $f(v)$ changes only if $v$ is a common neighbour of the two ends of some fill edge.
2. Adding one edge $ab$ to a graph where $a,b$ are not adjacent changes fill only at $a$, $b$
   and $C(a,b)$:
   $$f(c)\mathrel{-}=1\ \ (c\in C(a,b)),\qquad f(a)\mathrel{+}=\delta(a)-|C(a,b)|,\qquad
   f(b)\mathrel{+}=\delta(b)-|C(a,b)|,$$
   with degrees and $C(a,b)$ taken before the edge is added.
3. Deleting $z$ once $N$ is a clique changes fill only on $N$:
   $f(v)\mathrel{-}=\delta(v)-|N|$ for $v\in N$, with $\delta(v)$ taken before the deletion.

*Proof.*
1. Fill edges join two vertices of $N$, and the deleted vertex $z$ is not adjacent to $v$. So no
   edge at $v$ is added or removed. The pairs inside $N(v)$ are fixed; one of them changes
   adjacency only if it is a fill edge, which requires both of its ends in $N(v)$.
2. For $c\notin\{a,b\}$, $N(c)$ is unchanged, and the only pair whose adjacency changes is
   $\{a,b\}$. It lies inside $N(c)$ iff $c\in C(a,b)$, and it has just become adjacent. At $a$,
   $N(a)$ gains $b$. The old pairs in $N(a)$ keep their adjacency ($\{a,b\}$ is not among
   them). The new pairs are $\{b,x\}$ for $x\in N(a)$, and they are non-adjacent exactly when
   $x\notin N(b)$: there are $\delta(a)-|C(a,b)|$ of them. The same holds at $b$.
3. A vertex $v\notin N$ is not adjacent to $z$, so nothing changes at $v$. For $v\in N$, deleting
   $z$ from $N(v)$ removes the pairs $\{z,x\}$ with $x\in N(v)\setminus\{z\}$, and such a pair
   was non-adjacent iff $x\notin N$. Since $N$ is a clique, $N\setminus\{v\}\subseteq N(v)$, so
   exactly $(\delta(v)-1)-(|N|-1)$ of them were non-adjacent. $\square$

Elimination is the fill edges added one at a time, then the deletion of $z$, so applying (2)
to each fill edge in turn (against the graph so far) and then (3) gives the new fill exactly.
The order in which the fill edges are added does not matter, because the result is a function
of the final graph. Degrees and weights only change on $N$, so they are recomputed there.

This is what makes the star cheap. Eliminating a leaf adds no fill edge, so (2) never runs and
(3) is a single subtraction at the hub. Recomputing just the scores in $N\cup N(N)$ would not
be enough: the hub is in $N$ at every step, and recomputing its fill pairwise is still
$\Theta(d^2)$. The initial fills also avoid the pairwise count. $f(v)=\binom{\delta(v)}2-t(v)$,
where $t(v)$ is the number of edges inside $N(v)$, and
$t(v)=\tfrac12\sum_{u\in N(v)}|N(u)\cap N(v)|$. Each intersection costs
$O(\min(\delta(u),\delta(v)))$, so the hub costs $O(d)$, not $O(d^2)$.

**Proposition 6 (same choices).** Keep a heap of entries $(s,\mathrm{rank}(v),v)$, pushing a new
entry whenever an eligible vertex's score changes, and skip popped entries that are stale
(their $v$ is already eliminated, or $s$ is no longer $v$'s score). Then each step pops the
remaining eligible vertex minimising $(\mathrm{score},\mathrm{rank})$, which is exactly the vertex
the full recomputation chooses.

*Proof.* Every remaining eligible vertex $v$ has an entry carrying its current score $s(v)$: one
was pushed when $s(v)$ last changed (or at the start), and it has not been popped, because
popping a current entry eliminates $v$. Stale entries are skipped and never chosen. The ranks
are distinct, so the keys $(s(v),\mathrm{rank}(v))$ of the current entries are distinct, and the
heap's minimum among them is the minimiser. $\square$

## 4. Querying a subset

`VariableElimination` eliminates only the variables outside $Q\cup E$. Its graph is
$H(\Phi[e])$, and the query vertices stay in place. Propositions 3 and 4 are stated for full
elimination, and the tests apply them only in that setting.

## 5. The naive-Bayes example, as a graph

The moral graph of $C\to F_i$ is a star: $C$ is adjacent to every $F_i$, and no parents need
marrying. Take the query $P(F_1)$.

- **`min_fill` / `min_neighbours`:** each $F_i$ with $i\ge2$ is a leaf, with one neighbour and
  fill 0. They are eliminated first, at width 1. $C$ follows, with neighbour $F_1$.
- **$C$ first:** $N(C)=\{F_1,\ldots,F_n\}$. Eliminating $C$ adds $\binom n2$ fill edges and
  builds a table over all $n+1$ variables: width $n$, $2^{n+1}$ cells.

Lemma 2 predicts both traces exactly, and the tests check the prediction against measured runs.
