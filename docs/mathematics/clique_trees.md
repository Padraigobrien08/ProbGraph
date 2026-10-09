# P11 — Chordal graphs and clique trees

> Message passing needs a **tree** of clusters whose separators carry all the information that
> crosses them. Such a tree exists exactly when the graph is **chordal**. This note has two
> parts:
>
> - **Part 1 (M3.4):** how to make a graph chordal by triangulating it along an elimination
>   order, how to recognise a chordal graph by maximum cardinality search, and how to read off
>   its maximal cliques.
> - **Part 2 (M3.5):** connecting those cliques into a clique tree with the running intersection
>   property, why one exists exactly for chordal graphs, and why every maximum-weight spanning
>   tree is one.

Prerequisites: `elimination_orders.md` (vertex elimination, fill edges, Lemma 2, and
Proposition 4 on chordal graphs).

---

# Part 1 — Chordal graphs

## 1. Definitions

A graph is **chordal** if every cycle of length at least 4 has a **chord**, an edge joining two
non-consecutive vertices of the cycle. Equivalently, it has no **induced** cycle of length
$\ge4$. The brute-force test oracle checks exactly this second form.

An ordering $\pi=(v_1,\ldots,v_n)$ of all vertices is a **perfect elimination ordering (PEO)**
if, for every $v_i$, its **later neighbours** $\{v_j\in N(v_i):j>i\}$ form a clique. In other
words, eliminating vertices in order $\pi$ never adds a fill edge.

## 2. Triangulation along an elimination order

Given a graph $H$ and an ordering $\pi$ of all its vertices, the **triangulation** $H^+_\pi$ is
$H$ plus every fill edge created when its vertices are eliminated in the order $\pi$
(`elimination_orders.md` §2).

**Lemma 1.** $\pi$ is a PEO of $H^+_\pi$.

*Proof.* Let $N_i$ be the neighbours of $v_i$ in the elimination graph at the moment $v_i$ is
eliminated. Eliminating $v_i$ connects $N_i$ into a clique, so $N_i$ is a clique in $H^+_\pi$. It
remains to show that $N_i$ is exactly the set of later neighbours of $v_i$ in $H^+_\pi$.

- Every $u\in N_i$ is still present, so it comes later than $v_i$, and the edge $v_i$–$u$ is in $H^+_\pi$.
- Conversely, let $u$ come later than $v_i$ with $v_i$–$u$ in $H^+_\pi$. That edge is either in
  $H$ or was added as fill when some vertex $w$ was eliminated, and $w$ is earlier than both
  endpoints, because both were still present. Edges disappear only with their vertices, and
  neither endpoint has been eliminated by the time $v_i$ is. So the edge is present at that
  moment, and $u\in N_i$.

(A fill edge at $v_i$ can also lead to an *earlier* vertex. That edge is part of the earlier
vertex's set of later neighbours, not of $v_i$'s.) $\square$

**Lemma 2.** A graph with a PEO is chordal.

*Proof.* Take any cycle of length $\ge4$, and let $v$ be the cycle vertex that comes **first** in
the PEO. Its two neighbours on the cycle come later, so they are adjacent (later neighbours form
a clique). That edge is a chord, because the cycle has length at least 4. $\square$

**Theorem 3 (Fulkerson & Gross, 1965).** A graph is chordal **if and only if** it has a PEO.

*Proof.* ($\Leftarrow$) is Lemma 2. ($\Rightarrow$) Every chordal graph has a simplicial vertex
(Dirac, 1961; already used in `elimination_orders.md` Proposition 4). Put it first, delete it,
and repeat. Induced subgraphs of chordal graphs are chordal. $\square$

**Consequences.**

- Every $H^+_\pi$ is chordal (Lemmas 1 and 2), and it contains $H$. This is invariant **J4**.
- The width of $\pi$, that is, the largest set of later neighbours, is the same number that
  M2's `simulate_elimination` reports. The cliques $\{v\}\cup(\text{later neighbours of }v)$ are
  exactly the elimination-step scopes $S_{z}$ of `elimination_orders.md` Lemma 2. This is
  invariant **J5**, checked in M3.5.

## 3. Recognising chordal graphs: maximum cardinality search

**Maximum cardinality search (MCS)** (Tarjan & Yannakakis, 1984) visits the vertices one at a
time. At each step it visits the unvisited vertex with the **most visited neighbours**, breaking
ties by insertion order. Let $\sigma$ be the visit order.

**Theorem 4 (Tarjan & Yannakakis, 1984).** If $H$ is chordal, then the **reverse** of $\sigma$ is
a PEO.

*Sketch.* The proof is an invariant about MCS on chordal graphs: when a vertex $v$ is visited,
its already-visited neighbours form a clique. Otherwise there would be two non-adjacent visited
neighbours $a,b$ of $v$, and the visit history yields a path from $a$ to $b$ through vertices
visited earlier. Together with $v$ that path closes a cycle of length at least 4, and the MCS
priority rule ensures the cycle can have no chord, contradicting chordality. (See the paper for
the full argument.) Reversing $\sigma$ turns "visited earlier" into "eliminated later", which is
the PEO condition.

**The test, and why it is sound.** Run MCS, reverse the order, and check the PEO condition
directly. Each vertex's later neighbours must form a clique, which is a mechanical check.

- If the check **passes**, the graph has a PEO, so it **is** chordal by Lemma 2. That direction
  needs no trust in Theorem 4.
- If the check **fails**, Theorem 4 says the graph is not chordal. This is the direction that
  relies on the cited theorem. The tests confirm it exhaustively: on every labelled graph with
  at most 6 vertices, the test agrees with the brute-force "no induced cycle of length $\ge4$"
  definition.

**Independent confirmation.** The number of labelled chordal graphs on $n$ vertices is known
(OEIS A058862): $1, 2, 8, 61, 822, 18154$ for $n=1,\ldots,6$. The brute-force oracle reproduces
these counts, and so must `is_chordal`.

**Cost.** This implementation scans the unvisited vertices at each step, which is $O(n^2+m)$. The
original uses bucket queues to reach $O(n+m)$. The PEO check costs $O(\sum_v d_v^2)$ edge
lookups. Both are negligible next to inference.

## 4. Maximal cliques of a chordal graph

Given a PEO $\pi$, define for each vertex the candidate
$C_v=\{v\}\cup(\text{later neighbours of }v)$.

**Proposition 5.** Every $C_v$ is a clique, and every maximal clique equals some $C_v$. So the
maximal cliques are the inclusion-maximal sets among $\{C_v\}$, and a chordal graph has at most
$n$ of them.

*Proof.* $C_v$ is a clique, because later neighbours form a clique (PEO) and all of them are
adjacent to $v$. Let $K$ be a maximal clique, and let $v$ be its vertex that comes first in
$\pi$. The other vertices of $K$ are neighbours of $v$ that come later, so $K\subseteq C_v$.
$C_v$ is a clique and $K$ is maximal, so $K=C_v$. $\square$

General graphs can have exponentially many maximal cliques. Chordality is what makes this
linear, and `maximal_cliques` refuses a non-chordal graph rather than returning something
misleading. The tests compare against brute-force enumeration of every clique, on every chordal
graph with at most 6 vertices.

# Part 2 — Clique trees

## 5. Definitions

A **cluster tree** over a graph $H$ is a tree whose nodes are vertex sets ("clusters")
$C_1,\ldots,C_k$, with separators $S_{ij}=C_i\cap C_j$ on its edges. It is a **clique tree** (or
junction tree) when:

- **J1** it is a tree;
- **J2 running intersection property (RIP):** for every vertex $x$, the clusters containing $x$
  form a connected subtree;
- **J3** the clusters are exactly the maximal cliques of a chordal graph $H^+\supseteq H$. In
  particular, every edge of $H$ lies inside some cluster.

RIP is what makes message passing exact (P12). A variable summed out of a message never
reappears further along, because every cluster that contains it lies on one connected piece of
the tree.

## 6. Construction from an elimination order

Fix an ordering $\pi$ of all vertices, the triangulation $H^+_\pi$, and the clusters
$C_v=\{v\}\cup L(v)$, where $L(v)$ is the set of later neighbours of $v$ in $H^+_\pi$. These are
exactly the elimination-step scopes, so the cluster sizes are the M2 trace (J5).

**Step 1: the elimination tree.** For each $v$ with $L(v)\neq\varnothing$, let its **parent**
$p(v)$ be the vertex of $L(v)$ that comes first in $\pi$, and join $C_v$ to $C_{p(v)}$. This is
where the message $\tau_v$ created by eliminating $v$ is consumed: $p(v)$ is the first variable
in $\tau_v$'s scope to be eliminated. Every parent comes later than its child, so there are no
cycles, and the result is a forest with one root per connected component. Chaining the roots
with empty separators makes it a tree. That is harmless, because components share no vertex.

**Lemma 6.** The elimination tree satisfies RIP.

*Proof.* Fix a vertex $x$. The clusters containing $x$ are $C_x$, together with $C_v$ for every
$v$ earlier than $x$ with $x\in L(v)$. Follow parents from such a $C_v$. Let $u=p(v)$. If $u=x$,
we have reached $C_x$. Otherwise $u$ is earlier than $x$. $L(v)$ is a clique (Lemma 1), so $x$ is
adjacent to $u$, and since $x$ comes after $u$, $x\in L(u)\subseteq C_u$. The walk stays inside
clusters containing $x$, moves strictly later in $\pi$, and so reaches $C_x$. All of them are
therefore connected to $C_x$ through clusters containing $x$. $\square$

**Step 2: remove non-maximal clusters.**

**Lemma 7.** In a cluster tree with RIP, if $C_i\subseteq C_j$ ($i\ne j$), then $C_i\subseteq C_m$
for the neighbour $m$ of $i$ on the path to $j$.

*Proof.* Every $x\in C_i$ is in both endpoints of the path. By RIP it is in every cluster on the
path, including $C_m$. $\square$

So while some cluster $C_i$ is contained in another, it is contained in a **neighbour** $C_m$.
**Contract** it: delete $C_i$ and join its other neighbours to $C_m$. This keeps a tree, and it
keeps RIP. For $x\in C_i$ we have $x\in C_m$, so paths that went through $C_i$ now go through
$C_m$. For $x\notin C_i$, no path among clusters containing $x$ used $C_i$. When no cluster is
contained in another, every $C_v$ that remains is maximal. By Proposition 5, these are exactly
the maximal cliques of $H^+_\pi$. This gives J3.

(Distinct vertices never produce *equal* clusters. If $u$ comes after $v$, then $v\notin C_u$,
because $C_u$ holds only $u$ and vertices after it, while $v\in C_v$. So "contained in" and "strictly
contained in" coincide here. During testing, an equivalent mutant that switched between the
two passed every test, as this predicts.)

**Family preservation (J6).** Every factor's scope is a clique of $H\subseteq H^+_\pi$, so it lies
inside some maximal clique. `CliqueTree.assign` gives each scope to the first clique (in tree
order) that contains it.

## 7. Clique trees exist exactly for chordal graphs

**Theorem 8.** The maximal cliques of a graph can be arranged into a tree with RIP **if and only
if** the graph is chordal.

*Proof.* ($\Leftarrow$) is §6 with $\pi$ a PEO, so that $H^+_\pi=H$. ($\Rightarrow$) Take a leaf
clique $L$ with neighbour $N$. A vertex in $L\setminus N$ lies in no other clique, because by RIP
any other clique containing it would have to be connected to $L$ through $N$. So all of its
neighbours lie in $L$, which is a clique: the vertex is **simplicial**. Eliminate those vertices
first, delete the leaf, and repeat on the smaller tree. This produces a PEO, so the graph is
chordal by Lemma 2. $\square$

## 8. The independent oracle: maximum-weight spanning trees

For a cluster tree $T$, let $w(T)=\sum_{(i,j)\in T}|S_{ij}|$, and let $k_x$ be the number of
clusters containing $x$.

**Proposition 9.** For **every** spanning tree $T$ on a fixed set of clusters (with every vertex
in at least one cluster), $w(T)\le\sum_x(k_x-1)=\sum_i|C_i|-n$, with equality **if and only if**
$T$ has RIP.

*Proof.* Count $w(T)$ vertex by vertex: $w(T)=\sum_xe_x$, where $e_x$ is the number of tree edges
whose two endpoints both contain $x$. Those edges form a subforest on the $k_x$ clusters that
contain $x$, so $e_x\le k_x-1$, with equality exactly when that subforest is connected. That is
RIP for $x$. Sum over $x$. $\square$

**Corollary (Jensen & Jensen, 1994).** Over the maximal cliques of a chordal graph, a spanning
tree is a junction tree **iff** it has maximum weight. A RIP tree exists (§6) and attains the
bound, so the maximum weight is exactly $\sum_i|C_i|-n$.

This gives the tests three independent checks:

1. RIP by brute force: for each vertex, check that its clusters are connected in the tree;
2. $w(T)=\sum_i|C_i|-n$, the closed form;
3. $w(T)=$ the weight of a Kruskal maximum spanning tree computed in the test, which shares no
   code with the construction.
