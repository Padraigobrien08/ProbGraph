# P11 — Chordal graphs and clique trees

> Message passing needs a **tree** of clusters whose separators carry all the information that
> crosses them. Such a tree exists exactly when the graph is **chordal**. This note has two
> parts:
>
> - **Part 1 (M3.4):** how to make a graph chordal by triangulating it along an elimination
>   order, how to recognise a chordal graph by maximum cardinality search, and how to read off
>   its maximal cliques.
> - **Part 2 (M3.5):** connecting those cliques into a clique tree with the running intersection
>   property.

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

*(Added in M3.5: the running intersection property, construction from an elimination order,
existence if and only if the graph is chordal, and the maximum-weight spanning tree as an
independent oracle.)*
