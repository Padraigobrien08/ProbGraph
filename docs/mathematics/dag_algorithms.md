# DAG algorithms — correctness arguments

This note justifies the three algorithms in `probgraph.graphs.dag`:
transactional cycle rejection, ancestor and descendant traversal, and Kahn's
topological sort. It also explains the independent test oracle used in
`tests/test_dag.py`.

Notation: $G=(V,E)$, with $|V|=n$ and $|E|=m$. $u\rightsquigarrow v$ means there is a
directed path of length $\ge 1$ from $u$ to $v$.

---

## 1. Cycle rejection on edge insertion (G1, G5)

**Lemma 1.** Let $G$ be acyclic and $u,v\in V$. Then $G+(u\rightarrow v)$ contains
a cycle **iff** $u=v$ or $v\rightsquigarrow u$ in $G$.

*Proof.* ($\Leftarrow$) If $u=v$, the new edge is a self-loop. If $v\rightsquigarrow u$,
then $u\rightarrow v\rightsquigarrow u$ is a cycle.
($\Rightarrow$) $G$ has no cycle, so any cycle in $G+(u\rightarrow v)$ must use the
new edge. Write it as $u\rightarrow v\rightsquigarrow u$. The return path
$v\rightsquigarrow u$ (when $u\ne v$) uses only old edges, because a simple cycle
uses each edge at most once. $\square$

**Algorithm.** Before changing any state, run a depth-first search from $v$ along
outgoing edges. If it reaches $u$, raise `CycleError` and report the cycle.
Otherwise insert the edge into both adjacency maps.

**Consequences.**
- *G1 is preserved by induction.* The empty graph is acyclic, and Lemma 1 shows
  that every accepted insertion keeps it acyclic. Removing edges cannot create
  a cycle.
- *G5 (transactional).* The check runs before any mutation, so a rejected edge
  leaves the graph unchanged.
- Cost: $O(n+m)$ per insertion in the worst case.

---

## 2. Ancestors and descendants (G3, G4)

Every edge is stored twice: as $v\in\text{children}[u]$ and as
$u\in\text{parents}[v]$. Both entries are written and removed together, so
**G3 holds by construction.**

$\mathrm{desc}(u)=\{v: u\rightsquigarrow v\}$ is computed by graph search over
`children`, and $\mathrm{anc}(v)=\{u: u\rightsquigarrow v\}$ by graph search over
`parents`. G3 makes walking backwards through `parents` the same as reversing
the edges. So
$$u\in\mathrm{anc}(v)\iff u\rightsquigarrow v\iff v\in\mathrm{desc}(u),$$
which is **G4**. Neither set contains the node itself: that would require
$u\rightsquigarrow u$, which is a cycle.

---

## 3. Kahn's algorithm (G2)

**Lemma 2.** Every finite non-empty DAG has a *source*, meaning a node with
in-degree 0.

*Proof.* Suppose every node has a parent. Start anywhere and keep stepping to a
parent. After $n$ steps we have visited $n+1$ nodes, so by the pigeonhole
principle some node repeats. That gives a cycle, a contradiction. $\square$

**Algorithm.** Repeatedly remove a source, append it to the output, and decrement
the in-degree of each of its children.

**Correctness.**
- *Completeness.* Removing a node from a DAG leaves a DAG. By Lemma 2 a source
  always exists while nodes remain, so all $n$ nodes are output.
- *Validity (G2).* A node becomes a source only after all of its parents have
  been removed, so for every edge $u\rightarrow v$, $u$ is output before $v$.
- *Cycle detection, as a defensive check.* If $G$ had a cycle, no node on it
  could ever reach in-degree 0, so the output would be short. Insertion already
  prevents cycles, so the implementation treats a short output as an internal
  error.

**Determinism.** When several sources are ready, the implementation takes the one
that was **inserted earliest**, using a min-heap keyed on insertion index. The
output is then a pure function of the construction history, at a cost of
$O((n+m)\log n)$ instead of $O(n+m)$. This rule picks one valid ordering among
many. It is not a mathematical requirement, and tests should not rely on it
except to check reproducibility.

---

## 4. Independent oracle used in tests

The tests do not check the graph code against itself. They build the adjacency
matrix $A$ ($A_{ij}=1$ iff $i\rightarrow j$) and use linear algebra:

- $(A^k)_{ij}$ counts the walks of length $k$ from $i$ to $j$.
- **$G$ is acyclic $\iff A^n = 0$** (that is, $A$ is nilpotent). In a DAG every
  walk is a path, and a path has at most $n-1$ edges, so $A^n=0$. If there is a
  cycle, going around it repeatedly gives walks of every length, so $A^k\ne0$
  for every $k$.
- **Reachability.** $u\rightsquigarrow v \iff \big(\sum_{k=1}^{n}A^k\big)_{uv}>0$.
  This gives an independent ground truth for `ancestors`/`descendants` and for
  Lemma 1's accept or reject decision.
