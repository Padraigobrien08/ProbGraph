# P28 — Markov equivalence classes

> Two DAGs that imply the same conditional independences cannot be told apart by any amount of
> observational data. M4 already showed the score side of this: BDeu and BIC tie on equivalent graphs.
> This note characterises the classes. They are determined by the **skeleton** and the
> **v-structures**, and are represented by a partially directed graph, the **CPDAG**, whose directed
> edges are exactly those that every member agrees on. Meek's rules compute it from a DAG, and
> Dor–Tarsi's procedure turns it back into a member.

Prerequisites: P7 (d-separation), P17 §4 (covered edges, score equivalence).

---

## 1. Definitions

DAGs $G$ and $H$ on the same nodes are **Markov equivalent** if they have the same d-separation
statements: $X\perp_GY\mid Z\iff X\perp_HY\mid Z$ for all disjoint $X,Y,Z$. The **skeleton** is the
undirected graph obtained by forgetting directions. A **v-structure** is a triple $a\to c\leftarrow b$ with
$a$ and $b$ not adjacent (an *unshielded collider*).

## 2. Two facts read off from separating sets

**Lemma 1 (adjacency).** In a DAG, $a$ and $b$ are adjacent iff no set d-separates them.

*Proof.* An edge is a trail of length one, with no intermediate nodes to block it, so adjacent nodes are
never d-separated. Conversely, if $a$ and $b$ are not adjacent, one of them, say $a$, is not a descendant
of the other. By the local Markov property (P1, P7), $a$ is d-separated from its non-descendants given its
parents, and $b$ is one of those non-descendants. So $\mathrm{pa}(a)$ separates them. $\square$

**Lemma 2 (unshielded triples).** Let $a-c-b$ be adjacent pairs with $a,b$ not adjacent. If $a\to c\leftarrow b$,
then **no** separating set of $a$ and $b$ contains $c$. Otherwise (a chain or fork at $c$), **every**
separating set contains $c$.

*Proof.* If $c$ is a collider, any set containing $c$ activates the trail $a\to c\leftarrow b$. If $c$ is a chain
or fork node, the trail $a-c-b$ is active unless $c$ is in the conditioning set. $\square$

**Theorem 1 (Verma–Pearl).** $G$ and $H$ are Markov equivalent iff they have the same skeleton and the
same v-structures.

*Proof of "only if".* The d-separation statements determine the skeleton, by Lemma 1. They also
determine, for each unshielded triple, whether $c$ lies in the separating sets of $a$ and $b$, and so, by
Lemma 2, whether the triple is a v-structure. The "if" direction is stated, and checked exhaustively in
the tests by comparing complete lists of d-separation statements on every 4-node DAG. A proof follows
from Chickering's theorem (P17 §4): equivalent DAGs are joined by covered-edge reversals, and each
reversal preserves the skeleton, the v-structures and the d-separations. $\square$

Lemmas 1 and 2 are also exactly the two steps of the PC algorithm (P30): find adjacencies by searching for
separating sets, then orient unshielded triples by checking whether the middle node was used to separate.

## 3. The CPDAG

An edge of $G$ is **compelled** if it has the same direction in every DAG of the class, and **reversible**
otherwise. The **CPDAG** (completed partially directed acyclic graph, or essential graph) has the skeleton
of the class, with compelled edges directed and reversible edges undirected. Equivalent DAGs have the same
CPDAG by definition, and different classes have different CPDAGs because their skeletons or v-structures
differ.

**Computing it.** Start from the **pattern**: the skeleton, with only the v-structure edges directed. Then
apply Meek's rules until nothing changes. Each rule orients an undirected edge $b-c$ whose other orientation
would create a new v-structure or a cycle, so each orientation is forced in every member:

| Rule | Configuration | Conclusion | Why the other direction is impossible |
|---|---|---|---|
| R1 | $a\to b$, $b-c$, $a,c$ not adjacent | $b\to c$ | $c\to b$ would make a new v-structure $a\to b\leftarrow c$ |
| R2 | $a\to b\to c$, $a-c$ | $a\to c$ | $c\to a$ would close the cycle $a\to b\to c\to a$ |
| R3 | $a-b$, $a-c$, $a-d$, $c\to b$, $d\to b$, $c,d$ not adjacent | $a\to b$ | if $b\to a$, then avoiding the cycles $c\to b\to a\to c$ and $d\to b\to a\to d$ forces $c\to a\leftarrow d$, a new v-structure |

**Theorem 2 (Meek, 1995; stated).** The closure of the pattern under R1–R3 is the CPDAG. A fourth rule is
needed only when background knowledge orients extra edges, which M7 does not use. The tests verify Theorem 2
exhaustively on 4 nodes: an edge is directed in the computed CPDAG iff every DAG of the class orients it
the same way.

## 4. Back from a CPDAG to a member: Dor–Tarsi

To find a DAG with the given skeleton, directed edges and v-structures, repeatedly pick a node $x$ such that:

1. $x$ has no outgoing directed edge (it can be a sink), and
2. every undirected neighbour of $x$ is adjacent to every other node adjacent to $x$.

Orient all of $x$'s undirected edges into $x$, then remove $x$. Condition 2 ensures that the new parents of
$x$ are pairwise adjacent to its other parents, so no new v-structure appears. Condition 1 and the removal
order ensure acyclicity. If at some point no such $x$ exists, the PDAG has no consistent extension (Dor and
Tarsi, 1992). The chordless undirected 4-cycle is the standard example: every acyclic orientation of it has
a sink with two non-adjacent parents.

## 5. Enumerating a class, and distance between classes

For small graphs, `members()` tries every orientation of the undirected edges and keeps those that are
acyclic and have the same CPDAG. **Structural Hamming distance** compares two CPDAGs pair by pair. Each pair
of nodes has one of four marks: absent, undirected, $a\to b$, or $b\to a$. The distance counts the pairs
whose marks differ. It is a metric on CPDAGs, so it compares what data can determine, not arbitrary choices
within a class.

## 6. How the tests check this independently

- **Fixture F2:** the late network's CPDAG, with Traffic → Late compelled by R1, and a class of 2 members.
- **Counts (F1):** grouping every DAG on 1–5 nodes by its computed CPDAG gives 1, 2, 11, 185, 8782 classes,
  the same partition as grouping by skeleton and v-structures computed independently.
- **Semantics:** on 4 nodes, two DAGs have the same CPDAG iff they have the same complete list of
  d-separation statements, computed with M2's Bayes ball.
- **Compelled edges (Theorem 2):** directed in the CPDAG iff fixed across the whole class, on every 4-node class.
- **Each Meek rule** on its own hand example; Dor–Tarsi on every class, and its failure on the 4-cycle.
- **SHD:** hand examples, symmetry, and the triangle inequality on random CPDAGs.
