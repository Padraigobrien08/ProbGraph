# P29 — Score-based structure search

> M4 gave every graph a decomposable score: one term per family. Structure learning maximises it over
> all DAGs, whose number grows super-exponentially (543 on 4 nodes, 29,281 on 5, about $4\times10^{18}$ on 10).
> Two methods are developed here. **Hill climbing** moves between neighbouring DAGs, and decomposability
> makes each move cost one or two family scores. It always stops, but only at a *local* optimum.
> **Exact search** by dynamic programming over subsets finds the global optimum in $O(n\,2^n)$ steps after
> scoring parent sets, which is feasible for about a dozen variables. It is the oracle for hill climbing.

Prerequisites: P15 Part 2 (BDeu), P17 (BIC, decomposability, consistency), P28 (equivalence).

---

## 1. The objective

For data $D$ over variables $X_1,\ldots,X_n$, a decomposable score is
$$\mathrm{score}(G)=\sum_is(X_i,\mathrm{Pa}_G(X_i)),$$
where the family term $s(X,U)$ depends only on the counts of $X$ and $U$ (P17 §3): BIC or BDeu. Both are
score-equivalent (P15 §11, P17 §4), so the optimum is a whole equivalence class, and search returns one
member of it.

## 2. Moves and their deltas (S1)

From a DAG $G$, three moves reach a neighbour:

| Move | Legal when | Change in score |
|---|---|---|
| add $a\to b$ | not adjacent; no directed path $b\leadsto a$; $\lvert\mathrm{Pa}(b)\rvert<$ `max_parents`; not forbidden | $s(b,\mathrm{Pa}(b)\cup\{a\})-s(b,\mathrm{Pa}(b))$ |
| delete $a\to b$ | not required | $s(b,\mathrm{Pa}(b)\setminus\{a\})-s(b,\mathrm{Pa}(b))$ |
| reverse $a\to b$ | not required; $b\to a$ not forbidden; no path $a\leadsto b$ other than the edge; parent limit for $a$ | the delete term for $b$, plus $s(a,\mathrm{Pa}(a)\cup\{b\})-s(a,\mathrm{Pa}(a))$ |

**Proposition 1.** Each delta equals $\mathrm{score}(G')-\mathrm{score}(G)$.

*Proof.* Only the families whose parent sets change contribute: $b$'s for an addition or deletion, and
$b$'s and $a$'s for a reversal. Every other term is the same in $G$ and $G'$. $\square$

**Acyclicity.** Adding $a\to b$ creates a cycle iff $b$ already reaches $a$. Reversing $a\to b$ creates one
iff $a$ reaches $b$ by some other path, since the new edge $b\to a$ would then close it. Both are checked by
a depth-first search.

Family scores are cached by (child, parent set). Hill climbing revisits the same families constantly, so
`evaluations`, the number of cache misses, is far below the number of moves considered.

## 3. Hill climbing (S2)

Plain greedy search repeatedly takes the move with the largest **strictly positive** delta. Ties go to
the first in a fixed order: by variable order of $a$, then $b$, then add before delete before reverse.

**Proposition 2.** It terminates, its score history strictly increases, and its result is a **local
optimum**: no legal single move improves it.

*Proof.* Each step raises the score of a DAG, and there are finitely many DAGs, so no DAG is visited
twice and the process stops. It stops exactly when no move has a positive delta. $\square$

**Escapes** (spec ⚑3, opt-in):

- **Tabu search** keeps the last `tabu_length` graphs. It may take the best non-improving move to an
  unvisited graph, and stops after `tabu_length` consecutive steps without improving on the best score
  seen. It returns the best graph seen, not the last.
- **Random restarts** perturb the best graph found so far by several random legal moves, run hill climbing
  again, and keep the best result.

Neither makes the method exact. They make the trap of fixture F5 rarer, and the tests measure by how much.

## 4. Exact search by dynamic programming (S3)

Two observations make the exact optimum tractable for small $n$ (Silander and Myllymäki, 2006).

**Best parents.** For each variable $X$ and each candidate set $C\subseteq V\setminus\{X\}$, let
$$\mathrm{bps}(X,C)=\max_{U\subseteq C}s(X,U).$$
This satisfies $\mathrm{bps}(X,C)=\max\big(s(X,C),\max_{Y\in C}\mathrm{bps}(X,C\setminus\{Y\})\big)$, so all of
them follow from the $2^{n-1}$ family scores of $X$, with `max_parents` limiting $\lvert U\rvert$.

**Best orders.** Every DAG has a topological order, and the best DAG compatible with an order lets each
variable choose its best parents among its predecessors. Let $F(S)$ be the best score of a DAG on the subset
$S$, with every node drawing its parents from $S$ only. Its last node (a sink) $X$ can take any parents in
$S\setminus\{X\}$, so
$$F(\varnothing)=0,\qquad F(S)=\max_{X\in S}\Big(F(S\setminus\{X\})+\mathrm{bps}(X,S\setminus\{X\})\Big).$$

**Proposition 3.** $F(V)$ is the maximum score over all DAGs on $V$, and following the maximising choices back
from $V$ recovers an optimal DAG.

*Proof.* Every DAG on $S$ has a sink $X$. Removing $X$ leaves a DAG on $S\setminus\{X\}$, and $X$'s parents lie in
$S\setminus\{X\}$, so its score is at most $F(S\setminus\{X\})+\mathrm{bps}(X,S\setminus\{X\})$. That gives $\le$. Conversely, the
choices achieving the maximum assemble into a DAG (each node's parents precede it in the order recovered),
achieving the bound. $\square$

The table $F$ has $2^n$ entries, each a maximum over at most $n$ terms, plus the parent-set work: $O(n\,2^n)$
lookups after $n\,2^{n-1}$ family scores. Twelve variables means about 25,000 family scores (spec ⚑4).

## 5. Why greedy search gets trapped, and consistency

Hill climbing over DAGs can stop where every single change is worse, but a pair of changes would help. A
typical case is an edge in the wrong direction whose reversal is only worthwhile once a third edge is added.
In fixture F5, plain hill climbing from the empty graph ended below the exhaustive optimum in 16 of 60
random 4-variable problems.

**Consistency (stated).** With BIC or BDeu, as $N\to\infty$ the exact optimum lies in the true equivalence
class with probability tending to 1 (P17 §5). Hill climbing has no such guarantee, which is what GES (P31)
restores.

## 6. How the tests check this independently

- **Deltas** for every legal move on random graphs, against rescoring the whole graph with `family_scores`.
- **Local optimality** of every plain hill-climbing result, by trying every single move independently.
- **Constraints:** acyclicity, `max_parents`, required and forbidden edges, on every result.
- **Exhaustive optima** by scoring all 543 DAGs on 4 nodes. On the tests' 30 problems, plain greedy search
  is trapped in 6; tabu (length 10) alone leaves 2, 10 random restarts alone leave 2, 30 restarts leave
  none, and tabu with 5 restarts leaves 1. Each step is checked to take the steepest move.
- **Exact search (M7.3)** against exhaustive scoring on 4 and 5 nodes, and recovery of known networks'
  equivalence classes from large samples.
