# P7 — d-separation

> **Question.** Given a DAG $G$ and disjoint sets of variables $X$, $Y$ and $Z$, does
> $X\perp Y\mid Z$ hold in **every** distribution that factorises over $G$? d-separation answers
> this from the graph alone. Bayes ball decides it in time linear in the size of the graph.

This note covers the definition, the three local structures, and the correctness of Bayes ball
(M2.7). Soundness, generic completeness and the moralised-ancestral criterion are added in
M2.8 (§5).

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

## 5. Soundness and completeness (added in M2.8)

This section will contain:

- **Soundness:** $X\perp_G Y\mid Z\Rightarrow X\perp Y\mid Z$ in every $P$ that factorises over
  $G$, proved through the moralised ancestral graph.
- **Completeness for generic parameters** (Meek, 1995).
- **Equivalence** of Bayes ball and the moralised-ancestral criterion.
