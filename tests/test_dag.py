import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probgraph import DAG
from probgraph.exceptions import CycleError, UnknownNodeError, ValidationError

# ---------------------------------------------------------------------------
# Helpers and independent oracle (see docs/mathematics/dag_algorithms.md §4)
# ---------------------------------------------------------------------------


def chain(*nodes: str) -> DAG:
    return DAG(nodes=nodes, edges=list(zip(nodes, nodes[1:])))


def adjacency(dag: DAG) -> tuple[list[str], np.ndarray]:
    nodes = list(dag.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for u, v in dag.edges():
        a[index[u], index[v]] = 1
    return nodes, a


def reachability(a: np.ndarray) -> np.ndarray:
    """R[i, j] is True iff there is a directed path of length >= 1 from i to j."""
    n = len(a)
    reach = np.zeros_like(a, dtype=bool)
    power = np.eye(n, dtype=np.int64)
    for _ in range(n):
        power = np.minimum(power @ a, 1)  # clip to keep entries bounded
        reach |= power.astype(bool)
    return reach


def is_nilpotent(a: np.ndarray) -> bool:
    n = len(a)
    return n == 0 or not np.linalg.matrix_power(a, n).any()


def assert_valid_topological_order(dag: DAG, order: list[str]) -> None:
    assert sorted(order) == sorted(dag.nodes()), "must be a permutation of the nodes"
    position = {node: i for i, node in enumerate(order)}
    for u, v in dag.edges():
        assert position[u] < position[v], f"edge {u}->{v} violates the ordering"


def snapshot(dag: DAG):
    return dag.nodes(), dag.edges(), {n: (dag.parents(n), dag.children(n)) for n in dag.nodes()}


# ---------------------------------------------------------------------------
# Specification table (§3.2)
# ---------------------------------------------------------------------------


def test_chain_is_valid_dag():
    dag = chain("A", "B", "C")
    assert dag.edges() == (("A", "B"), ("B", "C"))
    assert dag.topological_sort() == ["A", "B", "C"]


def test_three_cycle_is_rejected():
    dag = chain("A", "B", "C")
    # The reported cycle starts at the offending edge's source.
    with pytest.raises(CycleError, match="C -> A -> B -> C"):
        dag.add_edge("C", "A")


def test_self_loop_is_rejected():
    dag = DAG(nodes=["A"])
    with pytest.raises(CycleError):
        dag.add_edge("A", "A")


def test_cycle_error_is_a_validation_error():
    dag = DAG(nodes=["A"])
    with pytest.raises(ValidationError):
        dag.add_edge("A", "A")


def test_fork_root_precedes_children():
    dag = DAG(nodes="ABC", edges=[("A", "B"), ("A", "C")])
    order = dag.topological_sort()
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")


def test_disconnected_nodes_are_included_in_topological_sort():
    dag = DAG(nodes=["X", "A", "B", "Y"], edges=[("A", "B")])
    order = dag.topological_sort()
    assert_valid_topological_order(dag, order)
    assert set(order) == {"X", "A", "B", "Y"}


def test_repeated_edge_is_idempotent():
    dag = chain("A", "B")
    dag.add_edge("A", "B")
    assert dag.edges() == (("A", "B"),)
    assert dag.children("A") == {"B"}


def test_repeated_node_is_idempotent():
    dag = chain("A", "B")
    dag.add_node("A")
    assert dag.nodes() == ("A", "B")
    assert dag.children("A") == {"B"}


def test_remove_nonexistent_edge_is_noop():
    dag = chain("A", "B", "C")
    before = snapshot(dag)
    dag.remove_edge("A", "C")
    dag.remove_edge("C", "B")
    assert snapshot(dag) == before


def test_remove_edge_updates_both_directions():
    dag = chain("A", "B", "C")
    dag.remove_edge("A", "B")
    assert dag.children("A") == set()
    assert dag.parents("B") == set()
    dag.add_edge("C", "A")  # now legal: no path A ~> C remains
    assert_valid_topological_order(dag, dag.topological_sort())


def test_ancestor_is_not_direct_parent():
    dag = chain("A", "B", "C")
    assert "A" in dag.ancestors("C")
    assert "A" not in dag.parents("C")
    assert dag.ancestors("C") == {"A", "B"}
    assert dag.descendants("A") == {"B", "C"}


def test_node_is_not_its_own_ancestor_or_descendant():
    dag = chain("A", "B", "C")
    for node in dag.nodes():
        assert node not in dag.ancestors(node)
        assert node not in dag.descendants(node)


def test_collider_structure():
    dag = DAG(nodes=["A", "B", "C"], edges=[("A", "C"), ("B", "C")])
    assert dag.parents("C") == {"A", "B"}
    assert dag.parents("A") == set() and dag.parents("B") == set()
    assert dag.ancestors("A") == set()  # A and B are unrelated structurally
    assert "B" not in dag.descendants("A")


# ---------------------------------------------------------------------------
# Unknown nodes, input validation, encapsulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        lambda d: d.add_edge("A", "Z"),
        lambda d: d.add_edge("Z", "A"),
        lambda d: d.remove_edge("A", "Z"),
        lambda d: d.parents("Z"),
        lambda d: d.children("Z"),
        lambda d: d.ancestors("Z"),
        lambda d: d.descendants("Z"),
    ],
)
def test_unknown_node_is_rejected(operation):
    dag = chain("A", "B")
    before = snapshot(dag)
    with pytest.raises(UnknownNodeError):
        operation(dag)
    assert snapshot(dag) == before


@pytest.mark.parametrize("bad", ["", None, 1])
def test_invalid_node_key_is_rejected(bad):
    with pytest.raises(ValidationError):
        DAG().add_node(bad)


def test_returned_sets_are_copies():
    dag = chain("A", "B", "C")
    dag.parents("B").add("C")
    dag.children("A").clear()
    dag.ancestors("C").clear()
    dag.descendants("A").add("Z")
    assert dag.parents("B") == {"A"}
    assert dag.children("A") == {"B"}
    assert dag.ancestors("C") == {"A", "B"}
    assert dag.descendants("A") == {"B", "C"}


def test_nodes_and_edges_preserve_insertion_order():
    dag = DAG(nodes=["C", "A", "B"])
    dag.add_edge("C", "B")
    dag.add_edge("A", "B")
    assert dag.nodes() == ("C", "A", "B")
    assert dag.edges() == (("C", "B"), ("A", "B"))


def test_topological_sort_is_deterministic():
    def build() -> DAG:
        return DAG(nodes=["D", "B", "A", "C"], edges=[("B", "C"), ("A", "C"), ("D", "C")])

    assert build().topological_sort() == build().topological_sort()
    # Ties are broken by insertion order.
    assert build().topological_sort() == ["D", "B", "A", "C"]


def test_empty_graph():
    dag = DAG()
    assert dag.nodes() == ()
    assert dag.edges() == ()
    assert dag.topological_sort() == []


def test_topological_sort_matches_brute_force_enumeration():
    """The result is one of the orderings found by exhaustive search."""
    dag = DAG(nodes="ABCD", edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    valid = set()
    for perm in itertools.permutations(dag.nodes()):
        pos = {n: i for i, n in enumerate(perm)}
        if all(pos[u] < pos[v] for u, v in dag.edges()):
            valid.add(perm)
    assert valid == {("A", "B", "C", "D"), ("A", "C", "B", "D")}
    assert tuple(dag.topological_sort()) in valid


# ---------------------------------------------------------------------------
# Property-based tests: invariants G1–G5 on random edge-insertion histories
# ---------------------------------------------------------------------------

NODE_NAMES = [f"X{i}" for i in range(8)]


@st.composite
def insertion_histories(draw):
    n = draw(st.integers(min_value=1, max_value=len(NODE_NAMES)))
    nodes = NODE_NAMES[:n]
    pairs = st.tuples(st.sampled_from(nodes), st.sampled_from(nodes))
    return nodes, draw(st.lists(pairs, max_size=30))


@settings(max_examples=300, deadline=None)
@given(insertion_histories())
def test_invariants_hold_under_random_insertions(history):
    nodes, attempts = history
    dag = DAG(nodes=nodes)

    for u, v in attempts:
        names, a = adjacency(dag)
        idx = {name: i for i, name in enumerate(names)}
        would_cycle = u == v or reachability(a)[idx[v], idx[u]]
        before = snapshot(dag)

        if would_cycle:
            # Lemma 1 + G5: rejected, and the graph is untouched.
            with pytest.raises(CycleError):
                dag.add_edge(u, v)
            assert snapshot(dag) == before
        else:
            dag.add_edge(u, v)
            assert (u, v) in dag.edges()

    names, a = adjacency(dag)
    reach = reachability(a)

    # G1 — acyclicity, verified independently via nilpotency of the adjacency matrix.
    assert is_nilpotent(a)

    # G2 — every edge goes forward in the topological ordering.
    assert_valid_topological_order(dag, dag.topological_sort())

    for i, u in enumerate(names):
        # G3 — parent/child reciprocity.
        for v in dag.children(u):
            assert u in dag.parents(v)
        for p in dag.parents(u):
            assert u in dag.children(p)

        # G4 — ancestor/descendant reciprocity, and agreement with the oracle.
        expected_desc = {names[j] for j in np.flatnonzero(reach[i])}
        expected_anc = {names[j] for j in np.flatnonzero(reach[:, i])}
        assert dag.descendants(u) == expected_desc
        assert dag.ancestors(u) == expected_anc
        for v in dag.descendants(u):
            assert u in dag.ancestors(v)


@settings(max_examples=200, deadline=None)
@given(insertion_histories(), st.data())
def test_removal_preserves_invariants(history, data):
    nodes, attempts = history
    dag = DAG(nodes=nodes)
    for u, v in attempts:
        try:
            dag.add_edge(u, v)
        except CycleError:
            pass
    for u, v in data.draw(st.lists(st.sampled_from(list(dag.edges()) or [(nodes[0], nodes[0])]))):
        dag.remove_edge(u, v)
        assert (u, v) not in dag.edges()
        assert v not in dag.children(u) and u not in dag.parents(v)

    names, a = adjacency(dag)
    assert is_nilpotent(a)
    assert_valid_topological_order(dag, dag.topological_sort())
