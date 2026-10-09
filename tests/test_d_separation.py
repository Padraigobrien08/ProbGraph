"""M2.7: d-separation by Bayes ball (spec task 7).

The main oracle implements docs/mathematics/d_separation.md §1 literally: it
enumerates every simple trail and checks each interior node. It is exhaustive on
all labelled DAGs with 4 nodes, and randomised on larger graphs.
"""

import itertools

import numpy as np
import pytest
from support import conditionally_independent, late_network, random_network

from probgraph import DAG
from probgraph.exceptions import UnknownNodeError, ValidationError

# ---------------------------------------------------------------------------
# Definition oracle: enumerate simple trails (test-local, exponential)
# ---------------------------------------------------------------------------


def trails(dag: DAG, start: str, end: str):
    """Every simple trail from ``start`` to ``end``, ignoring edge directions."""
    neighbours = {n: dag.parents(n) | dag.children(n) for n in dag.nodes()}

    def extend(path):
        if path[-1] == end:
            yield list(path)
            return
        for nxt in sorted(neighbours[path[-1]]):
            if nxt not in path:
                yield from extend([*path, nxt])

    yield from extend([start])


def trail_is_active(dag: DAG, trail: list[str], given: set[str]) -> bool:
    for prev, node, nxt in zip(trail, trail[1:], trail[2:], strict=False):
        collider = dag.has_edge(prev, node) and dag.has_edge(nxt, node)
        if collider:
            if node not in given and not (dag.descendants(node) & given):
                return False
        elif node in given:
            return False
    return True


def oracle_d_separated(dag: DAG, xs, ys, given) -> bool:
    given = set(given)
    return not any(
        trail_is_active(dag, t, given) for x in xs for y in ys for t in trails(dag, x, y)
    )


def all_dags(nodes: str):
    """Every labelled DAG on ``nodes``: each pair is absent, ->, or <-; cyclic ones are skipped."""
    pairs = list(itertools.combinations(nodes, 2))
    for choice in itertools.product((None, 0, 1), repeat=len(pairs)):
        edges = [
            (u, v) if c == 0 else (v, u)
            for (u, v), c in zip(pairs, choice, strict=True)
            if c is not None
        ]
        try:
            yield DAG(nodes=nodes, edges=edges)
        except Exception:  # CycleError
            continue


# ---------------------------------------------------------------------------
# The three local structures
# ---------------------------------------------------------------------------


def test_chain():
    dag = DAG(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    assert not dag.d_separated({"A"}, {"C"})
    assert dag.d_separated({"A"}, {"C"}, given={"B"})


def test_fork():
    dag = DAG(nodes="ABC", edges=[("B", "A"), ("B", "C")])
    assert not dag.d_separated({"A"}, {"C"})
    assert dag.d_separated({"A"}, {"C"}, given={"B"})


def test_collider_and_its_descendants():
    dag = DAG(nodes="ABCDE", edges=[("A", "B"), ("C", "B"), ("B", "D"), ("D", "E")])
    assert dag.d_separated({"A"}, {"C"})
    assert not dag.d_separated({"A"}, {"C"}, given={"B"})
    assert not dag.d_separated({"A"}, {"C"}, given={"D"})  # a child of the collider
    assert not dag.d_separated({"A"}, {"C"}, given={"E"})  # a grandchild
    # Observing the collider's descendant opens it, but observing B's parent blocks again.
    dag2 = DAG(nodes="ABCDP", edges=[("P", "A"), ("A", "B"), ("C", "B"), ("B", "D")])
    assert not dag2.d_separated({"P"}, {"C"}, given={"D"})
    assert dag2.d_separated({"P"}, {"C"}, given={"D", "A"})


def test_disconnected_nodes_are_d_separated():
    dag = DAG(nodes="ABC", edges=[("A", "B")])
    assert dag.d_separated({"A"}, {"C"})
    assert dag.d_separated({"A"}, {"C"}, given={"B"})


# ---------------------------------------------------------------------------
# Fixture facts: structural answers agree with the numbers (M2.2)
# ---------------------------------------------------------------------------

INDEPENDENCE_FACTS = [
    (["Accident"], ["Rain"], [], True),
    (["Accident"], ["Rain"], ["Traffic"], False),
    (["Accident"], ["Rain"], ["Late"], False),
    (["Accident"], ["Umbrella"], ["Late"], False),
    (["Accident"], ["Late"], ["Traffic"], True),
    (["Umbrella"], ["Late"], ["Rain"], True),
    (["Umbrella"], ["Late"], ["Traffic"], True),
    (["Umbrella"], ["Accident"], ["Traffic"], False),
    (["Umbrella"], ["Accident"], ["Traffic", "Rain"], True),
]


@pytest.mark.parametrize(("xs", "ys", "zs", "separated"), INDEPENDENCE_FACTS)
def test_fixture_facts(xs, ys, zs, separated):
    model = late_network()
    assert model.graph.d_separated(xs, ys, zs) is separated
    assert conditionally_independent(model, xs, ys, zs) is separated


# ---------------------------------------------------------------------------
# Agreement with the definition
# ---------------------------------------------------------------------------


def test_exhaustive_agreement_on_all_four_node_dags():
    nodes = "ABCD"
    checked = 0
    dags = list(all_dags(nodes))
    assert len(dags) == 543  # the number of labelled DAGs on 4 nodes (OEIS A003024)
    for dag in dags:
        for x, y in itertools.combinations(nodes, 2):
            rest = [n for n in nodes if n not in (x, y)]
            for k in range(len(rest) + 1):
                for given in itertools.combinations(rest, k):
                    expected = oracle_d_separated(dag, [x], [y], given)
                    assert dag.d_separated({x}, {y}, set(given)) is expected, (
                        dag,
                        x,
                        y,
                        given,
                    )
                    checked += 1
    assert checked == 543 * 6 * 4


@pytest.mark.parametrize("seed", range(60))
def test_agreement_with_sets_on_random_dags(seed):
    rng = np.random.default_rng(seed)
    dag = random_network(seed, n_vars=(3, 7), edge_prob=0.35).graph
    nodes = list(dag.nodes())
    perm = [str(n) for n in rng.permutation(nodes)]
    nx = int(rng.integers(1, 3))
    ny = int(rng.integers(1, 3))
    xs, ys = perm[:nx], perm[nx : nx + ny]
    if not ys:
        return
    given = [n for n in perm[nx + ny :] if rng.random() < 0.4]
    assert dag.d_separated(xs, ys, given) is oracle_d_separated(dag, xs, ys, given)
    assert dag.d_separated(ys, xs, given) is dag.d_separated(xs, ys, given)  # D1 symmetry


@pytest.mark.parametrize("seed", range(40))
def test_d_connected_nodes_matches_definition(seed):
    rng = np.random.default_rng(seed + 1000)
    dag = random_network(seed, n_vars=(2, 7), edge_prob=0.35).graph
    nodes = list(dag.nodes())
    x = str(rng.choice(nodes))
    given = {n for n in nodes if n != x and rng.random() < 0.3}
    expected = {
        y
        for y in nodes
        if y != x and y not in given and not oracle_d_separated(dag, [x], [y], given)
    }
    assert dag.d_connected_nodes(x, given) == expected


# ---------------------------------------------------------------------------
# D2: input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xs", "ys", "zs", "error", "match"),
    [
        ({"A"}, {"A"}, set(), ValidationError, "disjoint"),
        ({"A"}, {"C"}, {"A"}, ValidationError, "disjoint"),
        ({"A"}, {"C"}, {"C"}, ValidationError, "disjoint"),
        (set(), {"C"}, set(), ValidationError, "empty"),
        ({"A"}, set(), set(), ValidationError, "empty"),
        ({"Z"}, {"C"}, set(), UnknownNodeError, "Z"),
        ({"A"}, {"C"}, {"Z"}, UnknownNodeError, "Z"),
        ("A", {"C"}, set(), ValidationError, "string"),
    ],
)
def test_invalid_d_separation_queries(xs, ys, zs, error, match):
    dag = DAG(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    with pytest.raises(error, match=match):
        dag.d_separated(xs, ys, zs)


def test_d_connected_nodes_validation():
    dag = DAG(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    with pytest.raises(ValidationError, match="given"):
        dag.d_connected_nodes("A", {"A"})
    with pytest.raises(UnknownNodeError):
        dag.d_connected_nodes("Z")
    assert dag.d_connected_nodes("A") == {"B", "C"}
    assert dag.d_connected_nodes("A", {"B"}) == set()
