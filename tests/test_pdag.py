"""M7.1: PDAGs and CPDAGs (E1–E4; F1, F2; equivalence_classes.md)."""

import itertools
from collections import defaultdict

import numpy as np
import pytest

from probgraph import DAG
from probgraph.exceptions import ValidationError
from probgraph.structure import PDAG, cpdag, markov_equivalent, structural_hamming_distance


def all_dags(n: int):
    """Every labelled DAG on nodes "0".."n-1" (by orienting each pair, keeping acyclic ones)."""
    nodes = [str(i) for i in range(n)]
    pairs = list(itertools.combinations(nodes, 2))
    for choice in itertools.product((0, 1, 2), repeat=len(pairs)):
        edges = [(a, b) if c == 1 else (b, a) for (a, b), c in zip(pairs, choice, strict=True) if c]
        try:
            yield DAG(nodes=nodes, edges=edges)
        except ValidationError:
            continue


def signature(dag: DAG):
    """Skeleton and v-structures, computed directly (Theorem 1)."""
    skeleton = frozenset(frozenset(e) for e in dag.edges())
    v = set()
    for c in dag.nodes():
        for a, b in itertools.combinations(sorted(dag.parents(c)), 2):
            if frozenset((a, b)) not in skeleton:
                v.add((a, c, b))
    return skeleton, frozenset(v)


def separations(dag: DAG) -> frozenset:
    """Every statement X ⊥ Y | Z for single X, Y and any Z, by Bayes ball."""
    nodes = dag.nodes()
    out = set()
    for x, y in itertools.combinations(nodes, 2):
        rest = [n for n in nodes if n not in (x, y)]
        for k in range(len(rest) + 1):
            for z in itertools.combinations(rest, k):
                if dag.d_separated({x}, {y}, set(z)):
                    out.add((x, y, z))
    return frozenset(out)


def late() -> DAG:
    return DAG(
        nodes=["Rain", "Accident", "Traffic", "Late", "Umbrella"],
        edges=[
            ("Rain", "Traffic"),
            ("Accident", "Traffic"),
            ("Traffic", "Late"),
            ("Rain", "Umbrella"),
        ],
    )


# ---------------------------------------------------------------------------
# F2: the late network
# ---------------------------------------------------------------------------


def test_f2_late_network_cpdag():
    result = cpdag(late())
    assert result.directed == {("Rain", "Traffic"), ("Accident", "Traffic"), ("Traffic", "Late")}
    assert result.undirected == {frozenset({"Rain", "Umbrella"})}
    members = list(result.members())
    assert len(members) == 2
    assert {("Rain", "Umbrella") in m.edges() for m in members} == {True, False}


# ---------------------------------------------------------------------------
# E1, E4, F1: CPDAGs partition DAGs exactly by skeleton and v-structures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "dags", "classes"), [(1, 1, 1), (2, 3, 2), (3, 25, 11), (4, 543, 185)]
)
def test_f1_class_counts_and_partition(n, dags, classes):
    graphs = list(all_dags(n))
    assert len(graphs) == dags
    by_cpdag = defaultdict(set)
    by_signature = defaultdict(set)
    for i, g in enumerate(graphs):
        by_cpdag[cpdag(g)].add(i)
        by_signature[signature(g)].add(i)
    assert len(by_cpdag) == classes
    assert sorted(map(sorted, by_cpdag.values())) == sorted(map(sorted, by_signature.values()))


def test_f1_five_nodes():
    graphs = list(all_dags(5))
    assert len(graphs) == 29281
    assert len({cpdag(g) for g in graphs}) == 8782


def test_same_cpdag_iff_same_d_separations():
    """Verma–Pearl in both directions, semantically, on every 4-node DAG."""
    graphs = list(all_dags(4))
    by_cpdag = defaultdict(set)
    by_statements = defaultdict(set)
    for i, g in enumerate(graphs):
        by_cpdag[cpdag(g)].add(i)
        by_statements[separations(g)].add(i)
    assert sorted(map(sorted, by_cpdag.values())) == sorted(map(sorted, by_statements.values()))


def test_markov_equivalent():
    a = DAG(nodes=["x", "y", "z"], edges=[("x", "y"), ("y", "z")])
    b = DAG(nodes=["x", "y", "z"], edges=[("z", "y"), ("y", "x")])
    c = DAG(nodes=["x", "y", "z"], edges=[("x", "y"), ("z", "y")])
    assert markov_equivalent(a, b) and not markov_equivalent(a, c)


# ---------------------------------------------------------------------------
# E2: compelled edges (Meek's theorem), and E3: members and extensions
# ---------------------------------------------------------------------------


def test_compelled_edges_are_exactly_the_directed_ones():
    classes = defaultdict(list)
    for g in all_dags(4):
        classes[cpdag(g)].append(g)
    for pdag, members in classes.items():
        orientations = defaultdict(set)
        for g in members:
            for a, b in g.edges():
                orientations[frozenset((a, b))].add((a, b))
        for pair, seen in orientations.items():
            if len(seen) == 1:
                assert next(iter(seen)) in pdag.directed
            else:
                assert pair in pdag.undirected
        assert sorted(sorted(m.edges()) for m in pdag.members()) == sorted(
            sorted(g.edges()) for g in members
        )
        assert cpdag(pdag.consistent_extension()) == pdag


def test_a_chordless_four_cycle_has_no_extension():
    nodes = ("a", "b", "c", "d")
    cycle = PDAG(
        nodes,
        frozenset(),
        frozenset(frozenset(p) for p in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]),
    )
    with pytest.raises(ValidationError, match="no consistent extension"):
        cycle.consistent_extension()
    assert list(cycle.members()) == []


# ---------------------------------------------------------------------------
# Each Meek rule on its own
# ---------------------------------------------------------------------------


def test_meek_rule_1():
    """a -> b <- e (v-structure), b - c, c not adjacent to a: b -> c is compelled."""
    g = DAG(nodes=["a", "e", "b", "c"], edges=[("a", "b"), ("e", "b"), ("b", "c")])
    assert ("b", "c") in cpdag(g).directed


def test_meek_rule_2():
    """a -> b <- d is a v-structure; b -> c follows by R1 (d, c not adjacent). Then a - c can
    only be oriented by R2: a -> b -> c with a - c gives a -> c (c -> a would close a cycle)."""
    g = DAG(nodes=["a", "b", "c", "d"], edges=[("a", "b"), ("d", "b"), ("b", "c"), ("a", "c")])
    result = cpdag(g)
    assert result.directed == {("a", "b"), ("d", "b"), ("b", "c"), ("a", "c")}
    assert len(list(result.members())) == 1


def test_meek_rule_3():
    """c -> b <- d (v-structure), a adjacent to b, c, d (undirected): a -> b."""
    g = DAG(
        nodes=["a", "b", "c", "d"],
        edges=[("c", "b"), ("d", "b"), ("a", "b"), ("a", "c"), ("a", "d")],
    )
    result = cpdag(g)
    assert ("a", "b") in result.directed
    assert frozenset(("a", "c")) in result.undirected and frozenset(("a", "d")) in result.undirected


# ---------------------------------------------------------------------------
# Structural Hamming distance
# ---------------------------------------------------------------------------


def test_shd_by_hand():
    base = cpdag(late())
    assert structural_hamming_distance(base, base) == 0
    no_accident = cpdag(
        DAG(nodes=late().nodes(), edges=[e for e in late().edges() if e[0] != "Accident"])
    )
    # Accident - Traffic goes, and with it the v-structure: Rain - Traffic - Late become undirected.
    assert structural_hamming_distance(base, no_accident) == 3
    assert structural_hamming_distance(no_accident, base) == 3


def test_shd_is_a_metric_on_random_cpdags():
    rng = np.random.default_rng(0)
    graphs = list(all_dags(4))
    for _ in range(200):
        a, b, c = (cpdag(graphs[int(i)]) for i in rng.integers(0, len(graphs), 3))
        ab = structural_hamming_distance(a, b)
        assert ab == structural_hamming_distance(b, a)
        assert (ab == 0) == (a == b)
        assert structural_hamming_distance(a, c) <= ab + structural_hamming_distance(b, c)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("directed", "undirected", "match"),
    [
        ({("a", "z")}, set(), "unknown"),
        ({("a", "a")}, set(), "self-loop"),
        ({("a", "b"), ("b", "a")}, set(), "both directions"),
        ({("a", "b")}, {frozenset(("a", "b"))}, "both directed and undirected"),
    ],
)
def test_pdag_validation(directed, undirected, match):
    with pytest.raises(ValidationError, match=match):
        PDAG(("a", "b"), frozenset(directed), frozenset(undirected))


def test_shd_needs_the_same_nodes():
    with pytest.raises(ValidationError, match="same nodes"):
        structural_hamming_distance(cpdag(late()), PDAG(("a",), frozenset(), frozenset()))
