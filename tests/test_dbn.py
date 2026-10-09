"""M5.8: dynamic Bayesian networks (N1, N2; dbn.md)."""

import itertools

import numpy as np
import pytest
from support import joint_table, posterior, random_evidence
from test_forward_backward import random_hmm

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import CycleError, ValidationError
from probgraph.inference import JunctionTree
from probgraph.temporal import DynamicBayesianNetwork, previous


def random_cpd(rng, child: DiscreteVariable, parents) -> TabularCPD:
    shape = (child.cardinality, *(p.cardinality for p in parents))
    columns = rng.dirichlet(np.ones(child.cardinality), size=int(np.prod(shape[1:], dtype=int)))
    return TabularCPD(child, tuple(parents), columns.T.reshape(shape))


def random_dbn(seed: int) -> DynamicBayesianNetwork:
    """2–3 template variables; random intra-slice edges (in a fixed order) and inter-slice edges."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 4))
    template = [
        DiscreteVariable(f"V{i}", ("a", "b", "c")[: int(rng.integers(2, 4))]) for i in range(n)
    ]
    edges = [(f"V{i}", f"V{j}") for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    initial = BayesianNetwork(template, edges)
    for v in template:
        parents = [p for p in template if p.name in initial.parents(v.name)]
        initial.add_cpd(random_cpd(rng, v, parents))
    transition = []
    for j, v in enumerate(template):
        parents = [template[i] for i in range(j) if rng.random() < 0.4]
        parents += [previous(u) for u in template if rng.random() < 0.5]
        transition.append(random_cpd(rng, v, parents))
    return DynamicBayesianNetwork(initial, transition)


# ---------------------------------------------------------------------------
# N1: an HMM written as a 2-TBN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_unrolling_an_hmm_gives_the_hmm_network(seed):
    hmm = random_hmm(seed, k=3, m=2)
    dbn = DynamicBayesianNetwork.from_hmm(hmm)
    assert dbn.interface == ("X",)
    for length in (1, 2, 5):
        ours, theirs = dbn.unroll(length), hmm.to_bayesian_network(length)
        assert {v.name for v in ours.variables} == {v.name for v in theirs.variables}
        assert set(ours.edges()) == set(theirs.edges())
        for name, cpd in theirs.cpds.items():
            assert ours.cpds[name].parent_names == cpd.parent_names
            assert ours.cpds[name].values == pytest.approx(cpd.values, abs=0)


# ---------------------------------------------------------------------------
# Proposition 1: the unrolled joint is the product formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_unrolled_joint_is_the_slice_product(seed):
    dbn = random_dbn(seed)
    length = 3
    network = dbn.unroll(length)
    network.validate()
    table = joint_table(network)
    assert table.sum() == pytest.approx(1.0, abs=1e-12)
    position = {v.name: i for i, v in enumerate(network.variables)}
    template = dbn.variables
    for index in itertools.product(*(range(v.cardinality) for v in network.variables)):
        value = {name: index[i] for name, i in position.items()}

        def state(name: str, t: int, value: dict[str, int] = value) -> str:
            v = next(u for u in template if u.name == name)
            return v.states[value[f"{name}_{t}"]]

        p = 1.0
        for v in template:  # slice 1: the initial network
            cpd = dbn.initial.cpds[v.name]
            p *= cpd.probability(state(v.name, 1), {u: state(u, 1) for u in cpd.parent_names})
        for t in range(2, length + 1):
            for cpd in dbn.transition:
                given = {}
                for u in cpd.parent_names:
                    if u.endswith("[t-1]"):
                        given[u] = state(u[: -len("[t-1]")], t - 1)
                    else:
                        given[u] = state(u, t)
                p *= cpd.probability(state(cpd.variable.name, t), given)
        assert table[index] == pytest.approx(p, abs=1e-15)


# ---------------------------------------------------------------------------
# Proposition 2: the interface d-separates past and future
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_the_interface_separates_past_and_future(seed):
    dbn = random_dbn(seed + 100)
    length = 4
    graph = dbn.unroll(length).graph
    names = [v.name for v in dbn.variables]
    for t in range(1, length):
        interface = {f"{name}_{t}" for name in dbn.interface}
        past = {f"{n}_{s}" for n in names for s in range(1, t + 1)} - interface
        future = {f"{n}_{s}" for n in names for s in range(t + 1, length + 1)}
        if past and future:
            assert graph.d_separated(past, future, interface)


def test_interface_is_the_set_of_variables_with_children_in_the_next_slice():
    a, b, c = (DiscreteVariable(n, ("0", "1")) for n in "ABC")
    initial = BayesianNetwork([a, b, c], [("A", "B")])
    initial.add_cpd(TabularCPD(a, (), [0.5, 0.5]))
    initial.add_cpd(TabularCPD(b, (a,), [[0.9, 0.2], [0.1, 0.8]]))
    initial.add_cpd(TabularCPD(c, (), [0.3, 0.7]))
    transition = [
        TabularCPD(a, (previous(c),), [[0.6, 0.1], [0.4, 0.9]]),
        TabularCPD(b, (a,), [[0.9, 0.2], [0.1, 0.8]]),
        TabularCPD(c, (previous(c),), [[0.7, 0.4], [0.3, 0.6]]),
    ]
    dbn = DynamicBayesianNetwork(initial, transition)
    assert dbn.interface == ("C",)  # in template order
    assert set(dbn.unroll(3).edges()) >= {("C_1", "A_2"), ("C_2", "C_3"), ("A_3", "B_3")}


# ---------------------------------------------------------------------------
# N2: exact inference on the unrolled network, and its cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_junction_tree_on_the_unrolled_network_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    network = random_dbn(seed + 200).unroll(3)
    evidence = random_evidence(network, rng)
    tree = JunctionTree(network, evidence)
    for v in network.variables:
        if v.name not in evidence:
            assert tree.marginal(v.name).values == pytest.approx(
                posterior(network, [v.name], evidence).values, abs=1e-12
            )


def factorial_hmm(n_chains: int, seed: int = 0) -> DynamicBayesianNetwork:
    """n independent binary chains, and one observation Y with every chain as a parent."""
    rng = np.random.default_rng(seed)
    chains = [DiscreteVariable(f"C{k}", ("0", "1")) for k in range(n_chains)]
    y = DiscreteVariable("Y", ("0", "1", "2"))
    initial = BayesianNetwork([*chains, y], [(c.name, "Y") for c in chains])
    for c in chains:
        initial.add_cpd(TabularCPD(c, (), [0.5, 0.5]))
    emission = random_cpd(rng, y, chains)
    initial.add_cpd(emission)
    transition = [random_cpd(rng, c, [previous(c)]) for c in chains] + [emission]
    return DynamicBayesianNetwork(initial, transition)


def largest_clique(network: BayesianNetwork, evidence) -> int:
    return max(len(c) for c in JunctionTree(network, evidence).tree.cliques)


def observe_y(length: int) -> dict[str, str]:
    return {f"Y_{t}": ("0", "2", "1")[t % 3] for t in range(1, length + 1)}


def test_clique_size_does_not_grow_with_length():
    dbn = factorial_hmm(3)
    sizes = [largest_clique(dbn.unroll(t), observe_y(t)) for t in (6, 10, 16)]
    assert sizes[0] == sizes[1] == sizes[2]


def test_clique_size_grows_with_the_number_of_chains():
    sizes = [largest_clique(factorial_hmm(n).unroll(8), observe_y(8)) for n in (1, 2, 3, 4)]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0] + 2
    assert all(size >= n + 1 for size, n in zip(sizes, (1, 2, 3, 4), strict=True))


def test_entanglement_observations_couple_independent_chains():
    """A priori the chains are independent; observing their common child Y couples them."""
    network = factorial_hmm(2, seed=3).unroll(5)
    ve = VariableElimination(network)
    prior = ve.query(["C0_5", "C1_5"]).values
    assert prior == pytest.approx(np.outer(prior.sum(axis=1), prior.sum(axis=0)), abs=1e-14)
    belief = ve.query(["C0_5", "C1_5"], observe_y(5)).values
    product = np.outer(belief.sum(axis=1), belief.sum(axis=0))
    assert np.abs(belief - product).max() > 1e-2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def two_variables():
    a, b = DiscreteVariable("A", ("0", "1")), DiscreteVariable("B", ("0", "1"))
    initial = BayesianNetwork([a, b], [])
    initial.add_cpd(TabularCPD(a, (), [0.5, 0.5]))
    initial.add_cpd(TabularCPD(b, (), [0.5, 0.5]))
    return a, b, initial


def test_validation():
    a, b, initial = two_variables()
    ok_a = TabularCPD(a, (previous(a),), [[0.9, 0.1], [0.1, 0.9]])
    ok_b = TabularCPD(b, (a,), [[0.9, 0.1], [0.1, 0.9]])
    DynamicBayesianNetwork(initial, [ok_a, ok_b])
    with pytest.raises(ValidationError, match="no transition CPD for 'B'"):
        DynamicBayesianNetwork(initial, [ok_a])
    with pytest.raises(ValidationError, match="two transition CPDs for 'A'"):
        DynamicBayesianNetwork(initial, [ok_a, ok_a, ok_b])
    other = DiscreteVariable("Z", ("0", "1"))
    with pytest.raises(ValidationError, match="'Z'"):
        DynamicBayesianNetwork(initial, [ok_a, TabularCPD(b, (other,), [[0.5, 0.5], [0.5, 0.5]])])
    with pytest.raises(ValidationError, match="previous-slice"):
        DynamicBayesianNetwork(initial, [ok_a, ok_b, TabularCPD(previous(a), (), [0.5, 0.5])])
    cyclic_a = TabularCPD(a, (b,), [[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(CycleError):
        DynamicBayesianNetwork(initial, [cyclic_a, ok_b])
    with pytest.raises(ValidationError, match="length"):
        DynamicBayesianNetwork(initial, [ok_a, ok_b]).unroll(0)


def test_initial_network_must_be_complete():
    a, b, _ = two_variables()
    incomplete = BayesianNetwork([a, b], [])
    incomplete.add_cpd(TabularCPD(a, (), [0.5, 0.5]))
    with pytest.raises(ValidationError):
        DynamicBayesianNetwork(
            incomplete, [TabularCPD(a, (), [0.5, 0.5]), TabularCPD(b, (), [0.5, 0.5])]
        )


def test_previous_names_the_copy():
    a = DiscreteVariable("A", ("0", "1"))
    assert previous(a).name == "A[t-1]" and previous(a).states == a.states
