"""M7.4: G² independence tests and exact chi-square tails (T1–T3; F3, F4; P30 Part 1)."""

import itertools
import math

import numpy as np
import pytest

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.learning import Dataset
from probgraph.structure import IndependenceTest, chi_square_survival, g_squared_test

A, B = DiscreteVariable("A", ("0", "1")), DiscreteVariable("B", ("0", "1"))


def table_dataset(table, variables=(A, B)) -> Dataset:
    """Rows reproducing an (|A|, |B|) count table."""
    rows = []
    for i, j in np.ndindex(np.shape(table)):
        rows += [
            {variables[0].name: variables[0].states[i], variables[1].name: variables[1].states[j]}
        ] * int(table[i][j])
    return Dataset(list(variables), rows)


# ---------------------------------------------------------------------------
# T1, F4: chi-square tails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "dof"),
    [
        (3.841458820694124, 1),
        (5.991464547107979, 2),
        (7.814727903251178, 3),
        (9.487729036781154, 4),
    ],
)
def test_f4_five_percent_critical_values(x, dof):
    assert chi_square_survival(x, dof) == pytest.approx(0.05, abs=1e-14)


@pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 3.0, 10.0, 40.0])
def test_closed_forms_for_small_dof(x):
    y = x / 2
    assert chi_square_survival(x, 1) == pytest.approx(math.erfc(math.sqrt(y)), rel=1e-13, abs=0)
    assert chi_square_survival(x, 2) == pytest.approx(math.exp(-y), rel=1e-13, abs=0)
    assert chi_square_survival(x, 3) == pytest.approx(
        math.erfc(math.sqrt(y)) + math.exp(-y) * math.sqrt(y) / math.gamma(1.5), rel=1e-13, abs=0
    )
    assert chi_square_survival(x, 4) == pytest.approx(math.exp(-y) * (1 + y), rel=1e-13, abs=0)


@pytest.mark.parametrize("dof", [1, 2, 5, 10, 31])
def test_the_recurrence_q_d_plus_2(dof):
    """Q_{d+2}(x) = Q_d(x) + e^{-y} y^{d/2} / Γ(d/2 + 1)."""
    for x in (0.3, 4.0, 25.0, 90.0):
        y = x / 2
        step = math.exp(-y + dof / 2 * math.log(y) - math.lgamma(dof / 2 + 1))
        assert chi_square_survival(x, dof + 2) == pytest.approx(
            chi_square_survival(x, dof) + step, rel=1e-12, abs=0
        )


@pytest.mark.parametrize("dof", [1, 3, 8, 20, 50])
def test_tails_match_simulation(dof):
    """Against numpy's chi-square sampler: 400k draws give a standard error below 0.0008."""
    draws = np.random.default_rng(dof).chisquare(dof, 400_000)
    for q in (0.5, 0.9, 0.99):
        x = float(np.quantile(draws, q))
        assert chi_square_survival(x, dof) == pytest.approx(1 - q, abs=0.004)


def test_far_tail_and_edges():
    assert chi_square_survival(1400.0, 1) == pytest.approx(
        math.erfc(math.sqrt(700)), rel=1e-10, abs=0
    )
    deep = chi_square_survival(2000.0, 1)  # erfc itself underflows; the asymptotic form does not
    assert 0.0 <= deep < 1e-300
    assert chi_square_survival(0.0, 3) == 1.0 and chi_square_survival(math.inf, 3) == 0.0
    assert chi_square_survival(200.0, 200) == pytest.approx(0.4867, abs=0.002)  # near the median
    xs = np.linspace(0.1, 60, 200)
    values = [chi_square_survival(float(x), 7) for x in xs]
    assert all(a > b for a, b in itertools.pairwise(values))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_dof_validation(bad):
    with pytest.raises(ValidationError, match="dof"):
        chi_square_survival(1.0, bad)


# ---------------------------------------------------------------------------
# T2, F3: the statistic
# ---------------------------------------------------------------------------


def test_f3_by_hand():
    result = g_squared_test(table_dataset([[30, 10], [10, 30]]), "A", "B")
    assert isinstance(result, IndependenceTest)
    assert result.statistic == pytest.approx(
        2 * (60 * math.log(1.5) + 20 * math.log(0.5)), abs=1e-12
    )
    assert result.statistic == pytest.approx(20.929925750581912, abs=1e-12)
    assert result.dof == 1
    assert result.p_value == pytest.approx(4.763938479565465e-06, rel=1e-10, abs=0)


def conditional_mutual_information(counts: np.ndarray) -> float:
    """Î(X; Y | Z) in nats from a (|X|, |Y|, strata) table, by the definition."""
    p = counts / counts.sum()
    total = 0.0
    for x, y, z in np.ndindex(p.shape):
        if p[x, y, z] > 0:
            pz, pxz, pyz = p[:, :, z].sum(), p[x, :, z].sum(), p[:, y, z].sum()
            total += p[x, y, z] * math.log(p[x, y, z] * pz / (pxz * pyz))
    return total


@pytest.mark.parametrize("seed", range(10))
def test_g_squared_is_twice_n_times_conditional_mutual_information(seed):
    rng = np.random.default_rng(seed)
    variables = [
        DiscreteVariable(n, tuple(str(i) for i in range(int(rng.integers(2, 4))))) for n in "XYZW"
    ]
    rows = [
        {v.name: v.states[int(rng.integers(v.cardinality))] for v in variables} for _ in range(300)
    ]
    data = Dataset(variables, rows)
    result = g_squared_test(data, "X", "Y", ["Z", "W"])
    counts = data.counts(["X", "Y", "Z", "W"]).values
    table = counts.reshape(counts.shape[0], counts.shape[1], -1)
    assert result.statistic == pytest.approx(
        2 * 300 * conditional_mutual_information(table), rel=1e-11
    )


def test_independent_table_gives_zero_and_p_one():
    result = g_squared_test(table_dataset([[20, 20], [10, 10]]), "A", "B")
    assert result.statistic == pytest.approx(0.0, abs=1e-12) and result.p_value == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------------------
# Degrees of freedom adjusted for sparse tables
# ---------------------------------------------------------------------------


def test_dof_adjustment():
    x = DiscreteVariable("X", ("0", "1", "2"))
    y = DiscreteVariable("Y", ("0", "1"))
    z = DiscreteVariable("Z", ("a", "b", "c"))
    rows = (
        [{"X": "0", "Y": "0", "Z": "a"}] * 5
        + [{"X": "1", "Y": "1", "Z": "a"}] * 5
        + [{"X": "2", "Y": "0", "Z": "a"}]
        * 5  # stratum a: 3 x-values, 2 y-values -> (3-1)(2-1) = 2
        + [{"X": "0", "Y": "0", "Z": "b"}] * 5
        + [{"X": "0", "Y": "1", "Z": "b"}] * 5  # stratum b: 1 x-value -> 0
    )  # stratum c: empty -> 0
    result = g_squared_test(Dataset([x, y, z], rows), "X", "Y", ["Z"])
    assert result.dof == 2  # unadjusted would be (3-1)(2-1) x 3 strata = 6


def test_no_degrees_of_freedom_gives_p_one():
    rows = [{"A": "0", "B": "0"}] * 10 + [{"A": "0", "B": "1"}] * 10  # A constant
    result = g_squared_test(Dataset([A, B], rows), "A", "B")
    assert result.dof == 0 and result.p_value == 1.0


# ---------------------------------------------------------------------------
# T3: calibration under H0, and power
# ---------------------------------------------------------------------------


def fork(dependent: bool) -> BayesianNetwork:
    """Z -> X, Z -> Y (so X ⊥ Y | Z); with dependent=True also X -> Y."""
    z = DiscreteVariable("Z", ("0", "1", "2"))
    x = DiscreteVariable("X", ("0", "1"))
    y = DiscreteVariable("Y", ("0", "1"))
    edges = [("Z", "X"), ("Z", "Y")] + ([("X", "Y")] if dependent else [])
    model = BayesianNetwork([z, x, y], edges)
    model.add_cpd(TabularCPD(z, (), [0.3, 0.3, 0.4]))
    model.add_cpd(TabularCPD(x, (z,), [[0.8, 0.5, 0.2], [0.2, 0.5, 0.8]]))
    if dependent:
        model.add_cpd(
            TabularCPD(
                y,
                (z, x),
                [[[0.9, 0.3], [0.6, 0.2], [0.5, 0.1]], [[0.1, 0.7], [0.4, 0.8], [0.5, 0.9]]],
            )
        )
    else:
        model.add_cpd(TabularCPD(y, (z,), [[0.7, 0.4, 0.3], [0.3, 0.6, 0.7]]))
    return model


def test_p_values_are_calibrated_under_conditional_independence():
    model = fork(dependent=False)
    p_values = []
    for seed in range(600):
        data = Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(300))
        p_values.append(g_squared_test(data, "X", "Y", ["Z"]).p_value)
    p_values = np.array(p_values)
    assert 0.025 <= (p_values < 0.05).mean() <= 0.08  # about 5% (sd ≈ 0.9%)
    assert 0.07 <= (p_values < 0.10).mean() <= 0.14
    assert 0.42 <= (p_values < 0.5).mean() <= 0.58
    # Marginally, X and Y are dependent through Z: the unconditional test rejects.
    data = Dataset.from_samples(model, AncestralSampler(model, seed=1).sample(2000))
    assert g_squared_test(data, "X", "Y").p_value < 1e-6


def test_power_under_dependence():
    model = fork(dependent=True)
    data = Dataset.from_samples(model, AncestralSampler(model, seed=2).sample(1000))
    assert g_squared_test(data, "X", "Y", ["Z"]).p_value < 1e-10


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation():
    data = table_dataset([[1, 2], [3, 4]])
    with pytest.raises(UnknownNodeError):
        g_squared_test(data, "A", "C")
    with pytest.raises(ValidationError, match="distinct"):
        g_squared_test(data, "A", "A")
    with pytest.raises(ValidationError, match="distinct"):
        g_squared_test(data, "A", "B", ["A"])
    with pytest.raises(ValidationError, match="Dataset"):
        g_squared_test("data", "A", "B")


@pytest.mark.parametrize("z", [25.2, 25.8, 26.4])
def test_the_asymptotic_erfc_is_accurate_where_it_takes_over(z):
    """For 25 < z < 26.5 the library uses the asymptotic series, yet math.erfc is still
    representable (about 1e-290), so the two can be compared directly."""
    exact = math.erfc(z)
    assert exact > 0
    assert chi_square_survival(2 * z * z, 1) == pytest.approx(exact, rel=1e-9, abs=0)
