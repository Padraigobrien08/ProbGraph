"""M4.1: Dataset (spec §3.1; D1–D4; likelihood.md Part 1)."""

import itertools
from collections import Counter

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, DiscreteVariable
from probgraph.exceptions import UnknownNodeError, UnknownStateError, ValidationError
from probgraph.learning import Dataset

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))
Y, N = "yes", "no"


def f1_rows():
    """Fixture F1 (milestone-4.md §4): 10 complete rows over Rain → Traffic."""
    return (
        [{"Rain": Y, "Traffic": Y}] * 3
        + [{"Rain": Y, "Traffic": N}] * 1
        + [{"Rain": N, "Traffic": Y}] * 1
        + [{"Rain": N, "Traffic": N}] * 5
    )


def f1() -> Dataset:
    return Dataset([RAIN, TRAFFIC], f1_rows())


# ---------------------------------------------------------------------------
# Construction (D1, D2)
# ---------------------------------------------------------------------------


def test_construction():
    data = f1()
    assert data.n_rows == len(data) == 10
    assert data.names == ("Rain", "Traffic")
    assert data.is_complete
    assert data.missing_count == 0


def test_missing_values_are_none():
    data = Dataset([RAIN, TRAFFIC], [{"Rain": Y, "Traffic": None}, {"Rain": None, "Traffic": N}])
    assert not data.is_complete
    assert data.missing_count == 2
    assert data.codes.tolist() == [[1, -1], [-1, 0]]


@pytest.mark.parametrize(
    ("row", "error", "match"),
    [
        ({"Rain": "maybe", "Traffic": Y}, UnknownStateError, "maybe"),
        ({"Rain": "", "Traffic": Y}, UnknownStateError, "''"),  # empty string is not "missing"
        ({"Rain": float("nan"), "Traffic": Y}, UnknownStateError, "nan"),
        ({"Rain": Y}, ValidationError, r"missing.*Traffic"),  # a forgotten variable
        ({"Rain": Y, "Traffic": Y, "Snow": N}, UnknownNodeError, "Snow"),
    ],
)
def test_invalid_rows_are_rejected(row, error, match):
    with pytest.raises(error, match=match):
        Dataset([RAIN, TRAFFIC], [row])


def test_rows_must_be_mappings_and_variables_valid():
    with pytest.raises(ValidationError):
        Dataset([RAIN, TRAFFIC], [[Y, N]])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="duplicate"):
        Dataset([RAIN, RAIN], [])
    with pytest.raises(ValidationError):
        Dataset(["Rain"], [])  # type: ignore[list-item]


def test_error_names_the_row():
    rows = [*f1_rows(), {"Rain": "drizzle", "Traffic": Y}]
    with pytest.raises(UnknownStateError, match="Row 10"):
        Dataset([RAIN, TRAFFIC], rows)


def test_empty_dataset():
    data = Dataset([RAIN, TRAFFIC], [])
    assert data.n_rows == 0 and data.is_complete
    assert data.counts(["Rain"]).values.tolist() == [0.0, 0.0]
    assert data.counts([]).value({}) == 0.0


# ---------------------------------------------------------------------------
# D3: immutability
# ---------------------------------------------------------------------------


def test_dataset_is_immutable():
    rows = f1_rows()
    data = Dataset([RAIN, TRAFFIC], rows)
    rows.append({"Rain": Y, "Traffic": Y})
    assert data.n_rows == 10
    with pytest.raises(ValueError):
        data.codes[0, 0] = 0
    with pytest.raises(AttributeError):
        data.variables = ()  # type: ignore[misc]


def test_rows_round_trip():
    rows = [{"Rain": Y, "Traffic": None}, {"Rain": N, "Traffic": N}]
    assert list(Dataset([RAIN, TRAFFIC], rows).rows()) == rows


# ---------------------------------------------------------------------------
# D4: counts
# ---------------------------------------------------------------------------


def test_f1_counts():
    data = f1()
    joint = data.counts(["Rain", "Traffic"])
    assert joint.value({"Rain": Y, "Traffic": Y}) == 3
    assert joint.value({"Rain": Y, "Traffic": N}) == 1
    assert joint.value({"Rain": N, "Traffic": Y}) == 1
    assert joint.value({"Rain": N, "Traffic": N}) == 5
    assert data.counts(["Rain"]).values.tolist() == [6.0, 4.0]
    assert data.counts(["Traffic", "Rain"]).names == ("Traffic", "Rain")
    assert data.counts([]).value({}) == 10


def test_counts_use_only_fully_observed_rows():
    rows = [
        {"Rain": Y, "Traffic": Y},
        {"Rain": Y, "Traffic": None},
        {"Rain": None, "Traffic": N},
        {"Rain": N, "Traffic": N},
    ]
    data = Dataset([RAIN, TRAFFIC], rows)
    assert data.counts(["Rain"]).values.tolist() == [1.0, 2.0]
    assert data.counts(["Traffic"]).values.tolist() == [2.0, 1.0]
    assert data.counts(["Rain", "Traffic"]).total() == 2


@pytest.mark.parametrize("bad", [["Snow"], ["Rain", "Rain"], "Rain"])
def test_counts_validate_names(bad):
    with pytest.raises((UnknownNodeError, ValidationError)):
        f1().counts(bad)


@pytest.mark.parametrize("seed", range(30))
def test_counts_match_brute_force_on_random_data(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(2, 6), cards=(1, 3))
    samples = AncestralSampler(model, seed=seed).sample(int(rng.integers(0, 300)))
    data = Dataset.from_samples(model, samples).with_missing(float(rng.uniform(0, 0.4)), seed=seed)
    names = [v.name for v in model.variables]
    for k in range(len(names) + 1):
        for subset in itertools.combinations(names, k):
            subset = list(subset)
            counts = data.counts(subset)
            tally = Counter(
                tuple(row[n] for n in subset)
                for row in data.rows()
                if all(row[n] is not None for n in subset)
            )
            assert counts.total() == sum(tally.values())
            if subset:
                for states, c in tally.items():
                    assert counts.value(dict(zip(subset, states, strict=True))) == c


# ---------------------------------------------------------------------------
# from_samples and MCAR masking
# ---------------------------------------------------------------------------


def test_from_samples():
    model = late_network()
    samples = AncestralSampler(model, seed=1).sample(50)
    data = Dataset.from_samples(model, samples)
    assert data.names == tuple(v.name for v in model.variables)
    assert data.n_rows == 50 and data.is_complete
    assert list(data.rows()) == samples


def test_with_missing_extremes_and_reproducibility():
    data = Dataset.from_samples(
        late_network(), AncestralSampler(late_network(), seed=2).sample(200)
    )
    assert data.with_missing(0.0, seed=1).codes.tolist() == data.codes.tolist()
    assert data.with_missing(1.0, seed=1).missing_count == data.n_rows * len(data.names)
    a, b = data.with_missing(0.3, seed=5), data.with_missing(0.3, seed=5)
    assert a.codes.tolist() == b.codes.tolist()
    assert a.codes.tolist() != data.with_missing(0.3, seed=6).codes.tolist()


def test_with_missing_only_removes_information():
    data = Dataset.from_samples(
        late_network(), AncestralSampler(late_network(), seed=3).sample(300)
    )
    once = data.with_missing(0.2, seed=1)
    twice = once.with_missing(0.2, seed=2)
    observed_before = once.codes >= 0
    assert ((twice.codes == once.codes) | (twice.codes == -1)).all()
    assert (twice.codes[~observed_before] == -1).all()  # missing stays missing
    assert (data.codes[observed_before] == once.codes[observed_before]).all()  # values never change


def test_with_missing_rate_is_within_the_bernstein_bound():
    data = Dataset.from_samples(
        late_network(), AncestralSampler(late_network(), seed=4).sample(4000)
    )
    masked = data.with_missing(0.25, seed=9)
    k = data.n_rows * len(data.names)
    p_hat = masked.missing_count / k
    tolerance = 5 * np.sqrt(0.25 * 0.75 / k) + 25 / (3 * k)  # M1's Bernstein bound
    assert abs(p_hat - 0.25) <= tolerance


def test_with_missing_is_independent_of_values_sampled_with_the_same_seed():
    """Regression: with one shared stream, a cell was hidden exactly when the uniform that
    produced its value was small, so P(hidden) depended on the value (not MCAR)."""
    model = late_network()
    data = Dataset.from_samples(model, AncestralSampler(model, seed=0).sample(20_000))
    masked = data.with_missing(0.3, seed=0)
    for column in range(len(data.names)):
        for value in (0, 1):
            rows = data.codes[:, column] == value
            n = int(rows.sum())
            hidden = float((masked.codes[rows, column] == -1).mean())
            assert abs(hidden - 0.3) <= 5 * np.sqrt(0.3 * 0.7 / n) + 25 / (3 * n)


def test_with_missing_on_selected_variables():
    data = Dataset.from_samples(
        late_network(), AncestralSampler(late_network(), seed=5).sample(200)
    )
    masked = data.with_missing(1.0, seed=0, variables=["Traffic"])
    column = masked.names.index("Traffic")
    assert (masked.codes[:, column] == -1).all()
    others = [i for i in range(len(masked.names)) if i != column]
    assert (masked.codes[:, others] >= 0).all()


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_with_missing_validates_fraction(fraction):
    with pytest.raises(ValidationError):
        f1().with_missing(fraction, seed=0)
