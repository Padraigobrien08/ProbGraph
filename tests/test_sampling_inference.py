"""Task 10: rejection sampling and likelihood weighting (invariants A1–A3).

Exact variable elimination is the oracle. Tolerances come from
docs/mathematics/sampling_inference.md. Each check fails for a correct sampler
with probability at most about 1.5e-5, and every seed is fixed.
"""

import itertools

import numpy as np
import pytest
from support import late_network, random_evidence, random_network

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import (
    InsufficientSamplesError,
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
)
from probgraph.sampling import ApproximatePosterior, LikelihoodWeighting, RejectionSampler

Z = 5.0
Y = "yes"


def tol(p: float, n: int) -> float:
    """M1's Bernstein tolerance for a [0,1]-bounded mean with mean p from n draws."""
    return Z * np.sqrt(p * (1 - p) / n) + Z**2 / (3 * n)


def assignments(names, model):
    variables = [model.variable(n) for n in names]
    for states in itertools.product(*(v.states for v in variables)):
        yield dict(zip(names, states, strict=True))


def check_rejection(model, query, evidence, result: ApproximatePosterior, n: int):
    ve = VariableElimination(model)
    p_e = ve.probability_of_evidence(evidence)
    accepted = result.effective_sample_size
    assert accepted == round(accepted)  # an integer count for rejection sampling
    assert abs(result.evidence_estimate - p_e) <= tol(p_e, n)  # A1 (1)
    exact = ve.query(query, evidence)
    for cell in assignments(query, model):  # A1 (2): a binomial proportion out of `accepted`
        p = exact.value(cell)
        assert abs(result.estimate.value(cell) - p) <= tol(p, int(accepted)), cell


def check_likelihood_weighting(model, query, evidence, result: ApproximatePosterior, n: int):
    ve = VariableElimination(model)
    p_e = ve.probability_of_evidence(evidence)
    t_d = tol(p_e, n)
    assert abs(result.evidence_estimate - p_e) <= t_d  # A2
    for cell in assignments(query, model):  # A3: the ratio interval
        joint = ve.probability_of_evidence({**evidence, **cell})  # P(q, e)
        t_n = tol(joint, n)
        assert p_e - t_d > 0
        lo, hi = (joint - t_n) / (p_e + t_d), (joint + t_n) / (p_e - t_d)
        assert lo <= result.estimate.value(cell) <= hi, (cell, lo, hi)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

FIXTURE_QUERIES = [
    (["Accident"], {"Late": Y}),
    (["Accident"], {"Late": Y, "Umbrella": Y}),
    (["Rain", "Accident"], {"Late": Y}),
    (["Late"], {}),
]


@pytest.mark.parametrize(("query", "evidence"), FIXTURE_QUERIES)
def test_rejection_sampling_on_fixture(query, evidence):
    model = late_network()
    n = 200_000
    result = RejectionSampler(model).query(query, evidence, n, seed=1)
    assert result.n == n
    assert result.estimate.names == tuple(query)
    assert result.estimate.total() == pytest.approx(1.0)
    check_rejection(model, query, evidence, result, n)


@pytest.mark.parametrize(("query", "evidence"), FIXTURE_QUERIES)
def test_likelihood_weighting_on_fixture(query, evidence):
    model = late_network()
    n = 200_000
    result = LikelihoodWeighting(model).query(query, evidence, n, seed=2)
    assert result.estimate.names == tuple(query)
    assert result.estimate.total() == pytest.approx(1.0)
    check_likelihood_weighting(model, query, evidence, result, n)


def test_explaining_away_is_visible_in_both_estimates():
    model = late_network()
    for sampler in (RejectionSampler(model), LikelihoodWeighting(model)):
        late = sampler.query(["Accident"], {"Late": Y}, 200_000, seed=3)
        both = sampler.query(["Accident"], {"Late": Y, "Umbrella": Y}, 200_000, seed=3)
        # 65/371 ≈ 0.175 drops to 475/3787 ≈ 0.125 once the umbrella is seen.
        assert late.estimate.value({"Accident": Y}) > both.estimate.value({"Accident": Y}) + 0.03


# ---------------------------------------------------------------------------
# Random networks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_both_methods_on_random_networks(seed):
    model = random_network(seed, n_vars=(2, 6), cards=(2, 3), edge_prob=0.4)
    rng = np.random.default_rng(seed + 130_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = random_evidence(model, rng, exclude=query)
    n = 60_000
    check_rejection(
        model, query, evidence, RejectionSampler(model).query(query, evidence, n, seed), n
    )
    check_likelihood_weighting(
        model, query, evidence, LikelihoodWeighting(model).query(query, evidence, n, seed), n
    )


# ---------------------------------------------------------------------------
# A3: the likelihood-weighting posterior is biased for finite n
# ---------------------------------------------------------------------------


def test_single_sample_likelihood_weighting_is_biased():
    """With n = 1 the weight cancels, so E[R̂] = q(A=1) = P(A=1) = 0.1, not 65/371 ≈ 0.175."""
    sampler = LikelihoodWeighting(late_network())
    runs = 10_000  # tolerance ≈ 0.016, far below the 0.075 bias
    mean = np.mean(
        [
            sampler.query(["Accident"], {"Late": Y}, 1, seed=s).estimate.value({"Accident": Y})
            for s in range(runs)
        ]
    )
    assert abs(mean - 0.1) <= tol(0.1, runs)
    assert abs(mean - 65 / 371) > 0.05


# ---------------------------------------------------------------------------
# §4: effective sample size
# ---------------------------------------------------------------------------


def test_evidence_on_roots_gives_full_ess_and_exact_draws():
    model = late_network()
    evidence = {"Rain": Y, "Accident": "no"}  # both are roots
    n = 10_000
    result = LikelihoodWeighting(model).query(["Late"], evidence, n, seed=4)
    assert result.effective_sample_size == pytest.approx(n)
    assert result.evidence_estimate == pytest.approx(0.3 * 0.9)  # every weight is P(e) exactly


def test_downstream_evidence_lowers_ess():
    model = late_network()
    n = 20_000
    result = LikelihoodWeighting(model).query(["Accident"], {"Late": Y, "Umbrella": Y}, n, seed=5)
    assert 1 <= result.effective_sample_size < 0.9 * n


def test_rejection_ess_is_the_accepted_count():
    model = late_network()
    n = 50_000
    result = RejectionSampler(model).query(["Accident"], {"Late": Y}, n, seed=6)
    assert result.effective_sample_size == pytest.approx(result.evidence_estimate * n)


# ---------------------------------------------------------------------------
# Failure modes and validation
# ---------------------------------------------------------------------------


def impossible_evidence_model() -> BayesianNetwork:
    x = DiscreteVariable("X", ("a", "b"))
    s = DiscreteVariable("Switch", ("off", "on"))
    model = BayesianNetwork([x, s], [("X", "Switch")])
    model.add_cpd(TabularCPD(x, (), [0.5, 0.5]))
    model.add_cpd(TabularCPD(s, (x,), [[1.0, 1.0], [0.0, 0.0]]))  # "on" is impossible
    return model


def test_no_accepted_samples_raises():
    with pytest.raises(InsufficientSamplesError, match="accepted"):
        RejectionSampler(impossible_evidence_model()).query(["X"], {"Switch": "on"}, 1_000, seed=0)


def test_zero_total_weight_raises():
    with pytest.raises(InsufficientSamplesError, match="weight"):
        LikelihoodWeighting(impossible_evidence_model()).query(
            ["X"], {"Switch": "on"}, 1_000, seed=0
        )


@pytest.mark.parametrize("cls", [RejectionSampler, LikelihoodWeighting])
@pytest.mark.parametrize(
    ("query", "evidence", "n", "error"),
    [
        (["Snow"], {}, 10, UnknownNodeError),
        ([], {}, 10, ValidationError),
        ("Rain", {}, 10, ValidationError),
        (["Rain"], {"Rain": Y}, 10, ValidationError),
        (["Rain"], {"Late": "very"}, 10, UnknownStateError),
        (["Rain"], {}, 0, ValidationError),
        (["Rain"], {}, -5, ValidationError),
        (["Rain"], {}, 2.5, ValidationError),
    ],
)
def test_validation(cls, query, evidence, n, error):
    with pytest.raises(error):
        cls(late_network()).query(query, evidence, n, seed=0)


@pytest.mark.parametrize("cls", [RejectionSampler, LikelihoodWeighting])
def test_reproducible_with_a_seed(cls):
    sampler = cls(late_network())
    a = sampler.query(["Accident"], {"Late": Y}, 5_000, seed=9)
    b = sampler.query(["Accident"], {"Late": Y}, 5_000, seed=9)
    c = sampler.query(["Accident"], {"Late": Y}, 5_000, seed=10)
    assert a.estimate.allclose(b.estimate, atol=0.0)
    assert (a.evidence_estimate, a.effective_sample_size) == (
        b.evidence_estimate,
        b.effective_sample_size,
    )
    assert not a.estimate.allclose(c.estimate, atol=0.0)


@pytest.mark.parametrize("cls", [RejectionSampler, LikelihoodWeighting])
def test_samplers_snapshot_the_model(cls):
    model = late_network()
    sampler = cls(model)
    model.add_cpd(TabularCPD(model.variable("Accident"), (), [0.0, 1.0]))
    result = sampler.query(["Accident"], {}, 20_000, seed=0)
    assert result.estimate.value({"Accident": Y}) == pytest.approx(0.1, abs=tol(0.1, 20_000))
