from importlib.metadata import version

import probgraph


def test_version_matches_installed_metadata():
    assert probgraph.__version__ == version("probgraph")


def test_public_api():
    assert set(probgraph.__all__) == {
        "AncestralSampler",
        "BayesianNetwork",
        "DAG",
        "DiscreteFactor",
        "DiscreteVariable",
        "TabularCPD",
    }
    for name in probgraph.__all__:
        assert getattr(probgraph, name).__module__.startswith("probgraph.")
