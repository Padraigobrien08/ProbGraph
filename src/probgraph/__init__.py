"""ProbGraph: discrete directed probabilistic graphical models from first principles."""

from probgraph.distributions import TabularCPD
from probgraph.graphs import DAG
from probgraph.variables import DiscreteVariable

__all__ = ["DAG", "DiscreteVariable", "TabularCPD"]
__version__ = "0.1.0.dev0"
