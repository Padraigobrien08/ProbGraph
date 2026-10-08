# ProbGraph

Discrete Bayesian networks built from first principles: DAGs, conditional probability
tables, joint factorisation and ancestral sampling. No graph or Bayesian-network
libraries are used.

**Status:** Milestone 1 units M1.1–M1.4 are done: variables, DAGs, CPDs, Bayesian networks with
joint probability, and ancestral sampling. Packaging, examples and CI (M1 task 9) are still to do.

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest
```

```python
from probgraph import DAG, DiscreteVariable

rain = DiscreteVariable("Rain", ("no", "yes"))
dag = DAG(nodes=["Rain", "Accident", "Traffic"],
          edges=[("Rain", "Traffic"), ("Accident", "Traffic")])
dag.topological_sort()   # ['Rain', 'Accident', 'Traffic']
```

The mathematical background is in [`docs/mathematics/`](docs/mathematics/).
