"""Exception hierarchy for ProbGraph.

Every error raised deliberately by the library derives from ``ProbGraphError``,
so callers can catch library failures without catching unrelated bugs.
"""


class ProbGraphError(Exception):
    """Base class for all ProbGraph errors."""


class ValidationError(ProbGraphError, ValueError):
    """An object would violate one of its mathematical invariants."""


class CycleError(ValidationError):
    """A graph operation would introduce a directed cycle (invariant G1)."""


class UnknownNodeError(ProbGraphError, LookupError):
    """A graph operation referenced a node that has not been added."""


class UnknownStateError(ProbGraphError, LookupError):
    """A state label does not belong to a variable's domain."""
