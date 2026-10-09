"""Shared validation of partial assignments (evidence)."""

from __future__ import annotations

from collections.abc import Mapping

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.variables import DiscreteVariable


def check_partial_assignment(
    variables: Mapping[str, DiscreteVariable], assignment: object, what: str = "Evidence"
) -> dict[str, str]:
    """Validate a partial assignment against ``variables`` and return it as a plain dict.

    Raises ``ValidationError`` for a non-mapping, ``UnknownNodeError`` for an
    unknown variable, and ``UnknownStateError`` for a state outside its domain.
    """
    if not isinstance(assignment, Mapping):
        raise ValidationError(f"{what} must be a mapping, got {assignment!r}.")
    unknown = [name for name in assignment if name not in variables]
    if unknown:
        raise UnknownNodeError(f"{what} names unknown variables {unknown}.")
    for name, state in assignment.items():
        variables[name].index_of(state)
    return dict(assignment)
