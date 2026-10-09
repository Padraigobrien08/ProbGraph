"""Shared validation of partial assignments (evidence)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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


def check_query(variables: Mapping[str, DiscreteVariable], query: object) -> list[str]:
    """Validate a non-empty sequence of distinct, known variable names."""
    if isinstance(query, str):
        raise ValidationError(
            f"Query must be a sequence of variable names, not the string {query!r}."
        )
    if not isinstance(query, Iterable):
        raise ValidationError(f"Query must be a sequence of variable names, got {query!r}.")
    names = list(query)
    if not names:
        raise ValidationError("Query must name at least one variable.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValidationError(f"Query has duplicate variables {duplicates}.")
    unknown = [n for n in names if n not in variables]
    if unknown:
        raise UnknownNodeError(f"Query names unknown variables {unknown}.")
    return names


def check_query_and_evidence(
    variables: Mapping[str, DiscreteVariable], query: object, evidence: object
) -> tuple[list[str], dict[str, str]]:
    """Validate a query and its evidence together: both valid, and disjoint."""
    names = check_query(variables, query)
    observed = check_partial_assignment(variables, evidence if evidence is not None else {})
    both = [n for n in names if n in observed]
    if both:
        raise ValidationError(f"Variables {both} are both queried and observed.")
    return names, observed
