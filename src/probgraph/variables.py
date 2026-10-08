"""Finite discrete random variables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field
from types import MappingProxyType

from probgraph.exceptions import UnknownStateError, ValidationError


@dataclass(frozen=True)
class DiscreteVariable:
    """A random variable with a finite, ordered domain of named states.

    The order of ``states`` matters: it fixes which index each state gets on
    the variable's axis in every CPD table. For that reason the domain is
    stored as an immutable tuple, and unordered inputs such as sets are
    rejected.

    Invariants:
        - ``name`` is a non-empty string.
        - ``states`` is a non-empty tuple of unique, non-empty strings.
          A domain with a single state is allowed.
    """

    name: str
    states: tuple[str, ...]
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError(f"Variable name must be a non-empty string, got {self.name!r}.")

        states = _as_ordered_tuple(self.name, self.states)
        if not states:
            raise ValidationError(f"Variable {self.name!r} must have at least one state.")
        for state in states:
            if not isinstance(state, str) or not state:
                raise ValidationError(
                    f"Variable {self.name!r}: state labels must be non-empty strings, got {state!r}."
                )
        if len(set(states)) != len(states):
            duplicates = sorted({s for s in states if states.count(s) > 1})
            raise ValidationError(f"Variable {self.name!r} has duplicate states: {duplicates}.")

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "_index", MappingProxyType({s: i for i, s in enumerate(states)}))

    @property
    def cardinality(self) -> int:
        """The number of states, |X|."""
        return len(self.states)

    def index_of(self, state: str) -> int:
        """Return the position of ``state`` along this variable's axis."""
        try:
            return self._index[state]
        except (KeyError, TypeError):
            raise UnknownStateError(
                f"{state!r} is not a state of {self.name!r}; expected one of {self.states}."
            ) from None


def _as_ordered_tuple(name: str, states: Iterable[str]) -> tuple:
    if isinstance(states, str):
        raise ValidationError(
            f"Variable {name!r}: states must be a sequence of labels, not a single string."
        )
    if isinstance(states, (Set, Mapping)):
        raise ValidationError(
            f"Variable {name!r}: states must have a deterministic order; "
            f"got unordered {type(states).__name__}."
        )
    try:
        return tuple(states)
    except TypeError:
        raise ValidationError(
            f"Variable {name!r}: states must be an iterable of strings, got {states!r}."
        ) from None
