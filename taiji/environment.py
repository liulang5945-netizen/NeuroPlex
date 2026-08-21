"""Minimal active-environment contract for native Taiji interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, runtime_checkable


@dataclass(frozen=True)
class EnvironmentOutcome:
    """What the environment returns after one Taiji motor action."""

    sensation: int
    reward: float
    terminal: bool = False


@runtime_checkable
class TaijiEnvironment(Protocol):
    """Protocol for environments whose transitions depend on Taiji actions."""

    def reset(self) -> Tuple[int, Sequence[int]]:
        """Return the initial sensation and currently afforded actions."""

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        """Execute an action and return its sensation, reward and terminal flag."""
