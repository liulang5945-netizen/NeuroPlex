"""Taiji: persistent event-driven population dynamics.

This package is intentionally isolated from the legacy resonance runtime until
its state, locality and population contracts have passed their ablations.
"""

from .cell import TaijiCell
from .config import TaijiConfig
from .events import EventKind, EventMode, TaijiEvent
from .field import TaijiField
from .runtime import TaijiRuntime
from .state import (
    AssociationLearningResult,
    TaijiCellState,
    TaijiFieldState,
    TaijiState,
    TaijiStepResult,
)

__all__ = [
    "EventKind",
    "EventMode",
    "AssociationLearningResult",
    "TaijiCell",
    "TaijiCellState",
    "TaijiConfig",
    "TaijiEvent",
    "TaijiField",
    "TaijiFieldState",
    "TaijiRuntime",
    "TaijiState",
    "TaijiStepResult",
]
