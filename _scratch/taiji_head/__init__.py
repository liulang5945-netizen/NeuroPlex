"""Taiji: a native persistent predictive-computing architecture.

This top-level package is independent of the legacy NeuroPlex Transformer
runtime.  PyTorch is used only as a tensor execution engine.
"""

from .config import TaijiConfig
from .environment import EnvironmentOutcome, TaijiEnvironment
from .fabric import TaijiFabric
from .memory import EpisodicField, EpisodicReplay, EpisodicWrite
from .model import Taiji
from .organs import ByteMotor, ByteSensor, SparseReceptorBank
from .sparse import SparseSynapses
from .state import (
    MemoryRecall,
    MemoryState,
    PendingAction,
    PendingExperience,
    RegionState,
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
)

__all__ = [
    "ByteMotor",
    "ByteSensor",
    "EnvironmentOutcome",
    "EpisodicField",
    "EpisodicReplay",
    "EpisodicWrite",
    "MemoryRecall",
    "MemoryState",
    "PendingAction",
    "PendingExperience",
    "RegionState",
    "SparseReceptorBank",
    "SparseSynapses",
    "Taiji",
    "TaijiConfig",
    "TaijiConsolidation",
    "TaijiDecision",
    "TaijiEnvironment",
    "TaijiFabric",
    "TaijiOutcome",
    "TaijiState",
    "TaijiStep",
]
