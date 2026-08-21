"""Taiji: a native persistent predictive-computing architecture.

This top-level package is independent of the legacy NeuroPlex Transformer
runtime.  PyTorch is used only as a tensor execution engine.
"""

from .config import TaijiConfig
from .fabric import TaijiFabric
from .model import Taiji
from .organs import ByteMotor, ByteSensor
from .sparse import SparseSynapses
from .state import RegionState, TaijiState, TaijiStep

__all__ = [
    "ByteMotor",
    "ByteSensor",
    "RegionState",
    "SparseSynapses",
    "Taiji",
    "TaijiConfig",
    "TaijiFabric",
    "TaijiState",
    "TaijiStep",
]
