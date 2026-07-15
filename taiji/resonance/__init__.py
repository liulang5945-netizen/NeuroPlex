"""Resonance Field Architecture — 共振场架构模块.

A minimal, testable implementation of the core resonance field
mechanism for the Taiji Neuron project.

Core components:
- ResonanceField: shared 4096-dim field with L2-normalized writes
- ResonanceNeuron: wraps Transformer backbone with field_write/field_read
- ResonanceEnsemble: multi-round resonance loop with confidence gating +
  early stop + dynamic threshold filtering
- ConfidenceGate: only resonate when prediction uncertainty is high
- EarlyStopResonance: stop iterating when logits converge
- ResonanceTrigger: full trigger conditions (uncertainty + diversity + error room)
- QualityFilter: exclude weak neurons from resonance
- TokenTranslator + TokenizerHub: domain-specific tokenizer hot-swap
- ScaleLayering + ClusterDominance + DivisionPath: division-of-labor

All gating mechanisms (Experiment 12) are fully integrated into
ResonanceEnsemble.forward().
"""

from .field import ResonanceField
from .neuron import ResonanceNeuron
from .ensemble import ResonanceEnsemble
from .config import NeuronConfig, COMPACT, STANDARD, EXPERT, TINY_TEST
from .gating import ConfidenceGate, EarlyStopResonance, ResonanceTrigger
from .quality import QualityFilter
from .translator import TokenTranslator, TokenizerHub
from .division import ScaleLayering, ClusterDominance, DivisionPath

__all__ = [
    # Core
    "ResonanceField",
    "ResonanceNeuron",
    "ResonanceEnsemble",
    "NeuronConfig",
    "COMPACT",
    "STANDARD",
    "EXPERT",
    "TINY_TEST",
    # Gating (Experiment 12)
    "ConfidenceGate",
    "EarlyStopResonance",
    "ResonanceTrigger",
    # Quality
    "QualityFilter",
    # Translator
    "TokenTranslator",
    "TokenizerHub",
    # Division-of-labor
    "ScaleLayering",
    "ClusterDominance",
    "DivisionPath",
]
