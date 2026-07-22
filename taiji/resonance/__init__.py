"""Resonance Field Architecture — 共振场架构模块.

三层架构：
- Layer 1 (共享感官): shared_embedding(256000, 512)，所有 neuron 共用
- Layer 2 (认知空间): per-neuron embed_adapter + Transformer body
- Layer 3 (神经语言): 4096-dim 共振场

TokenTranslator 桥接域 tokenizer → 通用 token → 共享嵌入。
通用词表可热插拔，不影响 neuron 内部。

Core components:
- ResonanceField: shared 4096-dim field with L2-normalized writes
- ResonanceNeuron: wraps Transformer backbone with field_write/field_read
- ResonanceEnsemble: multi-round resonance loop
- TokenizerHub: domain-specific tokenizer hot-swap
"""

from .field import ResonanceField
from .neuron import ResonanceNeuron
from .ensemble import ResonanceEnsemble
from .config import (
    NeuronConfig, COMPACT, STANDARD, FOUNDATION, EXPERT, TINY_TEST,
    DEFAULT_NEURON_SPEC, get_default_neuron_config,
    DOMAIN_VOCAB_SIZES, GENERAL_TOKENIZER_DOMAIN, get_domain_neuron_config,
)
from .translator import (
    TokenTranslator, TokenizerHub,
    build_position_alignment, batch_align_and_embed,
)
from .lifecycle import LifecycleManager, ApoptosisTracker, MaturityTracker, NeurogenesisTrigger
from .stdp import STDPTracker, STDPRule, FiringRecord
from .neuro_modulation import NeuromodulatorState, SleepConsolidator
from .cognitive_enhancements import (
    CorticalColumn, ColumnRegistry, AttentionBeam, ThresholdPlasticity,
)
from .gamma_oscillator import GammaOscillator, apply_gamma_gate
from .geometry import NeuronGeometry

__all__ = [
    # Core
    "ResonanceField",
    "ResonanceNeuron",
    "ResonanceEnsemble",
    "NeuronConfig",
    "COMPACT", "STANDARD", "FOUNDATION",
    "EXPERT",
    "TINY_TEST",
    "DEFAULT_NEURON_SPEC",
    "get_default_neuron_config",
    "DOMAIN_VOCAB_SIZES",
    "GENERAL_TOKENIZER_DOMAIN",
    "get_domain_neuron_config",
    # Translator
    "TokenTranslator",
    "TokenizerHub",
    "build_position_alignment",
    "batch_align_and_embed",
    # 生命周期
    "LifecycleManager",
    "ApoptosisTracker",
    "MaturityTracker",
    "NeurogenesisTrigger",
    # STDP
    "STDPTracker", "STDPRule", "FiringRecord",
    # 神经调质
    "NeuromodulatorState", "SleepConsolidator",
    # 认知增强
    "CorticalColumn", "ColumnRegistry", "AttentionBeam", "ThresholdPlasticity",
    # Gamma 同步
    "GammaOscillator", "apply_gamma_gate",
    # RSGN 几何
    "NeuronGeometry",
]
