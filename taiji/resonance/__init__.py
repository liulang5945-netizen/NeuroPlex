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
from .config import NeuronConfig, COMPACT, STANDARD, FOUNDATION, EXPERT, TINY_TEST
from .gating import ConfidenceGate, EarlyStopResonance, ResonanceTrigger
from .quality import QualityFilter
from .translator import TokenTranslator, TokenizerHub
from .division import ScaleLayering, ClusterDominance, DivisionPath
from .lifecycle import LifecycleManager, ApoptosisTracker, MaturityTracker, NeurogenesisTrigger
from .stdp import STDPTracker, STDPRule, FiringRecord
from .neuro_modulation import NeuromodulatorState, SleepConsolidator
from .cognitive_enhancements import (
    CorticalColumn, ColumnRegistry, AttentionBeam, ThresholdPlasticity,
)
from .neurogenesis_creator import NeurogenesisCreator
from .domain_router import DomainRouter
from .thalamic_router import ThalamicRouter
from .standalone_embedding import StandaloneEmbedding
from .gamma_oscillator import GammaOscillator, apply_gamma_gate
from .tribal import CoactivationTracker
from .self_evolving_encoder import (
    SharedContextEncoder,
    HebbianUpdater,
    ContrastiveLoss,
    MLMLoss,
    SelfEvolver,
)

__all__ = [
    # Core
    "ResonanceField",
    "ResonanceNeuron",
    "ResonanceEnsemble",
    "NeuronConfig",
    "COMPACT",
    "STANDARD","FOUNDATION",
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
    # 生命周期（人脑启发：凋亡与新生）
    "LifecycleManager",
    "ApoptosisTracker",
    "MaturityTracker",
    "NeurogenesisTrigger",
    # STDP 局部学习（人脑启发：脉冲时序依赖可塑性）
    "STDPTracker",
    "STDPRule",
    "FiringRecord",
    # 神经调质与睡眠巩固
    "NeuromodulatorState",
    "SleepConsolidator",
    # 认知增强（功能柱、注意力、阈值可塑性）
    "CorticalColumn",
    "ColumnRegistry",
    "AttentionBeam",
    "ThresholdPlasticity",
    # 神经新生创建器（P2-7 分场景教师选择）
    "NeurogenesisCreator",
    # P2-2: Domain-aware routing
    "DomainRouter",
    "ThalamicRouter",
    "StandaloneEmbedding",
    # P6-3: Gamma 同步绑定
    "GammaOscillator",
    "apply_gamma_gate",
    # P1-1: 共激活追踪（动态部落化 + side_channels 强化依据）
    "CoactivationTracker",
    # P6-6: 自主进化 encoder
    "SharedContextEncoder",
    "HebbianUpdater",
    "ContrastiveLoss",
    "MLMLoss",
    "SelfEvolver",
]
