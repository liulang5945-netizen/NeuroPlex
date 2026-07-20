"""Neuron specification configurations for the resonance field architecture.

Three standard specs (compact / standard / expert), each designed
as the smallest configuration that can independently understand
text while staying trainable on CPU (see docs Chapter 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class NeuronConfig:
    """Configuration for a single resonance neuron."""

    # ── Transformer body ──
    hidden_size: int = 768
    num_hidden_layers: int = 10
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2304
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0

    # ── Embedding (shared across all neurons) ──
    vocab_size: int = 256000
    base_embed_dim: int = 512

    # ── Field interface ──
    field_dim: int = 4096

    # ── Domain extension (0 = disabled) ──
    num_domain_concepts: int = 0

    # ── Metadata ──
    spec: str = "standard"
    neuron_id: Optional[str] = None

    # ── 神经元类型（人脑启发：兴奋性/抑制性分化）──
    # excitatory: 默认，对场做正向贡献（类比谷氨酸能）
    # inhibitory: 对场做负向贡献，抑制过度共振（类比 GABA 能）
    # 约 20% 神经元应为 inhibitory，由 CoactivationTracker 自动转化过度兴奋的神经元
    neuron_type: Literal["excitatory", "inhibitory"] = "excitatory"

    # ── 不应期配置（人脑启发：refractory period）──
    # 写入场后进入不应期，rounds_cooldown 轮内只能读场不能写
    # 防止强神经元垄断场，强制信息轮替
    refractory_cooldown: int = 2

    # ── lm_head 低秩分解配置 ──
    # 共享基矩阵 W_base [hidden, vocab] + 每个神经元低秩残差 U_i @ V_i
    # 保留 per-neuron 输出差异，同时控制参数量
    # lm_head_rank=0 表示禁用低秩分解（使用传统 per-neuron lm_head）
    lm_head_rank: int = 64

    # ── Approximate parameter count (excluding shared embedding) ──
    @property
    def approx_params_m(self) -> float:
        """Rough parameter count in millions (transformer body + field projections)."""
        d = self.hidden_size
        n = self.num_hidden_layers
        h = self.num_attention_heads
        # Per-layer: 4*(d^2) for Q/K/V/O (GQA saves on K/V) + 3*(d*intermediate) for SwiGLU
        kv_ratio = self.num_key_value_heads / self.num_attention_heads
        attn_params = d * d * (2 + 2 * kv_ratio)  # Q(1) + K(kv_ratio) + V(kv_ratio) + O(1)
        ffn_params = 3 * d * self.intermediate_size
        per_layer = attn_params + ffn_params
        # Norm params are negligible
        body = n * per_layer
        # Embed adapter: base_embed_dim -> hidden_size
        adapter = self.base_embed_dim * d
        # Field write: hidden_size -> field_dim
        field_w = d * self.field_dim
        # Field read: field_dim -> hidden_size per layer
        field_r = n * self.field_dim * d
        # LM head: hidden_size -> vocab_size (per-neuron, not shared)
        lm_head = d * self.vocab_size
        total = body + adapter + field_w + field_r + lm_head
        return total / 1_000_000


# ── Standard specs ──

COMPACT = NeuronConfig(
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1536,
    spec="compact",
    # H9: unified field_dim=4096 across all specs so every neuron can
    # co-resonate directly (no padding, no complementarity noise).
    field_dim=4096,
)

STANDARD = NeuronConfig(
    hidden_size=768,
    num_hidden_layers=10,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2304,
    spec="standard",
    # H9: unified field_dim=4096; old 3072 checkpoints are legacy-only.
    field_dim=4096,
)


FOUNDATION = NeuronConfig(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    num_key_value_heads=2,
    intermediate_size=1152,
    spec='foundation',
    # H9: unified field_dim=4096 across all v3 neurons.
    field_dim=4096,
)

EXPERT = NeuronConfig(
    hidden_size=1024,
    num_hidden_layers=14,
    num_attention_heads=16,
    num_key_value_heads=4,
    intermediate_size=3072,
    spec="expert",
    # Existing code/math checkpoints use field_dim=4096.
    field_dim=4096,
)

# ── Tiny test config (for smoke-testing the code) ──

TINY_TEST = NeuronConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    field_dim=512,
    spec="tiny_test",
)
