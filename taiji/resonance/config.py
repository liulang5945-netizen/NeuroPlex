"""Neuron specification configurations for the resonance field architecture.

Three standard specs (compact / standard / expert), each designed
as the smallest configuration that can independently understand
text while staying trainable on CPU (see docs Chapter 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
        total = body + adapter + field_w + field_r
        return total / 1_000_000


# ── Standard specs ──

COMPACT = NeuronConfig(
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1536,
    spec="compact",
    # Field dim must match STANDARD so compact/standard neurons can share
    # one resonance group and be distilled/ensambled together.
    field_dim=3072,
)

STANDARD = NeuronConfig(
    hidden_size=768,
    num_hidden_layers=10,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2304,
    spec="standard",
    # Existing zh/en/general checkpoints were distilled at field_dim=3072.
    # Keeping this default means re-distillation stays compatible with them.
    field_dim=3072,
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
