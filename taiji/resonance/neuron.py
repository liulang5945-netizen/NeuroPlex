"""Resonance neuron — wraps a ModelSelfBackbone with field I/O.

Each neuron:
1. Receives shared base embeddings (Level 0 → Level 1)
2. Projects through its own adapter into a per-neuron concept space
3. Processes through a standard Transformer (layers.py, zero changes)
4. Writes a normalised field vector (Level 1 → Level 2)
5. Reads field state for conditioning (Level 2 → Level 1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.config import ModelConfig
from taiji.layers import RMSNorm, TransformerBlock

from .config import NeuronConfig


class ResonanceNeuron(nn.Module):
    """A single resonance neuron — independent Transformer + field interface.

    Built on top of the existing TransformerBlock and RMSNorm from layers.py
    (zero changes to existing code).  Adds:
    - embed_adapter: shared embedding → neuron's internal dimension
    - field_write: final hidden → field vector (L2-normalised)
    - field_read_layers: field state → per-layer conditioning residual
    - lm_head: vocab projection (for PPL evaluation / pretraining)
    - fingerprint: frozen direction vector for prescreening (future)
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()
        self.config = neuron_config
        c = neuron_config

        # ── Embedding adapter (P0: shared base → per-neuron concept space) ──
        self.embed_adapter = nn.Linear(c.base_embed_dim, c.hidden_size, bias=False)

        # ── Transformer body (reuses layers.py, zero changes) ──
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_attention_heads,
                num_kv_heads=c.num_key_value_heads,
                intermediate_size=c.intermediate_size,
                rms_norm_eps=c.rms_norm_eps,
                bias=c.attention_bias,
            )
            for _ in range(c.num_hidden_layers)
        ])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

        # ── Field write projection ──
        self.field_write = nn.Linear(c.hidden_size, c.field_dim, bias=False)

        # Attention pooling for field write (v2)
        # Learn a query that pools over all positions instead of only using the last token.
        # This surfaces the most salient conceptual content for field communication.
        self.field_pool_query = nn.Parameter(torch.randn(c.hidden_size) * 0.02)
        self.field_pool_scale = c.hidden_size ** -0.5

        # ── Field read projections (one per layer, for conditioning) ──
        self.field_read_layers = nn.ModuleList([
            nn.Linear(c.field_dim, c.hidden_size, bias=False)
            for _ in range(c.num_hidden_layers)
        ])

        # Position gate for field read (v2)
        # Each position decides how much field conditioning to absorb,
        # replacing the old broadcast (same vector to all positions).
        self.field_read_gate = nn.Linear(c.hidden_size, 1, bias=True)

        # ── Language modelling head (for PPL / pretraining) ──
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

        # ── Direction fingerprint (frozen, for future prescreening) ──
        self.register_buffer("fingerprint", torch.zeros(c.hidden_size))

        # ── Side channel interface (reserved for P1) ──
        self.side_channels: Dict[int, nn.Linear] = {}

        # v1 compatibility: use last-token write + broadcast read (for old ckpts)
        self.v1_compat: bool = False

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        field_state: Optional[torch.Tensor] = None,
        round_num: int = 1,
        return_logits: bool = False,
        side_signals: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the neuron.

        Args:
            shared_embeddings: [B, L, base_embed_dim] from the shared embedding table
            field_state: [D] current field state vector (from round 2 onward)
            round_num: current resonance round (1 = independent, 2+ = conditioned)
            return_logits: if True, also return lm_head logits (for PPL)
            side_signals: optional {neuron_id: vector} for side-channel communication

        Returns:
            dict with keys:
            - field_vector: [B, D] L2-normalised write vector
            - hidden_before_write: [B, hidden] for diversity loss
            - logits: [B, L, vocab] (only if return_logits=True)
        """
        # ── Step 1: Embedding adapter (shared base → neuron concept space) ──
        h = self.embed_adapter(shared_embeddings)  # [B, L, hidden]

        # ── Step 2: Transformer layers + field conditioning ──
        bsz, seqlen, _ = h.shape

        for i, block in enumerate(self.layers):
            # Standard transformer block (layers.py, zero changes)
            h_normed = block.attention_norm(h)
            attn_out, _ = block.attention(h_normed)
            h = h + attn_out
            h = h + block.feed_forward(block.ffn_norm(h))

            # Field conditioning (round 2+ only)
            if field_state is not None and round_num > 1:
                conditioning = self.field_read_layers[i](field_state)  # [D] → [hidden]
                if conditioning.dim() == 1:
                    conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1,1,H] -> all B,L
                else:
                    conditioning = conditioning.unsqueeze(1)  # [B,1,H] -> over seq
                if self.v1_compat:
                    # v1: broadcast same conditioning to all positions
                    h = h + conditioning
                else:
                    # v2: per-position gated read
                    gate = torch.sigmoid(self.field_read_gate(h))  # [B, L, 1]
                    h = h + gate * conditioning

            # Side channel injection (reserved for P1)
            if side_signals is not None:
                for src_id, signal in side_signals.items():
                    if src_id in self.side_channels:
                        h = h + 0.1 * self.side_channels[src_id](signal)

        # ── Step 3: Final norm ──
        h = self.norm(h)

        # ── Step 4: Field write ──
        if self.v1_compat:
            # v1: last-token write (matches old checkpoint training distribution)
            hidden_last = h[:, -1, :]  # [B, hidden]
            v_raw = self.field_write(hidden_last)  # [B, D]
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": hidden_last,
            }
        else:
            # v2: attention-pooled field write
            attn_scores = torch.matmul(h, self.field_pool_query) * self.field_pool_scale  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L]
            pooled = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # [B, hidden]
            v_raw = self.field_write(pooled)  # [B, D]
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": pooled,
                "field_attn_weights": attn_weights,
            }

        # ── Step 5: Optional logits (for PPL evaluation) ──
        if return_logits:
            result["logits"] = self.lm_head(h)  # [B, L, vocab]

        return result

    def freeze_fingerprint(self) -> None:
        """Compute and freeze the direction fingerprint.

        Fingerprint = L2-normalised mean of field_write weight rows.
        Used for lightweight prescreening (P1).
        """
        with torch.no_grad():
            fp = self.field_write.weight.mean(dim=0)  # [hidden]
            self.fingerprint.copy_(fp / (fp.norm() + 1e-8))

    def establish_side_channel(self, peer_id: int) -> None:
        """Create a side channel to a frequently co-active peer neuron."""
        if peer_id not in self.side_channels:
            self.side_channels[peer_id] = nn.Linear(
                self.config.field_dim, self.config.hidden_size, bias=False
            )

    @torch.no_grad()
    def quick_probe(self, shared_embeddings: torch.Tensor) -> torch.Tensor:
        """Lightweight forward pass for prescreening (skip full Transformer).

        Runs only the adapter + field_write, no Transformer layers.
        Returns a rough field_vector direction for candidate selection.
        """
        h = self.embed_adapter(shared_embeddings)
        # Use mean pooling over sequence as a rough representation
        h_pooled = h.mean(dim=1)  # [B, hidden]
        v_raw = self.field_write(h_pooled)
        return v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
