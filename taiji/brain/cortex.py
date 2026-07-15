"""Cortex — consciousness center via resonance field state.

Phase 4: Replaces the monolithic ModelSelf with ResonanceEnsemble.
The field state IS the consciousness — not a single model's hidden state,
but the collective resonance pattern across domain-specialized neurons.

Architecture:
    Input tokens → Shared Embedding (256K) → ResonanceEnsemble
        ├── zh neuron (standard)
        ├── en neuron (standard)
        ├── code neuron (expert)
        ├── math neuron (expert)
        └── general neuron (standard)
            ↓
    Resonance Field (shared consciousness)
            ↓
    Weighted Logits → Token prediction

Usage:
    cortex = Cortex(neurons_dir="data/neurons")
    output = cortex.generate("今天天气怎么样？", max_tokens=256)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig,
    ConfidenceGate, EarlyStopResonance, QualityFilter, DivisionPath,
)


class Cortex:
    """Resonance-field-based consciousness center.

    Wraps ResonanceEnsemble with a high-level generate() interface
    compatible with the existing API and agent systems.
    """

    def __init__(
        self,
        neurons_dir: str = "data/neurons",
        device: str = "cpu",
        max_rounds: int = 3,
        confidence_threshold: float = 0.9,
        enable_gating: bool = True,
    ):
        self.device = device
        self.neurons_dir = neurons_dir
        self.max_rounds = max_rounds
        self.enable_gating = enable_gating

        # ── Load neurons ──
        self.neurons: Dict[str, ResonanceNeuron] = {}
        self._load_neurons()

        # ── Create field and ensemble ──
        field_dim = max(n.config.field_dim for n in self.neurons.values()) if self.neurons else 4096
        self.field = _AdaptiveField(dim=field_dim)
        self.ensemble = ResonanceEnsemble(
            self.neurons, self.field,
            max_rounds=max_rounds,
            confidence_gate=ConfidenceGate(threshold=confidence_threshold) if enable_gating else None,
            early_stop=EarlyStopResonance() if enable_gating else None,
        )

        # ── Shared embedding (placeholder — in production, use SVD-initialized embedding) ──
        self._shared_embedding: Optional[torch.nn.Embedding] = None
        self._tokenizer = None

        # ── State ──
        self.is_loaded = len(self.neurons) > 0
        print(f"[Cortex] Loaded {len(self.neurons)} neurons, field_dim={field_dim}")

    def _load_neurons(self):
        """Load all distilled neurons from disk."""
        for domain in ["zh", "en", "code", "math", "general"]:
            ckpt_path = os.path.join(self.neurons_dir, f"neuron_{domain}.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                cfg: NeuronConfig = ckpt["neuron_config"]
                neuron = ResonanceNeuron(cfg).to(self.device)
                neuron.load_state_dict(ckpt["state_dict"])
                neuron.eval()
                neuron.freeze_fingerprint()
                self.neurons[domain] = neuron
                n_params = sum(p.numel() for p in neuron.parameters())
                print(f"  [{domain}] {cfg.spec} neuron: {n_params/1e6:.0f}M params")

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for encode/decode."""
        self._tokenizer = tokenizer

    def set_shared_embedding(self, embedding: torch.nn.Embedding) -> None:
        """Set the shared embedding table."""
        self._shared_embedding = embedding

    def think(self, input_ids: torch.Tensor) -> Dict:
        """Run one round of resonance thinking.

        Args:
            input_ids: [B, L] token IDs.

        Returns:
            dict with field_state, weighted_logits, final_scores, n_rounds.
        """
        if self._shared_embedding is not None:
            shared_emb = self._shared_embedding(input_ids.to(self.device))
        else:
            # Fallback: random embedding (for testing)
            shared_emb = torch.randn(
                input_ids.shape[0], input_ids.shape[1], 512,
                device=self.device,
            )

        result = self.ensemble.forward(shared_emb, return_logits=True, enable_gating=self.enable_gating)
        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate text using resonance ensemble.

        Args:
            prompt: input text.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: top-k sampling.

        Returns:
            generated text string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call cortex.set_tokenizer() first.")

        # Encode
        input_ids = self._tokenizer.encode(prompt)
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = []
        for _ in range(max_tokens):
            # Get logits from ensemble
            result = self.think(ids_tensor)

            if "weighted_logits" not in result:
                break

            logits = result["weighted_logits"][:, -1, :] / temperature  # [B, vocab]

            # Top-k sampling
            if top_k > 0:
                top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx_in_topk = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, sampled_idx_in_topk[0]].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Check for EOS
            if hasattr(self._tokenizer, 'eos_id'):
                if next_token == self._tokenizer.eos_id():
                    break

            generated.append(next_token)
            ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        # Decode
        return self._tokenizer.decode(generated) if generated else ""

    def get_field_state(self) -> torch.Tensor:
        """Get current resonance field state (consciousness snapshot)."""
        return self.field.get_state()

    def get_dominant_domain(self) -> Optional[str]:
        """Identify which domain is dominating the current thought."""
        if not self.field.scores:
            return None
        return max(self.field.scores, key=self.field.scores.get)


class _AdaptiveField(ResonanceField):
    """Field that auto-pads neuron vectors to match field dim."""

    def _pad(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        vd = vector.shape[-1]
        if vd < self.dim:
            pad = torch.zeros(*vector.shape[:-1], self.dim - vd, device=vector.device, dtype=vector.dtype)
            return torch.cat([vector, pad], dim=-1)
        return vector[..., :self.dim] if vd > self.dim else vector

    def write(self, nid, v):
        return super().write(nid, self._pad(v))

    def score(self, v):
        return super().score(self._pad(v))
