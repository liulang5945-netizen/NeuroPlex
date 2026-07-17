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
        if self.neurons:
            dims = {n.config.field_dim for n in self.neurons.values()}
            if len(dims) > 1:
                raise ValueError(
                    f"[Cortex] neurons disagree on field_dim: {dims}. "
                    f"Re-distill legacy checkpoints under H9 (field_dim=4096) before loading."
                )
            field_dim = dims.pop()
        else:
            field_dim = 4096
        self.field = ResonanceField(dim=field_dim)
        self.ensemble = ResonanceEnsemble(
            self.neurons, self.field,
            max_rounds=max_rounds,
            confidence_gate=ConfidenceGate(threshold=confidence_threshold) if enable_gating else None,
            early_stop=EarlyStopResonance() if enable_gating else None,
        )

        # ── Shared embedding (placeholder — in production, use SVD-initialized embedding) ──
        self._shared_embedding: Optional[torch.nn.Embedding] = None
        self._embed_pipeline = None  # H10: callable(input_ids) -> [B, L, base_embed_dim]
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
        """Set the shared embedding table (highest precedence source)."""
        self._shared_embedding = embedding

    def set_teacher_pipeline(self, teacher_model, shared_proj) -> None:
        """H10: register a teacher-model + SharedEmbedProj chain to derive
        shared embeddings from token IDs. Used when no preloaded embedding
        table is available. shared_proj must be an nn.Module mapping
        [..., 2048] -> [..., base_embed_dim].
        """
        import torch as _torch
        from taiji.training.checkpoint_bridge import extract_hidden_states

        def _pipeline(input_ids: _torch.Tensor) -> _torch.Tensor:
            with _torch.no_grad():
                hidden = extract_hidden_states(teacher_model, input_ids)  # [B, L, 2048]
                return shared_proj(hidden)                                  # [B, L, 512]

        self._embed_pipeline = _pipeline

    def think(self, input_ids: torch.Tensor) -> Dict:
        """Run one round of resonance thinking.

        Args:
            input_ids: [B, L] token IDs.

        Returns:
            dict with field_state, weighted_logits, final_scores, n_rounds.
        """
        # H10: shared_emb must come from the teacher-SVD embedding path.
        # Precedence: (a) explicit set_shared_embedding()  (b) SharedEmbedProj on
        # teacher hidden states  (c) fail loudly rather than emit random noise.
        input_ids = input_ids.to(self.device)
        if self._shared_embedding is not None:
            shared_emb = self._shared_embedding(input_ids)
        elif self._embed_pipeline is not None:
            shared_emb = self._embed_pipeline(input_ids)
        else:
            raise RuntimeError(
                "[Cortex.think] no shared_embedding source. Either call "
                "set_shared_embedding(embedding_table) with a preloaded table, "
                "or set_teacher_pipeline(teacher_model, shared_proj) to derive "
                "embeddings from teacher hidden states on the fly."
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


# _AdaptiveField removed: field_dim is unified under H9; no padding needed.
