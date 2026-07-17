"""Resonance field core - shared communication medium for all neurons.

The field is a D-dimensional vector space where neurons write
L2-normalised vectors and read the accumulated state.  It is the
"neural language" of the architecture - completely independent of
the tokenizer (Level 0) and the per-neuron concept spaces (Level 1).

Key properties:
- All writes are L2-normalised -> neuron size does not determine "loudness"
- Cosine similarity between a neuron's write and the field state is the
  scoring mechanism (resonance = alignment with the collective)
- W_cond is now ACTIVE: it projects the field state to a "conditioned"
  subspace before scoring, so it learns which cross-neuron patterns matter.

Fixes (this version):
  H2: state is [B, D], one independent field per sample (no cross-sample bleed)
  H5: score() uses leave-one-out (excludes the neuron's own contribution)
  H6: prediction_complementarity_score(): how much another neuron *corrects*
      this one's mistakes, measured on logits (not orthogonal geometry).
  H8: W_cond is now applied: conditioned = sigmoid(state @ W_cond) * state
      before scoring - the dead parameter becomes a learned gate on the field.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ResonanceField(nn.Module):
    """Shared resonance field - the "neural language" of the architecture.

    State shape is now [B, D] (H2 fix): each sample in the batch gets its own
    independent field, so there is no cross-sample contamination.

    W_cond (H8 fix): a learnable [D, D] parameter applied as a multiplicative
    gate on the field state before scoring.  This was previously a dead
    parameter; it is now used in score() and complementarity().
    """

    def __init__(self, dim: int = 4096, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self._device = device or torch.device("cpu")

        self.register_buffer("state", torch.zeros(dim))
        # H2: per-sample field state, set lazily on first write with batch dim
        self._batch_size: int = 1

        self.W_cond = nn.Parameter(torch.randn(dim, dim) * 0.02)

        self._write_history: Dict[str, List[torch.Tensor]] = {}
        self._contributions: Dict[str, torch.Tensor] = {}
        self.scores: Dict[str, float] = {}
        self.n_active: int = 0

    def reset(self, batch_size: int = 1) -> None:
        # Promote state to [B, D] up front for batch_size > 1 (H2): each sample
        # gets an independent field, so there is never cross-sample bleed.
        if batch_size > 1:
            self.state = torch.zeros(batch_size, self.dim)
        else:
            self.state = torch.zeros(self.dim)
        self._batch_size = batch_size
        self._contributions.clear()
        self.scores.clear()
        self.n_active = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def write(self, neuron_id: str, vector: torch.Tensor) -> torch.Tensor:
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        v_norm = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
        B = v_norm.shape[0]
        if B == 1 and self._batch_size == 1:
            self.state = self.state + v_norm.squeeze(0)
        elif self.state.dim() == 1 and self._batch_size == 1:
            # auto-promote single [D] field to per-sample [B, D] (H2): robust when
            # neurons write batched vectors without an explicit reset(batch_size=...).
            self.state = self.state.unsqueeze(0).expand(B, -1).clone()
            self.state = self.state + v_norm
            self._batch_size = B
        elif B == self._batch_size:
            if self.state.dim() == 1:
                self.state = self.state.unsqueeze(0).expand(self._batch_size, -1).clone()
            self.state = self.state + v_norm
        else:
            raise ValueError(f"vector batch {B} != field batch {self._batch_size}")
        self._contributions[neuron_id] = v_norm.detach()
        self.n_active += 1
        if neuron_id not in self._write_history:
            self._write_history[neuron_id] = []
        self._write_history[neuron_id].append(v_norm.detach())
        return v_norm

    def _leave_one_out_state(self, exclude_id: str) -> torch.Tensor:
        """Field state with one neuron's contribution removed (H5 fix)."""
        contrib = self._contributions.get(exclude_id)
        if contrib is None:
            return self.state
        if self.state.dim() == 1:
            return self.state - contrib.squeeze(0)
        return self.state - contrib

    def _condition(self, state: torch.Tensor) -> torch.Tensor:
        """Apply W_cond as a multiplicative gate (H8 fix)."""
        if state.norm() < 1e-8:
            return state
        cond = torch.sigmoid(state @ self.W_cond)
        return state * cond

    def score(self, vector: torch.Tensor, neuron_id: Optional[str] = None) -> float:
        score_state = self._leave_one_out_state(neuron_id) if neuron_id else self.state
        cond = self._condition(score_state)
        # Normalise cond to a 2-D [..., D] tensor. When the field state is [D]
        # (batch_size=1) cond is [1, D] and broadcasts against a [B, D] vector;
        # when the state is per-sample [B, D] cond matches it sample-for-sample.
        # (Old code did cond.unsqueeze(0) which, for a 2-D [B, D] state, produced
        # [1, B, D] and silently formed a BxB outer product over the batch axis.)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        vec2 = vector if vector.dim() == 2 else vector.unsqueeze(0)
        v_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (v_norm * cond).sum(dim=-1) / (cond.norm(dim=-1, keepdim=True) + 1e-8)
        return float(sims.mean().item())

    def complementarity_score(self, vector: torch.Tensor, neuron_id: Optional[str] = None) -> float:
        """Geometric orthogonality (legacy): kept for diagnostics.

        Not used directly by routing; the ensemble uses prediction_complementarity.
        """
        score_state = self._leave_one_out_state(neuron_id) if neuron_id else self.state
        if vector.dim() == 2:
            vector = vector.mean(dim=0)
        v_norm = vector / (vector.norm() + 1e-8)
        if score_state.dim() == 2:
            score_state = score_state.mean(dim=0)
        if score_state.norm() < 1e-8:
            return 1.0
        f_norm = score_state / (score_state.norm() + 1e-8)
        alignment = float(torch.dot(v_norm, f_norm).item())
        orthogonal = v_norm - alignment * f_norm
        return float(orthogonal.norm().item())

    def prediction_complementarity(
        self,
        neuron_a_logits: torch.Tensor,
        neuron_b_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> float:
        """How much neuron B corrects neuron A's mistakes (H6 fix).

        Measures the log-loss reduction when B is consulted. A token is a
        "mistake" if A assigns low probability to the truth. We measure how
        often B is more right on those tokens.

        Without targets, falls back to disagreement-driven complementarity:
        tokens where A and B disagree give B "weight if B is more confident".
        With targets, it is the actual log-loss reduction from B on A's errors.
        """
        logp_a = torch.log_softmax(neuron_a_logits, dim=-1)
        logp_b = torch.log_softmax(neuron_b_logits, dim=-1)
        pa = torch.exp(logp_a)
        pb = torch.exp(logp_b)

        if targets is not None:
            shift_t = targets[:, 1:].contiguous() if targets.shape == neuron_a_logits.shape[:2] else targets
            shift_a = logp_a[:, :-1, :]
            shift_b = logp_b[:, :-1, :]
            if shift_t.dim() == 2:
                tflat = shift_t.reshape(-1)
                nll_a = -shift_a.reshape(-1, shift_a.size(-1)).gather(-1, tflat.unsqueeze(-1)).squeeze(-1)
                nll_b = -shift_b.reshape(-1, shift_b.size(-1)).gather(-1, tflat.unsqueeze(-1)).squeeze(-1)
            else:
                nll_a = -shift_a.gather(-1, shift_t)
                nll_b = -shift_b.gather(-1, shift_t)
            reduction = (nll_a - nll_b).clamp(min=0.0).mean()
            return float(reduction.item())
        raise_prob_b = (pb.max(dim=-1).values > pa.max(dim=-1).values)
        boost = raise_prob_b.float().mean()
        return float(boost.item())

    def combined_score(self, vector: torch.Tensor, alpha: float = 0.5, neuron_id: Optional[str] = None) -> float:
        align = self.score(vector, neuron_id=neuron_id)
        comp = self.complementarity_score(vector, neuron_id=neuron_id)
        align_01 = (align + 1.0) / 2.0
        return (1.0 - alpha) * align_01 + alpha * comp

    def directional_congestion(self, vector: torch.Tensor, active_vectors: List[torch.Tensor]) -> float:
        if not active_vectors:
            return 0.0
        if vector.dim() == 2:
            vector = vector.mean(dim=0)
        v_norm = vector / (vector.norm() + 1e-8)
        sims = []
        for av in active_vectors:
            av_clean = av.mean(dim=0) if av.dim() == 2 else av
            av_norm = av_clean / (av_clean.norm() + 1e-8)
            sims.append(max(0.0, float(torch.dot(v_norm, av_norm).item())))
        return sum(sims) / len(sims)

    def compute_threshold(self, directional_congestion: float) -> float:
        return 0.30 + directional_congestion * 3.0

    def get_state(self) -> torch.Tensor:
        return self.state

    def get_normalised_state(self) -> torch.Tensor:
        if self.state.dim() == 1:
            if self.state.norm() < 1e-8:
                return self.state
            return self.state / (self.state.norm() + 1e-8)
        # [B, D] per-sample normalisation (H2): each sample's field uses its own
        # norm, not the global Frobenius norm of the whole batch.
        norms = self.state.norm(dim=-1, keepdim=True)
        return self.state / (norms + 1e-8)

    def write_history(self, neuron_id: str) -> List[torch.Tensor]:
        return self._write_history.get(neuron_id, [])

    def clear_history(self) -> None:
        self._write_history.clear()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_writes={self.n_active}"
