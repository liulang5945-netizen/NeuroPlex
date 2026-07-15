"""Resonance field core — shared communication medium for all neurons.

The field is a D-dimensional vector space where neurons write
L2-normalised vectors and read the accumulated state.  It is the
"neural language" of the architecture — completely independent of
the tokenizer (Level 0) and the per-neuron concept spaces (Level 1).

Key properties:
- All writes are L2-normalised → neuron size does not determine "loudness"
- Cosine similarity between a neuron's write and the field state is the
  natural scoring mechanism (resonance = alignment with the collective)
- W_cond learns over time which patterns of co-activation produce good outputs
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ResonanceField(nn.Module):
    """Shared resonance field — the "neural language" of the architecture.

    Dimensions:
        D_field = hidden_size(standard) * N_standard / compression_ratio
        For 5 standard neurons (hidden=768): 768*5/6 ≈ 640 → use 1024
        For 30+ neurons: 768*30/6 = 3840 → use 4096

    The field dimension should be large enough to hold N_active independent
    directions without excessive overlap, but not so large that the field
    becomes too sparse for resonance to occur.
    """

    def __init__(self, dim: int = 4096, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self._device = device or torch.device("cpu")

        # Field state: accumulated write vectors (reset each inference)
        self.register_buffer("state", torch.zeros(dim))

        # Conditioning matrix: learns which neuron combinations work well
        self.W_cond = nn.Parameter(torch.randn(dim, dim) * 0.02)

        # Per-neuron write history (for debugging and fingerprinting)
        self._write_history: Dict[str, List[torch.Tensor]] = {}

        # Resonance scores from the most recent round
        self.scores: Dict[str, float] = {}

        # Crowding tracking
        self.n_active: int = 0

    def reset(self) -> None:
        """Reset the field state for a new inference."""
        self.state.zero_()
        self.scores.clear()
        self.n_active = 0

    def write(self, neuron_id: str, vector: torch.Tensor) -> torch.Tensor:
        """Write a neuron's field vector (L2-normalised) into the field.

        Args:
            neuron_id: stable identifier for the writing neuron
            vector: [B, D] or [D] tensor

        Returns:
            The normalised vector that was written
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        # L2 normalise — all neurons have equal "loudness"
        v_norm = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)

        # Add to field state
        self.state = self.state + v_norm.squeeze(0) if v_norm.shape[0] == 1 else self.state + v_norm.sum(dim=0)

        # Track
        self.n_active += 1
        if neuron_id not in self._write_history:
            self._write_history[neuron_id] = []
        self._write_history[neuron_id].append(v_norm.squeeze(0).detach())

        return v_norm

    def score(self, vector: torch.Tensor) -> float:
        """Compute the resonance score (cosine similarity with current field state).

        Score ∈ [-1, 1].  Higher = more aligned with the collective.
        If vector is [B, D], averages over the batch dimension.
        """
        if vector.dim() == 2:
            vector = vector.mean(dim=0)  # [B, D] -> [D]
        v_norm = vector / (vector.norm() + 1e-8)
        f_norm = self.state / (self.state.norm() + 1e-8)
        return float(torch.dot(v_norm, f_norm).item())

    def directional_congestion(self, vector: torch.Tensor, active_vectors: List[torch.Tensor]) -> float:
        """Compute how crowded this neuron's direction is.

        Higher values mean many other neurons are writing in similar
        directions — this neuron faces a higher threshold to stay active.
        If inputs are [B, D], averages over batch before comparison.
        """
        if not active_vectors:
            return 0.0
        # Normalise to [D]
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
        """Dynamic threshold for resonance filtering.

        T_i = 0.30 + directional_congestion_i * 3.0

        Low congestion (0.1) → T ≈ 0.60  (easy to enter)
        High congestion (0.85) → T ≈ 2.85 (nearly impossible)
        """
        return 0.30 + directional_congestion * 3.0

    def get_state(self) -> torch.Tensor:
        """Return the current field state vector."""
        return self.state

    def get_normalised_state(self) -> torch.Tensor:
        """Return L2-normalised field state."""
        return self.state / (self.state.norm() + 1e-8)

    def write_history(self, neuron_id: str) -> List[torch.Tensor]:
        """Get the write history for a neuron (for fingerprinting / debugging)."""
        return self._write_history.get(neuron_id, [])

    def clear_history(self) -> None:
        """Clear write history (call periodically to avoid memory growth)."""
        self._write_history.clear()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_writes={self.n_active}"
