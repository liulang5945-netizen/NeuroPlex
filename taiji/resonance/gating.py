"""Resonance gating mechanisms — confidence gate, early stop, trigger conditions.

Experiment 12 revealed that 1+1>2 is not the default. Resonance only helps when:
1. Prediction uncertainty is high (top-k probability distribution is flat)
2. Multiple neurons have complementary knowledge
3. There is sufficient room for improvement (not near-perfect predictions)

This module implements three mechanisms discovered in Experiment 12:
- ConfidenceGate: skip resonance when the prediction is already confident
- EarlyStopResonance: stop iterating when logits converge
- Trigger conditions: only resonate when all three conditions are met
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ConfidenceGate:
    """Confidence gate: only resonate when prediction uncertainty is high.

    If the model is already very confident about its prediction (max_prob > threshold),
    multi-round resonance would only introduce field noise and degrade the output.
    In that case, skip resonance entirely and return the single-neuron output directly.

    Position: called at the start of ResonanceEnsemble.forward(), after round 1.
    """

    def __init__(self, threshold: float = 0.9):
        """Args:
            threshold: top-1 probability threshold. If max_prob > threshold,
                       resonance is skipped (prediction is already confident).
        """
        self.threshold = threshold

    def should_resonate(self, logits: torch.Tensor) -> bool:
        """Check whether resonance should be activated for this prediction.

        Args:
            logits: [B, L, vocab] or [L, vocab] from the best-matching neuron.

        Returns:
            True = need resonance (prediction is uncertain)
            False = skip resonance (prediction is already confident enough)
        """
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean()  # average top-1 probability
        return float(max_prob.detach()) < self.threshold

    def get_confidence(self, logits: torch.Tensor) -> float:
        """Return the confidence score (max_prob) for logging/analysis."""
        probs = torch.softmax(logits, dim=-1)
        return float(probs.max(dim=-1).values.mean().detach())


class EarlyStopResonance:
    """Early stop: stop multi-round resonance when logits converge.

    Without early stopping, the ensemble always runs max_rounds iterations.
    Experiment 12 showed that for near-perfect predictions, additional rounds
    introduce noise (PPL went from 15.66 to 19.88, +27%).

    Position: called after each round in ResonanceEnsemble.forward().
    """

    def __init__(self, threshold: float = 1e-3, min_rounds: int = 2):
        """Args:
            threshold: relative L2-norm difference between consecutive logits.
                       If diff < threshold, logits have converged.
            min_rounds: minimum number of resonance rounds before early stop
                        is allowed (prevents premature termination).
        """
        self.threshold = threshold
        self.min_rounds = min_rounds

    def should_stop(self, logits_history: list[torch.Tensor]) -> bool:
        """Check whether logits have converged across rounds.

        Args:
            logits_history: list of weighted_logits from each round.
                            Most recent is at the end.

        Returns:
            True = logits have converged, stop iterating.
            False = continue resonance.
        """
        if len(logits_history) < self.min_rounds:
            return False

        # Compare the last two rounds
        current = logits_history[-1]
        previous = logits_history[-2]
        diff = torch.norm(current - previous) / (torch.norm(current) + 1e-8)
        return float(diff) < self.threshold

    def get_convergence_rate(self, logits_history: list[torch.Tensor]) -> float:
        """Return the convergence rate for analysis (lower = more converged)."""
        if len(logits_history) < 2:
            return float("inf")
        current = logits_history[-1]
        previous = logits_history[-2]
        return float(
            torch.norm(current - previous) / (torch.norm(current) + 1e-8)
        )


class ResonanceTrigger:
    """Full trigger conditions for resonance activation.

    Combines the three conditions from Experiment 12:
    1. Prediction uncertainty is high (flat top-k distribution)
    2. Multiple neurons have complementary knowledge (diverse field vectors)
    3. Sufficient error room (not near-perfect prediction)

    Position: called at the start of ResonanceEnsemble.forward(), after round 1.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        diversity_threshold: float = 0.3,
        min_active_neurons: int = 2,
    ):
        self.confidence_gate = ConfidenceGate(threshold=confidence_threshold)
        self.diversity_threshold = diversity_threshold  # min pairwise cosine distance
        self.min_active_neurons = min_active_neurons

    def should_resonate(
        self,
        logits: torch.Tensor,
        field_vectors: dict[str, torch.Tensor],
    ) -> tuple[bool, str]:
        """Full trigger check.

        Args:
            logits: logits from the best-matching neuron (round 1).
            field_vectors: {neuron_id: field_vector} from round 1.

        Returns:
            (should_resonate, reason) tuple.
            should_resonate: True if resonance should be activated.
            reason: human-readable explanation for the decision.
        """
        # Condition 1: Prediction uncertainty
        if not self.confidence_gate.should_resonate(logits):
            confidence = self.confidence_gate.get_confidence(logits)
            return False, f"confident prediction (max_prob={confidence:.3f})"

        # Condition 2: Multiple neurons with diverse knowledge
        n_active = len(field_vectors)
        if n_active < self.min_active_neurons:
            return False, f"too few active neurons ({n_active} < {self.min_active_neurons})"

        # Check diversity of field vectors
        vecs = list(field_vectors.values())
        if len(vecs) >= 2:
            # Compute average pairwise cosine similarity
            similarities = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    v_i = vecs[i] / (vecs[i].norm() + 1e-8)
                    v_j = vecs[j] / (vecs[j].norm() + 1e-8)
                    sim = float(torch.dot(v_i.squeeze(), v_j.squeeze()))
                    similarities.append(sim)
            avg_sim = sum(similarities) / len(similarities)
            avg_dist = 1 - avg_sim  # cosine distance
            if avg_dist < self.diversity_threshold:
                return False, f"insufficient neuron diversity (cos_dist={avg_dist:.3f})"

        # Condition 3: Sufficient error room (not near-perfect)
        # If all neurons agree closely, there's not enough room for resonance to help.
        # This is implicitly checked by Condition 1 (confidence).

        return True, "triggered: uncertain + diverse + has error room"
