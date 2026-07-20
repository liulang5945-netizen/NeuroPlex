"""Division-of-labor path for resonance ensemble.

The current consensus path (weighted average of all neurons) has a
fundamental problem: weak neurons dilute strong ones. The human brain
does not work by "voting" — it divides labor across specialized regions.

Two complementary strategies:

Strategy A: Scale Layering
    Different neuron sizes take different roles:
    - Expert (118M): decision-making + quality gatekeeper
    - Standard (59M): main task execution
    - Compact (24M): auxiliary task support

Strategy B: Cluster Dominance
    The best-fit cluster takes the lead, others assist:
    - Compute cluster fit score (internal coherence * external relevance)
    - Dominant cluster gets higher weight
    - Auxiliary clusters provide supplementary knowledge

Combined: cluster dominance * internal scale layering = true division-of-labor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _to_1d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 1D (collapse batch dim if present via mean)."""
    if t.dim() > 1:
        t = t.mean(dim=0)  # [B, D] -> [D]
    return t.reshape(-1)


class ScaleLayering:
    """Strategy A: Different neuron sizes take different roles.

    Expert neurons (118M) make decisions, standard neurons (59M) execute,
    compact neurons (24M) assist. This mirrors the human brain's hierarchy:
    prefrontal cortex (decision) → motor cortex (execution).
    """

    SPEC_WEIGHTS = {
        "expert": 3.0,
        "standard": 2.0,
        "compact": 1.0,
    }

    def compute_weights(
        self,
        neuron_specs: Dict[str, str],
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute scale-layered weights for each neuron."""
        raw_weights = {}
        for nid, spec in neuron_specs.items():
            base = self.SPEC_WEIGHTS.get(spec, 1.0)
            score = scores.get(nid, 0.0)
            raw_weights[nid] = base * max(score, 0.01)

        total = sum(raw_weights.values())
        if total > 0:
            return {nid: w / total for nid, w in raw_weights.items()}
        n = len(raw_weights)
        return {nid: 1.0 / n for nid in raw_weights}

    def assign_roles(
        self, neuron_specs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Assign neurons to roles based on their specs."""
        roles = {"decision": [], "execution": [], "auxiliary": []}
        for nid, spec in neuron_specs.items():
            if spec == "expert":
                roles["decision"].append(nid)
            elif spec == "standard":
                roles["execution"].append(nid)
            else:
                roles["auxiliary"].append(nid)
        return roles


class ClusterDominance:
    """Strategy B: Best-fit cluster dominates, others assist."""

    def compute_cluster_fit(
        self,
        input_vector: torch.Tensor,
        cluster_neurons: Dict[str, torch.Tensor],
    ) -> float:
        """Compute how well a cluster fits the input.

        fit = internal_coherence * external_relevance
        """
        if not cluster_neurons:
            return 0.0

        # Ensure all vectors are 1D
        vecs = [_to_1d(v) for v in cluster_neurons.values()]
        in_vec = _to_1d(input_vector)

        # Internal coherence（矩阵化：原为 O(M) Python 列表推导 + 每次 .detach() 同步）
        if len(vecs) >= 2:
            stacked = torch.stack(vecs, dim=0)  # [M, D]
            centroid = stacked.mean(dim=0)
            centroid_norm = centroid / (centroid.norm() + 1e-8)
            stacked_norm = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-8)
            # 一次矩阵乘法得到所有成员与质心的 cosine similarity [M]
            coherence = float((stacked_norm @ centroid_norm).mean().item())
        else:
            coherence = 1.0

        # External relevance
        centroid = torch.stack(vecs).mean(dim=0)
        centroid_norm = _to_1d(centroid / (centroid.norm() + 1e-8))
        input_norm = _to_1d(in_vec / (in_vec.norm() + 1e-8))
        relevance = float(torch.dot(input_norm, centroid_norm).detach())
        relevance = max(relevance, 0.0)

        return coherence * relevance

    def compute_cluster_weights(
        self,
        input_vector: torch.Tensor,
        clusters: Dict[str, Dict[str, torch.Tensor]],
        dominant_weight: float = 0.7,
    ) -> Tuple[str, Dict[str, float]]:
        """Compute per-cluster weights."""
        fits = {}
        for cname, neurons in clusters.items():
            fits[cname] = self.compute_cluster_fit(input_vector, neurons)

        if not fits:
            return "", {}

        dominant = max(fits, key=fits.get)
        n_others = len(fits) - 1

        weights = {}
        remaining = 1.0 - dominant_weight
        others_fit_sum = sum(fits[c] for c in fits if c != dominant)

        for cname in fits:
            if cname == dominant:
                weights[cname] = dominant_weight
            elif n_others > 0 and others_fit_sum > 0:
                weights[cname] = remaining * (fits[cname] / others_fit_sum)
            else:
                weights[cname] = 0.0

        return dominant, weights


class DivisionPath:
    """Combined division-of-labor: cluster dominance * internal scale layering."""

    def __init__(self):
        self.scale_layering = ScaleLayering()
        self.cluster_dominance = ClusterDominance()

    def compute_final_weights(
        self,
        input_vector: torch.Tensor,
        clusters: Dict[str, Dict[str, torch.Tensor]],
        neuron_specs: Dict[str, str],
        resonance_scores: Dict[str, float],
        dominant_weight: float = 0.7,
    ) -> Dict[str, float]:
        """Compute final per-neuron weights."""
        # Step 1: Cluster-level weighting
        dominant, cluster_weights = self.cluster_dominance.compute_cluster_weights(
            input_vector, clusters, dominant_weight
        )

        # Step 2: Within-cluster scale layering
        final_weights = {}
        for cname, cweight in cluster_weights.items():
            cneurons = clusters.get(cname, {})
            if not cneurons:
                continue

            cspecs = {nid: neuron_specs.get(nid, "compact") for nid in cneurons}
            cscores = {nid: resonance_scores.get(nid, 0.0) for nid in cneurons}
            inner_weights = self.scale_layering.compute_weights(cspecs, cscores)

            for nid, iweight in inner_weights.items():
                final_weights[nid] = cweight * iweight

        # Re-normalize
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {nid: w / total for nid, w in final_weights.items()}

        return final_weights
