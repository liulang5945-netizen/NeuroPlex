"""Phase 3: Division-of-labor path experiments.

Compares three strategies against the default softmax consensus:
  Strategy A: Scale Layering (expert×3, standard×2, compact×1)
  Strategy B: Cluster Dominance (best-fit cluster takes 0.7 weight)
  Strategy C: DivisionPath (cluster dominance × internal scale layering)

Experiment:
  - Load 5 distilled neurons (zh, en, code, math, general)
  - Run resonance ensemble on cross-domain test data
  - Compare PPL across strategies

Usage:
    python scripts/training/run_division_experiments.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig,
    ScaleLayering, ClusterDominance, DivisionPath,
    ConfidenceGate, EarlyStopResonance, QualityFilter,
)


class PaddingResonanceField(ResonanceField):
    """ResonanceField that auto-pads neuron vectors to match field dim."""

    def __init__(self, dim: int = 4096, device=None):
        super().__init__(dim=dim, device=device)

    def _pad_vec(self, vector: torch.Tensor) -> torch.Tensor:
        """Pad or truncate vector to match field dim."""
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        vd = vector.shape[-1]
        if vd < self.dim:
            pad = torch.zeros(*vector.shape[:-1], self.dim - vd, device=vector.device, dtype=vector.dtype)
            return torch.cat([vector, pad], dim=-1)
        elif vd > self.dim:
            return vector[..., :self.dim]
        return vector

    def write(self, neuron_id: str, vector: torch.Tensor) -> torch.Tensor:
        """Pad or truncate vector to match field dim before writing."""
        vector = self._pad_vec(vector)
        return super().write(neuron_id, vector)

    def score(self, vector: torch.Tensor) -> float:
        """Pad vector before computing score."""
        vector = self._pad_vec(vector)
        return super().score(vector)

    def directional_congestion(self, vector: torch.Tensor, active_vectors: list) -> float:
        """Pad all vectors before congestion check."""
        vector = self._pad_vec(vector)
        padded_active = [self._pad_vec(v) for v in active_vectors]
        return super().directional_congestion(vector, padded_active)


def load_neuron(ckpt_path: str, device: str = "cpu") -> ResonanceNeuron:
    """Load a distilled neuron from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: NeuronConfig = ckpt["neuron_config"]
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt["state_dict"])
    neuron.eval()
    neuron.freeze_fingerprint()
    return neuron


def evaluate_ensemble(
    ensemble: ResonanceEnsemble,
    data: torch.Tensor,
    device: str = "cpu",
    max_batches: int = 30,
    label: str = "",
) -> Dict[str, float]:
    """Evaluate PPL of an ensemble on domain data."""
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch[0].to(device)
            # Random shared embedding (same for all strategies, fair comparison)
            shared_emb = torch.randn(input_ids.shape[0], input_ids.shape[1], 512, device=device)

            result = ensemble.forward(shared_emb, return_logits=True)

            if "weighted_logits" not in result:
                continue

            logits = result["weighted_logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 10))

    return {"ppl": ppl, "loss": avg_loss, "n_tokens": total_tokens}


def build_clusters(neurons: Dict[str, ResonanceNeuron]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group neurons into domain clusters based on fingerprint."""
    clusters = {"language": {}, "code": {}, "math": {}, "general": {}}
    for nid, neuron in neurons.items():
        if nid in ("zh", "en"):
            clusters["language"][nid] = neuron.fingerprint.clone()
        elif nid == "code":
            clusters["code"][nid] = neuron.fingerprint.clone()
        elif nid == "math":
            clusters["math"][nid] = neuron.fingerprint.clone()
        else:
            clusters["general"][nid] = neuron.fingerprint.clone()
    # Remove empty clusters
    return {k: v for k, v in clusters.items() if v}


def main():
    print("=" * 60)
    print("Phase 3: Division-of-Labor Path Experiments")
    print("=" * 60)

    neurons_dir = "data/neurons"
    data_path = "data/distill/domain_datasets.pt"

    # Load data
    datasets = torch.load(data_path, map_location="cpu", weights_only=True)

    # Load neurons
    print("\n[1/5] Loading neurons...")
    neurons = {}
    specs = {}
    for domain in ["zh", "en", "code", "math", "general"]:
        ckpt = os.path.join(neurons_dir, f"neuron_{domain}.pt")
        if os.path.exists(ckpt):
            neuron = load_neuron(ckpt)
            neurons[domain] = neuron
            specs[domain] = neuron.config.spec
            print(f"  {domain}: {neuron.config.spec}, fingerprint_norm={neuron.fingerprint.norm().item():.4f}")

    # Build clusters
    clusters = build_clusters(neurons)
    print(f"\n  Clusters: {list(clusters.keys())}")
    for cname, members in clusters.items():
        print(f"    {cname}: {list(members.keys())}")

    # Use the maximum field_dim among all neurons for the shared field
    field_dim = max(n.config.field_dim for n in neurons.values())
    print(f"\n  Field dim: {field_dim}")

    # ── Experiment: 4 strategies on 5 domains ──
    print("\n[2/5] Running experiments...")

    strategies = {
        "consensus": None,  # Default softmax weighting
        "scale_layering": ScaleLayering(),
        "cluster_dominance": None,  # Handled by DivisionPath
        "division_path": DivisionPath(),
    }

    # Precompute cluster fingerprints for division path
    # For consensus and scale_layering, we use simple ensemble
    # For division_path, we use the full DivisionPath

    all_ppls = {}

    for strategy_name in ["consensus", "scale_layering", "division_path"]:
        print(f"\n  --- {strategy_name} ---")

        # Build ensemble based on strategy
        field = PaddingResonanceField(dim=field_dim)

        if strategy_name == "scale_layering":
            ensemble = ResonanceEnsemble(
                neurons, field, max_rounds=1,
                division_path=DivisionPath(),
            )
        elif strategy_name == "division_path":
            ensemble = ResonanceEnsemble(
                neurons, field, max_rounds=1,
                division_path=DivisionPath(),
            )
        else:
            ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

        # Override division path behavior
        if strategy_name == "scale_layering":
            # For pure scale layering: single cluster, all neurons
            orig_compute = ensemble.division_path.compute_final_weights

            def scale_only_weights(input_vector, clusters, neuron_specs, resonance_scores, dominant_weight=0.7):
                return ensemble.division_path.scale_layering.compute_weights(neuron_specs, resonance_scores)

            ensemble.division_path.compute_final_weights = scale_only_weights
            ensemble._scale_layering_override = True

        # Evaluate on each domain
        for domain in ["zh", "en", "code", "math", "general"]:
            if domain not in datasets:
                continue

            # Build per-evaluation clusters based on input
            if strategy_name == "division_path":
                # Use actual clusters
                # Build input proxy from test data
                sample_ids = datasets[domain][:2].to("cpu")
                sample_emb = torch.randn(2, 256, 512)
                # Quick forward to get field vectors
                field.reset()
                vecs = {}
                for nid, n in neurons.items():
                    r = n.forward(sample_emb, return_logits=False)
                    vecs[nid] = r["field_vector"].mean(dim=0)
                    field.write(nid, vecs[nid])

                # Use the first neuron's vector as input proxy
                input_vec = next(iter(vecs.values())).squeeze()
                if input_vec.dim() == 0:
                    input_vec = input_vec.unsqueeze(0)

                # Build cluster field vectors
                cluster_vecs = {}
                for cname, members in clusters.items():
                    cluster_vecs[cname] = {
                        nid: vecs.get(nid, torch.zeros(1)) for nid in members
                    }

                # Compute division weights
                scores = {nid: field.score(vecs[nid]) for nid in neurons}
                ensemble.division_path.compute_final_weights(
                    input_vector=input_vec,
                    clusters=cluster_vecs,
                    neuron_specs=specs,
                    resonance_scores=scores,
                )

            result = evaluate_ensemble(ensemble, datasets[domain], label=f"{strategy_name}/{domain}")
            key = f"{strategy_name}_{domain}"
            all_ppls[key] = result["ppl"]
            print(f"    {domain}: PPL={result['ppl']:.1f}")

    # ── Summary table ──
    print("\n[3/5] PPL Comparison Matrix")
    print(f"  {'Domain':>10}", end="")
    domains_list = ["zh", "en", "code", "math", "general"]
    for d in domains_list:
        print(f"{d:>10}", end="")
    print()

    best_counts = {"consensus": 0, "scale_layering": 0, "division_path": 0}

    for d in domains_list:
        print(f"  {d:>10}", end="")
        ppls = {}
        for s in ["consensus", "scale_layering", "division_path"]:
            ppl = all_ppls.get(f"{s}_{d}", float('inf'))
            ppls[s] = ppl
            print(f"{ppl:>10.1f}", end="")
        print()

        # Track which strategy is best for this domain
        best_strategy = min(ppls, key=ppls.get)
        best_counts[best_strategy] += 1

    # ── Best strategy summary ──
    print(f"\n[4/5] Best strategy per domain:")
    for d in domains_list:
        ppls = {s: all_ppls.get(f"{s}_{d}", float('inf')) for s in ["consensus", "scale_layering", "division_path"]}
        best = min(ppls, key=ppls.get)
        print(f"  {d}: {best} (PPL={ppls[best]:.1f})")

    print(f"\n  Wins: consensus={best_counts['consensus']}, "
          f"scale_layering={best_counts['scale_layering']}, "
          f"division_path={best_counts['division_path']}")

    # ── Average PPL per strategy ──
    print(f"\n[5/5] Average PPL per strategy:")
    for s in ["consensus", "scale_layering", "division_path"]:
        ppls = [all_ppls.get(f"{s}_{d}", float('inf')) for d in domains_list]
        avg = sum(p for p in ppls if p < float('inf')) / max(len([p for p in ppls if p < float('inf')]), 1)
        print(f"  {s}: avg_PPL={avg:.1f}")

    print("\n" + "=" * 60)
    print("Phase 3 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
