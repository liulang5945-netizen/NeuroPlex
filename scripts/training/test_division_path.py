"""Phase 3: Division-of-labor path experiment.

Validates the three strategies from the roadmap:
- Strategy A: Scale Layering (expert decisions → standard executes → compact assists)
- Strategy B: Cluster Dominance (best-fit cluster leads)
- Combined: DivisionPath (cluster dominance × internal scale layering)

Compares against the default softmax weighting (consensus path).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance import (
    ResonanceField, ResonanceEnsemble,
    NeuronConfig, COMPACT, STANDARD, EXPERT, TINY_TEST,
    ScaleLayering, ClusterDominance, DivisionPath,
)


def create_test_neurons():
    """Create 3 neurons of different sizes for scale layering test."""
    neurons = {}
    specs = {}

    for spec_name, cfg in [("compact", COMPACT), ("standard", STANDARD), ("expert", EXPERT)]:
        cfg_copy = NeuronConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            spec=spec_name,
        )
        cfg_copy.field_dim = 256  # Small field for testing
        cfg_copy.vocab_size = 256000
        cfg_copy.base_embed_dim = 512

        from taiji.resonance.neuron import ResonanceNeuron
        neuron = ResonanceNeuron(cfg_copy)
        neurons[spec_name] = neuron
        specs[spec_name] = spec_name

    return neurons, specs


def test_scale_layering():
    """Test A: Scale layering gives expert higher weight."""
    print("=" * 60)
    print("Test A: Scale Layering")
    print("=" * 60)

    sl = ScaleLayering()
    specs = {"expert": "expert", "standard": "standard", "compact": "compact"}
    scores = {"expert": 0.5, "standard": 0.5, "compact": 0.5}  # equal resonance scores

    weights = sl.compute_weights(specs, scores)
    print(f"  Weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Expert should have highest weight
    assert weights["expert"] > weights["standard"] > weights["compact"], \
        f"Scale layering failed: {weights}"

    # Test when scores differ
    scores2 = {"expert": 0.2, "standard": 0.8, "compact": 0.5}
    weights2 = sl.compute_weights(specs, scores2)
    print(f"  Weights (varied scores): { {k: f'{v:.3f}' for k, v in weights2.items()} }")

    # Standard should be highest now (high score × mid weight)
    assert weights2["standard"] > weights2["compact"], \
        f"Score influence failed: {weights2}"

    print("  PASS\n")
    return True


def test_cluster_dominance():
    """Test B: Cluster dominance picks the best-fit cluster."""
    print("=" * 60)
    print("Test B: Cluster Dominance")
    print("=" * 60)

    cd = ClusterDominance()

    # Create two clusters with different "directions"
    input_vec = torch.tensor([1.0, 0.0, 0.0])  # pointing to x-axis
    cluster_a = {
        "a1": torch.tensor([0.9, 0.1, 0.0]),
        "a2": torch.tensor([0.8, 0.2, 0.0]),
    }
    cluster_b = {
        "b1": torch.tensor([0.0, 0.9, 0.1]),
        "b2": torch.tensor([0.1, 0.8, 0.1]),
    }
    clusters = {"code": cluster_a, "math": cluster_b}

    dominant, weights = cd.compute_cluster_weights(input_vec, clusters)
    print(f"  Dominant cluster: {dominant}")
    print(f"  Cluster weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Cluster A (code) should dominate because it's closer to input
    assert dominant == "code", f"Wrong dominant cluster: {dominant}"
    assert weights["code"] > weights["math"], \
        f"Dominant cluster should have higher weight: {weights}"

    print("  PASS\n")
    return True


def test_division_path():
    """Test C: Combined division-of-labor."""
    print("=" * 60)
    print("Test C: DivisionPath (combined)")
    print("=" * 60)

    dp = DivisionPath()
    input_vec = torch.tensor([1.0, 0.0, 0.0])

    # Two clusters: code (with expert+standard) and math (with compact only)
    clusters = {
        "code": {
            "code_expert": torch.tensor([0.9, 0.1, 0.0]),
            "code_standard": torch.tensor([0.7, 0.3, 0.0]),
        },
        "math": {
            "math_compact": torch.tensor([0.0, 0.8, 0.2]),
        },
    }
    specs = {
        "code_expert": "expert",
        "code_standard": "standard",
        "math_compact": "compact",
    }
    scores = {
        "code_expert": 0.6,
        "code_standard": 0.5,
        "math_compact": 0.3,
    }

    weights = dp.compute_final_weights(input_vec, clusters, specs, scores)
    print(f"  Final weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Code cluster should dominate, and within it, expert > standard
    assert weights["code_expert"] > weights["code_standard"], \
        f"Expert should lead within cluster: {weights}"
    assert weights["code_expert"] > weights["math_compact"], \
        f"Dominant cluster neuron should outrank others: {weights}"
    assert abs(sum(weights.values()) - 1.0) < 0.01, \
        f"Weights should sum to 1: {weights}"

    print("  PASS\n")
    return True


def test_ensemble_with_division():
    """Test D: Ensemble with DivisionPath integration."""
    print("=" * 60)
    print("Test D: Ensemble + DivisionPath")
    print("=" * 60)

    # Create tiny neurons for quick test
    from taiji.resonance.neuron import ResonanceNeuron

    neurons = {}
    specs = {}
    for name, spec in [("n_expert", "expert"), ("n_standard", "standard"), ("n_compact", "compact")]:
        cfg = NeuronConfig(
            hidden_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=512, field_dim=256,
            vocab_size=256000, base_embed_dim=512, spec=spec,
        )
        neurons[name] = ResonanceNeuron(cfg)
        specs[name] = spec

    field = ResonanceField(dim=256)
    dp = DivisionPath()

    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2, division_path=dp,
    )

    # Store specs for division path (ensemble uses neuron.config.spec)
    for nid, neuron in neurons.items():
        neuron.config.spec = specs[nid]

    dummy_emb = torch.randn(1, 8, 512)
    result = ensemble.forward(dummy_emb, return_logits=True)

    print(f"  n_rounds: {result['n_rounds']}")
    if "final_weights" in result:
        print(f"  Final weights: { {k: f'{v:.3f}' for k, v in result['final_weights'].items()} }")
    if result.get("weighted_logits") is not None:
        print(f"  weighted_logits shape: {result['weighted_logits'].shape}")
    print("  PASS\n")
    return True


if __name__ == "__main__":
    print("Phase 3: Division-of-Labor Path Experiment")
    print("=" * 60)
    print()

    results = {}
    results["scale_layering"] = test_scale_layering()
    results["cluster_dominance"] = test_cluster_dominance()
    results["division_path"] = test_division_path()
    results["ensemble_integration"] = test_ensemble_with_division()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
