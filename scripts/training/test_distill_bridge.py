"""Phase 2 verification: test that the 1.5B checkpoint can be loaded.

Verifies:
1. Checkpoint bridge can load the teacher model
2. Hidden states can be extracted
3. A tiny ResonanceNeuron can be created and distill one step
4. Gating mechanisms work end-to-end

Run: python scripts/training/test_distill_bridge.py
"""

import os
import sys

# Add taiji-neuron to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    NeuronConfig, TINY_TEST, ConfidenceGate, EarlyStopResonance,
    QualityFilter,
)


def test_load_checkpoint():
    """Test 1: Can we load the 1.5B checkpoint?"""
    print("=" * 60)
    print("Test 1: Load 1.5B checkpoint")
    print("=" * 60)

    checkpoint = "e:/taiji/checkpoint-400000"
    if not os.path.exists(checkpoint):
        print("  SKIP: checkpoint-400000 not found at e:/taiji/checkpoint-400000")
        return None, None

    try:
        model, embedding = load_teacher_model(checkpoint, device="cpu")
        print("  PASS: Teacher model loaded successfully")
        return model, embedding
    except Exception as e:
        print(f"  FAIL: {e}")
        return None, None


def test_extract_hidden_states(model):
    """Test 2: Can we extract hidden states?"""
    print("\n" + "=" * 60)
    print("Test 2: Extract hidden states")
    print("=" * 60)

    if model is None:
        print("  SKIP: No teacher model")
        return False

    try:
        dummy_ids = torch.randint(0, 256000, (1, 16))
        hidden = extract_hidden_states(model, dummy_ids)
        print(f"  PASS: Hidden states extracted, shape={hidden.shape}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_create_neuron():
    """Test 3: Create a tiny ResonanceNeuron."""
    print("\n" + "=" * 60)
    print("Test 3: Create ResonanceNeuron (tiny)")
    print("=" * 60)

    try:
        cfg = TINY_TEST
        cfg.vocab_size = 256000
        neuron = ResonanceNeuron(cfg)
        param_count = sum(p.numel() for p in neuron.parameters())
        print(f"  PASS: Neuron created, params={param_count:,}")
        return neuron
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return None


def test_forward_pass(neuron):
    """Test 4: Forward pass through neuron."""
    print("\n" + "=" * 60)
    print("Test 4: Neuron forward pass")
    print("=" * 60)

    if neuron is None:
        print("  SKIP: No neuron")
        return False

    try:
        dummy_emb = torch.randn(1, 8, 512)  # [B, L, base_embed_dim]
        result = neuron.forward(dummy_emb, return_logits=True)
        print(f"  field_vector: {result['field_vector'].shape}")
        print(f"  logits: {result['logits'].shape}")
        print("  PASS: Forward pass OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_ensemble_with_gating(neuron):
    """Test 5: Ensemble with gating mechanisms."""
    print("\n" + "=" * 60)
    print("Test 5: ResonanceEnsemble with gating")
    print("=" * 60)

    if neuron is None:
        print("  SKIP: No neuron")
        return False

    try:
        field = ResonanceField(dim=512)
        neurons = {"test_neuron": neuron}

        ensemble = ResonanceEnsemble(
            neurons, field,
            max_rounds=2,
            confidence_gate=ConfidenceGate(threshold=0.9),
            early_stop=EarlyStopResonance(),
        )

        dummy_emb = torch.randn(1, 8, 512)
        result = ensemble.forward(dummy_emb, return_logits=True, enable_gating=True)

        print(f"  n_rounds: {result['n_rounds']}")
        print(f"  skipped_resonance: {result.get('skipped_resonance', False)}")
        if result.get("weighted_logits") is not None:
            print(f"  weighted_logits shape: {result['weighted_logits'].shape}")
        print("  PASS: Ensemble with gating OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_quality_filter(neuron):
    """Test 6: Quality filter integration."""
    print("\n" + "=" * 60)
    print("Test 6: QualityFilter integration")
    print("=" * 60)

    if neuron is None:
        print("  SKIP: No neuron")
        return False

    try:
        field = ResonanceField(dim=512)
        neurons = {"n1": neuron}
        qf = QualityFilter(ppl_threshold=100)
        qf.set_ppls({"n1": 25.0})  # neuron is good quality

        ensemble = ResonanceEnsemble(neurons, field, quality_filter=qf)
        dummy_emb = torch.randn(1, 8, 512)
        result = ensemble.forward(dummy_emb, return_logits=True)

        print(f"  n_rounds: {result['n_rounds']}")
        print("  PASS: QualityFilter integration OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Phase 2: Distillation Bridge Verification")
    print("=" * 60)

    results = {}

    # Test 1: Load checkpoint
    model, embedding = test_load_checkpoint()
    results["load_checkpoint"] = model is not None

    # Test 2: Extract hidden states
    results["extract_hidden"] = test_extract_hidden_states(model)

    # Test 3: Create neuron
    neuron = test_create_neuron()
    results["create_neuron"] = neuron is not None

    # Test 4: Forward pass
    results["forward_pass"] = test_forward_pass(neuron)

    # Test 5: Ensemble with gating
    results["ensemble_gating"] = test_ensemble_with_gating(neuron)

    # Test 6: Quality filter
    results["quality_filter"] = test_quality_filter(neuron)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
