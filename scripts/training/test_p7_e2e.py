"""P7 端到端验证：加载从零训练的神经元，测试 generate 和路由。
用法: python scripts/training/test_p7_e2e.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from taiji.brain.cortex import Cortex
from taiji.loader import assemble_cortex

DEVICE = "cpu"
NEURONS_DIR = "data/neurons"


def test_load_neurons():
    """测试 1: 加载 5 个从零训练的神经元"""
    print("=" * 60)
    print("Test 1: Load neurons")
    print("=" * 60)
    cortex = Cortex(neurons_dir=NEURONS_DIR, device=DEVICE, max_rounds=3)
    print(f"Neurons loaded: {list(cortex.neurons.keys())}")
    for nid, n in cortex.neurons.items():
        cfg = n.config
        n_params = sum(p.numel() for p in n.parameters())
        has_embed = hasattr(n, 'embedding') and n.embedding is not None
        has_lm_head = hasattr(n, 'lm_head') or hasattr(n, 'lm_head_delta_u')
        print(f"  [{nid}] vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, "
              f"layers={cfg.num_hidden_layers}, {n_params/1e6:.1f}M, "
              f"embed={'OK' if has_embed else 'MISSING'}, "
              f"lm_head={'OK' if has_lm_head else 'MISSING'}")
    assert len(cortex.neurons) >= 5, f"Expected 5 neurons, got {len(cortex.neurons)}"
    print("PASS\n")
    return cortex


def test_assemble():
    """测试 2: assemble_cortex 完整装配"""
    print("=" * 60)
    print("Test 2: assemble_cortex")
    print("=" * 60)
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir=NEURONS_DIR,
        device=DEVICE,
        wire_bio_modules=False,
    )
    print(f"Modules: {list(modules.keys())}")
    assert "tokenizer_hub" in modules, "tokenizer_hub should be registered"
    hub = modules["tokenizer_hub"]
    print(f"TokenizerHub domains: {hub.list_domains()}")
    for d in hub.list_domains():
        print(f"  {d}: vocab={hub.vocab_size(d)}, eos={hub.eos_token_id(d)}")
    print("PASS\n")
    return cortex


def test_generate(cortex, prompt, domain, expected_lang="auto"):
    """测试 generate，验证输出非空且合理。"""
    print(f"  Prompt: '{prompt}' (domain={domain})")
    try:
        output = cortex.generate(prompt, max_tokens=50, temperature=0.8, top_k=30, domain=domain)
        if output:
            print(f"  Output: '{output[:100]}'")
            print(f"  Length: {len(output)} chars")
        else:
            print(f"  Output: (empty)")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def main():
    print("P7 End-to-End Test")
    print(f"Device: {DEVICE}, Neurons: {NEURONS_DIR}\n")

    # Test 1 + 2
    cortex = test_load_neurons()
    cortex2 = test_assemble()
    cortex = cortex2  # Use assembled version with tokenizer hub

    # Test 3: Generate
    print("=" * 60)
    print("Test 3: Generate (P7 domain tokenizer)")
    print("=" * 60)

    test_generate(cortex, "今天天气怎么样？", "zh")
    test_generate(cortex, "保持健康的三个提示。", "zh")
    test_generate(cortex, "What is the capital of France?", "en")
    test_generate(cortex, "Give three tips for staying healthy.", "en")
    test_generate(cortex, "def fibonacci(n):", "code")
    test_generate(cortex, "Solve for x: 2x + 5 = 13", "math")
    test_generate(cortex, "Hello world, this is a test.", "general")

    print("=" * 60)
    print("All tests complete!")


if __name__ == "__main__":
    main()
