"""验证 contrastive phase 修复：三信号（route/proto/align）不再趋零。

直接调用 _train_contrastive_phase 多轮，观察 loss 是否产生有效梯度信号。
对比修复前后：原版三信号≈0（死代码），修复版应产生非零且下降的信号。
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

from taiji.loader import assemble_cortex
from taiji.life.sleep_engine import get_sleep_engine


def main():
    print("=" * 60)
    print("Contrastive phase 修复验证 — 三信号非零检查")
    print("=" * 60)

    print("\n[1] 装配 Cortex...")
    cortex, _, _ = assemble_cortex()
    print(f"  Neurons: {list(cortex.neurons.keys())}")

    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex

    print("\n[2] 连续运行 contrastive phase 8 轮...")
    print("  （观察 route/proto/align 是否非零且 proto 下降）")
    for i in range(8):
        loss = sleep_engine._train_contrastive_phase(cortex)
        if loss is None:
            print(f"  Round {i+1}: SKIPPED (None)")
        else:
            print(f"  Round {i+1}: total_contrastive={loss:.4f}")

    # 检查 domain_prototype 是否被 EMA 更新（非零）
    print("\n[3] 检查 domain_prototype 是否被激活（非零）...")
    for nid, neuron in cortex.neurons.items():
        proto = neuron.domain_prototype
        norm = proto.norm().item()
        status = "✓ 已激活" if norm > 1e-4 else "✗ 仍为零（死代码）"
        print(f"  {nid}: prototype norm={norm:.4f} {status}")

    print("\n[4] 验证 L2 指纹路由是否产生区分度...")
    import torch
    test_prompts = {
        "zh": "神经元共振场架构",
        "en": "neural resonance architecture",
        "code": "def resonance(): pass",
        "math": "integral of sin(x)",
    }
    for domain, prompt in test_prompts.items():
        general_ids = cortex._general_sp.encode(prompt) or [0]
        active = cortex._fingerprint_route(general_ids, top_k=2)
        print(f"  [{domain}] '{prompt[:25]}' → route: {active}")

    print("\n" + "=" * 60)
    print("验证完成：若 route/proto/align 均 > 0 且 prototype 已激活，")
    print("则机械塞入的死代码已被修复为有效融合信号。")


if __name__ == "__main__":
    main()
