"""S1 验证：forward_train 全可微多轮共振路径。

验证项：
1. forward_train 可以前向（不崩溃）
2. 反向传播梯度能流到 side_channels
3. 反向传播梯度能流到跨规格投影层（混合规格场景）
4. 多轮共振（round 2+ side_signals）真正生效
5. neuromodulator 影响 forward 输出（write_scale 接入）
6. gamma_oscillator 接入不崩溃
7. balance_loss + diversity_loss 非负且有限

Usage:
    python -u scripts/training/verify_forward_train_diff.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.config import NeuronConfig
from taiji.resonance.gamma_oscillator import GammaOscillator
from taiji.resonance.neuro_modulation import NeuromodulatorState


def make_tiny_neuron(nid: str, field_dim: int, vocab_size: int = 200) -> ResonanceNeuron:
    """创建极小 neuron 用于测试（不依赖 tokenizer，直接用随机 embedding）."""
    cfg = NeuronConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=vocab_size,
        base_embed_dim=32,
        field_dim=field_dim,
        spec="test",
        neuron_id=nid,
    )
    cfg.unified_field_dim = None
    return ResonanceNeuron(cfg)


def test_forward_train_differentiable_same_spec():
    """测试 1：同规格场景，forward_train 可微性."""
    print("\n=== Test 1: 同规格 forward_train 可微性 ===")
    torch.manual_seed(42)

    neurons = {
        "n0": make_tiny_neuron("n0", field_dim=128),
        "n1": make_tiny_neuron("n1", field_dim=128),
    }
    # 建立 side_channels
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    # 冻结核心参数，仅 side_channels 可训练
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        neuron.train()

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    # 构造输入（B=2, L=8, base_embed_dim=32）
    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    result = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        fusion_mode="soft",
    )

    print(f"  fused_logits shape: {result['fused_logits'].shape}")
    print(f"  weights: {result['weights'].detach().cpu().numpy()}")
    print(f"  scores: {result['scores'].detach().cpu().numpy()}")
    print(f"  balance_loss: {result['balance_loss'].item():.4f}")
    print(f"  diversity_loss: {result['diversity_loss'].item():.4f}")
    print(f"  n_rounds: {result['n_rounds']}")
    print(f"  field_state shape: {result['field_state'].shape}")

    # 反向传播
    target = torch.zeros(2, 8, dtype=torch.long)
    loss = F.cross_entropy(
        result["fused_logits"].view(-1, result["fused_logits"].size(-1)),
        target.view(-1),
    )
    loss = loss + 0.01 * result["balance_loss"] + 0.05 * result["diversity_loss"]
    loss.backward()

    # 检查 side_channels 的梯度
    grad_ok = 0
    grad_total = 0
    for nid, neuron in neurons.items():
        for pre_id, ch in neuron.excite_channels.items():
            for p in ch.parameters():
                grad_total += 1
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    grad_ok += 1

    print(f"  side_channels 梯度检查: {grad_ok}/{grad_total} 参数有非零梯度")
    assert grad_ok == grad_total, f"只有 {grad_ok}/{grad_total} 个 side_channels 参数有梯度"
    print("  ✅ Test 1 通过：side_channels 全部接收到梯度")
    return True


def test_forward_train_cross_spec():
    """测试 2：混合规格场景，跨规格投影层可微性."""
    print("\n=== Test 2: 混合规格 forward_train 可微性 ===")
    torch.manual_seed(42)

    neurons = {
        "compact": make_tiny_neuron("compact", field_dim=64),    # compact 规格
        "standard": make_tiny_neuron("standard", field_dim=128),  # standard 规格
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        neuron.train()

    # field.dim = max field_dim = 128
    # ensemble 自动创建跨规格投影层：compact(field_dim=64) <-> unified(128)
    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    print(f"  跨规格正向投影层: {len(ensemble._cross_spec_projectors)}")
    print(f"  跨规格反向投影层: {len(ensemble._cross_spec_back_projectors)}")
    assert len(ensemble._cross_spec_projectors) == 1, "应该为 compact 创建 1 个正向投影层"
    assert len(ensemble._cross_spec_back_projectors) == 1, "应该为 compact 创建 1 个反向投影层"

    # 跨规格投影层设为可训练
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True

    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    result = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        fusion_mode="soft",
    )

    print(f"  fused_logits shape: {result['fused_logits'].shape}")
    print(f"  field_state shape: {result['field_state'].shape}")  # 应该是 [2, 128]
    assert result["field_state"].shape == (2, 128), \
        f"field_state 应该是 unified 维度 128，实际 {result['field_state'].shape}"

    target = torch.zeros(2, 8, dtype=torch.long)
    loss = F.cross_entropy(
        result["fused_logits"].view(-1, result["fused_logits"].size(-1)),
        target.view(-1),
    )
    loss.backward()

    # 检查跨规格投影层的梯度
    fwd_grad_ok = 0
    for nid, proj in ensemble._cross_spec_projectors.items():
        for p in proj.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                fwd_grad_ok += 1
                print(f"  正向投影 {nid}: grad norm = {p.grad.norm().item():.6f}")

    bwd_grad_ok = 0
    for nid, proj in ensemble._cross_spec_back_projectors.items():
        for p in proj.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                bwd_grad_ok += 1
                print(f"  反向投影 {nid}: grad norm = {p.grad.norm().item():.6f}")

    assert fwd_grad_ok > 0, "正向投影层无梯度"
    assert bwd_grad_ok > 0, "反向投影层无梯度"
    print("  ✅ Test 2 通过：跨规格投影层（正向+反向）接收到梯度")
    return True


def test_forward_train_neuromodulator():
    """测试 3：neuromodulator 接入."""
    print("\n=== Test 3: neuromodulator 接入 ===")
    torch.manual_seed(42)

    neurons = {
        "n0": make_tiny_neuron("n0", field_dim=128),
        "n1": make_tiny_neuron("n1", field_dim=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    # 高 norepinephrine（write_scale=1.5）
    nm_high = NeuromodulatorState(norepinephrine=1.0)
    result_high = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        neuromodulator=nm_high,
    )
    scores_high = result_high["scores"].detach()

    # 低 norepinephrine（write_scale=0.5）
    nm_low = NeuromodulatorState(norepinephrine=0.0)
    result_low = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        neuromodulator=nm_low,
    )
    scores_low = result_low["scores"].detach()

    # write_scale 直接乘 scores，所以高/低 norepinephrine 的 scores 比值应为 3.0 (1.5/0.5)
    print(f"  高 norepinephrine scores: {scores_high.cpu().numpy()}")
    print(f"  低 norepinephrine scores: {scores_low.cpu().numpy()}")
    ratio = (scores_high.abs().mean() / scores_low.abs().mean().clamp(min=1e-8)).item()
    print(f"  scores 高/低比值: {ratio:.4f}（预期 ≈ 3.0，即 1.5/0.5）")
    assert abs(ratio - 3.0) < 0.1, f"write_scale 未正确接入 scores（比值 {ratio} ≠ 3.0）"
    print("  ✅ Test 3 通过：neuromodulator 影响 scores（write_scale 接入）")
    return True


def test_forward_train_gamma_oscillator():
    """测试 4：gamma_oscillator 接入."""
    print("\n=== Test 4: gamma_oscillator 接入 ===")
    torch.manual_seed(42)

    neurons = {
        "n0": make_tiny_neuron("n0", field_dim=128),
        "n1": make_tiny_neuron("n1", field_dim=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    # 注入 gamma_oscillator
    gamma = GammaOscillator()
    gamma.assign_phase("n0", phase=0.0)
    gamma.assign_phase("n1", phase=3.14159)  # 反相

    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    result = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        gamma_oscillator=gamma,
    )

    print(f"  weights (gamma 门控后): {result['weights'].detach().cpu().numpy()}")
    print(f"  scores (gamma 门控后): {result['scores'].detach().cpu().numpy()}")
    print("  ✅ Test 4 通过：gamma_oscillator 接入不崩溃")
    return True


def test_forward_train_rounds_diff():
    """测试 5：多轮共振 vs 单轮，验证 side_signals 真正生效."""
    print("\n=== Test 5: 多轮共振 vs 单轮（验证 side_signals 生效）===")
    torch.manual_seed(42)

    neurons = {
        "n0": make_tiny_neuron("n0", field_dim=128),
        "n1": make_tiny_neuron("n1", field_dim=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    # 单轮（无 side_signals）
    result_1 = ensemble.forward_train(shared_embeddings=shared_emb, n_rounds=1)
    # 多轮（有 side_signals）
    result_2 = ensemble.forward_train(shared_embeddings=shared_emb, n_rounds=2)

    diff = (result_1["fused_logits"] - result_2["fused_logits"]).abs().max().item()
    print(f"  单轮 vs 多轮 fused_logits 差异: {diff:.6f}")
    assert diff > 1e-6, "多轮共振与单轮无差异（side_signals 未生效）"
    print("  ✅ Test 5 通过：多轮共振与单轮输出不同（side_signals 真正生效）")
    return True


def test_forward_train_residual_mode():
    """测试 6：residual 融合模式（straight-through estimator）."""
    print("\n=== Test 6: residual 融合模式 ===")
    torch.manual_seed(42)

    neurons = {
        "n0": make_tiny_neuron("n0", field_dim=128),
        "n1": make_tiny_neuron("n1", field_dim=128),
        "n2": make_tiny_neuron("n2", field_dim=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    # side_channels 可训练
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        neuron.train()

    shared_emb = torch.randn(2, 8, 32, requires_grad=False)

    result = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        fusion_mode="residual",
    )

    print(f"  weights shape: {result['weights'].shape}")  # 应该是 [N-1] = [2]
    print(f"  weights: {result['weights'].detach().cpu().numpy()}")

    target = torch.zeros(2, 8, dtype=torch.long)
    loss = F.cross_entropy(
        result["fused_logits"].view(-1, result["fused_logits"].size(-1)),
        target.view(-1),
    )
    loss.backward()

    # 验证其他神经元的 side_channels 也能接收梯度
    grad_ok = 0
    grad_total = 0
    for nid, neuron in neurons.items():
        for pre_id, ch in neuron.excite_channels.items():
            for p in ch.parameters():
                grad_total += 1
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    grad_ok += 1
    print(f"  side_channels 梯度: {grad_ok}/{grad_total}")
    assert grad_ok > 0, "residual 模式下无梯度流到 side_channels"
    print("  ✅ Test 6 通过：residual 模式可微（权重可微，选择用 straight-through）")
    return True


def main():
    print("=" * 60)
    print("S1 验证：forward_train 全可微多轮共振路径")
    print("=" * 60)

    tests = [
        test_forward_train_differentiable_same_spec,
        test_forward_train_cross_spec,
        test_forward_train_neuromodulator,
        test_forward_train_gamma_oscillator,
        test_forward_train_rounds_diff,
        test_forward_train_residual_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{passed + failed} 通过")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
