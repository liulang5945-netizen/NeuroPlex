"""R3 共识投票融合 smoke test.

验证 _consensus_logit_fusion：
1. 方法存在
2. 单神经元退化（直接返回 logits）
3. 多神经元：输出形状正确 [B,L,V]
4. final_weights 来自共振分 softmax
5. consensus_votes 形状 [B,L,V]，值域 [0,N]
6. 高共识 token logit 被放大（与 per_position 模式不同）
7. fusion_mode == "consensus" 标记
8. forward(fusion_mode="consensus") 集成测试
9. 共识加成强度可调（consensus_alpha）
10. 共识票数正确（全员同意 → votes=N）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn as nn

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble


def _make_neuron(neuron_id="n0", seed=42) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 50
    cfg.neuron_id = neuron_id
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def _make_ensemble(n_neurons=3):
    neurons = {f"n{i}": _make_neuron(f"n{i}", seed=42 + i) for i in range(n_neurons)}
    field = ResonanceField(dim=TINY_TEST.field_dim)
    return ResonanceEnsemble(neurons, field, max_rounds=2)


def test_method_exists():
    """[1] _consensus_logit_fusion 方法存在。"""
    print("\n[1] 方法存在")
    assert hasattr(ResonanceEnsemble, "_consensus_logit_fusion")
    print("  PASS: _consensus_logit_fusion 方法存在")


def test_single_neuron_degenerate():
    """[2] 单神经元退化（直接返回 logits）。"""
    print("\n[2] 单神经元退化")
    ens = _make_ensemble(n_neurons=1)
    all_logits = {"n0": torch.randn(2, 4, 50)}
    scores = {"n0": 0.8}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    assert torch.allclose(result["weighted_logits"], all_logits["n0"])
    assert result["final_weights"] == {"n0": 1.0}
    print("  PASS: 单神经元直接返回 logits")


def test_output_shape():
    """[3] 多神经元：输出形状正确 [B,L,V]。"""
    print("\n[3] 输出形状")
    ens = _make_ensemble(n_neurons=3)
    B, L, V = 2, 4, 50
    all_logits = {f"n{i}": torch.randn(B, L, V) for i in range(3)}
    scores = {f"n{i}": 0.5 + i * 0.1 for i in range(3)}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    assert result["weighted_logits"].shape == (B, L, V), f"应为 [B,L,V], got {result['weighted_logits'].shape}"
    print(f"  PASS: 输出形状 {result['weighted_logits'].shape}")


def test_final_weights_from_scores():
    """[4] final_weights 来自共振分 softmax。"""
    print("\n[4] final_weights 来自共振分")
    ens = _make_ensemble(n_neurons=3)
    all_logits = {f"n{i}": torch.randn(2, 4, 50) for i in range(3)}
    scores = {"n0": 0.8, "n1": 0.3, "n2": 0.6}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    fw = result["final_weights"]
    # softmax([0.8, 0.3, 0.6])
    import torch.nn.functional as F
    expected = F.softmax(torch.tensor([0.8, 0.3, 0.6]), dim=0)
    assert abs(fw["n0"] - expected[0].item()) < 1e-5
    assert abs(fw["n1"] - expected[1].item()) < 1e-5
    assert abs(fw["n2"] - expected[2].item()) < 1e-5
    print(f"  PASS: final_weights={fw}（softmax 共振分）")


def test_consensus_votes_shape():
    """[5] consensus_votes 形状 [B,L,V]，值域 [0,N]。"""
    print("\n[5] consensus_votes 形状和值域")
    ens = _make_ensemble(n_neurons=3)
    B, L, V = 2, 4, 50
    all_logits = {f"n{i}": torch.randn(B, L, V) for i in range(3)}
    scores = {f"n{i}": 0.5 for i in range(3)}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    votes = result["consensus_votes"]
    assert votes.shape == (B, L, V), f"应为 [B,L,V], got {votes.shape}"
    assert votes.min() >= 0, f"票数应>=0, got min={votes.min()}"
    assert votes.max() <= 3, f"票数应<=N=3, got max={votes.max()}"
    print(f"  PASS: votes 形状={votes.shape}, 值域 [{votes.min():.0f}, {votes.max():.0f}]")


def test_high_consensus_amplified():
    """[6] 高共识 token logit 被放大（与 per_position 模式不同）。"""
    print("\n[6] 高共识 token 被放大")
    ens = _make_ensemble(n_neurons=3)
    B, L, V = 1, 1, 50
    # 构造 3 个神经元，它们在 token 0 上都有高 logit（高共识）
    base = torch.zeros(B, L, V)
    base[0, 0, 0] = 10.0  # token 0 高 logit
    all_logits = {f"n{i}": base.clone() + torch.randn(B, L, V) * 0.1 for i in range(3)}
    scores = {f"n{i}": 0.5 for i in range(3)}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))

    # token 0 应被所有 3 个神经元投票（votes=3），consensus_factor = 1 + 0.5*3/3 = 1.5
    votes_token0 = result["consensus_votes"][0, 0, 0].item()
    assert votes_token0 == 3.0, f"token 0 应有 3 票, got {votes_token0}"

    # 对比：无共识 token（votes=0），factor=1.0
    # token 0 的 logit 应比无共识 token 更被放大（相对值）
    fused = result["weighted_logits"][0, 0]
    base_fused = sum(0.5 * all_logits[f"n{i}"][0, 0] for i in range(3))  # 近似基础加权
    # token 0 的放大率 = fused[0] / base_fused[0] 应 > 1
    ratio_token0 = fused[0].item() / base_fused[0].item()
    assert ratio_token0 > 1.0, f"token 0 应被放大, ratio={ratio_token0}"
    print(f"  PASS: token 0 votes={votes_token0}, 放大率={ratio_token0:.3f}（共识加成生效）")


def test_fusion_mode_marker():
    """[7] fusion_mode == "consensus" 标记。"""
    print("\n[7] fusion_mode 标记")
    ens = _make_ensemble(n_neurons=2)
    all_logits = {"n0": torch.randn(1, 2, 50), "n1": torch.randn(1, 2, 50)}
    scores = {"n0": 0.7, "n1": 0.5}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    assert result.get("fusion_mode") == "consensus"
    print("  PASS: fusion_mode='consensus' 标记存在")


def test_forward_integration():
    """[8] forward(fusion_mode="consensus") 集成测试。"""
    print("\n[8] forward 集成测试")
    ens = _make_ensemble(n_neurons=2)
    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb, fusion_mode="consensus", return_logits=True)
    # 应有 weighted_logits
    assert "weighted_logits" in result, "应返回 weighted_logits"
    print(f"  PASS: forward(fusion_mode='consensus') 返回 weighted_logits")


def test_consensus_alpha_adjustable():
    """[9] 共识加成强度可调（consensus_alpha）。"""
    print("\n[9] consensus_alpha 可调")
    ens = _make_ensemble(n_neurons=3)
    ens.consensus_alpha = 0.0  # 关闭加成
    B, L, V = 1, 1, 50
    base = torch.zeros(B, L, V)
    base[0, 0, 0] = 10.0
    all_logits = {f"n{i}": base.clone() for i in range(3)}
    scores = {f"n{i}": 0.5 for i in range(3)}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    # alpha=0 时 consensus_factor=1.0，fused = base_fused
    fused = result["weighted_logits"][0, 0, 0].item()
    base_fused = 0.5 * 10.0 * 3  # 简化：3 个相同 logit，softmax 权重各 1/3
    # 但 softmax([0.5,0.5,0.5]) = [1/3, 1/3, 1/3]
    import torch.nn.functional as F
    w = F.softmax(torch.tensor([0.5, 0.5, 0.5]), dim=0)
    base_fused_val = (w[0] * 10.0 + w[1] * 10.0 + w[2] * 10.0).item()
    assert abs(fused - base_fused_val) < 1e-4, f"alpha=0 时 fused={fused} 应等于 base={base_fused_val}"
    print(f"  PASS: alpha=0 时 fused={fused:.4f} ≈ base_fused={base_fused_val:.4f}（无加成）")


def test_unanimous_votes():
    """[10] 共识票数正确（全员同意 → votes=N）。"""
    print("\n[10] 全员同意 votes=N")
    ens = _make_ensemble(n_neurons=3)
    B, L, V = 1, 1, 50
    # 3 个神经元在 token 5 上都有最高 logit
    base = torch.zeros(B, L, V)
    base[0, 0, 5] = 10.0
    all_logits = {f"n{i}": base.clone() for i in range(3)}
    scores = {f"n{i}": 0.5 for i in range(3)}
    result = {}
    ens._consensus_logit_fusion(all_logits, scores, result, ref=torch.tensor(0.0))
    votes_token5 = result["consensus_votes"][0, 0, 5].item()
    assert votes_token5 == 3.0, f"token 5 全员同意应有 3 票, got {votes_token5}"
    print(f"  PASS: token 5 全员同意 votes={votes_token5}")


def main():
    print("=" * 60)
    print("R3 共识投票融合 smoke test")
    print("=" * 60)

    test_method_exists()
    test_single_neuron_degenerate()
    test_output_shape()
    test_final_weights_from_scores()
    test_consensus_votes_shape()
    test_high_consensus_amplified()
    test_fusion_mode_marker()
    test_forward_integration()
    test_consensus_alpha_adjustable()
    test_unanimous_votes()

    print("\n" + "=" * 60)
    print("ALL 10/10 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
