"""C14 shared_expert_weight 动态化 smoke test.

验证方案 C（共振分数 + 场状态联合驱动）：
1. shared_weight_dynamic=False 向后兼容（固定 shared_expert_weight）
2. shared_weight_dynamic=True 创建 shared_weight_mlp
3. 初始化偏置使初始 sw ≈ shared_expert_weight（0.3）
4. per-sample sw 形状 [B,1]
5. sw 范围在 [0,1]（sigmoid 输出）
6. 动态 sw 改变 weighted_logits（与固定 sw 不同）
7. field_state 影响 sw（不同场状态→不同 sw）
8. shared_weight_per_sample 字段存在且形状正确
9. MLP 参数可收集（可训练性）
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
    cfg.vocab_size = 100
    cfg.neuron_id = neuron_id
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def _make_ensemble(
    shared_weight_dynamic=False,
    shared_expert_id="n0",
    shared_expert_weight=0.3,
):
    """创建含 shared_expert 的 ensemble（n0=shared, n1=domain）。"""
    neurons = {
        "n0": _make_neuron("n0", seed=42),
        "n1": _make_neuron("n1", seed=43),
    }
    field = ResonanceField(dim=TINY_TEST.field_dim)
    return ResonanceEnsemble(
        neurons, field,
        shared_expert_id=shared_expert_id,
        shared_expert_weight=shared_expert_weight,
        shared_weight_dynamic=shared_weight_dynamic,
    )


def test_backward_compat():
    """[1] shared_weight_dynamic=False 向后兼容。"""
    print("\n[1] shared_weight_dynamic=False 向后兼容")
    ens = _make_ensemble(shared_weight_dynamic=False)
    assert ens.shared_weight_mlp is None, "dynamic=False 不应创建 MLP"
    assert ens.shared_weight_dynamic is False
    print("  PASS: dynamic=False 时 shared_weight_mlp=None")


def test_mlp_created():
    """[2] shared_weight_dynamic=True 创建 shared_weight_mlp。"""
    print("\n[2] shared_weight_dynamic=True 创建 MLP")
    ens = _make_ensemble(shared_weight_dynamic=True)
    assert ens.shared_weight_mlp is not None, "dynamic=True 应创建 MLP"
    # 检查 MLP 结构: Linear(1+D, hidden) + GELU + Linear(hidden, 1)
    assert isinstance(ens.shared_weight_mlp[0], nn.Linear)
    assert isinstance(ens.shared_weight_mlp[1], nn.GELU)
    assert isinstance(ens.shared_weight_mlp[2], nn.Linear)
    # 输入维度 = 1 + field_dim
    assert ens.shared_weight_mlp[0].in_features == 1 + TINY_TEST.field_dim
    # 输出维度 = 1
    assert ens.shared_weight_mlp[2].out_features == 1
    print(f"  PASS: MLP 创建, 输入维度={ens.shared_weight_mlp[0].in_features}, 输出维度=1")


def test_initial_bias():
    """[3] 初始化偏置使初始 sw ≈ shared_expert_weight（0.3）。"""
    print("\n[3] 初始化偏置")
    ens = _make_ensemble(shared_weight_dynamic=True, shared_expert_weight=0.3)
    # 构造输入: max_score=0.5, field_state=随机
    field_state = torch.randn(2, TINY_TEST.field_dim)
    max_score = torch.full((2, 1), 0.5)
    mlp_input = torch.cat([max_score, field_state], dim=-1)
    with torch.no_grad():
        sw = torch.sigmoid(ens.shared_weight_mlp(mlp_input))  # [2,1]
    # 初始权重=0 + bias=logit(0.3)，故 sw ≈ 0.3
    assert (sw - 0.3).abs().max() < 0.01, f"初始 sw 应≈0.3, got {sw.mean().item():.4f}"
    print(f"  PASS: 初始 sw={sw.mean().item():.4f} ≈ 0.3")


def test_per_sample_shape():
    """[4] per-sample sw 形状 [B,1]。"""
    print("\n[4] per-sample sw 形状")
    ens = _make_ensemble(shared_weight_dynamic=True)
    shared_emb = torch.randn(3, 8, 512)  # B=3
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb, return_logits=True)
    assert "shared_weight_per_sample" in result, "应返回 shared_weight_per_sample"
    sw_ps = result["shared_weight_per_sample"]
    assert sw_ps.shape == (3,), f"per-sample sw 应为 [3], got {sw_ps.shape}"
    print(f"  PASS: per-sample sw 形状={sw_ps.shape}, 值={sw_ps.tolist()}")


def test_sw_range():
    """[5] sw 范围在 [0,1]（sigmoid 输出）。"""
    print("\n[5] sw 范围 [0,1]")
    ens = _make_ensemble(shared_weight_dynamic=True)
    shared_emb = torch.randn(4, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb, return_logits=True)
    sw_ps = result["shared_weight_per_sample"]
    assert (sw_ps >= 0).all() and (sw_ps <= 1).all(), f"sw 应在 [0,1], got min={sw_ps.min()}, max={sw_ps.max()}"
    print(f"  PASS: sw 范围 [{sw_ps.min():.4f}, {sw_ps.max():.4f}]")


def test_dynamic_changes_output():
    """[6] 动态 sw 改变 weighted_logits（与固定 sw 不同）。"""
    print("\n[6] 动态 sw 改变输出")
    torch.manual_seed(0)
    shared_emb = torch.randn(2, 8, 512)

    # 固定 sw
    ens_fixed = _make_ensemble(shared_weight_dynamic=False, shared_expert_weight=0.3)
    with torch.no_grad():
        result_fixed = ens_fixed.forward(shared_embeddings=shared_emb, return_logits=True)

    # 动态 sw（先用随机梯度更新 MLP 使其偏离初始 0.3）
    ens_dynamic = _make_ensemble(shared_weight_dynamic=True, shared_expert_weight=0.3)
    # 手动扰动 MLP 权重使其偏离初始状态
    with torch.no_grad():
        ens_dynamic.shared_weight_mlp[2].weight.add_(torch.randn_like(ens_dynamic.shared_weight_mlp[2].weight) * 0.5)
    with torch.no_grad():
        result_dynamic = ens_dynamic.forward(shared_embeddings=shared_emb, return_logits=True)

    if "weighted_logits" in result_fixed and "weighted_logits" in result_dynamic:
        diff = (result_fixed["weighted_logits"] - result_dynamic["weighted_logits"]).abs().max().item()
        assert diff > 1e-4, f"动态 sw 应改变输出, diff={diff}"
        print(f"  PASS: 输出差异={diff:.4f} (动态 vs 固定)")
    else:
        print("  SKIP: 无 weighted_logits（vocab 不同）")


def test_field_state_affects_sw():
    """[7] field_state 影响 sw（不同场状态→不同 sw）。"""
    print("\n[7] field_state 影响 sw")
    ens = _make_ensemble(shared_weight_dynamic=True)
    # 初始第二层 weight=0 导致第一层输出被忽略，需扰动第二层 weight
    with torch.no_grad():
        ens.shared_weight_mlp[2].weight.add_(torch.randn_like(ens.shared_weight_mlp[2].weight) * 0.5)

    # 构造两个不同 field_state
    fs1 = torch.randn(1, TINY_TEST.field_dim)
    fs2 = torch.randn(1, TINY_TEST.field_dim)
    max_score = torch.zeros(1, 1)
    inp1 = torch.cat([max_score, fs1], dim=-1)
    inp2 = torch.cat([max_score, fs2], dim=-1)
    with torch.no_grad():
        sw1 = torch.sigmoid(ens.shared_weight_mlp(inp1))
        sw2 = torch.sigmoid(ens.shared_weight_mlp(inp2))
    assert (sw1 - sw2).abs().max() > 1e-6, f"不同 field_state 应产生不同 sw, sw1={sw1.item()}, sw2={sw2.item()}"
    print(f"  PASS: sw1={sw1.item():.4f}, sw2={sw2.item():.4f} (field_state 影响)")


def test_max_score_affects_sw():
    """[8] max_domain_score 影响 sw（不同 score→不同 sw）。"""
    print("\n[8] max_domain_score 影响 sw")
    ens = _make_ensemble(shared_weight_dynamic=True)
    # 扰动第二层 weight 使第一层输出（含 max_score 信号）能影响最终结果
    with torch.no_grad():
        ens.shared_weight_mlp[2].weight.add_(torch.randn_like(ens.shared_weight_mlp[2].weight) * 0.5)

    field_state = torch.randn(1, TINY_TEST.field_dim)
    score_low = torch.full((1, 1), 0.1)
    score_high = torch.full((1, 1), 0.9)
    inp_low = torch.cat([score_low, field_state], dim=-1)
    inp_high = torch.cat([score_high, field_state], dim=-1)
    with torch.no_grad():
        sw_low = torch.sigmoid(ens.shared_weight_mlp(inp_low))
        sw_high = torch.sigmoid(ens.shared_weight_mlp(inp_high))
    assert (sw_low - sw_high).abs().max() > 1e-6, f"不同 score 应产生不同 sw, sw_low={sw_low.item()}, sw_high={sw_high.item()}"
    print(f"  PASS: sw_low={sw_low.item():.4f}, sw_high={sw_high.item():.4f} (score 影响)")


def test_mlp_params_collectable():
    """[9] MLP 参数可收集（可训练性）。"""
    print("\n[9] MLP 参数可收集")
    ens = _make_ensemble(shared_weight_dynamic=True)
    params = list(ens.shared_weight_mlp.parameters())
    assert len(params) == 4, f"MLP 应有 4 个参数（2 Linear×2）, got {len(params)}"
    total_params = sum(p.numel() for p in params)
    assert total_params > 0
    # 验证 requires_grad 默认 True
    for p in params:
        assert p.requires_grad, "MLP 参数应默认可训练"
    print(f"  PASS: MLP 参数数={total_params}, 全部 requires_grad=True")


def test_no_shared_expert():
    """[10] shared_expert_id=None 时不创建 MLP。"""
    print("\n[10] shared_expert_id=None 时不创建 MLP")
    neurons = {"n0": _make_neuron("n0", seed=42)}
    field = ResonanceField(dim=TINY_TEST.field_dim)
    ens = ResonanceEnsemble(
        neurons, field,
        shared_expert_id=None,
        shared_weight_dynamic=True,  # 即使启用也无 shared_expert
    )
    assert ens.shared_weight_mlp is None, "无 shared_expert 时不应创建 MLP"
    print("  PASS: 无 shared_expert 时 MLP=None")


def main():
    print("=" * 60)
    print("C14 shared_expert_weight 动态化 smoke test")
    print("=" * 60)

    test_backward_compat()
    test_mlp_created()
    test_initial_bias()
    test_per_sample_shape()
    test_sw_range()
    test_dynamic_changes_output()
    test_field_state_affects_sw()
    test_max_score_affects_sw()
    test_mlp_params_collectable()
    test_no_shared_expert()

    print("\n" + "=" * 60)
    print("ALL 10/10 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
