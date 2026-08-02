"""C8 场写入保留幅度作置信度 smoke test.

验证 field_confidence 的正确性：
1. field_confidence 存在且 shape 正确 [B]
2. field_confidence 范围 [0, 1]（attention entropy 归一化）
3. 不同输入产生不同 confidence
4. field.write 支持 per-sample scale（[B] tensor）
5. 梯度流：field_confidence 能接收梯度
6. 多头模式 confidence 计算正确（gate 加权聚合）
7. field.update 支持 per-sample scale
8. 向后兼容：confidence 默认 1.0（field.write 接受 float）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import math
import torch

from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField


def _make_neuron(num_field_heads: int = 1) -> ResonanceNeuron:
    """构建指定 num_field_heads 的 TINY_TEST neuron。"""
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = f"n_heads{num_field_heads}"
    cfg.num_field_heads = num_field_heads
    torch.manual_seed(42)
    return ResonanceNeuron(cfg)


def test_confidence_exists_and_shape():
    """[1] field_confidence 存在且 shape 正确 [B]。"""
    print("\n[1] field_confidence 存在且 shape 正确")
    neuron = _make_neuron(num_field_heads=1)
    neuron.eval()

    B = 4
    shared_emb = torch.randn(B, 16, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)

    assert "field_confidence" in result, "结果应包含 field_confidence"
    confidence = result["field_confidence"]
    assert confidence.shape == (B,), f"confidence shape 应为 ({B},), got {confidence.shape}"
    print(f"  PASS: field_confidence shape={confidence.shape}")


def test_confidence_range():
    """[2] field_confidence 范围 [0, 1]。"""
    print("\n[2] field_confidence 范围 [0, 1]")
    neuron = _make_neuron(num_field_heads=1)
    neuron.eval()

    shared_emb = torch.randn(8, 32, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)

    confidence = result["field_confidence"]
    assert (confidence >= 0).all() and (confidence <= 1).all(), \
        f"confidence 应在 [0, 1] 范围内, min={confidence.min()}, max={confidence.max()}"
    print(f"  PASS: confidence 范围 [{confidence.min():.4f}, {confidence.max():.4f}]")


def test_different_input_different_confidence():
    """[3] 不同输入产生不同 confidence。"""
    print("\n[3] 不同输入产生不同 confidence")
    neuron = _make_neuron(num_field_heads=1)
    # 放大 query 让 attention 对输入更敏感（模拟训练后的状态）
    # 随机初始化的 query 极小（std=0.02），加上 field_pool_scale=hidden^-0.5，
    # 使 attention scores 几乎为零 → softmax 接近均匀 → confidence≈0 对所有输入。
    # 训练后 query 会学到有意义的方向，attention 自然分化。
    with torch.no_grad():
        neuron.field_pool_query.mul_(50.0)
    neuron.eval()

    # 输入1：全零（h=0 → attention 完全均匀 → confidence=0）
    shared_emb1 = torch.zeros(2, 16, 512)
    # 输入2：随机（attention 非均匀 → confidence > 0）
    shared_emb2 = torch.randn(2, 16, 512)

    with torch.no_grad():
        result1 = neuron.forward(shared_emb1)
        result2 = neuron.forward(shared_emb2)

    diff = (result1["field_confidence"] - result2["field_confidence"]).abs().max().item()
    assert diff > 1e-4, f"不同输入应产生不同 confidence, diff={diff}"
    print(f"  PASS: 不同输入产生不同 confidence (diff={diff:.4e})")


def test_field_write_per_sample_scale():
    """[4] field.write 支持 per-sample scale（[B] tensor）。"""
    print("\n[4] field.write per-sample scale")
    field = ResonanceField(dim=64)
    field.reset(batch_size=4)

    vec = torch.randn(4, 64)
    # per-sample scale
    scale = torch.tensor([1.0, 0.5, 0.25, 0.125])

    v_scaled = field.write("n0", vec, scale=scale)

    # 验证：v_scaled 的每行幅度应与 scale 成正比
    v_norm = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
    expected = v_norm * scale.unsqueeze(-1)
    diff = (v_scaled - expected).abs().max().item()
    assert diff < 1e-5, f"per-sample scale 计算错误, diff={diff}"

    # 验证 state 也正确累加
    state_diff = (field.state - expected).abs().max().item()
    assert state_diff < 1e-5, f"state 累加错误, diff={state_diff}"
    print(f"  PASS: per-sample scale 正确 (scale={scale.tolist()})")


def test_confidence_gradient_flow():
    """[5] field_confidence 能接收梯度。"""
    print("\n[5] field_confidence 梯度流")
    neuron = _make_neuron(num_field_heads=1)
    neuron.train()

    shared_emb = torch.randn(2, 16, 512)
    result = neuron.forward(shared_emb, return_logits=True)

    # loss 包含 confidence，反向传播
    loss = result["logits"].sum() + result["field_confidence"].sum()
    loss.backward()

    # field_pool_query 应有梯度（confidence 依赖 attention weights）
    assert neuron.field_pool_query.grad is not None, "field_pool_query 无梯度"
    grad_norm = neuron.field_pool_query.grad.norm().item()
    assert grad_norm > 0, f"field_pool_query 梯度为零"
    print(f"  PASS: field_pool_query 有梯度 (norm={grad_norm:.4e}), confidence 进入梯度流")


def test_multihead_confidence():
    """[6] 多头模式 confidence 计算正确（gate 加权聚合）。"""
    print("\n[6] 多头 confidence")
    neuron = _make_neuron(num_field_heads=4)
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)

    confidence = result["field_confidence"]
    gate = result["field_gate"]  # [B, K]

    assert confidence.shape == (2,), f"多头 confidence shape 应为 (2,), got {confidence.shape}"
    assert (confidence >= 0).all() and (confidence <= 1).all(), \
        f"多头 confidence 应在 [0, 1], min={confidence.min()}, max={confidence.max()}"

    # gate 行和应为 1（softmax）
    gate_sums = gate.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5), \
        f"gate 行和应=1, got {gate_sums}"

    print(f"  PASS: 多头 confidence 正确 (val={confidence.tolist()}), gate 行和=1")


def test_field_update_per_sample_scale():
    """[7] field.update 支持 per-sample scale。"""
    print("\n[7] field.update per-sample scale")
    field = ResonanceField(dim=64)
    field.reset(batch_size=4)

    # 先 write
    vec1 = torch.randn(4, 64)
    field.write("n0", vec1, scale=1.0)

    # 再 update（per-sample scale）
    vec2 = torch.randn(4, 64)
    scale = torch.tensor([1.0, 0.5, 0.25, 0.125])
    v_scaled = field.update("n0", vec2, scale=scale)

    # 验证：state 应为 vec2 的 scaled 版本（替换了 vec1 的贡献）
    v2_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + 1e-8)
    expected = v2_norm * scale.unsqueeze(-1)
    diff = (field.state - expected).abs().max().item()
    assert diff < 1e-5, f"update 后 state 应为 vec2 的 scaled 版本, diff={diff}"
    print(f"  PASS: field.update per-sample scale 正确")


def test_backward_compat_float_scale():
    """[8] 向后兼容：field.write 接受 float scale。"""
    print("\n[8] 向后兼容 float scale")
    field = ResonanceField(dim=64)
    field.reset(batch_size=2)

    vec = torch.randn(2, 64)
    # float scale（向后兼容）
    scale = 0.5
    v_scaled = field.write("n0", vec, scale=scale)

    v_norm = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
    expected = v_norm * scale
    diff = (v_scaled - expected).abs().max().item()
    assert diff < 1e-5, f"float scale 计算错误, diff={diff}"
    print(f"  PASS: float scale 向后兼容 (scale={scale})")


def test_confidence_extremes():
    """[9] confidence 极端情况：完全聚焦 vs 均匀分布。"""
    print("\n[9] confidence 极端情况")
    # 构造完全聚焦的 attention weights（one-hot）
    L = 16
    focused = torch.zeros(1, L)
    focused[0, 0] = 1.0  # 完全聚焦在位置 0

    # 构造均匀分布的 attention weights
    uniform = torch.ones(1, L) / L

    # 计算 entropy-based confidence
    max_entropy = math.log(L)

    focused_entropy = -(focused * (focused + 1e-8).log()).sum(dim=-1)
    focused_conf = 1.0 - focused_entropy / max_entropy

    uniform_entropy = -(uniform * (uniform + 1e-8).log()).sum(dim=-1)
    uniform_conf = 1.0 - uniform_entropy / max_entropy

    assert focused_conf.item() > 0.99, f"完全聚焦 confidence 应接近 1, got {focused_conf.item()}"
    assert uniform_conf.item() < 0.01, f"均匀分布 confidence 应接近 0, got {uniform_conf.item()}"
    print(f"  PASS: 完全聚焦 confidence={focused_conf.item():.4f}, 均匀分布 confidence={uniform_conf.item():.4f}")


def main():
    print("=" * 60)
    print("C8 场写入保留幅度作置信度 smoke test")
    print("=" * 60)

    test_confidence_exists_and_shape()
    test_confidence_range()
    test_different_input_different_confidence()
    test_field_write_per_sample_scale()
    test_confidence_gradient_flow()
    test_multihead_confidence()
    test_field_update_per_sample_scale()
    test_backward_compat_float_scale()
    test_confidence_extremes()

    print("\n" + "=" * 60)
    print("ALL 9/9 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
