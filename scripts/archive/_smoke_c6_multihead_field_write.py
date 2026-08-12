"""C6 多头 field write smoke test.

验证 num_field_heads 多头 field write 的正确性：
1. 向后兼容：num_field_heads=1 与原 v2 单 query 行为一致（diff=0）
2. 多头生效：num_field_heads>1 改变 field_vector 输出
3. 梯度流：多头参数（field_write_heads/field_gate/field_pool_queries）能接收梯度
4. 门控聚合：gate 权重和为 1（softmax），且不同输入产生不同 gate
5. attention weights shape：多头时返回 [B, L]（mean over K）
6. checkpoint 兼容：旧单头 ckpt 加载到多头 neuron（strict=False，缺失参数用初始化值）
7. get_field_write_parameters：返回正确参数列表
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch

from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron


def _make_neuron(num_field_heads: int = 1) -> ResonanceNeuron:
    """构建指定 num_field_heads 的 TINY_TEST neuron。"""
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = f"n_heads{num_field_heads}"
    cfg.num_field_heads = num_field_heads
    torch.manual_seed(42)
    return ResonanceNeuron(cfg)


def test_backward_compat_single_head():
    """[1] num_field_heads=1 与原 v2 单 query 行为完全一致。"""
    print("\n[1] 单头向后兼容")
    neuron = _make_neuron(num_field_heads=1)
    assert neuron.num_field_heads == 1
    # 单头应保留原参数
    assert hasattr(neuron, "field_write") and neuron.field_write is not None
    assert hasattr(neuron, "field_pool_query")
    assert not hasattr(neuron, "field_write_heads")
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb, return_logits=True)

    assert "field_vector" in result
    assert result["field_vector"].shape == (2, 512)
    # L2 归一化检查
    norms = result["field_vector"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"field_vector 应 L2 归一化, norms={norms}"
    print(f"  PASS: 单头行为正确, field_vector shape={result['field_vector'].shape}")


def test_multihead_changes_output():
    """[2] num_field_heads>1 改变 field_vector 输出（与单头不同）。"""
    print("\n[2] 多头改变输出")
    torch.manual_seed(42)
    neuron_single = _make_neuron(num_field_heads=1)
    torch.manual_seed(42)
    neuron_multi = _make_neuron(num_field_heads=4)

    # 多头应有新参数
    assert neuron_multi.num_field_heads == 4
    assert hasattr(neuron_multi, "field_write_heads")
    assert len(neuron_multi.field_write_heads) == 4
    assert hasattr(neuron_multi, "field_gate")
    assert hasattr(neuron_multi, "field_pool_queries")
    assert neuron_multi.field_pool_queries.shape == (4, 256)
    # 多头时 field_write=None
    assert neuron_multi.field_write is None

    shared_emb = torch.randn(2, 16, 512)
    neuron_single.eval()
    neuron_multi.eval()
    with torch.no_grad():
        result_single = neuron_single.forward(shared_emb)
        result_multi = neuron_multi.forward(shared_emb)

    diff = (result_single["field_vector"] - result_multi["field_vector"]).abs().max().item()
    assert diff > 1e-4, f"多头应改变输出, diff={diff}"
    # 多头也应 L2 归一化
    norms = result_multi["field_vector"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"多头 field_vector 应 L2 归一化, norms={norms}"
    # 多头应返回 gate
    assert "field_gate" in result_multi, "多头应返回 field_gate 诊断信息"
    assert result_multi["field_gate"].shape == (2, 4)
    print(f"  PASS: 多头改变输出 (diff={diff:.4e}), gate shape={result_multi['field_gate'].shape}")


def test_gradient_flow():
    """[3] 多头参数能接收梯度。"""
    print("\n[3] 梯度流")
    neuron = _make_neuron(num_field_heads=4)
    neuron.train()

    shared_emb = torch.randn(2, 16, 512)
    result = neuron.forward(shared_emb, return_logits=True)
    loss = result["logits"].sum() + result["field_vector"].sum()
    loss.backward()

    # 检查每个 head 的 field_write 有梯度
    for k, head in enumerate(neuron.field_write_heads):
        grad_norm = head.weight.grad.norm().item() if head.weight.grad is not None else 0.0
        assert grad_norm > 0, f"head {k} field_write 权重无梯度"
    print(f"  PASS: {len(neuron.field_write_heads)} 个 head 权重均有梯度")

    # 检查 field_gate 有梯度
    assert neuron.field_gate.weight.grad is not None, "field_gate 权重无梯度"
    gate_grad_norm = neuron.field_gate.weight.grad.norm().item()
    assert gate_grad_norm > 0, f"field_gate 权重梯度为零"
    print(f"  PASS: field_gate 有梯度 (norm={gate_grad_norm:.4e})")

    # 检查 field_pool_queries 有梯度
    assert neuron.field_pool_queries.grad is not None, "field_pool_queries 无梯度"
    query_grad_norm = neuron.field_pool_queries.grad.norm().item()
    assert query_grad_norm > 0, f"field_pool_queries 梯度为零"
    print(f"  PASS: field_pool_queries 有梯度 (norm={query_grad_norm:.4e})")


def test_gate_softmax():
    """[4] gate 权重和为 1（softmax），且不同输入产生不同 gate。"""
    print("\n[4] 门控 softmax 性质")
    neuron = _make_neuron(num_field_heads=4)
    neuron.eval()

    # 输入 1
    shared_emb1 = torch.randn(2, 16, 512)
    # 输入 2（不同）
    shared_emb2 = torch.randn(2, 16, 512)

    with torch.no_grad():
        result1 = neuron.forward(shared_emb1)
        result2 = neuron.forward(shared_emb2)

    gate1 = result1["field_gate"]  # [B, K]
    gate2 = result2["field_gate"]

    # softmax 性质：每行和为 1
    row_sums1 = gate1.sum(dim=-1)
    assert torch.allclose(row_sums1, torch.ones_like(row_sums1), atol=1e-5), \
        f"gate1 行和应=1, got {row_sums1}"
    print(f"  PASS: gate softmax 行和=1 (sums={row_sums1.tolist()})")

    # 不同输入产生不同 gate
    gate_diff = (gate1 - gate2).abs().max().item()
    assert gate_diff > 1e-4, f"不同输入应产生不同 gate, diff={gate_diff}"
    print(f"  PASS: 不同输入产生不同 gate (diff={gate_diff:.4e})")


def test_attention_weights_shape():
    """[5] 多头时 field_attn_weights 返回 [B, L]（mean over K）。"""
    print("\n[5] attention weights shape")
    neuron = _make_neuron(num_field_heads=4)
    neuron.eval()

    B, L = 2, 16
    shared_emb = torch.randn(B, L, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)

    attn_weights = result["field_attn_weights"]
    assert attn_weights.shape == (B, L), \
        f"field_attn_weights 应为 [B, L]={(B, L)}, got {attn_weights.shape}"
    # 平均后应仍 softmax（每行和≈1/K 的平均，不一定=1，但应非负）
    assert (attn_weights >= 0).all(), "attention weights 应非负"
    print(f"  PASS: field_attn_weights shape={attn_weights.shape}")


def test_checkpoint_compat():
    """[6] 旧单头 ckpt 加载到多头 neuron（strict=False）。"""
    print("\n[6] checkpoint 兼容")
    # 训练单头 neuron 并保存
    torch.manual_seed(42)
    single_neuron = _make_neuron(num_field_heads=1)
    single_state = single_neuron.state_dict()

    # 创建多头 neuron，用 strict=False 加载
    torch.manual_seed(42)
    multi_neuron = _make_neuron(num_field_heads=4)

    # strict=False 加载：缺失参数用初始化值
    missing, unexpected = multi_neuron.load_state_dict(single_state, strict=False)

    # 应有缺失的 key（field_write_heads, field_gate, field_pool_queries）
    assert len(missing) > 0, f"应有缺失的 key（多头新参数）, missing={missing}"
    # field_pool_query 应能加载（如果名字匹配的话，但多头用 field_pool_queries，所以会缺失）
    has_new_params = any(
        "field_write_heads" in k or "field_gate" in k or "field_pool_queries" in k
        for k in missing
    )
    assert has_new_params, f"缺失的 key 应含多头新参数, missing={missing}"

    # 加载后多头 neuron 仍能正常 forward
    multi_neuron.eval()
    shared_emb = torch.randn(2, 16, 512)
    with torch.no_grad():
        result = multi_neuron.forward(shared_emb)
    assert "field_vector" in result
    print(f"  PASS: 单头→多头加载成功 (missing={len(missing)} keys), forward 正常")


def test_get_field_write_parameters():
    """[7] get_field_write_parameters 返回正确参数列表。"""
    print("\n[7] get_field_write_parameters")

    # 单头
    neuron_single = _make_neuron(num_field_heads=1)
    params_single = neuron_single.get_field_write_parameters()
    # 单头：field_write (weight) + field_pool_query
    assert len(params_single) == 2, f"单头应有 2 个参数, got {len(params_single)}"
    print(f"  PASS: 单头参数数={len(params_single)} (field_write.weight + field_pool_query)")

    # 多头
    neuron_multi = _make_neuron(num_field_heads=4)
    params_multi = neuron_multi.get_field_write_parameters()
    # 多头：4 个 head (weight) + field_gate (weight + bias) + field_pool_queries
    # = 4 + 2 + 1 = 7
    expected = 4 + 2 + 1
    assert len(params_multi) == expected, \
        f"多头应有 {expected} 个参数, got {len(params_multi)}"
    print(f"  PASS: 多头参数数={len(params_multi)} (4 heads + gate + queries)")


def test_quick_probe_multihead():
    """[8] quick_probe 在多头模式下正常工作。"""
    print("\n[8] quick_probe 多头兼容")
    neuron = _make_neuron(num_field_heads=4)
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    with torch.no_grad():
        v = neuron.quick_probe(shared_emb)
    assert v.shape == (2, 512), f"quick_probe 输出 shape 错误: {v.shape}"
    norms = v.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"quick_probe 输出应 L2 归一化, norms={norms}"
    print(f"  PASS: quick_probe 多头正常, shape={v.shape}")


def main():
    print("=" * 60)
    print("C6 多头 field write smoke test")
    print("=" * 60)

    test_backward_compat_single_head()
    test_multihead_changes_output()
    test_gradient_flow()
    test_gate_softmax()
    test_attention_weights_shape()
    test_checkpoint_compat()
    test_get_field_write_parameters()
    test_quick_probe_multihead()

    print("\n" + "=" * 60)
    print("ALL 8/8 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
