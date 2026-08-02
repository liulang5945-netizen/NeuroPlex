"""S10 树突化 Transformer smoke test.

验证树突化（Dendritic）扩展的正确性：
1. dendritic=False 时与标准 TransformerBlock 行为一致（向后兼容）
2. dendritic=True 时 apical 路径确实改变输出
3. field_state=None 时树突化块退化为标准行为（round 1 安全）
4. neuron 级别：dendritic neuron 接收 field_state 时输出改变
5. checkpoint 兼容：标准 ckpt 加载到 dendritic neuron（strict=False 不报错）

用 TINY_TEST neuron 避免加载真实模型，聚焦树突化逻辑正确性。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn as nn

from taiji.layers import TransformerBlock
from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron


def test_block_backward_compat():
    """[1] dendritic=False 时与标准 TransformerBlock 行为一致。"""
    print("\n[1] dendritic=False 向后兼容性")
    torch.manual_seed(42)
    block_std = TransformerBlock(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        intermediate_size=512, dendritic=False,
    )
    block_dend = TransformerBlock(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        intermediate_size=512, dendritic=False,  # False, 不创建 apical
    )
    # 复制权重确保相同
    block_dend.load_state_dict(block_std.state_dict())

    block_std.eval()
    block_dend.eval()
    x = torch.randn(2, 16, 256)
    with torch.no_grad():
        out_std, _ = block_std(x)
        out_dend, _ = block_dend(x)
    diff = (out_std - out_dend).abs().max().item()
    assert diff < 1e-6, f"dendritic=False 应与标准块一致, diff={diff}"
    print(f"  PASS: dendritic=False 与标准块一致 (diff={diff:.2e})")


def test_dendritic_changes_output():
    """[2] dendritic=True 时 apical 路径改变输出。"""
    print("\n[2] dendritic=True apical 路径生效")
    torch.manual_seed(42)
    block = TransformerBlock(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        intermediate_size=512, dendritic=True, apical_kv_dim=512,
    )
    block.eval()
    x = torch.randn(2, 16, 256)
    field_state = torch.randn(2, 512)  # [B, field_dim]

    with torch.no_grad():
        # field_state=None: 退化为基础行为（仅 basal）
        out_no_field, _ = block(x)
        # field_state 非 None: apical 路径激活
        out_with_field, _ = block(x, field_state=field_state)

    diff = (out_no_field - out_with_field).abs().max().item()
    assert diff > 1e-4, f"apical 路径应改变输出, diff={diff}"
    print(f"  PASS: apical 路径改变输出 (diff={diff:.4e})")


def test_dendritic_field_none_safe():
    """[3] field_state=None 时树突化块退化为标准行为（round 1 安全）。"""
    print("\n[3] field_state=None 时树突化块安全退化")
    torch.manual_seed(42)
    # 创建标准块和树突化块，复制 basal 权重
    block_std = TransformerBlock(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        intermediate_size=512, dendritic=False,
    )
    block_dend = TransformerBlock(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        intermediate_size=512, dendritic=True, apical_kv_dim=512,
    )
    # 只复制 basal 路径权重（apical 保持初始化）
    std_sd = block_std.state_dict()
    dend_sd = block_dend.state_dict()
    for k, v in std_sd.items():
        if k in dend_sd:
            dend_sd[k] = v
    block_dend.load_state_dict(dend_sd)

    block_std.eval()
    block_dend.eval()
    x = torch.randn(2, 16, 256)
    with torch.no_grad():
        out_std, _ = block_std(x)
        # field_state=None: 树突化块应退化为 basal-only
        out_dend, _ = block_dend(x, field_state=None)

    diff = (out_std - out_dend).abs().max().item()
    assert diff < 1e-6, f"field_state=None 时应与标准块一致, diff={diff}"
    print(f"  PASS: field_state=None 时树突化退化为 basal-only (diff={diff:.2e})")


def test_neuron_dendritic():
    """[4] neuron 级别：dendritic neuron 接收 field_state 时输出改变。"""
    print("\n[4] neuron 级别树突化")
    # 标准 neuron
    cfg_std = copy.deepcopy(TINY_TEST)
    cfg_std.vocab_size = 100
    cfg_std.neuron_id = "n_std"
    cfg_std.dendritic_enabled = False
    torch.manual_seed(42)
    neuron_std = ResonanceNeuron(cfg_std)

    # 树突化 neuron
    cfg_dend = copy.deepcopy(TINY_TEST)
    cfg_dend.vocab_size = 100
    cfg_dend.neuron_id = "n_dend"
    cfg_dend.dendritic_enabled = True
    torch.manual_seed(42)
    neuron_dend = ResonanceNeuron(cfg_dend)

    # 复制 basal 权重（共享参数名部分）
    std_sd = neuron_std.state_dict()
    dend_sd = neuron_dend.state_dict()
    shared_keys = set(std_sd.keys()) & set(dend_sd.keys())
    for k in shared_keys:
        if std_sd[k].shape == dend_sd[k].shape:
            dend_sd[k] = std_sd[k]
    neuron_dend.load_state_dict(dend_sd, strict=False)

    neuron_std.eval()
    neuron_dend.eval()
    shared_emb = torch.randn(2, 16, 512)
    field_state = torch.randn(2, 512)  # field_dim=512 for TINY_TEST

    with torch.no_grad():
        # 标准神经元（不传 field_state，round 1）
        result_std = neuron_std.forward(shared_emb, return_logits=True)
        # 树突化神经元，field_state=None（round 1，apical 不激活）
        result_dend_no_field = neuron_dend.forward(shared_emb, return_logits=True)
        # 树突化神经元，有 field_state（round 2+，apical 激活）
        result_dend_with_field = neuron_dend.forward(
            shared_emb, field_state=field_state, round_num=2, return_logits=True,
        )

    # field_state=None 时两者应接近（basal 权重相同）
    diff_no_field = (result_std["logits"] - result_dend_no_field["logits"]).abs().max().item()
    # 有 field_state 时输出应改变
    diff_with_field = (result_dend_no_field["logits"] - result_dend_with_field["logits"]).abs().max().item()

    print(f"  field_state=None: std vs dend diff={diff_no_field:.2e}")
    print(f"  field_state=非None: dend 改变 diff={diff_with_field:.4e}")
    assert diff_with_field > 1e-4, f"树突化 neuron 接收 field_state 时输出应改变, diff={diff_with_field}"
    print(f"  PASS: neuron 级别树突化生效")


def test_checkpoint_compat():
    """[5] checkpoint 兼容：标准 ckpt 加载到 dendritic neuron。"""
    print("\n[5] checkpoint 兼容性")
    # 训练一个标准 neuron，保存 state_dict
    cfg_std = copy.deepcopy(TINY_TEST)
    cfg_std.vocab_size = 100
    cfg_std.neuron_id = "n_ckpt"
    cfg_std.dendritic_enabled = False
    torch.manual_seed(42)
    neuron_std = ResonanceNeuron(cfg_std)
    std_sd = neuron_std.state_dict()

    # 创建树突化 neuron，用 strict=False 加载标准 ckpt
    cfg_dend = copy.deepcopy(TINY_TEST)
    cfg_dend.vocab_size = 100
    cfg_dend.neuron_id = "n_ckpt_dend"
    cfg_dend.dendritic_enabled = True
    torch.manual_seed(0)  # 不同种子，确保 apical 参数初始化值不同
    neuron_dend = ResonanceNeuron(cfg_dend)

    # 记录加载前的 apical 参数值
    apical_wq_before = neuron_dend.layers[0].apical_wq.weight.clone()

    # strict=False 加载（模拟旧 ckpt 加载到新 dendritic neuron）
    missing, unexpected = neuron_dend.load_state_dict(std_sd, strict=False)
    # 应有 missing keys（apical 参数）
    assert len(missing) > 0, f"应有 missing keys（apical 参数）, got {missing}"
    # 不应有 unexpected keys
    assert len(unexpected) == 0, f"不应有 unexpected keys, got {unexpected}"

    # apical 参数应保持初始化值（未被覆盖）
    apical_wq_after = neuron_dend.layers[0].apical_wq.weight.clone()
    apical_diff = (apical_wq_before - apical_wq_after).abs().max().item()
    assert apical_diff < 1e-8, f"apical 参数应保持初始化值, diff={apical_diff}"

    print(f"  PASS: 标准 ckpt 加载到 dendritic neuron (missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"  apical 参数保持初始化值 (diff={apical_diff:.2e})")


def main():
    print("=" * 70)
    print("S10 树突化 Transformer smoke test")
    print("=" * 70)

    test_block_backward_compat()
    test_dendritic_changes_output()
    test_dendritic_field_none_safe()
    test_neuron_dendritic()
    test_checkpoint_compat()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
