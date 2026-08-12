"""T6 跨规格投影层升级为 2 层 MLP smoke test.

验证：
1. CrossSpecProjector 类可 import
2. 结构正确: linear1 + gelu + linear2
3. 零初始化: linear2.weight 初始为 0
4. 初始等价性: y = linear1(x) + 0 = linear1(x)（与旧单层 Linear 一致）
5. 旧 checkpoint 兼容加载 (load_legacy_linear_state)
6. 新 checkpoint 格式 save/load 往返一致
7. 训练后非线性: linear2 学到非零权重后输出 ≠ linear1(x)
8. ensemble 自动创建 CrossSpecProjector（混合 field_dim 时）
9. _project_vec 调用正常（MLP 可调用）
10. finetune load 兼容旧格式（"weight" key → linear1.weight）
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
from taiji.resonance.ensemble import ResonanceEnsemble, CrossSpecProjector


def test_import():
    """[1] CrossSpecProjector 可 import。"""
    print("\n[1] CrossSpecProjector import")
    assert CrossSpecProjector is not None
    print("  PASS: CrossSpecProjector 可导入")


def test_structure():
    """[2] 结构正确: linear1 + gelu + linear2。"""
    print("\n[2] 结构检查")
    proj = CrossSpecProjector(64, 128)
    assert isinstance(proj.linear1, nn.Linear)
    assert proj.linear1.in_features == 64
    assert proj.linear1.out_features == 128
    assert isinstance(proj.gelu, nn.GELU)
    assert isinstance(proj.linear2, nn.Linear)
    assert proj.linear2.in_features == 128
    assert proj.linear2.out_features == 128
    print(f"  PASS: linear1({64}→{128}) + GELU + linear2({128}→{128})")


def test_zero_init():
    """[3] 零初始化: linear2.weight 初始为 0。"""
    print("\n[3] 零初始化")
    proj = CrossSpecProjector(64, 128)
    assert proj.linear2.weight.abs().max().item() == 0.0, "linear2.weight 应零初始化"
    print(f"  PASS: linear2.weight 零初始化 (max={proj.linear2.weight.abs().max().item():.6f})")


def test_initial_equivalence():
    """[4] 初始等价性: y = linear1(x)（与旧单层 Linear 一致）。"""
    print("\n[4] 初始等价性")
    torch.manual_seed(42)
    proj = CrossSpecProjector(64, 128)
    # 旧单层 Linear（相同权重）
    legacy_linear = nn.Linear(64, 128, bias=False)
    legacy_linear.weight.data.copy_(proj.linear1.weight.data)

    x = torch.randn(4, 64)
    y_mlp = proj(x)
    y_legacy = legacy_linear(x)
    diff = (y_mlp - y_legacy).abs().max().item()
    assert diff < 1e-6, f"初始 MLP 输出应与旧 Linear 一致, diff={diff}"
    print(f"  PASS: 初始 MLP ≡ 旧 Linear, diff={diff:.8f}")


def test_legacy_ckpt_compat():
    """[5] 旧 checkpoint 兼容加载 (load_legacy_linear_state)。"""
    print("\n[5] 旧 checkpoint 兼容加载")
    # 模拟旧单层 Linear 的权重
    legacy_weight = torch.randn(128, 64)  # [out_dim, in_dim]
    proj = CrossSpecProjector(64, 128)
    proj.load_legacy_linear_state(legacy_weight)
    # 验证 linear1.weight 已加载
    assert torch.allclose(proj.linear1.weight.data, legacy_weight)
    # 验证 linear2 仍零初始化
    assert proj.linear2.weight.abs().max().item() == 0.0
    # 验证加载后行为等价于旧 Linear
    x = torch.randn(4, 64)
    legacy_linear = nn.Linear(64, 128, bias=False)
    legacy_linear.weight.data.copy_(legacy_weight)
    diff = (proj(x) - legacy_linear(x)).abs().max().item()
    assert diff < 1e-6, f"加载后 MLP 应与旧 Linear 等价, diff={diff}"
    print(f"  PASS: 旧 ckpt 加载后 MLP ≡ 旧 Linear, diff={diff:.8f}")


def test_new_ckpt_roundtrip():
    """[6] 新 checkpoint 格式 save/load 往返一致。"""
    print("\n[6] 新 checkpoint 往返")
    proj1 = CrossSpecProjector(64, 128)
    # 扰动权重（模拟训练后状态）
    with torch.no_grad():
        proj1.linear1.weight.add_(torch.randn_like(proj1.linear1.weight) * 0.1)
        proj1.linear2.weight.add_(torch.randn_like(proj1.linear2.weight) * 0.1)
    sd = proj1.state_dict()
    # 验证新格式包含 linear1.weight 和 linear2.weight
    assert "linear1.weight" in sd
    assert "linear2.weight" in sd

    proj2 = CrossSpecProjector(64, 128)
    proj2.load_state_dict(sd)
    x = torch.randn(4, 64)
    diff = (proj1(x) - proj2(x)).abs().max().item()
    assert diff < 1e-6, f"save/load 往返应一致, diff={diff}"
    print(f"  PASS: 新格式 save/load 往返一致, diff={diff:.8f}")


def test_nonlinear_after_training():
    """[7] 训练后非线性: linear2 学到非零权重后输出 ≠ linear1(x)。"""
    print("\n[7] 训练后非线性")
    proj = CrossSpecProjector(64, 128)
    # 模拟训练: linear2 学到非零权重
    with torch.no_grad():
        proj.linear2.weight.add_(torch.randn_like(proj.linear2.weight) * 0.5)
    x = torch.randn(4, 64)
    y_mlp = proj(x)
    y_linear1_only = proj.linear1(x)
    diff = (y_mlp - y_linear1_only).abs().max().item()
    assert diff > 1e-4, f"训练后 MLP 应 ≠ linear1(x), diff={diff}"
    print(f"  PASS: 训练后 MLP ≠ linear1(x), diff={diff:.4f}（非线性生效）")


def test_ensemble_creates_mlp():
    """[8] ensemble 自动创建 CrossSpecProjector（混合 field_dim 时）。"""
    print("\n[8] ensemble 创建 MLP")
    # 创建两个不同 field_dim 的神经元
    cfg1 = copy.deepcopy(TINY_TEST)
    cfg1.vocab_size = 100
    cfg1.neuron_id = "n0"
    cfg1.field_dim = 64
    cfg2 = copy.deepcopy(TINY_TEST)
    cfg2.vocab_size = 100
    cfg2.neuron_id = "n1"
    cfg2.field_dim = 128  # 不同 field_dim

    torch.manual_seed(42)
    n0 = ResonanceNeuron(cfg1)
    n1 = ResonanceNeuron(cfg2)
    field = ResonanceField(dim=128)  # unified_dim = max
    ens = ResonanceEnsemble({"n0": n0, "n1": n1}, field)

    # n0 的 field_dim=64 ≠ unified=128，应创建 CrossSpecProjector
    assert "n0" in ens._cross_spec_projectors
    proj = ens._cross_spec_projectors["n0"]
    assert isinstance(proj, CrossSpecProjector)
    assert proj.linear1.in_features == 64
    assert proj.linear1.out_features == 128
    # n1 的 field_dim=128 = unified，不应创建投影层
    assert "n1" not in ens._cross_spec_projectors
    print(f"  PASS: n0 创建 CrossSpecProjector(64→128), n1 无投影层")


def test_project_vec_works():
    """[9] _project_vec 调用正常（MLP 可调用）。"""
    print("\n[9] _project_vec 调用")
    cfg1 = copy.deepcopy(TINY_TEST)
    cfg1.vocab_size = 100
    cfg1.neuron_id = "n0"
    cfg1.field_dim = 64
    cfg2 = copy.deepcopy(TINY_TEST)
    cfg2.vocab_size = 100
    cfg2.neuron_id = "n1"
    cfg2.field_dim = 128

    torch.manual_seed(42)
    n0 = ResonanceNeuron(cfg1)
    n1 = ResonanceNeuron(cfg2)
    field = ResonanceField(dim=128)
    ens = ResonanceEnsemble({"n0": n0, "n1": n1}, field)

    # n0 的 vec 应被投影到 128 维
    vec_raw = torch.randn(2, 64)
    vec_projected = ens._project_vec("n0", vec_raw)
    assert vec_projected.shape == (2, 128), f"投影后应为 [2,128], got {vec_projected.shape}"
    # n1 的 vec 应直接返回（无投影层）
    vec_n1 = torch.randn(2, 128)
    vec_n1_projected = ens._project_vec("n1", vec_n1)
    assert vec_n1_projected.shape == (2, 128)
    assert torch.allclose(vec_n1_projected, vec_n1)
    print(f"  PASS: n0 投影 {vec_raw.shape}→{vec_projected.shape}, n1 直通")


def test_finetune_legacy_load():
    """[10] finetune load 兼容旧格式（"weight" key → linear1.weight）。"""
    print("\n[10] finetune 旧格式兼容加载")
    # 模拟旧 checkpoint 的 cross_spec_state 格式
    legacy_sd = {"weight": torch.randn(128, 64)}  # 旧格式: {"weight": tensor}

    # 创建 ensemble 含 CrossSpecProjector
    cfg1 = copy.deepcopy(TINY_TEST)
    cfg1.vocab_size = 100
    cfg1.neuron_id = "n0"
    cfg1.field_dim = 64
    cfg2 = copy.deepcopy(TINY_TEST)
    cfg2.vocab_size = 100
    cfg2.neuron_id = "n1"
    cfg2.field_dim = 128
    torch.manual_seed(42)
    n0 = ResonanceNeuron(cfg1)
    n1 = ResonanceNeuron(cfg2)
    field = ResonanceField(dim=128)
    ens = ResonanceEnsemble({"n0": n0, "n1": n1}, field)

    # 模拟 finetune_cross_spec.py 的兼容加载逻辑
    proj = ens._cross_spec_projectors["n0"]
    if "weight" in legacy_sd and "linear1.weight" not in legacy_sd:
        proj.load_legacy_linear_state(legacy_sd["weight"])
    else:
        proj.load_state_dict(legacy_sd)

    # 验证加载成功
    assert torch.allclose(proj.linear1.weight.data, legacy_sd["weight"])
    assert proj.linear2.weight.abs().max().item() == 0.0
    print(f"  PASS: finetune 兼容加载旧格式, linear1 已加载, linear2 保持零初始化")


def main():
    print("=" * 60)
    print("T6 跨规格投影层升级为 2 层 MLP smoke test")
    print("=" * 60)

    test_import()
    test_structure()
    test_zero_init()
    test_initial_equivalence()
    test_legacy_ckpt_compat()
    test_new_ckpt_roundtrip()
    test_nonlinear_after_training()
    test_ensemble_creates_mlp()
    test_project_vec_works()
    test_finetune_legacy_load()

    print("\n" + "=" * 60)
    print("ALL 10/10 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
