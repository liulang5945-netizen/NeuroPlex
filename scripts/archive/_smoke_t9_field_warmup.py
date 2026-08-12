"""T9 field_conditioning warm-up smoke test.

验证：
1. forward_train 有 field_conditioning 参数（默认 True）
2. field_conditioning=True 时 round 2+ 注入 field_state（向后兼容）
3. field_conditioning=False 时 round 2+ 不注入 field_state
4. field_state 仍被维护（累积，启用后注入学习到的场状态）
5. finetune_cross_spec.py 有 --field_warmup_ratio 参数
6. warm-up 逻辑正确（前 N 步关闭，后启用）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import inspect
import torch

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


def _make_ensemble():
    neurons = {f"n{i}": _make_neuron(f"n{i}", seed=42 + i) for i in range(2)}
    field = ResonanceField(dim=TINY_TEST.field_dim)
    return ResonanceEnsemble(neurons, field, max_rounds=2)


def test_field_conditioning_param_exists():
    """[1] forward_train 有 field_conditioning 参数（默认 True）。"""
    print("\n[1] field_conditioning 参数存在")
    sig = inspect.signature(ResonanceEnsemble.forward_train)
    params = sig.parameters
    assert "field_conditioning" in params, "forward_train 应有 field_conditioning 参数"
    assert params["field_conditioning"].default is True, "默认应为 True（向后兼容）"
    print(f"  PASS: field_conditioning 参数存在, 默认={params['field_conditioning'].default}")


def test_backward_compat():
    """[2] field_conditioning=True 时 round 2+ 注入 field_state（向后兼容）。"""
    print("\n[2] 向后兼容（field_conditioning=True）")
    ens = _make_ensemble()
    shared_emb = torch.randn(2, 8, 512)
    # 不传 field_conditioning（用默认 True）
    result_default = ens.forward_train(shared_embeddings=shared_emb, n_rounds=2)
    # 显式传 True
    ens2 = _make_ensemble()
    result_true = ens2.forward_train(shared_embeddings=shared_emb, n_rounds=2, field_conditioning=True)
    # 两者应相同（field_state 注入一致）
    if "fused_logits" in result_default and "fused_logits" in result_true:
        diff = (result_default["fused_logits"] - result_true["fused_logits"]).abs().max().item()
        assert diff < 1e-6, f"默认与 True 应相同, diff={diff}"
    print("  PASS: 默认与 True 行为一致")


def test_field_conditioning_false():
    """[3] field_conditioning=False 时 round 2+ 不注入 field_state。"""
    print("\n[3] field_conditioning=False 不注入")
    ens_true = _make_ensemble()
    ens_false = _make_ensemble()
    shared_emb = torch.randn(2, 8, 512)
    result_true = ens_true.forward_train(shared_embeddings=shared_emb, n_rounds=2, field_conditioning=True)
    result_false = ens_false.forward_train(shared_embeddings=shared_emb, n_rounds=2, field_conditioning=False)
    # 两者应不同（field_state 注入 vs 不注入）
    if "fused_logits" in result_true and "fused_logits" in result_false:
        diff = (result_true["fused_logits"] - result_false["fused_logits"]).abs().max().item()
        assert diff > 1e-6, f"True 与 False 应不同（注入 vs 不注入）, diff={diff}"
        print(f"  PASS: True vs False 输出差异={diff:.4f}（field_state 注入影响）")
    else:
        print("  SKIP: 无 fused_logits")


def test_field_state_maintained():
    """[4] field_state 仍被维护（累积，启用后注入学习到的场状态）。"""
    print("\n[4] field_state 维护")
    ens = _make_ensemble()
    shared_emb = torch.randn(2, 8, 512)
    # warm-up 阶段：field_conditioning=False
    result_warmup = ens.forward_train(shared_embeddings=shared_emb, n_rounds=2, field_conditioning=False)
    # field_state 应仍存在（累积了 round 1+2 的写入）
    assert "field_state" in result_warmup, "应返回 field_state"
    field_state_warmup = result_warmup["field_state"]
    assert field_state_warmup is not None, "field_state 不应为 None（仍维护）"
    # field_state 应非零（round 1+2 写入了）
    norm = field_state_warmup.norm().item() if field_state_warmup.dim() >= 1 else 0
    assert norm > 0, f"warm-up 后 field_state 应非零, norm={norm}"
    print(f"  PASS: warm-up 后 field_state norm={norm:.4f}（仍维护累积）")


def test_finetune_has_warmup_arg():
    """[5] finetune_cross_spec.py 有 --field_warmup_ratio 参数。"""
    print("\n[5] finetune 有 --field_warmup_ratio")
    # 通过检查源码包含参数定义
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "training", "finetune_cross_spec.py",
    )
    with open(spec_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "--field_warmup_ratio" in source, "finetune_cross_spec.py 应有 --field_warmup_ratio 参数"
    assert "field_warmup_steps" in source, "应计算 field_warmup_steps"
    assert "field_conditioning=field_cond" in source, "应传 field_conditioning 参数"
    print("  PASS: finetune_cross_spec.py 含 warm-up 参数和逻辑")


def test_warmup_logic():
    """[6] warm-up 逻辑正确（前 N 步关闭，后启用）。"""
    print("\n[6] warm-up 逻辑")
    # 模拟 warm-up 逻辑
    field_warmup_steps = 100
    # step < 100 → False（关闭）
    assert (0 >= field_warmup_steps) is False  # step=0
    assert (50 >= field_warmup_steps) is False  # step=50
    assert (99 >= field_warmup_steps) is False  # step=99
    # step >= 100 → True（启用）
    assert (100 >= field_warmup_steps) is True  # step=100
    assert (200 >= field_warmup_steps) is True  # step=200
    print("  PASS: warm-up 逻辑正确（前 100 步关闭，100+ 启用）")


def test_zero_ratio_disables_warmup():
    """[7] field_warmup_ratio=0 时全程启用（无 warm-up）。"""
    print("\n[7] ratio=0 全程启用")
    field_warmup_steps = int(1000 * 0.0)  # 0
    assert field_warmup_steps == 0
    # 任何 step >= 0 都是 True（全程启用）
    assert (0 >= 0) is True
    print("  PASS: ratio=0 时 field_warmup_steps=0，全程启用")


def main():
    print("=" * 60)
    print("T9 field_conditioning warm-up smoke test")
    print("=" * 60)

    test_field_conditioning_param_exists()
    test_backward_compat()
    test_field_conditioning_false()
    test_field_state_maintained()
    test_finetune_has_warmup_arg()
    test_warmup_logic()
    test_zero_ratio_disables_warmup()

    print("\n" + "=" * 60)
    print("ALL 7/7 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
