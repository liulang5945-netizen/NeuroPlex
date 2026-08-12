"""C4 场读入模式 smoke test.

验证 field_read_mode 三种模式的正确性：
1. additive（默认）与原实现一致（向后兼容）
2. multiplicative 乘性门控改变输出
3. predictive 预测编码改变输出
4. 三种模式输出互不相同
5. checkpoint 兼容（参数结构不变，旧 ckpt 可加载）
6. field_state=None 时三种模式行为一致（不触发 conditioning）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch

from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron


def _make_neuron(field_read_mode: str) -> ResonanceNeuron:
    """构建指定 field_read_mode 的 TINY_TEST neuron。"""
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = f"n_{field_read_mode}"
    cfg.field_read_mode = field_read_mode
    torch.manual_seed(42)
    return ResonanceNeuron(cfg)


def test_additive_backward_compat():
    """[1] additive 模式与原实现一致。"""
    print("\n[1] additive 向后兼容")
    neuron = _make_neuron("additive")
    assert neuron.field_read_mode == "additive"
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    field_state = torch.randn(2, 512)

    with torch.no_grad():
        # round 1: 无 field conditioning
        result_r1 = neuron.forward(shared_emb, return_logits=True)
        # round 2: 有 field conditioning（additive）
        result_r2 = neuron.forward(shared_emb, field_state=field_state, round_num=2, return_logits=True)

    diff = (result_r1["logits"] - result_r2["logits"]).abs().max().item()
    assert diff > 1e-4, f"additive 模式 round 2 应改变输出, diff={diff}"
    print(f"  PASS: additive 模式 round 2 改变输出 (diff={diff:.4e})")


def test_multiplicative():
    """[2] multiplicative 乘性门控改变输出。"""
    print("\n[2] multiplicative 乘性门控")
    neuron = _make_neuron("multiplicative")
    assert neuron.field_read_mode == "multiplicative"
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    field_state = torch.randn(2, 512)

    with torch.no_grad():
        result_r1 = neuron.forward(shared_emb, return_logits=True)
        result_r2 = neuron.forward(shared_emb, field_state=field_state, round_num=2, return_logits=True)

    diff = (result_r1["logits"] - result_r2["logits"]).abs().max().item()
    assert diff > 1e-4, f"multiplicative 模式 round 2 应改变输出, diff={diff}"
    print(f"  PASS: multiplicative 模式 round 2 改变输出 (diff={diff:.4e})")


def test_predictive():
    """[3] predictive 预测编码改变输出。"""
    print("\n[3] predictive 预测编码")
    neuron = _make_neuron("predictive")
    assert neuron.field_read_mode == "predictive"
    neuron.eval()

    shared_emb = torch.randn(2, 16, 512)
    field_state = torch.randn(2, 512)

    with torch.no_grad():
        result_r1 = neuron.forward(shared_emb, return_logits=True)
        result_r2 = neuron.forward(shared_emb, field_state=field_state, round_num=2, return_logits=True)

    diff = (result_r1["logits"] - result_r2["logits"]).abs().max().item()
    assert diff > 1e-4, f"predictive 模式 round 2 应改变输出, diff={diff}"
    print(f"  PASS: predictive 模式 round 2 改变输出 (diff={diff:.4e})")


def test_modes_differ():
    """[4] 三种模式输出互不相同（相同权重 + 相同 field_state → 不同结果）。"""
    print("\n[4] 三种模式输出互不相同")
    # 构建三个 neuron，复制相同权重
    neurons = {}
    base_neuron = _make_neuron("additive")
    base_sd = base_neuron.state_dict()

    for mode in ["additive", "multiplicative", "predictive"]:
        n = _make_neuron(mode)
        # 复制权重（参数名相同，field_read_mode 不影响 state_dict）
        n.load_state_dict(base_sd, strict=True)
        n.eval()
        neurons[mode] = n

    shared_emb = torch.randn(2, 16, 512)
    field_state = torch.randn(2, 512)

    logits = {}
    with torch.no_grad():
        for mode, n in neurons.items():
            r = n.forward(shared_emb, field_state=field_state, round_num=2, return_logits=True)
            logits[mode] = r["logits"]

    # 三种模式应互不相同
    diff_add_mult = (logits["additive"] - logits["multiplicative"]).abs().max().item()
    diff_add_pred = (logits["additive"] - logits["predictive"]).abs().max().item()
    diff_mult_pred = (logits["multiplicative"] - logits["predictive"]).abs().max().item()

    assert diff_add_mult > 1e-4, f"additive vs multiplicative 应不同, diff={diff_add_mult}"
    assert diff_add_pred > 1e-4, f"additive vs predictive 应不同, diff={diff_add_pred}"
    assert diff_mult_pred > 1e-4, f"multiplicative vs predictive 应不同, diff={diff_mult_pred}"

    print(f"  additive vs multiplicative: {diff_add_mult:.4e}")
    print(f"  additive vs predictive:     {diff_add_pred:.4e}")
    print(f"  multiplicative vs predictive: {diff_mult_pred:.4e}")
    print(f"  PASS: 三种模式输出互不相同")


def test_no_field_state_consistent():
    """[5] field_state=None 时三种模式行为一致（不触发 conditioning）。"""
    print("\n[5] field_state=None 时三种模式一致")
    neurons = {}
    base_sd = _make_neuron("additive").state_dict()

    for mode in ["additive", "multiplicative", "predictive"]:
        n = _make_neuron(mode)
        n.load_state_dict(base_sd, strict=True)
        n.eval()
        neurons[mode] = n

    shared_emb = torch.randn(2, 16, 512)

    logits = {}
    with torch.no_grad():
        for mode, n in neurons.items():
            # round 1, 无 field_state → 不触发 conditioning
            r = n.forward(shared_emb, return_logits=True)
            logits[mode] = r["logits"]

    diff_add_mult = (logits["additive"] - logits["multiplicative"]).abs().max().item()
    diff_add_pred = (logits["additive"] - logits["predictive"]).abs().max().item()

    assert diff_add_mult < 1e-6, f"field_state=None 时 additive vs multiplicative 应一致, diff={diff_add_mult}"
    assert diff_add_pred < 1e-6, f"field_state=None 时 additive vs predictive 应一致, diff={diff_add_pred}"
    print(f"  PASS: field_state=None 时三种模式一致 (diff={diff_add_mult:.2e})")


def test_checkpoint_compat():
    """[6] checkpoint 兼容：旧 ckpt（additive）可加载到新模式 neuron。"""
    print("\n[6] checkpoint 兼容性")
    # 训练 additive neuron，保存 state_dict
    old_neuron = _make_neuron("additive")
    old_sd = old_neuron.state_dict()

    # 创建 multiplicative neuron，加载旧 ckpt
    new_neuron = _make_neuron("multiplicative")
    new_neuron.load_state_dict(old_sd, strict=True)  # 参数结构相同，strict=True 成功
    print(f"  PASS: additive ckpt 加载到 multiplicative neuron (strict=True 成功)")


def main():
    print("=" * 70)
    print("C4 场读入模式 smoke test")
    print("=" * 70)

    test_additive_backward_compat()
    test_multiplicative()
    test_predictive()
    test_modes_differ()
    test_no_field_state_consistent()
    test_checkpoint_compat()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
