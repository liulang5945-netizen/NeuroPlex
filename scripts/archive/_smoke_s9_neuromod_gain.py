"""S9 神经调质门控 attention/FFN smoke test.

验证神经调质（norepinephrine/dopamine）正确门控 Transformer 内部计算：
1. NeuromodulatorState.get_attention_temp_gain / get_ffn_gain 返回值正确
2. gain=1.0 时与不传 gain 输出一致（向后兼容）
3. gain≠1.0 时 Transformer 输出确实改变
4. ensemble.forward_train 正确传递 gain（调质极端值影响输出）

用 TINY_TEST neuron 避免加载真实模型，聚焦 gain 注入逻辑正确性。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.neuro_modulation import NeuromodulatorState


def make_tiny_neuron(neuron_id: str) -> ResonanceNeuron:
    """构造一个 TINY_TEST neuron，vocab 缩小到 100 加速测试。"""
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = neuron_id
    torch.manual_seed(42)
    return ResonanceNeuron(cfg)


def test_neuromodulator_gain_methods():
    """[1] NeuromodulatorState 新方法返回值正确性。"""
    print("\n[1] NeuromodulatorState.get_attention_temp_gain / get_ffn_gain 返回值")
    nm = NeuromodulatorState()

    # 默认 NE=0.5, DA=0.5
    assert abs(nm.get_attention_temp_gain() - 1.0) < 1e-6, "NE=0.5 → temp_gain 应为 1.0"
    assert abs(nm.get_ffn_gain() - 1.0) < 1e-6, "DA=0.5 → ffn_gain 应为 1.0（训练/推理中性基线）"

    # NE 极端值（直接设置属性，跳过 EMA 收敛等待）
    nm.norepinephrine = 1.0
    assert abs(nm.get_attention_temp_gain() - 1.5) < 1e-6, "NE=1.0 → temp_gain 应为 1.5"

    nm.norepinephrine = 0.0
    assert abs(nm.get_attention_temp_gain() - 0.5) < 1e-6, "NE=0.0 → temp_gain 应为 0.5"

    # DA 极端值
    nm2 = NeuromodulatorState()
    nm2.dopamine = 1.0
    assert abs(nm2.get_ffn_gain() - 2.0) < 1e-6, "DA=1.0 → ffn_gain 应为 2.0"

    nm2.dopamine = 0.0
    assert abs(nm2.get_ffn_gain() - 0.5) < 1e-6, "DA=0.0 → ffn_gain 应为 0.5"

    print("  PASS: temp_gain/ffn_gain 映射正确（NE/DA 极端值）")


def test_neuron_backward_compat():
    """[2] gain=1.0 时与不传 gain 输出一致（向后兼容）。"""
    print("\n[2] gain=1.0 向后兼容性")
    neuron = make_tiny_neuron("test_n0")
    neuron.eval()

    torch.manual_seed(0)
    shared_emb = torch.randn(2, 16, 512)  # [B, L, base_embed_dim]

    with torch.no_grad():
        # 不传 gain（默认值 1.0）
        result_default = neuron.forward(shared_emb, return_logits=True)
        # 显式传 gain=1.0
        result_explicit = neuron.forward(
            shared_emb, return_logits=True, temp_gain=1.0, ffn_gain=1.0,
        )

    logits_diff = (result_default["logits"] - result_explicit["logits"]).abs().max().item()
    vec_diff = (result_default["field_vector"] - result_explicit["field_vector"]).abs().max().item()
    assert logits_diff < 1e-6, f"gain=1.0 时 logits 应一致，diff={logits_diff}"
    assert vec_diff < 1e-6, f"gain=1.0 时 field_vector 应一致，diff={vec_diff}"
    print(f"  PASS: gain=1.0 与默认输出一致 (logits_diff={logits_diff:.2e}, vec_diff={vec_diff:.2e})")


def test_gain_changes_output():
    """[3] gain≠1.0 时 Transformer 输出确实改变。"""
    print("\n[3] gain≠1.0 改变 Transformer 输出")
    neuron = make_tiny_neuron("test_n1")
    neuron.eval()

    torch.manual_seed(0)
    shared_emb = torch.randn(2, 16, 512)

    with torch.no_grad():
        result_baseline = neuron.forward(shared_emb, return_logits=True, temp_gain=1.0, ffn_gain=1.0)
        # temp_gain 改变（注意力温度）
        result_temp = neuron.forward(shared_emb, return_logits=True, temp_gain=1.5, ffn_gain=1.0)
        # ffn_gain 改变（FFN 输出强度）
        result_ffn = neuron.forward(shared_embeddings=shared_emb, return_logits=True, temp_gain=1.0, ffn_gain=1.5)

    temp_diff = (result_baseline["logits"] - result_temp["logits"]).abs().max().item()
    ffn_diff = (result_baseline["logits"] - result_ffn["logits"]).abs().max().item()
    assert temp_diff > 1e-4, f"temp_gain=1.5 应改变 logits，diff={temp_diff}"
    assert ffn_diff > 1e-4, f"ffn_gain=1.5 应改变 logits，diff={ffn_diff}"
    print(f"  PASS: temp_gain=1.5 改变输出 (diff={temp_diff:.4e})")
    print(f"  PASS: ffn_gain=1.5 改变输出 (diff={ffn_diff:.4e})")


def test_ensemble_forward_train_passes_gain():
    """[4] ensemble.forward_train 正确传递 gain（调质极端值影响输出）。"""
    print("\n[4] ensemble.forward_train 传递 gain")
    n0 = make_tiny_neuron("n0")
    n1 = make_tiny_neuron("n1")
    neurons = {"n0": n0, "n1": n1}

    field = ResonanceField(dim=512)
    field.reset(batch_size=2)

    # 构造 side_channels（让协作路径生效）
    n0.establish_side_channel("n1", n1, channel_type="excite")
    n1.establish_side_channel("n0", n0, channel_type="excite")

    shared_emb = torch.randn(2, 16, 512)

    # 场景 A：中性调质（NE=DA=0.5 → temp_gain=1.0, ffn_gain=1.0）
    nm_neutral = NeuromodulatorState()
    nm_neutral.dopamine = 0.5
    nm_neutral.norepinephrine = 0.5

    ensemble_a = ResonanceEnsemble(
        neurons={"n0": n0, "n1": n1},
        field=field,
        neuromodulator=nm_neutral,
    )

    # 场景 B：极端调质（NE=1.0, DA=1.0 → temp_gain=1.5, ffn_gain=2.0）
    nm_extreme = NeuromodulatorState()
    nm_extreme.dopamine = 1.0
    nm_extreme.norepinephrine = 1.0

    ensemble_b = ResonanceEnsemble(
        neurons={"n0": n0, "n1": n1},
        field=field,
        neuromodulator=nm_extreme,
    )

    with torch.no_grad():
        result_a = ensemble_a.forward_train(shared_embeddings=shared_emb, n_rounds=2)
        result_b = ensemble_b.forward_train(shared_embeddings=shared_emb, n_rounds=2)

    fused_diff = (result_a["fused_logits"] - result_b["fused_logits"]).abs().max().item()
    assert fused_diff > 1e-4, f"不同调质水平应改变 fused_logits，diff={fused_diff}"
    print(f"  PASS: 调质极端值改变 ensemble 输出 (fused_diff={fused_diff:.4e})")


def main():
    print("=" * 70)
    print("S9 神经调质门控 attention/FFN smoke test")
    print("=" * 70)

    test_neuromodulator_gain_methods()
    test_neuron_backward_compat()
    test_gain_changes_output()
    test_ensemble_forward_train_passes_gain()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
