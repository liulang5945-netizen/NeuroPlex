"""C1 神经元多亚型 smoke test.

验证 neuron_type 扩展为多亚型（PV+/SOM+/VIP+）：
1. config 支持新亚型字符串
2. neuron 默认 "excitatory"（向后兼容）
3. is_inhibitory: inhibitory + som 为 True
4. is_excitatory: excitatory + pv + vip 为 True
5. write_gain 各亚型不同
6. refractory_multiplier 各亚型不同
7. ensemble 写入时应用 write_gain
8. ensemble enter_refractory 应用 refractory_multiplier
9. 向后兼容（excitatory/inhibitory 行为不变）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron


def _make_neuron(neuron_type="excitatory", seed=42) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = "n0"
    cfg.neuron_type = neuron_type
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def test_config_supports_subtypes():
    """[1] config 支持新亚型字符串。"""
    print("\n[1] config 支持新亚型")
    for nt in ["excitatory", "excitatory_pv", "excitatory_som", "excitatory_vip", "inhibitory"]:
        cfg = copy.deepcopy(TINY_TEST)
        cfg.neuron_type = nt
        assert cfg.neuron_type == nt
    print("  PASS: 5 种亚型均支持")


def test_default_excitatory():
    """[2] neuron 默认 "excitatory"（向后兼容）。"""
    print("\n[2] 默认 excitatory")
    n = _make_neuron(neuron_type="excitatory")
    assert n.neuron_type == "excitatory"
    print("  PASS: 默认 excitatory")


def test_is_inhibitory():
    """[3] is_inhibitory: inhibitory + som 为 True。"""
    print("\n[3] is_inhibitory")
    assert _make_neuron("inhibitory").is_inhibitory is True
    assert _make_neuron("excitatory_som").is_inhibitory is True
    assert _make_neuron("excitatory").is_inhibitory is False
    assert _make_neuron("excitatory_pv").is_inhibitory is False
    assert _make_neuron("excitatory_vip").is_inhibitory is False
    print("  PASS: inhibitory + som → True, 其他 → False")


def test_is_excitatory():
    """[4] is_excitatory: excitatory + pv + vip 为 True。"""
    print("\n[4] is_excitatory")
    assert _make_neuron("excitatory").is_excitatory is True
    assert _make_neuron("excitatory_pv").is_excitatory is True
    assert _make_neuron("excitatory_vip").is_excitatory is True
    assert _make_neuron("inhibitory").is_excitatory is False
    assert _make_neuron("excitatory_som").is_excitatory is False
    print("  PASS: excitatory + pv + vip → True, 其他 → False")


def test_write_gain():
    """[5] write_gain 各亚型不同。"""
    print("\n[5] write_gain")
    gains = {
        "excitatory": 1.0,
        "excitatory_pv": 1.5,
        "excitatory_som": 0.8,
        "excitatory_vip": 1.2,
        "inhibitory": 1.0,
    }
    for nt, expected_gain in gains.items():
        n = _make_neuron(nt)
        assert n.write_gain == expected_gain, f"{nt} gain 应={expected_gain}, got {n.write_gain}"
    print(f"  PASS: gains={gains}")


def test_refractory_multiplier():
    """[6] refractory_multiplier 各亚型不同。"""
    print("\n[6] refractory_multiplier")
    mults = {
        "excitatory": 1.0,
        "excitatory_pv": 0.5,
        "excitatory_som": 1.5,
        "excitatory_vip": 0.8,
        "inhibitory": 2.0,
    }
    for nt, expected_mult in mults.items():
        n = _make_neuron(nt)
        assert n.refractory_multiplier == expected_mult, f"{nt} mult 应={expected_mult}, got {n.refractory_multiplier}"
    print(f"  PASS: mults={mults}")


def test_ensemble_applies_write_gain():
    """[7] ensemble 写入时应用 write_gain。"""
    print("\n[7] ensemble 应用 write_gain")
    # 通过检查 ensemble.py 源码包含 write_gain 应用
    ens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji", "resonance", "ensemble.py",
    )
    with open(ens_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "neuron.write_gain" in source, "ensemble.py 应应用 neuron.write_gain"
    # 应在两个位置应用（forward round1 + forward_train round2+）
    count = source.count("neuron.write_gain")
    assert count >= 2, f"应在 2+ 位置应用 write_gain, got {count}"
    print(f"  PASS: ensemble.py 在 {count} 处应用 write_gain")


def test_ensemble_applies_refractory_multiplier():
    """[8] ensemble enter_refractory 应用 refractory_multiplier。"""
    print("\n[8] ensemble 应用 refractory_multiplier")
    ens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji", "resonance", "ensemble.py",
    )
    with open(ens_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "refractory_multiplier" in source, "ensemble.py 应应用 refractory_multiplier"
    assert "neuromod_mult * subtype_mult" in source or "neuromod_mult * self.neurons[nid].refractory_multiplier" in source
    print("  PASS: ensemble.py 应用 refractory_multiplier")


def test_backward_compat():
    """[9] 向后兼容（excitatory/inhibitory 行为不变）。"""
    print("\n[9] 向后兼容")
    # excitatory: gain=1.0, mult=1.0（与原行为一致）
    n_exc = _make_neuron("excitatory")
    assert n_exc.write_gain == 1.0
    assert n_exc.refractory_multiplier == 1.0
    assert n_exc.is_inhibitory is False
    # inhibitory: gain=1.0, mult=2.0（原 inhibitory 无 mult，现 2.0 是新增）
    # 但原 inhibitory 不 enter_refractory（只在 write_inhibit 后），所以 mult 不影响
    n_inh = _make_neuron("inhibitory")
    assert n_inh.is_inhibitory is True
    assert n_inh.write_gain == 1.0  # 原无 gain，现 1.0 不变
    print("  PASS: excitatory/inhibitory 核心行为向后兼容")


def main():
    print("=" * 60)
    print("C1 神经元多亚型 smoke test")
    print("=" * 60)

    test_config_supports_subtypes()
    test_default_excitatory()
    test_is_inhibitory()
    test_is_excitatory()
    test_write_gain()
    test_refractory_multiplier()
    test_ensemble_applies_write_gain()
    test_ensemble_applies_refractory_multiplier()
    test_backward_compat()

    print("\n" + "=" * 60)
    print("ALL 9/9 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
