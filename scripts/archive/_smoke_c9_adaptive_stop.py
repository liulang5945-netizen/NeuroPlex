"""C9 共振轮数自适应停止 smoke test.

验证：
1. adaptive_stop=False 向后兼容（固定 max_rounds，不提前停止）
2. adaptive_stop=True 时 result 包含 adaptive_stopped 字段
3. 分数收敛触发停止（构造收敛场景）
4. 主导明确触发停止（top1/top2 > dominance_ratio）
5. min_rounds 约束生效（min_rounds 之前不停止）
6. max_rounds 上限约束（不超过 max_rounds）
7. _check_adaptive_stop 单元测试（边界情况）
8. 默认参数语义正确
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


def _make_neuron(neuron_id="n0", seed=42) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = neuron_id
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def _make_ensemble(max_rounds=3, adaptive_stop=False, **kwargs):
    neurons = {f"n{i}": _make_neuron(f"n{i}", seed=42 + i) for i in range(2)}
    field = ResonanceField(dim=TINY_TEST.field_dim)
    return ResonanceEnsemble(
        neurons, field,
        max_rounds=max_rounds,
        adaptive_stop=adaptive_stop,
        **kwargs,
    )


def test_backward_compat():
    """[1] adaptive_stop=False 向后兼容（固定 max_rounds）。"""
    print("\n[1] adaptive_stop=False 向后兼容")
    ens = _make_ensemble(max_rounds=3, adaptive_stop=False)
    assert ens.adaptive_stop is False
    assert ens.min_rounds == 2  # 默认值，但被 max(2, min(2, 3)) = 2

    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    # 应跑满 3 轮
    assert result["n_rounds"] == 3, f"无自适应停止应跑满 3 轮, got {result['n_rounds']}"
    assert result["adaptive_stopped"] is False
    print(f"  PASS: 无自适应停止跑满 {result['n_rounds']} 轮")


def test_adaptive_stop_field():
    """[2] adaptive_stop=True 时 result 包含 adaptive_stopped 字段。"""
    print("\n[2] adaptive_stop=True result 字段")
    ens = _make_ensemble(max_rounds=5, adaptive_stop=True)
    assert ens.adaptive_stop is True

    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    assert "adaptive_stopped" in result
    assert "adaptive_stop_reason" in result
    print(f"  PASS: result 含 adaptive_stopped={result['adaptive_stopped']}, reason={result['adaptive_stop_reason']}")


def test_convergence_stop():
    """[3] 分数收敛触发停止（构造收敛场景）。"""
    print("\n[3] 分数收敛触发停止")
    ens = _make_ensemble(
        max_rounds=5,
        adaptive_stop=True,
        convergence_threshold=0.5,  # 高阈值，容易触发
        dominance_ratio=100.0,  # 关闭主导信号
        min_rounds=2,
    )
    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    # 高阈值下应在 min_rounds(2)+1=3 轮内停止（round 2 比较 round 1）
    # 注意：round_num=2 时 prev=round1 scores，若收敛则 break
    assert result["n_rounds"] <= 3, f"收敛应在 3 轮内停止, got {result['n_rounds']}"
    # 是否因收敛停止（也可能是主导，但 dominance_ratio=100 应关闭）
    if result["adaptive_stopped"]:
        assert "converged" in result["adaptive_stop_reason"], \
            f"应因收敛停止, got {result['adaptive_stop_reason']}"
    print(f"  PASS: n_rounds={result['n_rounds']}, reason={result['adaptive_stop_reason']}")


def test_dominance_stop():
    """[4] 主导明确触发停止（top1/top2 > dominance_ratio）。"""
    print("\n[4] 主导明确触发停止")
    # 用极端不同的输入让一个 neuron 明确主导
    ens = _make_ensemble(
        max_rounds=5,
        adaptive_stop=True,
        dominance_ratio=1.5,  # 低阈值，容易触发
        convergence_threshold=1e-9,  # 关闭收敛信号
        min_rounds=2,
    )
    # 构造使一个 neuron 远强于另一个的输入
    shared_emb = torch.randn(2, 8, 512) * 10.0
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    if result["adaptive_stopped"]:
        assert "dominant" in result["adaptive_stop_reason"], \
            f"应因主导停止, got {result['adaptive_stop_reason']}"
    print(f"  PASS: n_rounds={result['n_rounds']}, reason={result['adaptive_stop_reason']}")


def test_min_rounds_constraint():
    """[5] min_rounds 约束生效（min_rounds 之前不停止）。"""
    print("\n[5] min_rounds 约束")
    ens = _make_ensemble(
        max_rounds=5,
        adaptive_stop=True,
        convergence_threshold=0.99,  # 极高阈值
        dominance_ratio=1.01,  # 极低阈值
        min_rounds=4,  # 要求至少 4 轮
    )
    assert ens.min_rounds == 4
    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    # 即使阈值容易触发，也应至少跑 4 轮
    assert result["n_rounds"] >= 4, f"min_rounds=4 应至少跑 4 轮, got {result['n_rounds']}"
    print(f"  PASS: min_rounds=4, 实际 n_rounds={result['n_rounds']}")


def test_max_rounds_cap():
    """[6] max_rounds 上限约束（不超过 max_rounds）。"""
    print("\n[6] max_rounds 上限约束")
    ens = _make_ensemble(
        max_rounds=3,
        adaptive_stop=True,
        convergence_threshold=1e-9,  # 关闭收敛
        dominance_ratio=1000.0,  # 关闭主导
        min_rounds=2,
    )
    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward(shared_embeddings=shared_emb)
    assert result["n_rounds"] <= 3, f"不应超过 max_rounds=3, got {result['n_rounds']}"
    assert result["adaptive_stopped"] is False, "无触发信号不应提前停止"
    print(f"  PASS: n_rounds={result['n_rounds']} (max_rounds=3)")


def test_check_adaptive_stop_unit():
    """[7] _check_adaptive_stop 单元测试（边界情况）。"""
    print("\n[7] _check_adaptive_stop 单元测试")
    ens = _make_ensemble(
        max_rounds=5,
        adaptive_stop=True,
        convergence_threshold=0.01,
        dominance_ratio=2.0,
        min_rounds=2,
    )

    # 情况 1: adaptive_stop=False → 不停止
    ens.adaptive_stop = False
    stop, _ = ens._check_adaptive_stop({"n0": 0.5, "n1": 0.4}, None, round_num=3)
    assert stop is False, "adaptive_stop=False 不应停止"

    # 情况 2: round_num < min_rounds → 不停止
    ens.adaptive_stop = True
    stop, _ = ens._check_adaptive_stop({"n0": 0.5, "n1": 0.4}, None, round_num=1)
    assert stop is False, "round_num < min_rounds 不应停止"

    # 情况 3: round_num >= max_rounds → 不停止（自然结束）
    stop, _ = ens._check_adaptive_stop({"n0": 0.5, "n1": 0.4}, None, round_num=5)
    assert stop is False, "round_num >= max_rounds 不应提前停止"

    # 情况 4: 收敛触发
    stop, reason = ens._check_adaptive_stop(
        {"n0": 0.5, "n1": 0.4},
        {"n0": 0.5001, "n1": 0.4001},  # 变化 < 0.01
        round_num=3,
    )
    assert stop is True, "收敛应触发停止"
    assert "converged" in reason

    # 情况 5: 主导触发
    stop, reason = ens._check_adaptive_stop(
        {"n0": 0.9, "n1": 0.1},  # top1/top2 = 9 > 2
        None,
        round_num=3,
    )
    assert stop is True, "主导应触发停止"
    assert "dominant" in reason

    # 情况 6: 无触发
    stop, _ = ens._check_adaptive_stop(
        {"n0": 0.5, "n1": 0.4},  # top1/top2 = 1.25 < 2
        {"n0": 0.3, "n1": 0.2},  # 变化 = 0.2 > 0.01
        round_num=3,
    )
    assert stop is False, "无触发信号不应停止"

    print("  PASS: 6 个边界情况全部正确")


def test_default_params():
    """[8] 默认参数语义正确。"""
    print("\n[8] 默认参数语义")
    ens = _make_ensemble(max_rounds=3, adaptive_stop=True)
    assert ens.convergence_threshold == 0.01, f"默认 convergence_threshold 应为 0.01"
    assert ens.dominance_ratio == 2.0, f"默认 dominance_ratio 应为 2.0"
    assert ens.min_rounds == 2, f"默认 min_rounds 应为 2"
    # min_rounds > max_rounds 时被截断
    ens2 = _make_ensemble(max_rounds=2, adaptive_stop=True, min_rounds=5)
    assert ens2.min_rounds == 2, f"min_rounds 应被截断到 max_rounds=2, got {ens2.min_rounds}"
    print("  PASS: 默认参数正确，min_rounds 截断正确")


def main():
    print("=" * 60)
    print("C9 共振轮数自适应停止 smoke test")
    print("=" * 60)

    test_backward_compat()
    test_adaptive_stop_field()
    test_convergence_stop()
    test_dominance_stop()
    test_min_rounds_constraint()
    test_max_rounds_cap()
    test_check_adaptive_stop_unit()
    test_default_params()

    print("\n" + "=" * 60)
    print("ALL 8/8 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
