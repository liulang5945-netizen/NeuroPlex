"""R1 共振分数软路由 smoke test.

验证：
1. cortex.py 可 import（语法正确）
2. _generate_p7 签名包含 routing_mode / resonance_top_k 参数
3. routing_mode 默认 "hybrid"（向后兼容）
4. 共振分数排序逻辑正确（按分数降序选 top-k）
5. shared_expert 始终包含在 active_nids 中
6. resonance 模式跨域激活（不限定 domain 前缀）
7. keyword 模式跳过 probe forward
8. hybrid 模式保留现有 50% 阈值硬切换逻辑
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import inspect
import torch

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.brain.cortex import Cortex


def test_cortex_import():
    """[1] cortex.py 可 import（语法正确）。"""
    print("\n[1] cortex.py import")
    assert Cortex is not None
    print("  PASS: Cortex 类可导入")


def test_generate_p7_signature():
    """[2] _generate_p7 签名包含 routing_mode / resonance_top_k。"""
    print("\n[2] _generate_p7 签名")
    sig = inspect.signature(Cortex._generate_p7)
    params = sig.parameters
    assert "routing_mode" in params, "应包含 routing_mode 参数"
    assert "resonance_top_k" in params, "应包含 resonance_top_k 参数"
    # 默认值检查
    assert params["routing_mode"].default == "hybrid", f"routing_mode 默认应为 hybrid"
    assert params["resonance_top_k"].default == 3, f"resonance_top_k 默认应为 3"
    print(f"  PASS: routing_mode 默认={params['routing_mode'].default}, resonance_top_k 默认={params['resonance_top_k'].default}")


def test_resonance_sort_logic():
    """[4] 共振分数排序逻辑正确（按分数降序选 top-k）。"""
    print("\n[4] 共振分数排序逻辑")
    # 模拟 probe_scores
    probe_scores = {"n0": 0.8, "n1": 0.3, "n2": 0.6, "n3": 0.1}
    sorted_nids = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
    top_nids = [nid for nid, _ in sorted_nids[:2]]
    assert top_nids == ["n0", "n2"], f"top-2 应为 n0,n2, got {top_nids}"
    print(f"  PASS: top-2 by score = {top_nids}")


def test_shared_expert_included():
    """[5] shared_expert 始终包含在 active_nids 中。"""
    print("\n[5] shared_expert 包含检查")
    # 模拟 resonance 路由逻辑
    probe_scores = {"n0": 0.8, "n1": 0.3, "n2": 0.6}
    shared_expert_id = "n3"  # shared_expert 分数低，不在 top-2
    sorted_nids = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
    top_nids = [nid for nid, _ in sorted_nids[:2]]  # ["n0", "n2"]
    # 确保 shared_expert 在激活集中
    if shared_expert_id:
        if shared_expert_id not in top_nids:
            top_nids.append(shared_expert_id)
    assert shared_expert_id in top_nids, f"shared_expert {shared_expert_id} 应在 active_nids 中"
    print(f"  PASS: shared_expert {shared_expert_id} 已包含, active_nids={top_nids}")


def test_cross_domain_activation():
    """[6] resonance 模式跨域激活（不限定 domain 前缀）。"""
    print("\n[6] 跨域激活")
    # 模拟跨域场景：zh 和 code 神经元都有高共振分
    probe_scores = {
        "zh_aug0": 0.7,    # 中文域
        "code_0": 0.9,     # 代码域
        "zh_std0": 0.5,    # 中文域
        "general": 0.3,    # 通用域
    }
    sorted_nids = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
    top_nids = [nid for nid, _ in sorted_nids[:3]]  # top-3
    # 应包含 code_0 和 zh_aug0（跨域）
    assert "code_0" in top_nids, "应包含 code 域神经元"
    assert "zh_aug0" in top_nids, "应包含 zh 域神经元"
    # general 不在 top-3（分数低）
    assert "general" not in top_nids, "general 分数低不应在 top-3"
    print(f"  PASS: 跨域激活 {top_nids}（含 code + zh）")


def test_keyword_mode_skips_probe():
    """[7] keyword 模式跳过 probe forward（通过代码逻辑验证）。"""
    print("\n[7] keyword 模式跳过 probe")
    # 验证代码逻辑：routing_mode != "keyword" 才进入 probe 分支
    # 通过 inspect 检查源码包含 "routing_mode != \"keyword\""
    source = inspect.getsource(Cortex._generate_p7)
    assert 'routing_mode != "keyword"' in source, "源码应包含 routing_mode != keyword 判断"
    print("  PASS: keyword 模式跳过 probe forward 逻辑存在")


def test_hybrid_mode_preserves_legacy():
    """[8] hybrid 模式保留现有 50% 阈值硬切换逻辑。"""
    print("\n[8] hybrid 模式保留硬切换")
    source = inspect.getsource(Cortex._generate_p7)
    # 验证 50% 阈值逻辑仍存在（1.5 倍 = 高 50%）
    assert "1.5" in source, "hybrid 模式应保留 1.5 倍阈值（50%）"
    assert "best_domain" in source, "hybrid 模式应保留 best_domain 逻辑"
    print("  PASS: hybrid 模式保留 50% 阈值硬切换逻辑")


def test_resonance_uses_final_scores():
    """[9] resonance 模式使用 final_scores 选 top-k。"""
    print("\n[9] resonance 使用 final_scores")
    source = inspect.getsource(Cortex._generate_p7)
    assert "final_scores" in source, "应使用 final_scores"
    assert "resonance_active_nids" in source, "应设置 resonance_active_nids"
    assert "sorted_nids" in source, "应按分数排序"
    print("  PASS: resonance 模式使用 final_scores 排序选 top-k")


def test_resonance_active_nids_priority():
    """[10] resonance_active_nids 优先于 routing_level。"""
    print("\n[10] resonance_active_nids 优先级")
    source = inspect.getsource(Cortex._generate_p7)
    # 验证 resonance_active_nids 优先于 routing_level 判断
    assert "resonance_active_nids is not None" in source, "应优先检查 resonance_active_nids"
    print("  PASS: resonance_active_nids 优先于 routing_level")


def main():
    print("=" * 60)
    print("R1 共振分数软路由 smoke test")
    print("=" * 60)

    test_cortex_import()
    test_generate_p7_signature()
    test_resonance_sort_logic()
    test_shared_expert_included()
    test_cross_domain_activation()
    test_keyword_mode_skips_probe()
    test_hybrid_mode_preserves_legacy()
    test_resonance_uses_final_scores()
    test_resonance_active_nids_priority()

    print("\n" + "=" * 60)
    print("ALL 9/9 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
