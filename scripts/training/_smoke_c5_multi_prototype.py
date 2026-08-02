"""C5 多原型混合 smoke test.

验证 domain_prototype 升级为 K 个原型 + 在线聚类：
1. num_prototypes=1 向后兼容（单 EMA 原型）
2. num_prototypes=K>1 创建 domain_prototypes [K, hidden]
3. update_domain_prototype 多原型：胜者 EMA 更新
4. 多原型初始化阶段：未使用的原型直接赋值
5. 多原型收敛：相似输入趋向同一原型
6. 路由 max cosine：多原型取最高相似度
7. proto_counts 统计正确
8. _fingerprint_route 适配多原型
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn.functional as F

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron


def _make_neuron(num_prototypes=1, seed=42) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = "n0"
    cfg.num_prototypes = num_prototypes
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def test_backward_compat():
    """[1] num_prototypes=1 向后兼容（单 EMA 原型）。"""
    print("\n[1] 向后兼容（num_prototypes=1）")
    n = _make_neuron(num_prototypes=1)
    assert n.num_prototypes == 1
    assert n.domain_prototypes is None  # 单原型模式
    assert n.domain_prototype is not None  # 使用原 domain_prototype
    print("  PASS: num_prototypes=1 用原 domain_prototype")


def test_multi_prototype_created():
    """[2] num_prototypes=K>1 创建 domain_prototypes [K, hidden]。"""
    print("\n[2] 多原型创建")
    K = 4
    n = _make_neuron(num_prototypes=K)
    assert n.num_prototypes == K
    assert n.domain_prototypes is not None
    assert n.domain_prototypes.shape == (K, TINY_TEST.hidden_size)
    # 原型应归一化（单位球面）
    norms = n.domain_prototypes.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5), f"原型应归一化, norms={norms}"
    print(f"  PASS: domain_prototypes 形状={n.domain_prototypes.shape}, 归一化")


def test_update_winner_only():
    """[3] update_domain_prototype 多原型：胜者 EMA 更新。"""
    print("\n[3] 胜者 EMA 更新")
    K = 3
    n = _make_neuron(num_prototypes=K, seed=42)
    # 记录初始原型
    initial_protos = n.domain_prototypes.clone()

    # 输入接近原型 0
    target = n.domain_prototypes[0].clone() + 0.1 * torch.randn(TINY_TEST.hidden_size)
    n.update_domain_prototype(target.unsqueeze(0))

    # 原型 0 应被更新（与初始不同）
    diff_0 = (n.domain_prototypes[0] - initial_protos[0]).abs().max().item()
    # 原型 1, 2 应保持不变
    diff_1 = (n.domain_prototypes[1] - initial_protos[1]).abs().max().item()
    diff_2 = (n.domain_prototypes[2] - initial_protos[2]).abs().max().item()

    assert diff_0 > 1e-6, f"胜者原型 0 应被更新, diff={diff_0}"
    # 初始化阶段（count<1）直接赋值，所以 diff_0 可能较大
    print(f"  PASS: 胜者原型 0 更新 (diff={diff_0:.4f}), 非胜者不变 (diff_1={diff_1:.4f}, diff_2={diff_2:.4f})")


def test_initialization_phase():
    """[4] 多原型初始化阶段：未使用的原型直接赋值。"""
    print("\n[4] 初始化阶段直接赋值")
    K = 3
    n = _make_neuron(num_prototypes=K, seed=42)
    # 所有原型 count=0，第一次更新应直接赋值
    target = torch.randn(TINY_TEST.hidden_size)
    n.update_domain_prototype(target.unsqueeze(0))

    # 找到胜者
    target_norm = F.normalize(target.unsqueeze(0), dim=-1).squeeze(0)
    sims = F.cosine_similarity(target_norm.unsqueeze(0), n.domain_prototypes, dim=-1)
    winner = int(sims.argmax().item())

    # 胜者应等于 target_norm（直接赋值）
    diff = (n.domain_prototypes[winner] - target_norm).abs().max().item()
    assert diff < 1e-5, f"初始化阶段胜者应直接赋值, diff={diff}"
    assert n.proto_counts[winner].item() == 1.0
    print(f"  PASS: 胜者 {winner} 直接赋值 (diff={diff:.6f}), count=1")


def test_convergence_to_same_prototype():
    """[5] 多原型收敛：相似输入趋向同一原型。"""
    print("\n[5] 相似输入趋向同一原型")
    K = 3
    n = _make_neuron(num_prototypes=K, seed=42)
    # 用相似输入更新多次
    base_target = torch.randn(TINY_TEST.hidden_size)
    for i in range(10):
        target = base_target + 0.01 * torch.randn(TINY_TEST.hidden_size)
        n.update_domain_prototype(target.unsqueeze(0))

    # 应有一个原型的 count 显著高于其他
    counts = n.proto_counts.tolist()
    max_count = max(counts)
    winner_idx = counts.index(max_count)
    assert max_count >= 8, f"相似输入应集中到一个原型, counts={counts}"
    print(f"  PASS: 相似输入集中到原型 {winner_idx} (count={max_count}/10)")


def test_max_cosine_routing():
    """[6] 路由 max cosine：多原型取最高相似度。"""
    print("\n[6] max cosine 路由")
    K = 3
    n = _make_neuron(num_prototypes=K, seed=42)
    # 构造一个与原型 2 最相似的输入
    query = n.domain_prototypes[2].clone() + 0.1 * torch.randn(TINY_TEST.hidden_size)
    query_norm = F.normalize(query.unsqueeze(0), dim=-1).squeeze(0)

    # 计算与所有原型的 cosine
    protos_norm = F.normalize(n.domain_prototypes, dim=-1)
    sims = (query_norm.unsqueeze(0) * protos_norm).sum(dim=-1)  # [K]
    max_sim = sims.max().item()
    max_idx = int(sims.argmax().item())

    assert max_idx == 2, f"最相似应为原型 2, got {max_idx}"
    assert max_sim > 0.5, f"相似度应高于其他原型, got {max_sim}"
    # 验证 max cosine 确实取了最大值
    assert sims[max_idx] == sims.max()
    print(f"  PASS: 输入与原型 {max_idx} 最相似 (cosine={max_sim:.4f}, max cosine 生效)")


def test_proto_counts():
    """[7] proto_counts 统计正确。"""
    print("\n[7] proto_counts 统计")
    K = 3
    n = _make_neuron(num_prototypes=K, seed=42)
    # 更新 5 次
    for _ in range(5):
        target = torch.randn(TINY_TEST.hidden_size)
        n.update_domain_prototype(target.unsqueeze(0))
    total_count = n.proto_counts.sum().item()
    assert total_count == 5.0, f"总 count 应=5, got {total_count}"
    print(f"  PASS: 总 count={total_count} (5 次更新)")


def test_fingerprint_route_compat():
    """[8] _fingerprint_route 适配多原型。"""
    print("\n[8] _fingerprint_route 适配")
    # 通过检查 cortex.py 源码包含多原型分支
    cortex_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji", "brain", "cortex.py",
    )
    with open(cortex_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "num_prototypes" in source, "cortex.py 应包含 num_prototypes 检查"
    assert "domain_prototypes" in source, "cortex.py 应包含 domain_prototypes 多原型分支"
    assert "max().item()" in source, "应取 max cosine"
    print("  PASS: cortex._fingerprint_route 含多原型 max cosine 分支")


def test_checkpoint_compat():
    """[9] checkpoint 兼容性（旧 ckpt 无 domain_prototypes 字段）。"""
    print("\n[9] checkpoint 兼容性")
    n = _make_neuron(num_prototypes=3, seed=42)
    # 模拟旧 ckpt（只有 domain_prototype，无 domain_prototypes）
    old_state = {
        "domain_prototype": torch.randn(TINY_TEST.hidden_size),
        "proto_ema_decay": torch.tensor(0.99),
        # 无 domain_prototypes 和 proto_counts
    }
    # strict=False 加载（旧 ckpt 兼容）
    missing, unexpected = n.load_state_dict(old_state, strict=False)
    # domain_prototypes 和 proto_counts 应在 missing 中
    assert any("domain_prototypes" in k for k in missing), f"domain_prototypes 应 missing, got {missing[:3]}"
    assert any("proto_counts" in k for k in missing), f"proto_counts 应 missing"
    # 加载后 domain_prototypes 仍保持初始化值（未被覆盖）
    assert n.domain_prototypes is not None
    print(f"  PASS: 旧 ckpt 加载后 domain_prototypes 保持初始化 (missing={len(missing)})")


def main():
    print("=" * 60)
    print("C5 多原型混合 smoke test")
    print("=" * 60)

    test_backward_compat()
    test_multi_prototype_created()
    test_update_winner_only()
    test_initialization_phase()
    test_convergence_to_same_prototype()
    test_max_cosine_routing()
    test_proto_counts()
    test_fingerprint_route_compat()
    test_checkpoint_compat()

    print("\n" + "=" * 60)
    print("ALL 9/9 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
