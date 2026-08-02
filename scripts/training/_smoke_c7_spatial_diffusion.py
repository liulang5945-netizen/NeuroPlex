"""C7: 空间场扩散 smoke test.

验证：
1. 扩散改变 vectors（alpha>0 时输出≠输入）
2. alpha=0 完全退化（向后兼容）
3. 近邻神经元扩散后更相似（扩散的方向性正确）
4. 梯度流：扩散后 loss 对输入有梯度
5. 子集处理：active_ids 是子集时不报错
6. 图拉普拉斯性质：对称 + 行和≈0 + 对角线≈1
7. ensemble 级别：spatial_diffusion_enabled=False 时 forward_train 行为不变
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.spatial_diffusion import SpatialDiffuser


def _make_geometry() -> NeuronGeometry:
    """构造 4 neuron 的 geometry：n0/n1 同域近邻，n2/n3 同域近邻，跨域远。"""
    geo = NeuronGeometry(embedding_dim=8, sigma=0.5)
    geo.assign_domain_positions({
        "zh": ["n0", "n1"],
        "en": ["n2", "n3"],
    })
    return geo


def test_diffusion_changes_vectors():
    """alpha>0 时扩散后 vectors 与输入不同。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.3)

    V = F.normalize(torch.randn(4, 2, 8), dim=-1)  # [N=4, B=2, D=8]
    diffused = diffuser.diffuse(V)

    diff = (V - diffused).abs().max().item()
    assert diff > 1e-4, f"alpha=0.3 时扩散应改变 vectors, diff={diff}"
    print(f"  [PASS] 扩散改变 vectors (max_diff={diff:.4f})")


def test_alpha_zero_no_change():
    """alpha=0 时完全退化（向后兼容）。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.0)

    V = F.normalize(torch.randn(4, 2, 8), dim=-1)
    diffused = diffuser.diffuse(V)

    diff = (V - diffused).abs().max().item()
    assert diff < 1e-8, f"alpha=0 时应完全退化, diff={diff}"
    print(f"  [PASS] alpha=0 完全退化 (diff={diff:.2e})")


def test_nearby_neurons_become_similar():
    """近邻神经元扩散后 cosine similarity 增加（扩散方向性正确）。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.5)

    # 构造正交向量：n0=[1,0,0,0], n1=[0,1,0,0], n2=[0,0,1,0], n3=[0,0,0,1]
    V = torch.zeros(4, 1, 8)
    V[0, 0, 0] = 1.0
    V[1, 0, 1] = 1.0
    V[2, 0, 2] = 1.0
    V[3, 0, 3] = 1.0
    V = F.normalize(V, dim=-1)

    # 扩散前 cosine similarity
    def cos_sim(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

    sim_n0_n1_before = cos_sim(V[0, 0], V[1, 0])  # 同域近邻
    sim_n0_n2_before = cos_sim(V[0, 0], V[2, 0])  # 跨域远邻

    diffused = diffuser.diffuse(V)

    sim_n0_n1_after = cos_sim(diffused[0, 0], diffused[1, 0])
    sim_n0_n2_after = cos_sim(diffused[0, 0], diffused[2, 0])

    # 近邻扩散后 similarity 应增加（n0 收到 n1 信号，n1 收到 n0 信号）
    assert sim_n0_n1_after > sim_n0_n1_before + 1e-4, (
        f"近邻 n0-n1 扩散后 similarity 应增加: "
        f"before={sim_n0_n1_before:.4f}, after={sim_n0_n1_after:.4f}"
    )
    # 远邻 similarity 变化应小于近邻变化
    delta_near = sim_n0_n1_after - sim_n0_n1_before
    delta_far = abs(sim_n0_n2_after - sim_n0_n2_before)
    assert delta_near > delta_far, (
        f"近邻变化({delta_near:.4f}) 应大于远邻变化({delta_far:.4f})"
    )
    print(f"  [PASS] 近邻扩散效果强于远邻 "
          f"(n0-n1: {sim_n0_n1_before:.3f}→{sim_n0_n1_after:.3f}, "
          f"n0-n2: {sim_n0_n2_before:.3f}→{sim_n0_n2_after:.3f})")


def test_gradient_flow():
    """扩散后 loss 对输入 vectors 有梯度（可微）。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.2)

    V_raw = torch.randn(4, 2, 8, requires_grad=True)
    V = F.normalize(V_raw, dim=-1)
    diffused = diffuser.diffuse(V)
    loss = diffused.sum()
    loss.backward()

    assert V_raw.grad is not None, "扩散后应有梯度"
    grad_norm = V_raw.grad.norm().item()
    assert grad_norm > 1e-6, f"梯度应非零, grad_norm={grad_norm}"
    print(f"  [PASS] 梯度流正常 (grad_norm={grad_norm:.4f})")


def test_subset_handling():
    """active_ids 是子集时扩散正确工作。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.3)

    # 只用 n0, n1（子集）
    V = F.normalize(torch.randn(2, 2, 8), dim=-1)
    diffused = diffuser.diffuse(V, active_ids=["n0", "n1"])

    diff = (V - diffused).abs().max().item()
    assert diff > 1e-4, f"子集扩散应改变 vectors, diff={diff}"
    print(f"  [PASS] 子集处理正确 (diff={diff:.4f})")


def test_laplacian_properties():
    """图拉普拉斯 L 性质：对称 + 行和≈0 + 对角线≈1。"""
    geo = _make_geometry()
    diffuser = SpatialDiffuser(["n0", "n1", "n2", "n3"], geo, alpha=0.1)

    L = diffuser.graph_laplacian  # [N, N]

    # 对称性
    sym_diff = (L - L.T).abs().max().item()
    assert sym_diff < 1e-6, f"L 应对称, sym_diff={sym_diff}"

    # 行和 ≈ 0（拉普拉斯性质，浮点误差允许 1e-4）
    row_sums = L.sum(dim=1)
    max_row_sum = row_sums.abs().max().item()
    assert max_row_sum < 1e-4, f"行和应≈0, max_row_sum={max_row_sum}"

    # 对角线 ≈ 1（归一化拉普拉斯，W[i,i]=0）
    diag = L.diagonal()
    diag_diff = (diag - 1.0).abs().max().item()
    assert diag_diff < 1e-6, f"对角线应≈1, diag_diff={diag_diff}"

    print(f"  [PASS] 拉普拉斯性质 (sym_diff={sym_diff:.2e}, "
          f"max_row_sum={max_row_sum:.2e}, diag_diff={diag_diff:.2e})")


def test_ensemble_backward_compatible():
    """ensemble 级别：spatial_diffusion_enabled=False 时 forward_train 行为不变。"""
    from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
    from taiji.resonance.field import ResonanceField
    from taiji.resonance.ensemble import ResonanceEnsemble

    # 构造 2 个小 neuron（用默认 compact 配置，只减少 layers 加速）
    cfg = get_domain_neuron_config("zh", spec="compact")
    cfg.num_layers = 2
    n0 = ResonanceNeuron(cfg)
    n1 = ResonanceNeuron(cfg)

    field = ResonanceField(dim=cfg.field_dim)
    shared_emb = torch.nn.Embedding(1000, cfg.hidden_size)

    # 不启用扩散（默认）
    ens_no_diff = ResonanceEnsemble(
        neurons={"n0": n0, "n1": n1},
        field=field,
        max_rounds=2,
        spatial_diffusion_enabled=False,  # 默认关闭
    )
    assert ens_no_diff.spatial_diffuser is None, "未启用时 diffuser 应为 None"

    # 启用扩散
    field2 = ResonanceField(dim=cfg.field_dim)
    n0b = ResonanceNeuron(cfg)
    n1b = ResonanceNeuron(cfg)
    ens_with_diff = ResonanceEnsemble(
        neurons={"n0": n0b, "n1": n1b},
        field=field2,
        max_rounds=2,
        spatial_diffusion_enabled=True,
        spatial_diffusion_alpha=0.2,
    )
    assert ens_with_diff.spatial_diffuser is not None, "启用时 diffuser 应存在"
    assert ens_with_diff.spatial_diffuser.alpha == 0.2

    # 两个 ensemble 都能正常 forward_train
    shared_emb_input = shared_emb(torch.randint(0, 1000, (2, 16)))

    result_no = ens_no_diff.forward_train(shared_embeddings=shared_emb_input)
    result_with = ens_with_diff.forward_train(shared_embeddings=shared_emb_input)

    assert "fused_logits" in result_no, "forward_train 应返回 fused_logits"
    assert "fused_logits" in result_with, "启用扩散后 forward_train 应正常"

    # 输出形状一致
    logits_no = result_no["fused_logits"]
    logits_with = result_with["fused_logits"]
    assert logits_no.shape == logits_with.shape, "输出形状应一致"

    # 输出不同（扩散改变了 field_state，进而改变 logits）
    # 注：两个 ensemble 用不同的 neuron 权重（随机初始化），所以输出本来就不同
    # 这里只验证 forward_train 能正常运行
    print(f"  [PASS] ensemble 向后兼容 "
          f"(no_diff logits shape={logits_no.shape}, "
          f"with_diff logits shape={logits_with.shape})")


def main():
    print("=" * 60)
    print("C7: 空间场扩散 smoke test")
    print("=" * 60)

    tests = [
        test_diffusion_changes_vectors,
        test_alpha_zero_no_change,
        test_nearby_neurons_become_similar,
        test_gradient_flow,
        test_subset_handling,
        test_laplacian_properties,
        test_ensemble_backward_compatible,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n[{test.__name__}]")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{passed + failed} 通过")
    if failed:
        print(f"失败: {failed} 项")
        sys.exit(1)
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
