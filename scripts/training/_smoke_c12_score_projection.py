"""C12 评分投影 + 对比约束 smoke test.

验证：
1. score_dim=None 向后兼容（neuron 无 score_proj，forward 无 score_vec 输出）
2. score_dim=256 启用 score_proj，forward 输出 score_vec [B, score_dim]
3. ensemble 创建 field_score_proj（所有 neuron score_dim 一致时）
4. forward_train 用 score_vec 评分（与 score_dim=None 评分不同）
5. contrastive_loss：targets=None 时为 0（向后兼容）
6. contrastive_loss：targets 不为 None 时 > 0
7. 梯度流：score_proj 和 field_score_proj 有梯度
8. checkpoint 兼容性：旧 ckpt（无 score_proj）加载到 score_dim=256 neuron（strict=False）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn.functional as F

from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble


def _make_neuron(score_dim=None, neuron_id="n0", seed=42) -> ResonanceNeuron:
    """构建指定 score_dim 的 TINY_TEST neuron。"""
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = neuron_id
    cfg.score_dim = score_dim
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def _make_ensemble(score_dim=None, n_neurons=2):
    """构建 n_neurons 个 neuron 的 ensemble，统一 score_dim。"""
    neurons = {}
    for i in range(n_neurons):
        nid = f"n{i}"
        neurons[nid] = _make_neuron(score_dim=score_dim, neuron_id=nid, seed=42 + i)
    field = ResonanceField(dim=TINY_TEST.field_dim)
    return ResonanceEnsemble(neurons, field)


def test_backward_compat_no_score_dim():
    """[1] score_dim=None 向后兼容：neuron 无 score_proj，forward 无 score_vec。"""
    print("\n[1] score_dim=None 向后兼容")
    neuron = _make_neuron(score_dim=None)
    assert neuron.score_proj is None, "score_dim=None 时 score_proj 应为 None"

    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)
    assert "score_vec" not in result, "score_dim=None 时不应输出 score_vec"
    print("  PASS: score_dim=None 无 score_proj，无 score_vec 输出")


def test_score_dim_enabled():
    """[2] score_dim=256 启用 score_proj，forward 输出 score_vec [B, score_dim]。"""
    print("\n[2] score_dim=256 启用 score_proj")
    neuron = _make_neuron(score_dim=256)
    assert neuron.score_proj is not None, "score_dim=256 时 score_proj 应存在"
    assert neuron.score_proj.weight.shape == (256, TINY_TEST.field_dim), \
        f"score_proj shape 错误: {neuron.score_proj.weight.shape}"

    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb)
    assert "score_vec" in result, "score_dim=256 时应输出 score_vec"
    score_vec = result["score_vec"]
    assert score_vec.shape == (2, 256), f"score_vec shape 错误: {score_vec.shape}"
    # 验证 L2 归一化
    norms = score_vec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"score_vec 应 L2 归一化, norms={norms}"
    print(f"  PASS: score_proj shape={neuron.score_proj.weight.shape}, score_vec shape={score_vec.shape}")


def test_ensemble_creates_field_score_proj():
    """[3] ensemble 创建 field_score_proj（所有 neuron score_dim 一致时）。"""
    print("\n[3] ensemble 创建 field_score_proj")
    ens = _make_ensemble(score_dim=256, n_neurons=2)
    assert ens.field_score_proj is not None, "ensemble 应创建 field_score_proj"
    assert ens.score_dim == 256, f"ensemble score_dim 应为 256, got {ens.score_dim}"
    assert ens.field_score_proj.weight.shape == (256, TINY_TEST.field_dim), \
        f"field_score_proj shape 错误: {ens.field_score_proj.weight.shape}"

    # 混合 score_dim（一个 None 一个 256）→ 不创建
    n0 = _make_neuron(score_dim=None, neuron_id="n0")
    n1 = _make_neuron(score_dim=256, neuron_id="n1")
    field = ResonanceField(dim=TINY_TEST.field_dim)
    ens_mixed = ResonanceEnsemble({"n0": n0, "n1": n1}, field)
    assert ens_mixed.field_score_proj is None, "混合 score_dim 时不应创建 field_score_proj"
    print(f"  PASS: field_score_proj 创建正确（混合 score_dim 时禁用）")


def test_forward_train_uses_score_vec():
    """[4] forward_train 用 score_vec 评分（与 score_dim=None 评分不同）。"""
    print("\n[4] forward_train 用 score_vec 评分")
    # 构建 score_dim=None 的 ensemble
    ens_no = _make_ensemble(score_dim=None, n_neurons=2)
    # 构建 score_dim=256 的 ensemble
    ens_with = _make_ensemble(score_dim=256, n_neurons=2)

    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result_no = ens_no.forward_train(shared_embeddings=shared_emb, n_rounds=1)
        result_with = ens_with.forward_train(shared_embeddings=shared_emb, n_rounds=1)

    scores_no = result_no["scores"]
    scores_with = result_with["scores"]
    diff = (scores_no - scores_with).abs().max().item()
    assert diff > 1e-4, f"score_vec 评分应与无投影不同, diff={diff}"
    print(f"  PASS: score_vec 评分与无投影不同 (diff={diff:.4e})")


def test_contrastive_loss_none_targets():
    """[5] contrastive_loss：targets=None 时为 0（向后兼容）。"""
    print("\n[5] contrastive_loss targets=None")
    ens = _make_ensemble(score_dim=256, n_neurons=2)
    shared_emb = torch.randn(2, 8, 512)
    with torch.no_grad():
        result = ens.forward_train(shared_embeddings=shared_emb, n_rounds=1, targets=None)
    cl = result["contrastive_loss"]
    assert cl.item() == 0.0, f"targets=None 时 contrastive_loss 应为 0, got {cl.item()}"
    print("  PASS: targets=None → contrastive_loss=0")


def test_contrastive_loss_with_targets():
    """[6] contrastive_loss：targets 不为 None 时 > 0。"""
    print("\n[6] contrastive_loss with targets")
    ens = _make_ensemble(score_dim=256, n_neurons=2)
    B, L = 2, 8
    V = 100
    shared_emb = torch.randn(B, L, 512)
    targets = torch.randint(0, V, (B, L))
    with torch.no_grad():
        result = ens.forward_train(shared_embeddings=shared_emb, n_rounds=1, targets=targets)
    cl = result["contrastive_loss"]
    assert cl.item() > 0, f"targets 不为 None 时 contrastive_loss 应 > 0, got {cl.item()}"
    print(f"  PASS: contrastive_loss > 0 (val={cl.item():.4f})")


def test_gradient_flow():
    """[7] 梯度流：score_proj 和 field_score_proj 有梯度。"""
    print("\n[7] 梯度流")
    ens = _make_ensemble(score_dim=256, n_neurons=2)
    B, L, V = 2, 8, 100
    shared_emb = torch.randn(B, L, 512)
    targets = torch.randint(0, V, (B, L))

    result = ens.forward_train(shared_embeddings=shared_emb, n_rounds=1, targets=targets)
    loss = (
        result["fused_logits"].sum()
        + result["contrastive_loss"]
        + result["balance_loss"]
    )
    loss.backward()

    # 检查 neuron score_proj 梯度
    for nid, neuron in ens.neurons.items():
        assert neuron.score_proj.weight.grad is not None, f"{nid}.score_proj 无梯度"
        grad_norm = neuron.score_proj.weight.grad.norm().item()
        assert grad_norm > 0, f"{nid}.score_proj 梯度为零"
        print(f"  {nid}.score_proj grad_norm={grad_norm:.4e}")

    # 检查 field_score_proj 梯度
    assert ens.field_score_proj.weight.grad is not None, "field_score_proj 无梯度"
    grad_norm = ens.field_score_proj.weight.grad.norm().item()
    assert grad_norm > 0, f"field_score_proj 梯度为零"
    print(f"  field_score_proj grad_norm={grad_norm:.4e}")
    print("  PASS: 所有投影头有梯度")


def test_checkpoint_compat():
    """[8] checkpoint 兼容性：旧 ckpt（无 score_proj）加载到 score_dim=256 neuron。"""
    print("\n[8] checkpoint 兼容性")
    # 创建 score_dim=None 的 neuron，保存 state_dict
    neuron_old = _make_neuron(score_dim=None, neuron_id="n0")
    old_sd = neuron_old.state_dict()

    # 创建 score_dim=256 的 neuron，加载旧 sd（strict=False）
    neuron_new = _make_neuron(score_dim=256, neuron_id="n0")
    missing, unexpected = neuron_new.load_state_dict(old_sd, strict=False)

    # score_proj 应在 missing keys 中
    score_proj_missing = [k for k in missing if "score_proj" in k]
    assert len(score_proj_missing) > 0, f"score_proj 应在 missing keys 中, missing={missing}"
    print(f"  PASS: 旧 ckpt 加载到 score_dim=256 neuron (missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"  score_proj missing keys: {score_proj_missing}")


def main():
    print("=" * 60)
    print("C12 评分投影 + 对比约束 smoke test")
    print("=" * 60)

    test_backward_compat_no_score_dim()
    test_score_dim_enabled()
    test_ensemble_creates_field_score_proj()
    test_forward_train_uses_score_vec()
    test_contrastive_loss_none_targets()
    test_contrastive_loss_with_targets()
    test_gradient_flow()
    test_checkpoint_compat()

    print("\n" + "=" * 60)
    print("ALL 8/8 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
