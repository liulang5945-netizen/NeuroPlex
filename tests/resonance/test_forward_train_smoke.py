"""P2-1b 提取后的 forward_train 冒烟回归。

验证质量监督流水线提取（_compute_quality_supervision）后：
- forward_train 返回键完整（质量流水线产物全部在场）
- 全部 loss 有限（无 NaN/Inf）
- per_neuron_nll 形状正确（C20 原版 general 投影路径）
"""
import torch

from taiji.resonance import (
    ResonanceEnsemble,
    ResonanceField,
    ResonanceNeuron,
    get_domain_neuron_config,
)


def test_forward_train_smoke():
    torch.manual_seed(0)
    cfg = get_domain_neuron_config("zh", spec="compact")  # vocab=50000
    n1 = ResonanceNeuron(cfg)
    n1.config.neuron_id = "n0"
    n2 = ResonanceNeuron(cfg)
    n2.config.neuron_id = "n1"
    field = ResonanceField(dim=2048)
    ens = ResonanceEnsemble(neurons={"n0": n1, "n1": n2}, field=field, max_rounds=2)

    B, L, D = 2, 8, 512
    emb = torch.randn(B, L, D)
    tgt = torch.randint(0, 100, (B, L))
    am = torch.zeros(B, L, dtype=torch.bool)
    am[:, L // 2:] = True

    r = ens.forward_train(shared_embeddings=emb, targets=tgt, answer_mask=am, n_rounds=2)

    # 质量流水线产物必须全部在场（提取段返回 4 元组 + result 键）
    for key in ("fused_logits", "weights", "scores", "balance_loss",
                "diversity_loss", "contrastive_loss", "per_neuron_nll"):
        assert key in r, f"缺返回键: {key}"

    assert torch.isfinite(r["fused_logits"]).all(), "fused_logits 含 NaN/Inf"
    assert torch.isfinite(r["contrastive_loss"]), "contrastive_loss 非有限"
    assert torch.isfinite(r["balance_loss"]), "balance_loss 非有限"
    assert torch.isfinite(r["diversity_loss"]), "diversity_loss 非有限"
    assert r["per_neuron_nll"].shape == (2,), f"per_neuron_nll 形状异常: {r['per_neuron_nll'].shape}"
    assert torch.isfinite(r["per_neuron_nll"]).all(), "per_neuron_nll 含 NaN/Inf"
    # C15 质量 logits：无 quality_head 的构造 neuron 应为 None（向后兼容）
    assert r["quality_logits"] is None or torch.isfinite(r["quality_logits"]).all()
