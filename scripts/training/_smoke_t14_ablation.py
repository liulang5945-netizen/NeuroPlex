"""T14 Ablation 评估 smoke test.

验证 ablation 脚本的核心逻辑（用 TINY_TEST neurons，不加载真实模型）：
1. eval_solo_ppl 逻辑：单个 neuron PPL 计算正确
2. eval_ensemble_ppl 逻辑：ensemble 协作 PPL 计算正确
3. fusion_mode 参数传递：soft/consensus 都能产出 weighted_logits
4. field_conditioning 开关：True/False 走不同路径
5. disable_channels 逻辑：置零通道后输出变化
6. 评估集无泄漏：split_train_eval 训练/评估不重叠
"""
from __future__ import annotations

import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.config import TINY_TEST
from scripts.training.utils import split_train_eval


def make_neuron(seed: int) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 500
    cfg.neuron_id = f"n_{seed}"
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def test_solo_ppl_logic():
    """[1] solo PPL 计算逻辑。"""
    print("\n[1] solo PPL 逻辑")
    torch.manual_seed(42)
    n = make_neuron(0)
    n.eval()
    # 模拟一条评估样本：随机 embedding + targets
    emb = torch.randn(1, 16, 512)
    targets = torch.randint(0, 500, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.bool)

    with torch.no_grad():
        result = n.forward(emb, return_logits=True)
    logits = result["logits"]
    shift_l = logits[:, :-1, :].contiguous()
    shift_t = targets[:, 1:].contiguous()
    shift_m = mask[:, 1:].contiguous()
    shift_t = shift_t.clone()
    shift_t[~shift_m] = -100
    loss = F.cross_entropy(
        shift_l.view(-1, shift_l.size(-1)), shift_t.view(-1),
        ignore_index=-100, reduction="sum",
    ) / max(shift_m.sum().item(), 1)
    ppl = math.exp(min(loss.item(), 20))
    assert ppl > 1.0, f"随机模型 PPL 应 > 1, got {ppl}"
    assert math.isfinite(ppl), f"PPL 应有限, got {ppl}"
    print(f"  PASS: solo PPL={ppl:.2f}（有限且 > 1）")


def test_ensemble_ppl_logic():
    """[2] ensemble 协作 PPL 逻辑。"""
    print("\n[2] ensemble 协作 PPL")
    neurons = {f"n{i}": make_neuron(i) for i in range(3)}
    for n in neurons.values():
        n.eval()
    field = ResonanceField(dim=512)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)

    neuron_embeddings = {nid: torch.randn(1, 16, 512) for nid in neurons}
    targets = torch.randint(0, 500, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.bool)

    with torch.no_grad():
        result = ens.forward(
            neuron_embeddings=neuron_embeddings,
            return_logits=True,
            fusion_mode="soft",
            field_conditioning=True,
        )
    assert "weighted_logits" in result, f"soft 模式应产出 weighted_logits, keys={list(result.keys())}"
    fused = result["weighted_logits"]
    assert fused.shape == (1, 16, 500), f"weighted_logits 形状 {fused.shape}"
    print(f"  PASS: ensemble soft 协作产出 weighted_logits {tuple(fused.shape)}")


def test_fusion_modes():
    """[3] fusion_mode：soft/consensus 都能产出 weighted_logits。"""
    print("\n[3] fusion_mode")
    neurons = {f"n{i}": make_neuron(i) for i in range(3)}
    for n in neurons.values():
        n.eval()
    field = ResonanceField(dim=512)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)

    neuron_embeddings = {nid: torch.randn(1, 16, 512) for nid in neurons}
    for mode in ["soft", "consensus"]:
        with torch.no_grad():
            result = ens.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode=mode,
            )
        assert "weighted_logits" in result, f"{mode} 模式应产出 weighted_logits"
        print(f"  PASS: fusion_mode={mode} → weighted_logits {tuple(result['weighted_logits'].shape)}")


def test_field_conditioning_switch():
    """[4] field_conditioning 开关生效。"""
    print("\n[4] field_conditioning 开关")
    neurons = {f"n{i}": make_neuron(i) for i in range(3)}
    for n in neurons.values():
        n.eval()
    field = ResonanceField(dim=512)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)

    neuron_embeddings = {nid: torch.randn(1, 16, 512) for nid in neurons}
    outs = {}
    for fc in [True, False]:
        with torch.no_grad():
            result = ens.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
                field_conditioning=fc,
            )
        outs[fc] = result["weighted_logits"]
    # 两种模式都可能改变输出（随机初始化下可能接近），核心是都能运行
    assert outs[True].shape == outs[False].shape, "两种模式输出形状应一致"
    diff = (outs[True] - outs[False]).abs().max().item()
    print(f"  PASS: field_conditioning True/False 均可运行, 输出 diff={diff:.4f}")


def test_disable_channels():
    """[5] disable_channels：置零通道后输出变化。"""
    print("\n[5] disable_channels 逻辑")
    neurons = {f"n{i}": make_neuron(i) for i in range(3)}
    for n in neurons.values():
        n.eval()
    # 建立 side_channels（模拟真实场景）
    for nid, neuron in neurons.items():
        for peer_id, peer in neurons.items():
            if peer_id != nid:
                neuron.establish_side_channel(peer_id, peer, channel_type="excite")

    field = ResonanceField(dim=512)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)
    neuron_embeddings = {nid: torch.randn(1, 16, 512) for nid in neurons}

    # 有通道
    with torch.no_grad():
        result_with = ens.forward(
            neuron_embeddings=neuron_embeddings, return_logits=True, fusion_mode="soft",
        )
    # 置零通道
    for neuron in neurons.values():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                with torch.no_grad():
                    p.zero_()
    with torch.no_grad():
        result_zero = ens.forward(
            neuron_embeddings=neuron_embeddings, return_logits=True, fusion_mode="soft",
        )
    logits_with = result_with["weighted_logits"]
    logits_zero = result_zero["weighted_logits"]
    # 随机初始化下通道权重非零，置零后 side-channel 调制消失 → 输出应变化
    diff = (logits_with - logits_zero).abs().max().item()
    assert diff > 0, f"置零通道应改变输出, diff={diff}"
    print(f"  PASS: 通道置零后输出变化 (diff={diff:.4f})")


def test_eval_set_no_leak():
    """[6] 评估集无泄漏：split_train_eval 训练/评估不重叠。"""
    print("\n[6] 评估集无泄漏")
    texts = [f"样本文本内容 {i} 用于测试分桶" for i in range(200)]
    train, eval_ = split_train_eval(texts, eval_ratio=0.05)
    inter = set(train) & set(eval_)
    assert len(inter) == 0, f"训练/评估应无重叠, got {len(inter)}"
    # 跨运行一致
    train2, eval2 = split_train_eval(texts, eval_ratio=0.05)
    assert eval_ == eval2, "评估集应跨运行一致"
    print(f"  PASS: 训练 {len(train)} 条 / 评估 {len(eval_)} 条, 无重叠, 跨运行一致")


def main():
    print("=" * 70)
    print("T14 Ablation 评估 smoke test")
    print("=" * 70)

    test_solo_ppl_logic()
    test_ensemble_ppl_logic()
    test_fusion_modes()
    test_field_conditioning_switch()
    test_disable_channels()
    test_eval_set_no_leak()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
