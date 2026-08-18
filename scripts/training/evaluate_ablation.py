"""T14: Ablation 评估——定位收益来源。

4 组对照实验（每组只变一个变量，其余固定）：
1. 共振有效性: ensemble 协作 vs 最强个体（验证"协作 > 单神经元"）
2. 融合方式: soft 加权平均 vs consensus 共识投票（验证集体智慧）
3. side_channels: 有协作通道 vs 无（验证神经元间通信）
4. field_conditioning: 有场读入 vs 无（验证场调制）

评估集：T1 held-out（5% hash 分桶，无数据泄漏，跨运行一致）
指标：PPL（越低越好）+ 每组对照差异

Usage:
    python -u scripts/training/evaluate_ablation.py
    python -u scripts/training/evaluate_ablation.py --weights dialogue --n_eval 50
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.resonance import (
    ResonanceField, ResonanceEnsemble,
)
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    load_dialogue_texts_multi, split_train_eval,
)
from scripts.training.eval_dialogue import load_neurons_and_weights, load_cross_spec_weights
from scripts.training.experiment_config import (
    DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER,
)

DEVICE = "cpu"


def compute_ppl_for_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor) -> tuple:
    """计算 shift-CE loss 和 PPL。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    shift_targets_clone = shift_targets.clone()
    shift_targets_clone[~shift_mask] = -100
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets_clone.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_tokens = shift_mask.sum().item()
    return loss.item(), n_tokens


def eval_solo_ppl(neurons, shared_embeddings, eval_texts, domain_sp, general_sp) -> dict:
    """每个神经元的 solo PPL（无协作）。"""
    print("\n[Group 1] 个体 PPL（无协作基线）", flush=True)
    results = {}
    for nid, neuron in neurons.items():
        shared_emb = shared_embeddings[nid]
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in eval_texts:
                emb, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                result = neuron.forward(emb.to(DEVICE), return_logits=True)
                loss, n_tok = compute_ppl_for_logits(
                    result["logits"], targets.to(DEVICE), mask.to(DEVICE),
                )
                total_loss += loss
                total_tokens += n_tok
        ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))
        results[nid] = ppl
        print(f"  solo [{nid}]: PPL={ppl:.1f}", flush=True)
    return results


def eval_ensemble_ppl(neurons, shared_embeddings, eval_texts, domain_sp, general_sp,
                      fusion_mode: str = "soft", field_conditioning: bool = True,
                      disable_channels: bool = False) -> float:
    """ensemble 协作 PPL（可配置 fusion_mode / field_conditioning / channels）。"""
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    load_cross_spec_weights(ensemble, "dialogue")

    # Group 3: 禁用 side_channels（权重置零 = 无信号调制）
    if disable_channels:
        for nid, neuron in neurons.items():
            for ch in neuron.excite_channels.values():
                for p in ch.parameters():
                    with torch.no_grad():
                        p.zero_()
            for ch in neuron.inhibit_channels.values():
                for p in ch.parameters():
                    with torch.no_grad():
                        p.zero_()

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
            neuron_embeddings = {}
            targets, mask = None, None
            for nid, shared_emb in shared_embeddings.items():
                emb, tgt, msk = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                neuron_embeddings[nid] = emb.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)

            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode=fusion_mode,
                field_conditioning=field_conditioning,
            )
            if "weighted_logits" in result:
                fused = result["weighted_logits"]
            else:
                continue
            loss, n_tok = compute_ppl_for_logits(fused, targets, mask)
            total_loss += loss
            total_tokens += n_tok
    ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))
    return ppl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="dialogue",
                        choices=["dialogue", "cross_spec", "none"])
    parser.add_argument("--n_eval", type=int, default=100, help="评估样本数")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print("=" * 70, flush=True)
    print("T14 Ablation 评估（定位收益来源）", flush=True)
    print(f"weights={args.weights}, n_eval={args.n_eval}", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载神经元 + 权重
    neurons, shared_embeddings = load_neurons_and_weights(args.weights, topology_mode="hybrid")

    # 2. 加载 held-out 评估集（T1 hash 分桶）
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    dialogue_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh",
    )
    all_texts = load_dialogue_texts_multi(dialogue_dir, max_texts=args.n_eval * 30)
    _, eval_texts = split_train_eval(all_texts, eval_ratio=0.05)
    eval_texts = eval_texts[:args.n_eval]
    print(f"\n评估集(held-out 5% hash 分桶): {len(eval_texts)} 条对话", flush=True)

    results = {}

    # ── Group 1: 共振有效性（协作 vs 个体）──
    solo = eval_solo_ppl(neurons, shared_embeddings, eval_texts, domain_sp, general_sp)
    best_solo = min(solo.values())
    ens_soft = eval_ensemble_ppl(
        neurons, shared_embeddings, eval_texts, domain_sp, general_sp,
        fusion_mode="soft",
    )
    results["group1_resonance"] = {
        "solo_per_neuron": solo,
        "best_solo": best_solo,
        "ensemble_soft": ens_soft,
        "collab_improves": ens_soft < best_solo,
        "delta_ppl": ens_soft - best_solo,
    }
    print(f"\n[Group 1 结论] 协作({ens_soft:.1f}) vs 最强个体({best_solo:.1f}) → "
          f"{'协作更优' if ens_soft < best_solo else '个体更优'} (Δ={ens_soft - best_solo:+.1f})", flush=True)

    # ── Group 2: 融合方式（soft vs consensus）──
    ens_consensus = eval_ensemble_ppl(
        neurons, shared_embeddings, eval_texts, domain_sp, general_sp,
        fusion_mode="consensus",
    )
    results["group2_fusion"] = {
        "soft": ens_soft,
        "consensus": ens_consensus,
        "consensus_better": ens_consensus < ens_soft,
        "delta_ppl": ens_consensus - ens_soft,
    }
    print(f"[Group 2 结论] consensus({ens_consensus:.1f}) vs soft({ens_soft:.1f}) → "
          f"{'consensus 更优' if ens_consensus < ens_soft else 'soft 更优'} (Δ={ens_consensus - ens_soft:+.1f})", flush=True)

    # ── Group 3: side_channels 贡献（有 vs 无）──
    ens_no_ch = eval_ensemble_ppl(
        neurons, shared_embeddings, eval_texts, domain_sp, general_sp,
        fusion_mode="soft", disable_channels=True,
    )
    results["group3_side_channels"] = {
        "with_channels": ens_soft,
        "without_channels": ens_no_ch,
        "channels_help": ens_soft < ens_no_ch,
        "delta_ppl": ens_soft - ens_no_ch,
    }
    print(f"[Group 3 结论] 有通道({ens_soft:.1f}) vs 无通道({ens_no_ch:.1f}) → "
          f"{'通道有效' if ens_soft < ens_no_ch else '通道无益'} (Δ={ens_soft - ens_no_ch:+.1f})", flush=True)

    # ── Group 4: field_conditioning 贡献（有 vs 无）──
    ens_no_field = eval_ensemble_ppl(
        neurons, shared_embeddings, eval_texts, domain_sp, general_sp,
        fusion_mode="soft", field_conditioning=False,
    )
    results["group4_field_conditioning"] = {
        "with_field": ens_soft,
        "without_field": ens_no_field,
        "field_helps": ens_soft < ens_no_field,
        "delta_ppl": ens_soft - ens_no_field,
    }
    print(f"[Group 4 结论] 有场读入({ens_soft:.1f}) vs 无场读入({ens_no_field:.1f}) → "
          f"{'场调制有效' if ens_soft < ens_no_field else '场调制无益'} (Δ={ens_soft - ens_no_field:+.1f})", flush=True)

    # ── 汇总表 ──
    print("\n" + "=" * 70, flush=True)
    print("Ablation 汇总（PPL 越低越好）", flush=True)
    print("=" * 70, flush=True)
    print(f"  {best_solo:>8.1f}  最强个体 (baseline)", flush=True)
    print(f"  {ens_soft:>8.1f}  ensemble soft（默认协作）", flush=True)
    print(f"  {ens_consensus:>8.1f}  ensemble consensus（共识投票）", flush=True)
    print(f"  {ens_no_ch:>8.1f}  ensemble 无 side_channels", flush=True)
    print(f"  {ens_no_field:>8.1f}  ensemble 无 field_conditioning", flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}", flush=True)

    # 汇总收益来源
    print("\n[收益来源定位]", flush=True)
    gains = {
        "共振协作": best_solo - ens_soft,
        "共识投票": ens_soft - ens_consensus,
        "side_channels": ens_no_ch - ens_soft,
        "field_conditioning": ens_no_field - ens_soft,
    }
    for name, gain in sorted(gains.items(), key=lambda x: -x[1]):
        direction = "↑收益" if gain > 0 else "↓无效"
        print(f"  {name}: ΔPPL={gain:+.2f} {direction}", flush=True)

    print("\nAblation 评估完成。", flush=True)


if __name__ == "__main__":
    main()
