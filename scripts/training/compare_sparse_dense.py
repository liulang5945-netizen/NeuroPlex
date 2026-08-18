"""阶段4：稀疏 vs 稠密对比评估脚本（§4.0c/§4.0d 训练验证）。

对比**同一 checkpoint** 在两种协作模式下的表现：
  1. 稠密模式 (use_sparse_router=False): 所有神经元参与 round 2+ 协作
  2. 稀疏模式 (use_sparse_router=True):  Router 选 per-sample top-K 参与 round 2+

评估指标：
  - 个体 PPL / 协作 PPL（held-out 对话数据）
  - EMERGE（协作 vs 最强个体）
  - 每样本平均激活神经元数（稀疏性，稠密恒=N）
  - 推理速度（每 token 平均 forward 耗时）

⚠️ 前提：checkpoint 必须含 sparse_router_state（即用 --use_sparse_router 训练过）。
若 checkpoint 无 Router 状态，稀疏模式的 Router 是随机初始化的，对比不代表真实效果
（脚本会检测并警告，此时结果仅供流程验证）。

Usage:
    python -u scripts/training/compare_sparse_dense.py
    python -u scripts/training/compare_sparse_dense.py --ckpt_path data/neurons/cross_spec_dialogue.ckpt.pt
    python -u scripts/training/compare_sparse_dense.py --ckpt_path xxx.pt --n_eval 20 --device cpu
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.eval_dialogue import (
    load_neurons_and_weights, load_cross_spec_weights, _checkpoint_has_router,
)
from scripts.training.finetune_cross_spec import load_dialogue_texts
from scripts.training.experiment_config import (
    ENSEMBLE_DIALOGUE_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN,
)
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer

DEVICE = "cpu"


def _load_eval_texts(n_eval: int = 50):
    """加载 held-out 对话评估数据（与 eval_dialogue 一致）。"""
    dialogue_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh", "alpaca_zh_sft.jsonl",
    )
    all_texts = load_dialogue_texts(dialogue_path, max_texts=n_eval * 3)
    texts = all_texts[-n_eval:] if len(all_texts) > n_eval else all_texts
    print(f"  评估集(alpaca-zh SFT): {len(texts)} 条对话", flush=True)
    return texts


def _build_ensemble(neurons, use_router: bool, top_k: int, warmup_steps: int):
    """构建 ensemble（dense 或 sparse 模式）。"""
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    return ResonanceEnsemble(
        neurons, field, max_rounds=2,
        use_sparse_router=use_router,
        sparse_router_top_k=top_k,
        sparse_router_warmup_steps=warmup_steps,
    )


def _eval_individual_ppl(neurons, shared_embeddings, domain_sp, general_sp, texts):
    """个体 PPL（每神经元独立，不涉及协作/路由）。"""
    individual_ppls = {}
    with torch.no_grad():
        for nid, neuron in neurons.items():
            shared_emb = shared_embeddings[nid]
            total_loss = 0.0
            total_tokens = 0
            for text in texts:
                emb_out, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                emb_out = emb_out.to(DEVICE)
                targets = targets.to(DEVICE)
                mask = mask.to(DEVICE)
                result = neuron.forward(emb_out, return_logits=True)
                logits = result["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100, reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += shift_mask.sum().item()
            avg_loss = total_loss / max(total_tokens, 1)
            individual_ppls[nid] = math.exp(min(avg_loss, 20))
    return individual_ppls


def _eval_collab(ensemble, neurons, shared_embeddings, domain_sp, general_sp, texts):
    """协作 PPL + 激活数统计 + 推理速度（单次 forward 循环）。

    Returns:
        (collab_ppl, avg_loss, avg_active, tokens_per_sec)
    """
    total_loss = 0.0
    total_tokens = 0
    total_active = 0.0
    n_samples = 0
    fwd_time = 0.0
    with torch.no_grad():
        for text in texts:
            neuron_embeddings = {}
            targets = None
            mask = None
            for nid, shared_emb in shared_embeddings.items():
                emb_out, tgt, msk = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                neuron_embeddings[nid] = emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)

            t0 = time.perf_counter()
            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )
            fwd_time += time.perf_counter() - t0

            if "weighted_logits" in result:
                fused_logits = result["weighted_logits"]
            else:
                best_nid = max(result.get("final_scores", {}),
                               key=result["final_scores"].get, default=NEURON_IDS[0])
                fused_logits = neurons[best_nid].forward(
                    neuron_embeddings[best_nid], return_logits=True
                )["logits"]

            shift_logits = fused_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100, reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()
            n_samples += 1

            # 激活神经元数统计（稀疏模式用 Router k_per_sample，稠密恒=N）
            router_result = getattr(ensemble, "_last_router_result", None)
            if router_result is not None and "k_per_sample" in router_result:
                total_active += router_result["k_per_sample"].float().mean().item()
            else:
                total_active += len(neurons)

    avg_loss = total_loss / max(total_tokens, 1)
    collab_ppl = math.exp(min(avg_loss, 20))
    avg_active = total_active / max(n_samples, 1)
    tokens_per_sec = total_tokens / max(fwd_time, 1e-9)
    return collab_ppl, avg_loss, avg_active, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description="阶段4: 稀疏 vs 稠密对比评估")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="checkpoint 文件（默认 data/neurons/cross_spec_dialogue.pt）")
    parser.add_argument("--weights", type=str, default="dialogue",
                        choices=["dialogue", "cross_spec"],
                        help="weights_type（用于默认路径与拓扑推断）")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sparse_router_top_k", type=int, default=3,
                        help="稀疏模式 top-K（须与训练时一致）")
    parser.add_argument("--sparse_router_warmup_steps", type=int, default=2000,
                        help="warmup 步数（须与训练时一致，推理恒走 Phase 2 不受影响）")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print("=" * 70, flush=True)
    print("阶段4: 稀疏 vs 稠密对比评估", flush=True)
    print("=" * 70, flush=True)

    neurons, shared_embeddings = load_neurons_and_weights(
        args.weights, "hybrid", ckpt_path=args.ckpt_path)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    texts = _load_eval_texts(args.n_eval)

    # ⚠️ 前提检测：checkpoint 是否含 Router 状态
    has_router = _checkpoint_has_router(args.weights, args.ckpt_path)
    if not has_router:
        print("\n⚠️  警告: checkpoint 不含 sparse_router_state（训练时未启用 --use_sparse_router）。", flush=True)
        print("    稀疏模式的 Router 将随机初始化，对比结果**不代表** Router 真实效果。", flush=True)
        print("    仅用于流程验证；真实验证需先以 --use_sparse_router 训练产出 checkpoint。", flush=True)

    # 个体 PPL（两种模式共用）
    individual_ppls = _eval_individual_ppl(
        neurons, shared_embeddings, domain_sp, general_sp, texts)
    min_ind = min(individual_ppls.values())
    best_id = min(individual_ppls, key=individual_ppls.get)
    print(f"\n[个体] 最强 [{best_id}] PPL={min_ind:.1f}", flush=True)

    # ── 稠密模式 ──
    print("\n" + "-" * 70, flush=True)
    print("[稠密模式] use_sparse_router=False", flush=True)
    dense_ensemble = _build_ensemble(neurons, False, args.sparse_router_top_k,
                                     args.sparse_router_warmup_steps)
    load_cross_spec_weights(dense_ensemble, args.weights, args.ckpt_path)
    dense_ppl, dense_loss, dense_active, dense_tps = _eval_collab(
        dense_ensemble, neurons, shared_embeddings, domain_sp, general_sp, texts)
    dense_emerge = (min_ind - dense_ppl) / min_ind * 100
    print(f"  协作 PPL={dense_ppl:.1f} (loss={dense_loss:.4f})", flush=True)
    print(f"  平均激活神经元: {dense_active:.2f}/{len(neurons)}", flush=True)
    print(f"  推理速度: {dense_tps:.1f} tokens/s", flush=True)
    print(f"  EMERGE: {'✓' if dense_ppl < min_ind else '✗'} 协作比最强个体好 {dense_emerge:.1f}%", flush=True)

    # ── 稀疏模式 ──
    print("\n" + "-" * 70, flush=True)
    print(f"[稀疏模式] use_sparse_router=True (top_k={args.sparse_router_top_k})", flush=True)
    sparse_ensemble = _build_ensemble(neurons, True, args.sparse_router_top_k,
                                      args.sparse_router_warmup_steps)
    load_cross_spec_weights(sparse_ensemble, args.weights, args.ckpt_path)
    sparse_ppl, sparse_loss, sparse_active, sparse_tps = _eval_collab(
        sparse_ensemble, neurons, shared_embeddings, domain_sp, general_sp, texts)
    sparse_emerge = (min_ind - sparse_ppl) / min_ind * 100
    print(f"  协作 PPL={sparse_ppl:.1f} (loss={sparse_loss:.4f})", flush=True)
    print(f"  平均激活神经元: {sparse_active:.2f}/{len(neurons)}", flush=True)
    print(f"  推理速度: {sparse_tps:.1f} tokens/s", flush=True)
    print(f"  EMERGE: {'✓' if sparse_ppl < min_ind else '✗'} 协作比最强个体好 {sparse_emerge:.1f}%", flush=True)

    # ── 汇总对比 ──
    print("\n" + "=" * 70, flush=True)
    print("汇总对比", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'指标':<20} {'稠密':>12} {'稀疏':>12} {'差异':>12}", flush=True)
    print(f"  {'协作 PPL':<20} {dense_ppl:>12.1f} {sparse_ppl:>12.1f} "
          f"{dense_ppl - sparse_ppl:>+12.1f}", flush=True)
    print(f"  {'EMERGE %':<20} {dense_emerge:>12.1f} {sparse_emerge:>12.1f} "
          f"{dense_emerge - sparse_emerge:>+12.1f}", flush=True)
    print(f"  {'平均激活神经元':<20} {dense_active:>12.2f} {sparse_active:>12.2f} "
          f"{dense_active - sparse_active:>+12.2f}", flush=True)
    print(f"  {'推理速度 tok/s':<20} {dense_tps:>12.1f} {sparse_tps:>12.1f} "
          f"{sparse_tps - dense_tps:>+12.1f}", flush=True)

    # 结论
    print("\n[结论]", flush=True)
    if sparse_active < dense_active - 0.01:
        print(f"  ✓ 稀疏模式激活 {sparse_active:.1f} 神经元 < 稠密 {dense_active:.0f}，实现了算力聚焦", flush=True)
    else:
        print(f"  ✗ 稀疏未实现激活减少（可能 top-K ≥ N 或 Router 未训练）", flush=True)
    if has_router:
        if sparse_ppl <= dense_ppl * 1.05:
            print(f"  ✓ 稀疏 PPL 与稠密相当（{sparse_ppl:.1f} vs {dense_ppl:.1f}），"
                  f"算力节省未损质量", flush=True)
        else:
            print(f"  ✗ 稀疏 PPL 劣于稠密（{sparse_ppl:.1f} vs {dense_ppl:.1f}），"
                  f"Router 选型欠佳或需继续训练", flush=True)
    else:
        print(f"  - checkpoint 无 Router 状态，上述结论仅限流程验证", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("对比完成", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
