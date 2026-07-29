"""评估 standard 神经元 zh_std0 的生成质量。

支持两种模式：
1. 单神经元评估：zh_std0 独立生成
2. 混合协作评估：zh_std0 (standard) + zh_aug0~3 (compact) 协作

Usage:
    # 单神经元评估
    python -u scripts/training/eval_std_neuron.py --mode single

    # 混合协作评估（standard + compact）
    python -u scripts/training/eval_std_neuron.py --mode mixed

    # 两种都跑
    python -u scripts/training/eval_std_neuron.py --mode both
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts,
)

DOMAIN = "zh"
STD_NEURON_ID = "zh_std0"
COMPACT_NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]
DEVICE = "cpu"


def load_neuron(nid, spec):
    """加载单个神经元及其 shared_embedding。

    Args:
        nid: 神经元 ID
        spec: 期望的规格（用于创建配置，但实际以 checkpoint 中的 neuron_config 为准）
    """
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    if not os.path.exists(path):
        print(f"  [{nid}] WARN 未找到 checkpoint: {path}", flush=True)
        return None, None, None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # 优先使用 checkpoint 中的 neuron_config，回退到 spec 创建
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DOMAIN, spec=spec)
    cfg.unified_field_dim = None

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.eval()

    # 加载该神经元自己的 shared_embedding
    shared_emb = nn.Embedding(256000, 512)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE).eval()

    result = ckpt.get("result", {})
    print(f"  [{nid}] spec={cfg.spec}, params={sum(p.numel() for p in neuron.parameters())/1e6:.1f}M, "
          f"best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)

    return neuron, shared_emb, cfg


def eval_single_ppl(neuron, shared_emb, domain_sp, general_sp, n_eval=100):
    """单神经元 PPL 评估。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[单神经元 PPL] {STD_NEURON_ID}", flush=True)
    print("=" * 70, flush=True)

    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=n_eval * 3)
    texts = texts[-n_eval:] if len(texts) > n_eval else texts
    print(f"  评估集(simple_zh): {len(texts)} 条文本", flush=True)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            shared_emb_out, targets, mask = batch_align_and_embed(
                [text], domain_sp, general_sp, shared_emb,
            )
            shared_emb_out = shared_emb_out.to(DEVICE)
            targets = targets.to(DEVICE)
            mask = mask.to(DEVICE)
            result = neuron.forward(shared_emb_out, return_logits=True)
            logits = result["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    print(f"  {STD_NEURON_ID} PPL={ppl:.1f} (loss={avg_loss:.4f})", flush=True)
    return ppl


def eval_single_generation(neuron, shared_emb, domain_sp, general_sp):
    """单神经元生成质量评估。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[单神经元生成] {STD_NEURON_ID}", flush=True)
    print("=" * 70, flush=True)

    PROMPTS = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "深度学习在自然语言处理中的应用",
        "请解释神经网络的工作原理",
        "在公园里，阳光透过",
    ]

    for prompt in PROMPTS:
        general_ids = general_sp.EncodeAsIds(prompt)
        if not general_ids:
            print(f"  prompt: {prompt}\n  (empty)\n", flush=True)
            continue
        ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated_domain = []
            for _ in range(80):
                emb_input = shared_emb(ids)
                result = neuron.forward(emb_input, return_logits=True)
                logits = result["logits"][:, -1, :].float()

                # Repetition penalty
                if generated_domain:
                    for prev_token in set(generated_domain[-20:]):
                        logits[0, prev_token] /= 1.2

                # Temperature + top-k
                logits = logits / 0.8
                if 40 < logits.size(-1):
                    topk_vals, _ = torch.topk(logits, 40, dim=-1)
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if hasattr(domain_sp, 'eos_id'):
                    eos = domain_sp.eos_id() if callable(domain_sp.eos_id) else domain_sp.eos_id
                    if next_token == eos:
                        break
                generated_domain.append(next_token)
                piece = domain_sp.DecodeIds([next_token])
                gen_ids = general_sp.EncodeAsIds(piece)
                if gen_ids:
                    ids = torch.cat([ids, torch.tensor([gen_ids], dtype=torch.long, device=DEVICE)], dim=1)
                else:
                    break
                if ids.shape[1] > 200:
                    break
            text = domain_sp.DecodeIds(generated_domain)
        print(f"  prompt: {prompt}\n  {STD_NEURON_ID}: {text}\n", flush=True)


def eval_mixed_collab(std_neuron, std_emb, std_cfg,
                       compact_neurons, compact_embs, compact_cfg,
                       domain_sp, general_sp, n_eval=100):
    """混合协作评估：standard + compact 神经元。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[混合协作 PPL] {STD_NEURON_ID} (standard) + {list(compact_neurons.keys())} (compact)", flush=True)
    print("=" * 70, flush=True)

    # 由于 standard 和 compact 的 field_dim 不同（3072 vs 2048），
    # 无法在同一个 ResonanceField 中协作。改为 logits 融合方式。
    # standard 神经元获得固定权重，compact 神经元按共振分分配剩余权重。

    all_neurons = {**compact_neurons, STD_NEURON_ID: std_neuron}
    all_embs = {**compact_embs, STD_NEURON_ID: std_emb}

    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=n_eval * 3)
    texts = texts[-n_eval:] if len(texts) > n_eval else texts
    print(f"  评估集(simple_zh): {len(texts)} 条文本", flush=True)

    # 个体 PPL
    individual_ppls = {}
    for nid, neuron in all_neurons.items():
        shared_emb = all_embs[nid]
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                shared_emb_out, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                shared_emb_out = shared_emb_out.to(DEVICE)
                targets = targets.to(DEVICE)
                mask = mask.to(DEVICE)
                result = neuron.forward(shared_emb_out, return_logits=True)
                logits = result["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += shift_mask.sum().item()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        individual_ppls[nid] = ppl
        print(f"  个体 [{nid}]: PPL={ppl:.1f}", flush=True)

    # 协作 PPL（logits 加权融合，standard 获得固定权重 0.5）
    STD_WEIGHT = 0.5
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            # 每个神经元独立编码
            all_logits = {}
            targets = None
            mask = None
            for nid, shared_emb in all_embs.items():
                shared_emb_out, tgt, msk = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                shared_emb_out = shared_emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)
                result = all_neurons[nid].forward(shared_emb_out, return_logits=True)
                all_logits[nid] = result["logits"]

            # logits 加权融合：standard 固定权重，compact 平均分配剩余
            std_logits = all_logits[STD_NEURON_ID]
            compact_logits_list = [all_logits[nid] for nid in compact_neurons]
            compact_avg = torch.stack(compact_logits_list).mean(dim=0)
            fused_logits = STD_WEIGHT * std_logits + (1 - STD_WEIGHT) * compact_avg

            shift_logits = fused_logits[:, :-1, :].contiguous()
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
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    collab_ppl = math.exp(min(avg_loss, 20))
    print(f"  协作 [mixed, std_w={STD_WEIGHT}]: PPL={collab_ppl:.1f}", flush=True)

    min_ind = min(individual_ppls.values())
    best_id = min(individual_ppls, key=individual_ppls.get)
    print(f"\n  最强个体: [{best_id}] PPL={min_ind:.1f}", flush=True)
    print(f"  协作 PPL: {collab_ppl:.1f}", flush=True)
    if collab_ppl < min_ind:
        imp = (min_ind - collab_ppl) / min_ind * 100
        print(f"  EMERGE 协作比最强个体好 {imp:.1f}%", flush=True)
    else:
        print(f"  NO_EMERGE 协作 PPL={collab_ppl:.1f} >= 最强个体 {min_ind:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="评估 standard 神经元 zh_std0")
    parser.add_argument("--mode", choices=["single", "mixed", "both"], default="both",
                        help="评估模式：single(单神经元) / mixed(混合协作) / both")
    parser.add_argument("--n_eval", type=int, default=100, help="PPL 评估文本数")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("zh_std0 (standard) 神经元评估", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载 standard 神经元
    print("\n[1] 加载 standard 神经元...", flush=True)
    std_neuron, std_emb, std_cfg = load_neuron(STD_NEURON_ID, "standard")
    if std_neuron is None:
        print("ERROR: 未找到 zh_std0 checkpoint，退出", flush=True)
        return

    # 2. tokenizers
    print("\n[2] tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    # 3. 单神经元评估
    if args.mode in ("single", "both"):
        print("\n[3] 单神经元评估...", flush=True)
        std_ppl = eval_single_ppl(std_neuron, std_emb, domain_sp, general_sp, n_eval=args.n_eval)
        eval_single_generation(std_neuron, std_emb, domain_sp, general_sp)

    # 4. 混合协作评估
    if args.mode in ("mixed", "both"):
        print("\n[4] 加载 compact 神经元用于混合协作...", flush=True)
        compact_neurons = {}
        compact_embs = {}
        compact_cfg = None
        for nid in COMPACT_NEURON_IDS:
            neuron, emb, cfg = load_neuron(nid, "compact")
            if neuron is not None:
                compact_neurons[nid] = neuron
                compact_embs[nid] = emb
                compact_cfg = cfg

        if compact_neurons:
            eval_mixed_collab(std_neuron, std_emb, std_cfg,
                            compact_neurons, compact_embs, compact_cfg,
                            domain_sp, general_sp, n_eval=args.n_eval)
        else:
            print("  WARN: 未加载任何 compact 神经元，跳过混合协作评估", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("评估完成", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
