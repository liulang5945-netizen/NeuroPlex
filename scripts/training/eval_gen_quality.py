"""详细生成质量对比：协作(side_channels) vs 个体神经元。

测试维度：
  1. 多 prompt 生成对比（10 条）
  2. 生成质量指标：diversity、repetition、length
  3. 重复 n-gram 分析

Usage:
    python -u scripts/training/eval_gen_quality.py
"""
from __future__ import annotations

import os
import sys
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR,
)

DOMAIN = "zh"
NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]
DEVICE = "cpu"

PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
    "深度学习在自然语言处理中的应用",
    "请解释神经网络的工作原理",
    "在公园里，阳光透过",
    "今天天气真好，我想去",
    "中国的首都是哪里？",
    "请写一首关于春天的诗",
    "如何学习编程？",
    "从前有一座山，山里有个庙",
]


def load_neurons():
    cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None
    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        neuron = ResonanceNeuron(cfg).to(DEVICE)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.eval()
        neurons[nid] = neuron
        shared_emb = nn.Embedding(256000, 512)
        if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
            shared_emb.load_state_dict(ckpt["shared_embedding_state"])
        shared_emb.to(DEVICE).eval()
        shared_embeddings[nid] = shared_emb

    # side_channels
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    # 加载微调权重
    finetuned_path = os.path.join(OUTPUT_DIR, "side_channels_finetuned.pt")
    if os.path.exists(finetuned_path):
        side_state = torch.load(finetuned_path, map_location=DEVICE, weights_only=False)
        for nid, neuron in neurons.items():
            if nid in side_state:
                for pid, ch_state in side_state[nid].get("excite", {}).items():
                    if pid in neuron.excite_channels:
                        neuron.excite_channels[pid].load_state_dict(ch_state)
        print(f"  [side_channels] 已加载微调权重", flush=True)

    return neurons, shared_embeddings, cfg


def compute_metrics(text):
    """计算生成质量指标。"""
    if not text:
        return {"len": 0, "diversity": 0, "repetition": 0, "top1_ratio": 0, "top2_ratio": 0}
    chars = list(text)
    n = len(chars)
    unique = len(set(chars))
    diversity = unique / max(n, 1)
    counter = Counter(chars)
    top1_ratio = counter.most_common(1)[0][1] / max(n, 1)
    # bigram repetition
    bigrams = [(chars[i], chars[i+1]) for i in range(n-1)]
    bg_counter = Counter(bigrams)
    top2_ratio = bg_counter.most_common(1)[0][1] / max(len(bigrams), 1) if bigrams else 0
    return {
        "len": n,
        "diversity": diversity,
        "repetition": top1_ratio,
        "top2_ratio": top2_ratio,
    }


def generate_individual(prompt, neuron, shared_emb, domain_sp, general_sp, max_tokens=80):
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return ""
    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = []
        for _ in range(max_tokens):
            emb_input = shared_emb(ids)
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :]
            next_token = logits.argmax(dim=-1).item()
            eos = getattr(domain_sp, 'eos_id', None)
            if eos is not None:
                eos_id = eos() if callable(eos) else eos
                if next_token == eos_id:
                    break
            generated.append(next_token)
            piece = domain_sp.DecodeIds([next_token])
            gen_ids = general_sp.EncodeAsIds(piece)
            if gen_ids:
                ids = torch.cat([ids, torch.tensor([gen_ids], dtype=torch.long, device=DEVICE)], dim=1)
            else:
                break
            if ids.shape[1] > 200:
                break
        return domain_sp.DecodeIds(generated)


def generate_collab(prompt, neurons, shared_embeddings, ensemble, domain_sp, general_sp, max_tokens=80):
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return ""
    current_ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = []
        for _ in range(max_tokens):
            neuron_embeddings = {}
            for nid, shared_emb in shared_embeddings.items():
                neuron_embeddings[nid] = shared_emb(current_ids)
            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )
            if "weighted_logits" in result:
                logits = result["weighted_logits"][:, -1, :]
            elif "neuron_logits" in result:
                n_logits = list(result["neuron_logits"].values())
                if len(set(l.shape[-1] for l in n_logits)) == 1:
                    logits = torch.stack(n_logits).mean(dim=0)[:, -1, :]
                else:
                    best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                    logits = result["neuron_logits"][best_nid][:, -1, :]
            else:
                best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                logits = neurons[best_nid].forward(neuron_embeddings[best_nid], return_logits=True)["logits"][:, -1, :]

            next_token = logits.argmax(dim=-1).item()
            eos = getattr(domain_sp, 'eos_id', None)
            if eos is not None:
                eos_id = eos() if callable(eos) else eos
                if next_token == eos_id:
                    break
            generated.append(next_token)
            piece = domain_sp.DecodeIds([next_token])
            gen_ids = general_sp.EncodeAsIds(piece)
            if gen_ids:
                current_ids = torch.cat([current_ids, torch.tensor([gen_ids], dtype=torch.long, device=DEVICE)], dim=1)
            else:
                break
            if current_ids.shape[1] > 200:
                break
        return domain_sp.DecodeIds(generated)


def main():
    print("=" * 70, flush=True)
    print("详细生成质量对比 — 协作(side_channels) vs 个体", flush=True)
    print("=" * 70, flush=True)

    print("\n[1] 加载神经元...", flush=True)
    neurons, shared_embeddings, cfg = load_neurons()
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    print(f"\n[2] 生成对比（{len(PROMPTS)} 条 prompt）...\n", flush=True)

    # 汇总指标
    all_metrics = {"collab": [], "zh_aug0": [], "zh_aug1": [], "zh_aug2": [], "zh_aug3": []}

    for prompt in PROMPTS:
        print(f"{'─'*70}", flush=True)
        print(f"  Prompt: {prompt}", flush=True)
        print(f"{'─'*70}", flush=True)

        # 协作生成
        collab_text = generate_collab(prompt, neurons, shared_embeddings, ensemble, domain_sp, general_sp)
        collab_metrics = compute_metrics(collab_text)
        all_metrics["collab"].append(collab_metrics)
        print(f"  [协作] len={collab_metrics['len']:3d} div={collab_metrics['diversity']:.2f} "
              f"rep={collab_metrics['repetition']:.2f} bg_rep={collab_metrics['top2_ratio']:.2f}", flush=True)
        print(f"         {collab_text[:120]}", flush=True)

        # 个体生成
        for nid in NEURON_IDS:
            text = generate_individual(prompt, neurons[nid], shared_embeddings[nid], domain_sp, general_sp)
            m = compute_metrics(text)
            all_metrics[nid].append(m)
            print(f"  [{nid}] len={m['len']:3d} div={m['diversity']:.2f} "
                  f"rep={m['repetition']:.2f} bg_rep={m['top2_ratio']:.2f}", flush=True)
            print(f"         {text[:120]}", flush=True)
        print(flush=True)

    # 汇总
    print(f"\n{'='*70}", flush=True)
    print("汇总（10 条 prompt 平均）", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'来源':<12} {'avg_len':>8} {'avg_div':>8} {'avg_rep':>8} {'avg_bg_rep':>10}", flush=True)
    print(f"{'─'*50}", flush=True)
    for key, label in [("collab", "协作"), ("zh_aug0", "zh_aug0"), ("zh_aug1", "zh_aug1"), ("zh_aug2", "zh_aug2"), ("zh_aug3", "zh_aug3")]:
        ms = all_metrics[key]
        avg_len = sum(m["len"] for m in ms) / len(ms)
        avg_div = sum(m["diversity"] for m in ms) / len(ms)
        avg_rep = sum(m["repetition"] for m in ms) / len(ms)
        avg_bg = sum(m["top2_ratio"] for m in ms) / len(ms)
        print(f"{label:<12} {avg_len:8.1f} {avg_div:8.3f} {avg_rep:8.3f} {avg_bg:10.3f}", flush=True)


if __name__ == "__main__":
    main()
