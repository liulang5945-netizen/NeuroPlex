#!/usr/bin/env python3
"""zh 基座 PPL 口径诊断（2026-08-11）。

背景：培养期验证（verify_feed_sleep_progressive）测得 held-out zh 整句 PPL 10761
（≈ 随机 50K 词表），而 C24 报告 answer PPL 70.2——差异巨大。怀疑整句 PPL 被
非 answer（prompt）部分主导。本脚本在 foundation_v1_dual（C24 重训产物）上
对比三种口径的 zh PPL：
1. 全序列：所有 token 计 loss（培养期验证口径）
2. answer-masked：只对 answer 计 loss（C24 SFT 口径，sft_mask）
3. prompt-only：只对 prompt 部分计 loss

结论用于修正培养期验证指标口径 + 定位 zh 基座真实能力。

运行：python -u scripts/training/diag_zh_ppl_masks.py
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from taiji.resonance import ResonanceNeuron  # noqa: E402
from scripts.training.utils import load_general_tokenizer  # noqa: E402
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab  # noqa: E402
from scripts.training.train_domain_target_sft import (  # noqa: E402
    build_sample, build_batch, load_sft, DOMAIN_VOCAB, SEQ_LEN,
)

NEURON_DIR = "data/foundation_v1_dual"
DOMAIN = "zh"
N_EVAL = 60


def load_model():
    ck = torch.load(os.path.join(NEURON_DIR, f"neuron_{DOMAIN}.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ck["state_dict"], strict=False)
    neuron.eval()
    shared_emb = torch.nn.Embedding(cfg.base_embed_dim * 0 + 0, 0)  # placeholder
    sd = ck.get("shared_embedding_state")
    vocab = sd["weight"].shape[0]
    shared_emb = torch.nn.Embedding(vocab, cfg.base_embed_dim)
    shared_emb.weight.data.copy_(sd["weight"])
    return neuron, shared_emb, ck


def ppl_by_mask(logits, targets, sft_mask):
    """按 sft_mask 计算 PPL。mask=True 位置计 loss。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_sft = sft_mask[:, 1:].contiguous()
    vocab_size = logits.size(-1)
    shift_targets = shift_targets.clamp(0, vocab_size - 1)
    shift_targets = shift_targets.clone()
    shift_targets[~shift_sft] = -100
    loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                           shift_targets.view(-1), ignore_index=-100,
                           reduction="sum")
    n = shift_sft.sum().item()
    return loss.item(), n


def main():
    random.seed(42)
    print("=" * 60, flush=True)
    print(f"zh 基座 PPL 口径诊断（{NEURON_DIR}）", flush=True)
    print("=" * 60, flush=True)

    neuron, shared_emb, ck = load_model()
    print(f"  neuron: {sum(p.numel() for p in neuron.parameters())/1e6:.1f}M, "
          f"hidden={neuron.config.hidden_size}, vocab={neuron.config.vocab_size}",
          flush=True)
    print(f"  ckpt best_ppl: {ck.get('result')}", flush=True)

    domain_sp = load_tokenizer_for_vocab(DOMAIN, DOMAIN_VOCAB[DOMAIN])
    general_sp = load_general_tokenizer()
    samples = load_sft(DOMAIN)
    random.shuffle(samples)
    pairs = [(s["full"], s["prompt"]) for s in samples[:N_EVAL]]

    # 长度分层抽样：短/中/长 answer 各看口径差异
    buckets = {"short(<=60)": [], "mid(61-200)": [], "long(>200)": []}
    for s in samples[:200]:
        rl = len(s.get("response", ""))
        if rl <= 60:
            buckets["short(<=60)"].append((s["full"], s["prompt"]))
        elif rl <= 200:
            buckets["mid(61-200)"].append((s["full"], s["prompt"]))
        else:
            buckets["long(>200)"].append((s["full"], s["prompt"]))
    for k, v in buckets.items():
        random.shuffle(v)

    with torch.no_grad():
        for label, bp in [("全序列(培养期口径)", pairs),
                          ("短answer", buckets["short(<=60)"][:30]),
                          ("中answer", buckets["mid(61-200)"][:30]),
                          ("长answer", buckets["long(>200)"][:30])]:
            if not bp:
                continue
            tot_full, n_full = 0.0, 0
            tot_ans, n_ans = 0.0, 0
            tot_prompt, n_prompt = 0.0, 0
            for i in range(0, len(bp), 8):
                batch = bp[i:i + 8]
                emb, y, m, sm, g_ids = build_batch(batch, domain_sp, general_sp, shared_emb)
                r = neuron.forward(emb, return_logits=True)
                lg = r["logits"]
                # 全序列
                lg_s = lg[:, :-1, :].contiguous()
                y_s = y[:, 1:].clone().contiguous()
                m_s = m[:, 1:].contiguous()
                y_s = y_s.clamp(0, lg.size(-1) - 1)
                y_s[~m_s] = -100
                loss = F.cross_entropy(lg_s.view(-1, lg.size(-1)), y_s.view(-1),
                                       ignore_index=-100, reduction="sum")
                tot_full += loss.item()
                n_full += m_s.sum().item()
                # answer-masked（C24 口径）
                l, n = ppl_by_mask(lg, y, sm)
                tot_ans += l
                n_ans += n
                # prompt-only（sft mask 取反且 attn 内）
                pm = sm.clone()
                pm[:, 1:] = ~sm[:, 1:] & m[:, 1:]
                l2, n2 = ppl_by_mask(lg, y, pm)
                tot_prompt += l2
                n_prompt += n2
            full_ppl = math.exp(min(tot_full / max(n_full, 1), 20))
            ans_ppl = math.exp(min(tot_ans / max(n_ans, 1), 20))
            prompt_ppl = math.exp(min(tot_prompt / max(n_prompt, 1), 20))
            print(f"\n[{label}] n={len(bp)}", flush=True)
            print(f"  全序列 PPL = {full_ppl:8.1f} (loss {tot_full/max(n_full,1):.4f})", flush=True)
            print(f"  answer PPL = {ans_ppl:8.1f} (loss {tot_ans/max(n_ans,1):.4f})  ← C24 口径", flush=True)
            print(f"  prompt PPL = {prompt_ppl:8.1f} (loss {tot_prompt/max(n_prompt,1):.4f})", flush=True)


if __name__ == "__main__":
    main()
