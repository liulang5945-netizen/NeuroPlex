"""诊断脚本 v2：深入检查 round 2 的 side_signals 和 field_state 是否生效。

Usage:
    python -u scripts/training/diagnose_rounds.py
"""
from __future__ import annotations

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
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR,
)
from scripts.training.train_cortex_joint import load_simple_zh_texts
from scripts.training.finetune_side_channels import load_neuron_with_embedding

DOMAIN = "zh"
NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]
DEVICE = "cpu"


def compute_ppl(logits, targets, mask):
    shift_l = logits[:, :-1, :].contiguous()
    shift_t = targets[:, 1:].contiguous()
    shift_m = mask[:, 1:].contiguous()
    shift_t = shift_t.clone()
    shift_t[~shift_m] = -100
    n_tok = shift_m.sum().item()
    if n_tok == 0:
        return 0.0, 0
    loss = F.cross_entropy(
        shift_l.view(-1, shift_l.size(-1)),
        shift_t.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / n_tok
    return math.exp(min(loss.item(), 20)), n_tok


def main():
    print("=" * 60)
    print("诊断 v2：side_signals 和 field_state 深入检查")
    print("=" * 60)

    cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid, cfg, debug=True)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 建立 side_channels
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")

    # 冻结
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        neuron.eval()
    for emb in shared_embeddings.values():
        emb.eval()

    # 检查 v1_compat
    print(f"\nneuron v1_compat: {neurons['zh_aug0'].v1_compat}")
    print(f"neuron excite_channels: {list(neurons['zh_aug0'].excite_channels.keys())}")
    print(f"field_dim: {cfg.field_dim}")

    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=40)

    BATCH_SIZE = 4
    batch_texts = texts[:BATCH_SIZE]
    neuron_embeddings = {}
    targets = None
    mask = None
    for nid, shared_emb in shared_embeddings.items():
        emb_out, tgt, msk = batch_align_and_embed(batch_texts, domain_sp, general_sp, shared_emb)
        neuron_embeddings[nid] = emb_out.to(DEVICE)
        if targets is None:
            targets = tgt.to(DEVICE)
            mask = msk.to(DEVICE)

    # 单神经元 round 1 vs round 2（带 field_state + side_signals）
    print("\n--- 单神经元 round 1 vs round 2 对比 ---")
    nid = "zh_aug2"
    neuron = neurons[nid]
    emb = neuron_embeddings[nid]

    with torch.no_grad():
        # Round 1: 无 field_state, 无 side_signals
        r1 = neuron.forward(emb, field_state=None, round_num=1, return_logits=True)
        ppl1, _ = compute_ppl(r1["logits"], targets, mask)
        print(f"  {nid} round 1: PPL={ppl1:.1f}, h_norm={r1['hidden_before_write'].norm():.4f}")

        # Round 2: 模拟 field_state + side_signals
        # 构造一个简单的 field_state
        field = ResonanceField(dim=cfg.field_dim)
        # 模拟 round 1 写入
        for nid_w in NEURON_IDS:
            r_w = neurons[nid_w].forward(neuron_embeddings[nid_w], field_state=None, round_num=1, return_logits=False)
            field.write(nid_w, r_w["field_vector"])

        field_state = field.get_normalised_state()
        print(f"  field_state norm: {field_state.norm():.4f}")
        print(f"  field_state sample: {field_state[:5]}")

        # 构造 side_signals
        side_signals = {}
        for pre_id in NEURON_IDS:
            if pre_id == nid:
                continue
            r_pre = neurons[pre_id].forward(neuron_embeddings[pre_id], field_state=None, round_num=1, return_logits=False)
            side_signals[pre_id] = r_pre["field_vector"]
            print(f"  side_signal {pre_id}: norm={r_pre['field_vector'].norm():.4f}")

        # 检查 side_channel forward
        for pre_id, sig in side_signals.items():
            if pre_id in neuron.excite_channels:
                proj = neuron.excite_channels[pre_id](sig)
                print(f"  side_channel {pre_id}->{nid}: proj_norm={proj.norm():.6f}")

        # Round 2: 带 field_state + side_signals
        r2 = neuron.forward(emb, field_state=field_state, round_num=2, return_logits=True, side_signals=side_signals)
        ppl2, _ = compute_ppl(r2["logits"], targets, mask)
        print(f"  {nid} round 2: PPL={ppl2:.1f}, h_norm={r2['hidden_before_write'].norm():.4f}")

        # 对比 logits 差异
        logits_diff = (r1["logits"] - r2["logits"]).abs().max()
        print(f"  logits max diff: {logits_diff:.6f}")

        # Round 2 只带 field_state（无 side_signals）
        r2f = neuron.forward(emb, field_state=field_state, round_num=2, return_logits=True)
        ppl2f, _ = compute_ppl(r2f["logits"], targets, mask)
        print(f"  {nid} round 2 (field only): PPL={ppl2f:.1f}")

        # Round 2 只带 side_signals（field_state=None）
        r2s = neuron.forward(emb, field_state=None, round_num=2, return_logits=True, side_signals=side_signals)
        ppl2s, _ = compute_ppl(r2s["logits"], targets, mask)
        print(f"  {nid} round 2 (side only): PPL={ppl2s:.1f}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
