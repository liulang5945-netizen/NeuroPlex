#!/usr/bin/env python3
"""P1.2：不落盘验证 dialogue 目标链能否快速过拟合首答案 token。

固定一个 compact dialogue checkpoint 和 8 条未截断、汉字首答案样本，使用生产训练入口
的 ``batch_align_and_embed`` 与有效 aligned answer mask 训练 32 步。只在内存中修改，不写
checkpoint；比较训练前后的 answer loss、首 token NLL、rank 和 Top-1。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_micro_overfit.py
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceNeuron
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import batch_align_and_embed, build_position_alignment
from scripts.training.diag_dialogue_answer_data_quality import _data_rows
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


NEURON_ID = "zh_aug0_dialogue"
MAX_TEXTS = 100000
EVAL_CAP = 100
SAMPLE_COUNT = 8
STEPS = 32
LR = 1e-4


def _load_neuron():
    path = os.path.join("data", "neurons", f"neuron_{NEURON_ID}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"])
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    del ckpt
    return neuron, shared


def _batch_forward(neuron, shared, texts, domain_sp, general_sp):
    embeddings, targets, mask, sft_mask = batch_align_and_embed(
        texts,
        domain_sp,
        general_sp,
        shared,
        answer_marker=SFT_ANSWER_MARKER,
    )
    logits = neuron(embeddings, return_logits=True)["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    shift_sft_mask = sft_mask[:, 1:].contiguous()
    valid_mask = effective_sft_mask(shift_targets, shift_mask, shift_sft_mask)
    target_ids = shift_targets.clone()
    target_ids[~valid_mask] = -100
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-100,
    )
    return loss, logits, valid_mask


def _first_token_metrics(logits, texts, domain_sp, general_sp) -> dict:
    ranks = []
    nlls = []
    top1 = 0
    for row_index, text in enumerate(texts):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker + len(SFT_ANSWER_MARKER)]
        general_ids, targets = build_position_alignment(text, domain_sp, general_sp)
        answer_start = len(general_sp.encode(prompt))
        target_id = int(targets[answer_start])
        next_logits = logits[row_index, answer_start - 1]
        target_logit = next_logits[target_id]
        ranks.append(int((next_logits > target_logit).sum()))
        nlls.append(float(-next_logits[target_id] + torch.logsumexp(next_logits, dim=-1)))
        top1 += int(int(next_logits.argmax()) == target_id)
    ordered = sorted(ranks)
    return {
        "first_token_nll": round(sum(nlls) / max(len(nlls), 1), 6),
        "mean_rank_zero_based": round(sum(ranks) / max(len(ranks), 1), 2),
        "median_rank_zero_based": ordered[len(ordered) // 2] if ordered else None,
        "top1_rate": round(top1 / max(len(ranks), 1), 4),
    }


def _measure(neuron, shared, texts, domain_sp, general_sp) -> dict:
    neuron.eval()
    shared.eval()
    with torch.no_grad():
        loss, logits, valid_mask = _batch_forward(neuron, shared, texts, domain_sp, general_sp)
        metrics = _first_token_metrics(logits, texts, domain_sp, general_sp)
    metrics["answer_loss"] = round(float(loss), 6)
    metrics["effective_answer_tokens"] = int(valid_mask.sum())
    return metrics


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(20260819)
    all_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, eval_pool = split_train_eval(all_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    rows = _data_rows(eval_texts, domain_sp, general_sp, {})
    selected = [
        (row["sample_index"], eval_texts[row["sample_index"]])
        for row in rows
        if row["category"] == "han" and not row["truncated"]
    ][:SAMPLE_COUNT]
    if len(selected) < SAMPLE_COUNT:
        raise RuntimeError(f"only found {len(selected)} eligible micro-overfit samples")
    texts = [text for _, text in selected]

    neuron, shared = _load_neuron()
    report = {
        "contract": {
            "neuron_id": NEURON_ID,
            "sample_count": len(texts),
            "steps": STEPS,
            "lr": LR,
            "writes_checkpoint": False,
        },
        "samples": [
            {
                "eval_index": index,
                "question": text[:text.find(SFT_ANSWER_MARKER)],
                "answer_preview": text[text.find(SFT_ANSWER_MARKER) + len(SFT_ANSWER_MARKER):][:60],
            }
            for index, text in selected
        ],
    }
    report["before"] = _measure(neuron, shared, texts, domain_sp, general_sp)

    neuron.train()
    shared.train()
    optimizer = torch.optim.AdamW(
        list(neuron.parameters()) + list(shared.parameters()),
        lr=LR,
        weight_decay=0.0,
    )
    losses = []
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        loss, _, _ = _batch_forward(neuron, shared, texts, domain_sp, general_sp)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if step % 8 == 0:
            print(f"step {step}/{STEPS}: answer_loss={losses[-1]:.6f}", flush=True)

    report["after"] = _measure(neuron, shared, texts, domain_sp, general_sp)
    report["loss_trace"] = {
        "first": round(losses[0], 6),
        "last": round(losses[-1], 6),
        "min": round(min(losses), 6),
    }
    del neuron, shared, optimizer
    gc.collect()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
