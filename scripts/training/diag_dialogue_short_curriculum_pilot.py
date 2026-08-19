#!/usr/bin/env python3
"""P1.2：短答案 curriculum 的 500 步 in-memory pilot。

对五个 dialogue checkpoint 逐个执行同一短答案训练配方：

* 数据：``max_answer_chars=120`` 的短答案子集；
* 优化：复用 dialogue fine-tune 的 answer mask、aligned target、AdamW 和 WSD；
* 验证：固定原始数据 hash holdout，使用 corrected aligned-answer PPL 和首 token rank；
* 保护：不写 checkpoint，不修改 data/neurons 下的任何文件。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_short_curriculum_pilot.py
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
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    make_wsd_scheduler,
    split_train_eval,
)


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
MAX_TEXTS = 100000
MAX_ANSWER_CHARS = 120
EVAL_CAP = 100
STEPS = 500
BATCH_SIZE = 8
LR = 1e-4
WARMUP_STEPS = 20
MAX_SEQ_LEN = 128


def _load_neuron(neuron_id: str):
    path = os.path.join("data", "neurons", f"neuron_{neuron_id}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"])
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    del ckpt
    return neuron, shared


def _batch_loss(neuron, shared, texts, domain_sp, general_sp):
    embeddings, targets, mask, sft_mask = batch_align_and_embed(
        texts,
        domain_sp,
        general_sp,
        shared,
        max_seq_len=MAX_SEQ_LEN,
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
    return loss


def _metrics(neuron, shared, texts, domain_sp, general_sp) -> dict:
    neuron.eval()
    shared.eval()
    total_ce = 0.0
    total_tokens = 0
    ranks = []
    top1 = 0
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            embeddings, targets, mask, sft_mask = batch_align_and_embed(
                batch,
                domain_sp,
                general_sp,
                shared,
                max_seq_len=MAX_SEQ_LEN,
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
            total_ce += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            total_tokens += int(valid_mask.sum())

            for row_index, text in enumerate(batch):
                marker = text.find(SFT_ANSWER_MARKER)
                prompt = text[:marker + len(SFT_ANSWER_MARKER)]
                _, aligned_targets = build_position_alignment(text, domain_sp, general_sp)
                answer_start = len(general_sp.encode(prompt))
                target_id = int(aligned_targets[answer_start])
                next_logits = logits[row_index, answer_start - 1]
                target_logit = next_logits[target_id]
                ranks.append(int((next_logits > target_logit).sum()))
                top1 += int(int(next_logits.argmax()) == target_id)
    ordered = sorted(ranks)
    return {
        "corrected_ppl": round(math.exp(min(total_ce / max(total_tokens, 1), 20)), 4),
        "aligned_answer_tokens": total_tokens,
        "mean_first_target_rank_zero_based": round(sum(ranks) / max(len(ranks), 1), 2),
        "median_first_target_rank_zero_based": ordered[len(ordered) // 2] if ordered else None,
        "first_target_top1_rate": round(top1 / max(len(ranks), 1), 4),
    }


def _train_one(neuron_id, train_texts, eval_texts, domain_sp, general_sp) -> dict:
    neuron, shared = _load_neuron(neuron_id)
    before = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    neuron.train()
    shared.train()
    optimizer = torch.optim.AdamW(
        list(neuron.parameters()) + list(shared.parameters()),
        lr=LR,
        weight_decay=0.1,
    )
    scheduler = make_wsd_scheduler(
        optimizer,
        num_steps=STEPS,
        warmup_steps=WARMUP_STEPS,
        decay_ratio=0.85,
    )
    losses = []
    for step in range(STEPS):
        batch_start = (step * BATCH_SIZE) % len(train_texts)
        batch = [train_texts[(batch_start + offset) % len(train_texts)] for offset in range(BATCH_SIZE)]
        optimizer.zero_grad()
        loss = _batch_loss(neuron, shared, batch, domain_sp, general_sp)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(f"  [{neuron_id}] step {step + 1}/{STEPS} loss={losses[-1]:.4f}", flush=True)
    after = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    result = {
        "before": before,
        "after": after,
        "loss_trace": {
            "first": round(losses[0], 6),
            "last": round(losses[-1], 6),
            "min": round(min(losses), 6),
        },
    }
    del neuron, shared, optimizer, scheduler
    gc.collect()
    return result


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(20260819)
    full_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, full_eval_pool = split_train_eval(full_texts, eval_ratio=0.05, seed=42)
    eval_texts = full_eval_pool[:EVAL_CAP]
    short_texts = load_dialogue_texts_multi(
        "data/simple_zh",
        max_texts=MAX_TEXTS,
        max_answer_chars=MAX_ANSWER_CHARS,
    )
    short_train, short_eval = split_train_eval(short_texts, eval_ratio=0.05, seed=42)
    del short_eval
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    report = {
        "contract": {
            "population": DIALOGUE_IDS,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_answer_chars": MAX_ANSWER_CHARS,
            "max_seq_len": MAX_SEQ_LEN,
            "short_train_samples": len(short_train),
            "full_eval_samples": len(eval_texts),
            "writes_checkpoint": False,
        },
        "neurons": {},
    }
    for index, neuron_id in enumerate(DIALOGUE_IDS, start=1):
        print(f"[{index}/{len(DIALOGUE_IDS)}] short curriculum pilot: {neuron_id}", flush=True)
        report["neurons"][neuron_id] = _train_one(
            neuron_id,
            short_train,
            eval_texts,
            domain_sp,
            general_sp,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
