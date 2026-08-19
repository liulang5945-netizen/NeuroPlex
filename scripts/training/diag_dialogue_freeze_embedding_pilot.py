#!/usr/bin/env python3
"""P1.2：冻结 shared embedding 的 200 步 in-memory pilot。

只测试 ``zh_aug2_dialogue``，使用原始混合 dialogue 数据和 corrected holdout evaluator；
与当前 fine-tune 入口唯一差异是 shared embedding 不参与优化。不会写入 checkpoint。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.diag_dialogue_short_curriculum_pilot import (
    _load_neuron,
    _metrics,
)
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    make_wsd_scheduler,
    split_train_eval,
)


NEURON_ID = "zh_aug2_dialogue"
MAX_TEXTS = 100000
EVAL_CAP = 100
STEPS = 200
BATCH_SIZE = 8
LR = 1e-4
WARMUP_STEPS = 20


def _loss(neuron, shared, texts, domain_sp, general_sp):
    embeddings, targets, mask, sft_mask = batch_align_and_embed(
        texts,
        domain_sp,
        general_sp,
        shared,
        max_seq_len=128,
        answer_marker=SFT_ANSWER_MARKER,
    )
    logits = neuron(embeddings, return_logits=True)["logits"]
    shift_logits = logits[:, :-1].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    valid_mask = effective_sft_mask(
        shift_targets,
        mask[:, 1:].contiguous(),
        sft_mask[:, 1:].contiguous(),
    )
    target_ids = shift_targets.clone()
    target_ids[~valid_mask] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-100,
    )


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(20260819)
    full_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    train_texts, eval_pool = split_train_eval(full_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    neuron, shared = _load_neuron(NEURON_ID)
    before = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)

    for parameter in shared.parameters():
        parameter.requires_grad = False
    neuron.train()
    shared.eval()
    optimizer = torch.optim.AdamW(neuron.parameters(), lr=LR, weight_decay=0.1)
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
        loss = _loss(neuron, shared, batch, domain_sp, general_sp)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{STEPS} loss={losses[-1]:.4f}", flush=True)

    after = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    report = {
        "contract": {
            "neuron_id": NEURON_ID,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "shared_embedding_frozen": True,
            "train_samples": len(train_texts),
            "eval_samples": len(eval_texts),
            "writes_checkpoint": False,
        },
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
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
