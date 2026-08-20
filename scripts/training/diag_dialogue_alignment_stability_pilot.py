#!/usr/bin/env python3
"""corrected alignment 的低学习率/梯度裁剪稳定性 pilot（仅内存）。"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from scripts.training.diag_dialogue_alignment_fix_pilot import (
    BATCH_SIZE,
    EVAL_CAP,
    GENERATION_QUESTIONS,
    MAX_SEQ_LEN,
    MAX_TEXTS,
    NEURON_ID,
    SEED,
    _batch_loss,
    _generate,
    _load_neuron,
    _metrics,
)
from scripts.training.utils import (
    SequentialSampler,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    make_wsd_scheduler,
    split_train_eval,
)


STEPS = 320
LR = 3e-5
WARMUP_STEPS = 20
MAX_GRAD_NORM = 1.0


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    full_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, eval_pool = split_train_eval(full_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    neuron, shared = _load_neuron()

    before = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    generation_before = {
        question: _generate(neuron, shared, domain_sp, general_sp, question)
        for question in GENERATION_QUESTIONS
    }

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
    sampler = SequentialSampler(full_texts, BATCH_SIZE, seed=SEED)
    losses = []
    grad_norms = []
    for step in range(STEPS):
        batch = sampler.sample_batch()
        optimizer.zero_grad()
        loss = _batch_loss(neuron, shared, batch, domain_sp, general_sp)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(neuron.parameters()) + list(shared.parameters()), MAX_GRAD_NORM
        )
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        grad_norms.append(float(grad_norm))
        if (step + 1) % 40 == 0:
            print(
                f"  [{NEURON_ID}] step {step + 1}/{STEPS} "
                f"loss={losses[-1]:.4f} grad_norm={grad_norms[-1]:.4f}",
                flush=True,
            )

    after = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    generation_after = {
        question: _generate(neuron, shared, domain_sp, general_sp, question)
        for question in GENERATION_QUESTIONS
    }
    report = {
        "contract": {
            "neuron_id": NEURON_ID,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_seq_len": MAX_SEQ_LEN,
            "train_samples": len(full_texts),
            "eval_samples": len(eval_texts),
            "alignment": "deduplicated first-occurrence domain target",
            "writes_checkpoint": False,
        },
        "before": before,
        "after": after,
        "generation_before": generation_before,
        "generation_after": generation_after,
        "loss_trace": {
            "first": round(losses[0], 6),
            "last": round(losses[-1], 6),
            "min": round(min(losses), 6),
            "max_grad_norm_observed": round(max(grad_norms), 6),
            "mean_grad_norm": round(sum(grad_norms) / len(grad_norms), 6),
        },
    }
    out_path = os.path.join("reports", "production_dialogue_alignment_stability_pilot_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    del neuron, shared, optimizer, scheduler
    gc.collect()


if __name__ == "__main__":
    main()
