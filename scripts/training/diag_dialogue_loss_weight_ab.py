#!/usr/bin/env python3
"""P1.2：短 A/B 验证首答案 token 加权是否改善泛化。

两个 in-memory 配方使用同一个 ``zh_aug0_dialogue`` 初始 checkpoint、8 条训练样本和 8 条
评估样本：

* baseline：所有有效 answer token 的 mean CE；
* first_token_weighted：``0.8 * token_mean + 0.2 * first_answer_token``。

两组各训练 32 步，不保存 checkpoint，用于决定是否值得把首 token 权重纳入正式训练入口。
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
from neuroplex.resonance.translator import batch_align_and_embed, build_position_alignment
from scripts.training.diag_dialogue_answer_data_quality import _data_rows
from scripts.training.diag_dialogue_micro_overfit import _load_neuron, _first_token_metrics
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


STEPS = 32
LR = 1e-4
FIRST_TOKEN_WEIGHT = 0.2
MAX_TEXTS = 100000
EVAL_CAP = 100
TRAIN_COUNT = 8
EVAL_COUNT = 8


def _batch_objective(neuron, shared, texts, domain_sp, general_sp, weighted: bool):
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
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-100,
    )

    first_logits = []
    first_targets = []
    for row_index, text in enumerate(texts):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker + len(SFT_ANSWER_MARKER)]
        _, aligned_targets = build_position_alignment(text, domain_sp, general_sp)
        answer_start = len(general_sp.encode(prompt))
        target_id = int(aligned_targets[answer_start])
        first_logits.append(logits[row_index, answer_start - 1])
        first_targets.append(target_id)
    first_token_loss = F.cross_entropy(
        torch.stack(first_logits),
        torch.tensor(first_targets, dtype=torch.long),
    )
    objective = (
        (1.0 - FIRST_TOKEN_WEIGHT) * token_loss + FIRST_TOKEN_WEIGHT * first_token_loss
        if weighted else token_loss
    )
    return objective, token_loss, first_token_loss, logits, valid_mask


def _evaluate(neuron, shared, texts, domain_sp, general_sp) -> dict:
    neuron.eval()
    shared.eval()
    with torch.no_grad():
        _, token_loss, first_loss, logits, valid_mask = _batch_objective(
            neuron, shared, texts, domain_sp, general_sp, weighted=False,
        )
        metrics = _first_token_metrics(logits, texts, domain_sp, general_sp)
    metrics.update({
        "answer_loss": round(float(token_loss), 6),
        "first_token_nll": round(float(first_loss), 6),
        "effective_answer_tokens": int(valid_mask.sum()),
    })
    return metrics


def _run(config_name, train_texts, eval_texts, domain_sp, general_sp, weighted):
    neuron, shared = _load_neuron()
    before = _evaluate(neuron, shared, eval_texts, domain_sp, general_sp)
    neuron.train()
    shared.train()
    optimizer = torch.optim.AdamW(
        list(neuron.parameters()) + list(shared.parameters()),
        lr=LR,
        weight_decay=0.0,
    )
    losses = []
    for _ in range(STEPS):
        optimizer.zero_grad()
        objective, _, _, _, _ = _batch_objective(
            neuron, shared, train_texts, domain_sp, general_sp, weighted=weighted,
        )
        objective.backward()
        optimizer.step()
        losses.append(float(objective.detach()))
    after = _evaluate(neuron, shared, eval_texts, domain_sp, general_sp)
    result = {
        "config": config_name,
        "before": before,
        "after": after,
        "objective_trace": {
            "first": round(losses[0], 6),
            "last": round(losses[-1], 6),
            "min": round(min(losses), 6),
        },
    }
    del neuron, shared, optimizer
    gc.collect()
    return result


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
    eligible = [
        row["sample_index"]
        for row in rows
        if row["category"] == "han" and not row["truncated"]
    ]
    if len(eligible) < TRAIN_COUNT + EVAL_COUNT:
        raise RuntimeError(f"only found {len(eligible)} eligible samples")
    train_texts = [eval_texts[index] for index in eligible[:TRAIN_COUNT]]
    holdout_texts = [eval_texts[index] for index in eligible[TRAIN_COUNT:TRAIN_COUNT + EVAL_COUNT]]

    report = {
        "contract": {
            "neuron_id": "zh_aug0_dialogue",
            "train_samples": len(train_texts),
            "eval_samples": len(holdout_texts),
            "steps": STEPS,
            "lr": LR,
            "first_token_weight": FIRST_TOKEN_WEIGHT,
            "writes_checkpoint": False,
        },
        "baseline": _run(
            "token_mean", train_texts, holdout_texts, domain_sp, general_sp, weighted=False,
        ),
        "first_token_weighted": _run(
            "0.8_token_mean_0.2_first_token",
            train_texts,
            holdout_texts,
            domain_sp,
            general_sp,
            weighted=True,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
