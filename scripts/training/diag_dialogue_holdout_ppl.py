#!/usr/bin/env python3
"""P1.2：用有效 aligned answer token 重算五个 dialogue checkpoint 的 holdout PPL。

该脚本只做 CPU 前向评估，不修改 checkpoint、不执行训练。它同时输出：

* corrected PPL：分母只包含有效 domain target；
* legacy PPL：复现旧评估分母，把未对齐 answer position 也计入；
* 与 checkpoint 内保存的 best_val_ppl 对照。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_holdout_ppl.py
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
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
CHECKPOINT_DIR = os.path.join("data", "neurons")
MAX_TEXTS = 100000
EVAL_CAP = 100
BATCH_SIZE = 4
DEVICE = "cpu"


def _load_checkpoint(neuron_id: str):
    path = os.path.join(CHECKPOINT_DIR, f"neuron_{neuron_id}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"]).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared_embedding = create_shared_embedding(DEVICE)
    shared_embedding.load_state_dict(ckpt["shared_embedding_state"])
    neuron.eval()
    shared_embedding.eval()
    metadata = {
        "data_source": ckpt.get("data_source"),
        "best_val_ppl": (ckpt.get("result") or {}).get("best_val_ppl"),
        "best_step": (ckpt.get("result") or {}).get("best_step"),
        "steps": (ckpt.get("result") or {}).get("steps"),
    }
    del ckpt
    return neuron, shared_embedding, metadata


def _evaluate(neuron, shared_embedding, texts, domain_sp, general_sp) -> dict:
    total_ce = 0.0
    corrected_tokens = 0
    legacy_tokens = 0
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            embeddings, targets, mask, sft_mask = batch_align_and_embed(
                batch,
                domain_sp,
                general_sp,
                shared_embedding,
                answer_marker=SFT_ANSWER_MARKER,
            )
            logits = neuron(embeddings.to(DEVICE), return_logits=True)["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].to(DEVICE).contiguous()
            shift_mask = mask[:, 1:].to(DEVICE).contiguous()
            shift_sft_mask = sft_mask[:, 1:].to(DEVICE).contiguous()
            valid_mask = effective_sft_mask(shift_targets, shift_mask, shift_sft_mask)
            target_ids = shift_targets.clone()
            target_ids[~valid_mask] = -100
            total_ce += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            corrected_tokens += int(valid_mask.sum())
            legacy_tokens += int((shift_mask & shift_sft_mask).sum())

    corrected_nll = total_ce / max(corrected_tokens, 1)
    legacy_nll = total_ce / max(legacy_tokens, 1)
    return {
        "samples": len(texts),
        "cross_entropy_sum": round(total_ce, 4),
        "corrected_aligned_answer_tokens": corrected_tokens,
        "legacy_answer_mask_tokens": legacy_tokens,
        "ignored_unaligned_tokens": legacy_tokens - corrected_tokens,
        "corrected_ppl": round(math.exp(min(corrected_nll, 20)), 4),
        "legacy_ppl": round(math.exp(min(legacy_nll, 20)), 4),
        "legacy_to_corrected_token_ratio": round(legacy_tokens / max(corrected_tokens, 1), 6),
    }


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    all_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, eval_pool = split_train_eval(all_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    report = {
        "contract": {
            "max_texts": MAX_TEXTS,
            "deduplicated_loaded_texts": len(all_texts),
            "eval_pool": len(eval_pool),
            "eval_used": len(eval_texts),
            "max_seq_len": 128,
            "answer_marker": SFT_ANSWER_MARKER,
            "effective_token_definition": "attention_mask & sft_mask & target >= 0",
        },
        "neurons": {},
    }
    for index, neuron_id in enumerate(DIALOGUE_IDS, start=1):
        print(f"[{index}/{len(DIALOGUE_IDS)}] evaluating {neuron_id}...", flush=True)
        neuron, shared_embedding, metadata = _load_checkpoint(neuron_id)
        report["neurons"][neuron_id] = {
            "checkpoint": metadata,
            "holdout": _evaluate(neuron, shared_embedding, eval_texts, domain_sp, general_sp),
        }
        del neuron, shared_embedding
        gc.collect()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
