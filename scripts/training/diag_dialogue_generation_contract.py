#!/usr/bin/env python3
"""P1.2：核对 dialogue 训练样本与 P7 首 token 生成契约。

检查内容：
1. `答：` answer mask 的起点是否与 general token 前缀一致；
2. 完整 teacher-forcing 样本在 answer 起点前的 logits 是否等于生成 prompt logits；
3. 首个答案 token 的目标 id、Top-1 和域 token→general 回填是否可用。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_generation_contract.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import build_position_alignment
from scripts.training.utils import load_dialogue_texts_multi, split_train_eval


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]


def _heldout_samples(max_texts: int = 100000, limit: int = 32) -> list[str]:
    """Use the same deterministic hash split as dialogue fine-tuning.

    The checkpoint training entry defaults to 100K loaded examples followed by
    the shared 5% hash holdout.  Selecting from that holdout keeps this report
    comparable across the five dialogue checkpoints and avoids evaluating only
    the first source file.
    """
    texts = load_dialogue_texts_multi("data/simple_zh", max_texts=max_texts)
    _, eval_texts = split_train_eval(texts, eval_ratio=0.05, seed=42)
    eval_texts = [text for text in eval_texts if SFT_ANSWER_MARKER in text]
    return sorted(
        eval_texts,
        key=lambda text: hashlib.md5(text.encode("utf-8")).hexdigest(),
    )[:limit]


def main() -> None:
    logging.disable(logging.CRITICAL)
    texts = _heldout_samples()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    general = cortex._general_sp
    zh = cortex._tokenizer_hub.get_tokenizer("zh")
    report = {
        "sample_count": len(texts),
        "samples": [],
        "neurons": {},
    }

    for sample_index, text in enumerate(texts):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker + len(SFT_ANSWER_MARKER)]
        full_general, targets = build_position_alignment(text, zh, general)
        prompt_general = general.encode(prompt)
        prefix_matches = list(full_general[:len(prompt_general)]) == list(prompt_general)
        answer_start = len(prompt_general)
        target_id = int(targets[answer_start]) if answer_start < len(targets) else -100
        backfill = general.encode(zh.decode([target_id])) if target_id >= 0 else []
        report["samples"].append({
            "sample_index": sample_index,
            "prompt_preview": prompt[:120],
            "full_general_len": len(full_general),
            "prompt_general_len": len(prompt_general),
            "answer_start": answer_start,
            "prefix_matches": prefix_matches,
            "first_target_id": target_id,
            "first_target_piece": zh.id_to_piece(target_id) if target_id >= 0 else None,
            "first_target_backfill_general_ids": backfill,
            "backfill_nonempty": bool(backfill),
        })

        for nid in DIALOGUE_IDS:
            emb = cortex._neuron_shared_embeddings[nid]
            prompt_ids = torch.tensor([list(prompt_general)], dtype=torch.long)
            prompt_logits = cortex.neurons[nid](emb(prompt_ids), return_logits=True)["logits"].detach()
            prompt_next = prompt_logits[:, -1, :]
            target_rank = (
                int((prompt_next[0] > prompt_next[0, target_id]).sum())
                if target_id >= 0 else None
            )
            neuron_report = report["neurons"].setdefault(nid, {
                "target_ranks_zero_based": [],
                "top1_hits": 0,
                "backfill_nonempty": 0,
                "prefix_parity": [],
            })
            if target_rank is not None:
                neuron_report["target_ranks_zero_based"].append(target_rank)
            if int(prompt_next.argmax()) == target_id:
                neuron_report["top1_hits"] += 1
            if backfill:
                neuron_report["backfill_nonempty"] += 1

            # Full teacher-forcing parity is expensive; verify it on a fixed
            # prefix subset while ranking the complete held-out sample set.
            if sample_index < 8:
                full_ids = torch.tensor([list(full_general)], dtype=torch.long)
                full_logits = cortex.neurons[nid](emb(full_ids), return_logits=True)["logits"].detach()
                full_next = full_logits[:, answer_start - 1, :]
                diff = (full_next - prompt_next).abs()
                cosine = float(torch.nn.functional.cosine_similarity(full_next, prompt_next).item())
                neuron_report["prefix_parity"].append({
                    "sample_index": sample_index,
                    "max_abs_diff": round(float(diff.max()), 6),
                    "cosine": round(max(-1.0, min(1.0, cosine)), 8),
                })

    for neuron_report in report["neurons"].values():
        ranks = neuron_report.pop("target_ranks_zero_based")
        ranks_sorted = sorted(ranks)
        neuron_report["mean_target_rank_zero_based"] = round(sum(ranks) / max(len(ranks), 1), 2)
        neuron_report["median_target_rank_zero_based"] = ranks_sorted[len(ranks_sorted) // 2] if ranks_sorted else None
        neuron_report["top1_rate"] = round(neuron_report["top1_hits"] / max(len(ranks), 1), 4)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
