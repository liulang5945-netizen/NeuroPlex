#!/usr/bin/env python3
"""修复版 full9 与 full9_no_aug2 的 32 条 held-out 生成统计。"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from scripts.training.diag_dialogue_capacity_ab import _generate, _summarize, _text_metrics
from scripts.training.diag_dialogue_generation_contract import _heldout_samples


DIALOGUE_IDS = [
    "zh_aug0_dialogue",
    "zh_aug1_dialogue",
    "zh_aug2_dialogue",
    "zh_aug3_dialogue",
    "zh_std0_dialogue",
]
GENERAL_IDS = ["code", "en", "math", "zh"]
FULL9 = DIALOGUE_IDS + GENERAL_IDS
FULL9_NO_AUG2 = [nid for nid in FULL9 if nid != "zh_aug2_dialogue"]


def _questions(limit: int = 32) -> list[str]:
    questions = []
    for text in _heldout_samples(limit=limit):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker]
        if prompt.startswith("问："):
            prompt = prompt[len("问：") :]
        questions.append(prompt.rstrip("\n"))
    return questions


def main() -> None:
    logging.disable(logging.CRITICAL)
    started = time.time()
    questions = _questions()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=False,
        neuron_ids=DIALOGUE_IDS,
    )
    report = {
        "contract": {
            "question_count": len(questions),
            "question_source": "deterministic 5% held-out dialogue split",
            "context_update": "full_text_reencode_after_domain_piece",
            "seed": 20260820,
            "temperature": 0.55,
            "top_k": 15,
            "max_tokens": 8,
            "production_checkpoints_written": False,
        },
        "modes": {},
    }
    for mode, active_nids in {
        "full9": FULL9,
        "full9_no_aug2": FULL9_NO_AUG2,
    }.items():
        rows = []
        for question in questions:
            try:
                text = _generate(cortex, active_nids, question)
                error = None
            except Exception as exc:
                text = ""
                error = f"{type(exc).__name__}: {exc}"
            row = {"question": question, "text": text, **_text_metrics(text)}
            if error is not None:
                row["error"] = error
            rows.append(row)
        report["modes"][mode] = {
            "active_nids": active_nids,
            "summary": _summarize(rows),
            "rows": rows,
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join(
        "reports", "production_dialogue_context_group_eval_20260820.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
