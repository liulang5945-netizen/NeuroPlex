#!/usr/bin/env python3
"""验证 zh_aug2_dialogue 是否直接引入技术回答碎片。"""
from __future__ import annotations

import json
import logging
import os
import time

from neuroplex.loader import assemble_cortex
from diag_dialogue_capacity_ab import DIALOGUE_IDS, QUESTIONS, _generate, _text_metrics


DIALOGUE_5 = DIALOGUE_IDS
DIALOGUE_4_NO_AUG2 = [nid for nid in DIALOGUE_IDS if nid != "zh_aug2_dialogue"]
FULL_8_NO_ZH = DIALOGUE_IDS + ["code", "en", "math"]
FULL_7_NO_ZH_NO_AUG2 = DIALOGUE_4_NO_AUG2 + ["code", "en", "math"]


def main() -> None:
    logging.disable(logging.CRITICAL)
    started = time.time()
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
            "seed": 20260820,
            "questions": QUESTIONS,
            "max_generation_tokens": 8,
            "temperature": 0.55,
            "top_k": 15,
            "repetition_penalty": 1.4,
            "collab_mode": "continuous",
            "fusion_mode": "soft",
            "writes_checkpoint": False,
        },
        "modes": {},
    }
    modes = {
        "dialogue_5": DIALOGUE_5,
        "dialogue_4_no_aug2": DIALOGUE_4_NO_AUG2,
        "full_8_no_zh": FULL_8_NO_ZH,
        "full_7_no_zh_no_aug2": FULL_7_NO_ZH_NO_AUG2,
    }
    for mode, active_nids in modes.items():
        rows = []
        for question in QUESTIONS:
            try:
                generated = _generate(cortex, active_nids, question)
                rows.append({
                    "question": question,
                    "text": generated,
                    **_text_metrics(generated),
                })
            except Exception as exc:
                rows.append({
                    "question": question,
                    "text": "",
                    **_text_metrics(""),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["modes"][mode] = {
            "active_nids": active_nids,
            "rows": rows,
            "summary": {
                "count": len(rows),
                "mean_char_len": round(sum(row["char_len"] for row in rows) / len(rows), 4),
                "mean_cjk_ratio": round(sum(row["cjk_ratio"] for row in rows) / len(rows), 4),
                "mean_repeat_bigram_rate": round(
                    sum(row["repeat_bigram_rate"] for row in rows) / len(rows), 4
                ),
            },
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_aug2_ab_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
