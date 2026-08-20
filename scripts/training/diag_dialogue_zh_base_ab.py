#!/usr/bin/env python3
"""验证 ``zh`` 基础成员是否污染 dialogue continuous 路径。"""
from __future__ import annotations

import json
import logging
import os
import time

from neuroplex.loader import assemble_cortex
from diag_dialogue_capacity_ab import (
    DIALOGUE_IDS,
    QUESTIONS,
    _generate,
    _text_metrics,
)


FULL_9 = DIALOGUE_IDS + ["code", "en", "math", "zh"]
FULL_8_NO_ZH = DIALOGUE_IDS + ["code", "en", "math"]


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
            "temperature": 0.55,
            "top_k": 15,
            "max_generation_tokens": 8,
            "collab_mode": "continuous",
            "fusion_mode": "soft",
            "writes_checkpoint": False,
        },
        "modes": {},
    }
    for mode, active_nids in {
        "full_9": FULL_9,
        "full_8_no_zh": FULL_8_NO_ZH,
    }.items():
        rows = []
        for question in QUESTIONS:
            try:
                generated = _generate(cortex, active_nids, question)
                row = {"question": question, "text": generated, **_text_metrics(generated)}
            except Exception as exc:
                row = {
                    "question": question,
                    "text": "",
                    **_text_metrics(""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
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
    out_path = os.path.join("reports", "production_dialogue_zh_base_ab_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
