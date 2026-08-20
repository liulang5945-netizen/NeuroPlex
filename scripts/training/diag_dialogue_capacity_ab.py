#!/usr/bin/env python3
"""固定口径的 dialogue 容量/群体对照。

比较同一生产装配、同一 prompt/seed/解码参数下的：
  - 4 个 compact dialogue 单体
  - 1 个 standard dialogue 单体
  - 5 个 dialogue 群体
  - 完整 9 成员群体

该诊断只做推理，不训练、不写 checkpoint，用于区分单体容量瓶颈与群体
融合/解码瓶颈。
"""
from __future__ import annotations

import json
import logging
import os
import time

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import build_dialogue_prompt


DIALOGUE_IDS = [
    "zh_aug0_dialogue",
    "zh_aug1_dialogue",
    "zh_aug2_dialogue",
    "zh_aug3_dialogue",
    "zh_std0_dialogue",
]
COMPACT_IDS = DIALOGUE_IDS[:4]
STANDARD_IDS = ["zh_std0_dialogue"]
FULL_IDS = DIALOGUE_IDS + ["code", "en", "math", "zh"]
QUESTIONS = [
    "你好，请介绍一下你自己",
    "你是谁？",
    "什么是神经网络？",
    "什么是注意力机制？",
]
SEED = 20260820


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def _model_params_m(cortex, nid: str) -> float:
    return round(sum(p.numel() for p in cortex.neurons[nid].parameters()) / 1e6, 6)


def _text_metrics(text: str) -> dict:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {
            "char_len": 0,
            "cjk_ratio": 0.0,
            "latin_ratio": 0.0,
            "repeat_bigram_rate": 0.0,
        }
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in chars)
    latin = sum(("a" <= c.lower() <= "z") for c in chars)
    bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
    repeated = len(bigrams) - len(set(bigrams))
    return {
        "char_len": len(chars),
        "cjk_ratio": round(cjk / len(chars), 4),
        "latin_ratio": round(latin / len(chars), 4),
        "repeat_bigram_rate": round(repeated / len(bigrams), 4) if bigrams else 0.0,
    }


def _generate(cortex, active_nids: list[str], question: str) -> str:
    _reset(cortex)
    torch.manual_seed(SEED)
    return cortex.generate(
        prompt=build_dialogue_prompt(question),
        max_tokens=8,
        temperature=0.55,
        top_k=15,
        domain="zh",
        repetition_penalty=1.4,
        active_nids=active_nids,
        collab_mode="continuous",
        fusion_mode="soft",
        auto_memory=False,
        instance_routing=False,
    )


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    keys = ["char_len", "cjk_ratio", "latin_ratio", "repeat_bigram_rate"]
    result = {"count": len(rows)}
    for key in keys:
        result[f"mean_{key}"] = round(sum(row[key] for row in rows) / len(rows), 4)
    result["empty_count"] = sum(row["char_len"] == 0 for row in rows)
    return result


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

    modes = {
        **{f"single_{nid}": [nid] for nid in DIALOGUE_IDS},
        "dialogue_5": DIALOGUE_IDS,
        "full_9": FULL_IDS,
    }
    report = {
        "contract": {
            "seed": SEED,
            "questions": QUESTIONS,
            "max_generation_tokens": 8,
            "temperature": 0.55,
            "top_k": 15,
            "repetition_penalty": 1.4,
            "collab_mode": "continuous",
            "fusion_mode": "soft",
            "writes_checkpoint": False,
        },
        "neurons": {
            nid: {
                "spec": getattr(cortex.neurons[nid].config, "spec", None),
                "hidden_size": getattr(cortex.neurons[nid].config, "hidden_size", None),
                "layers": getattr(cortex.neurons[nid].config, "num_hidden_layers", None),
                "local_params_m": _model_params_m(cortex, nid),
            }
            for nid in DIALOGUE_IDS
        },
        "modes": {},
    }

    for mode, active_nids in modes.items():
        rows = []
        for question in QUESTIONS:
            try:
                text = _generate(cortex, active_nids, question)
                error = None
            except Exception as exc:  # diagnostic must preserve per-case failures
                text = ""
                error = f"{type(exc).__name__}: {exc}"
            row = {"question": question, "text": text, **_text_metrics(text)}
            if error is not None:
                row["error"] = error
            rows.append(row)
        report["modes"][mode] = {
            "active_nids": active_nids,
            "active_params_m": round(
                sum(_model_params_m(cortex, nid) for nid in active_nids if nid in cortex.neurons),
                6,
            ),
            "summary": _summarize(rows),
            "rows": rows,
        }

    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_capacity_ab_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
