#!/usr/bin/env python3
"""P1.2：固定生产 9 成员阵容的解码 Top-K 敏感性审计。

只改变 top-k（15/40/100/1），其余固定：temperature=0.55、repetition_penalty=1.4、
max_tokens=8、seed 和 soft fusion。每次生成前清空场与对话状态，不写任何运行状态。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_decode_topk.py
"""
from __future__ import annotations

import json
import logging
import time

import torch

from neuroplex.loader import assemble_cortex


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
QUESTIONS = ["你好", "你是谁？", "今天天气怎么样？", "帮我写一首关于春天的诗"]
TOP_K_VARIANTS = [15, 40, 100, 1]
SEED = 20260819


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    started = time.time()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    report = {
        "contract": {
            "population": list(cortex.neurons.keys()),
            "questions": QUESTIONS,
            "top_k": TOP_K_VARIANTS,
            "temperature": 0.55,
            "repetition_penalty": 1.4,
            "max_tokens": 8,
            "seed": SEED,
            "fusion_mode": "soft",
            "writes_runtime_state": False,
        },
        "outputs": {},
    }
    for question in QUESTIONS:
        prompt = f"问：{question}\n答："
        report["outputs"][question] = {}
        for top_k in TOP_K_VARIANTS:
            _reset(cortex)
            torch.manual_seed(SEED)
            try:
                output = cortex.generate(
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0.55,
                    top_k=top_k,
                    domain="zh",
                    repetition_penalty=1.4,
                    fusion_mode="soft",
                    auto_memory=False,
                    instance_routing=False,
                )
                error = None
            except Exception as exc:  # diagnostic should preserve failures per variant
                output = None
                error = repr(exc)
            report["outputs"][question][f"top_k_{top_k}"] = {
                "text": output,
                "error": error,
            }
            print(f"{question} | top_k={top_k} | {output or error}", flush=True)

    report["elapsed_s"] = round(time.time() - started, 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
