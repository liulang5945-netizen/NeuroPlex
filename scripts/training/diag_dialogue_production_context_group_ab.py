#!/usr/bin/env python3
"""修复后的 Cortex 入口：5-dialogue 与完整 9 成员生成 A/B。"""
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
GENERAL_IDS = ["code", "en", "math", "zh"]
GROUPS = {
    "dialogue5": DIALOGUE_IDS,
    "full9": DIALOGUE_IDS + GENERAL_IDS,
    "full9_no_aug2": [nid for nid in DIALOGUE_IDS if nid != "zh_aug2_dialogue"] + GENERAL_IDS,
}
QUESTIONS = [
    "你好，请介绍一下你自己",
    "你是谁？",
    "什么是神经网络？",
    "什么是注意力机制？",
]


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def main() -> None:
    logging.disable(logging.CRITICAL)
    started = time.time()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS + GENERAL_IDS,
    )
    report = {
        "contract": {
            "groups": GROUPS,
            "context_update": "full_text_reencode_after_domain_piece",
            "seed": 20260820,
            "temperature": 0.55,
            "top_k": 15,
            "max_tokens": 8,
            "production_checkpoints_written": False,
        },
        "groups": {},
    }
    for group_name, active_nids in GROUPS.items():
        generations = {}
        for question in QUESTIONS:
            _reset(cortex)
            torch.manual_seed(20260820)
            generations[question] = cortex.generate(
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
        report["groups"][group_name] = {
            "active_nids": active_nids,
            "generations": generations,
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join(
        "reports", "production_dialogue_context_group_ab_20260820.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
