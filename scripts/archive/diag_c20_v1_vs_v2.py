#!/usr/bin/env python3
"""对比 v1/v2 collab 装配的挂载生成质量（2026-08-11）。

诊断：C20 重训 v2 更新了 quality_head + phasor（phasors max|d|=1.82）。
quality_head 不参与生成；phasor 经 C23-B 场写入 binding 本体化影响生成路径。
用同一套 9 神经元 + 不同 collab 装配，跑 Q1-Q4 对比生成质量。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

QUESTIONS = [
    "你好",
    "你是谁？",
    "1+1等于几？",
    "帮我写一首关于春天的诗",
]
MAX_TOKENS = 30
TEMPERATURE = 0.55
TOP_K = 15
REPETITION_PENALTY = 1.4


def run_collab(collab_name: str):
    print(f"\n{'=' * 60}", flush=True)
    print(f"collab = {collab_name}", flush=True)
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=collab_name,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    for q in QUESTIONS:
        prompt = f"问：{q}\n答："
        text = cortex.generate(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            domain="zh",
            repetition_penalty=REPETITION_PENALTY,
            fusion_mode="soft",
        )
        print(f"  Q: {q}\n  A: {text}", flush=True)


if __name__ == "__main__":
    run_collab("collab_v3_c24v2.ckpt.pt")   # v2（今天 12:37 重训）
    run_collab("collab_v3_c24v2_v1.ckpt.pt")  # v1（8/10 备份）
