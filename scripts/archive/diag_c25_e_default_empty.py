#!/usr/bin/env python3
"""诊断：连续默认模式下 API 挂载对话空输出根因（2026-08-11）。

test_api_dialogue 用训练格式 prompt（"问：{q}\n答："）+ domain="zh"，
continuous 默认下 Q2-Q6 空输出。对比 executive/continuous 同 prompt 生成，
并检查 leader 选择与 weighted_logits 是否异常。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

QUESTIONS = ["你是谁？", "今天天气怎么样？", "1+1等于几？", "你好"]


def main():
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
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
        print(f"\n=== {q} ===", flush=True)
        for mode in ("executive", "continuous"):
            text = cortex.generate(
                prompt=prompt, max_tokens=30, temperature=0.55, top_k=15,
                domain="zh", repetition_penalty=1.4, fusion_mode="soft",
                collab_mode=mode,
            )
            print(f"  [{mode}] {text[:80]!r} len={len(text)}", flush=True)


if __name__ == "__main__":
    main()
