#!/usr/bin/env python3
"""C25-F 端到端阶段 2 内容断言失败诊断（2026-08-11）。

隔离变量，定位阶段 2（强制 code 域）为何输出英文自然语言：
A. stage2 prompt + domain=code + continuous  （复现失败场景）
B. stage2 prompt + domain=code + executive   （对照：域约束 vs 共振模式）
C. stage2 prompt + domain=None + continuous  （自动判定）
D. 纯英文 code prompt + domain=code + continuous（C25-E A/B 基线对照）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

PREV = "假设：斐波那契数列为 为偶数 ，其 C 的 的 的价格 价格会计会计师"

STAGE2_TPL = "根据需求：{prev}\n写出满足需求的完整 Python 函数代码。"
STAGE2_PROMPT = STAGE2_TPL.format(prev=PREV)
EN_PROMPT = "Write a Python function to compute the Fibonacci sequence"


def run(cortex, tag, prompt, domain, mode):
    text = cortex.generate(
        prompt=prompt,
        max_tokens=48,
        temperature=0.55,
        top_k=15,
        domain=domain,
        repetition_penalty=1.4,
        collab_mode=mode,
        fusion_mode="soft",
    )
    has_code = any(k in text for k in ("def", "return", "import", "lambda"))
    print(f"[{tag}] domain={domain} mode={mode}\n  → {text[:100]!r}\n  has_code={has_code}", flush=True)


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
    print("  judge EMA 预热完成\n", flush=True)

    # 预检：无 {prev} 时 code 域对中文指令的反应
    run(cortex, "A0", "写出满足需求的完整 Python 函数代码", "code", "continuous")
    run(cortex, "A", STAGE2_PROMPT, "code", "continuous")
    run(cortex, "B", STAGE2_PROMPT, "code", "executive")
    run(cortex, "C", STAGE2_PROMPT, None, "continuous")
    run(cortex, "D", EN_PROMPT, "code", "continuous")


if __name__ == "__main__":
    main()
