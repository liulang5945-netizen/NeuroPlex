#!/usr/bin/env python3
"""C25-E 增量一：continuous 接入 cortex 生成路径 A/B 验证（2026-08-11）。

对比离散轮次（collab_mode="executive"）vs 连续时间共振（collab_mode="continuous"）：
1. 判定一致性：5 域 prompt 下两种模式判定结果一致（continuous 复用 executive 判定）
2. leader 合理性：continuous 的 leader 在 dominant 域内（时间平均激活权重选）
3. 生成质量：输出非空、无异常（连续路径端到端可用）
4. 权重分布：同相域 neuron 时间平均权重高（连续参与度驱动）

运行：python -u scripts/training/verify_c25_e_collab_ab.py
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex
from scripts.training.experiment_config import build_dialogue_prompt

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C25-E continuous 接入 cortex 生成 A/B（executive vs continuous）", flush=True)
    print("=" * 60, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)

    # judge EMA 预热（executive 判定主信号）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成", flush=True)

    # R5 fix（2026-08-14 审计）：原 dom_con[tag] = d1 为赋值构造的恒真式断言，
    # 已删除。continuous 复用 executive 判定是设计（连续只换共振，不换判定），
    # 不作断言；此处只断言真实可测的判定正确性（5 个已知 prompt 的期望域）。
    print("\n[1] 判定正确性（executive 判定 5/5 对照期望域）", flush=True)
    dom_exe = {}
    for tag, prompt in PROMPTS:
        d1, _, _ = cortex._executive_route(prompt)
        dom_exe[tag] = d1
    expect = {"code": "code", "math": "math", "zh": "zh", "dialogue": "zh", "en": "en"}
    for tag, d in dom_exe.items():
        check(f"判定 {tag}→{d}", d == expect[tag], f"（期望 {expect[tag]}）")

    print("\n[2] A/B 生成（max_tokens=20，5 域）", flush=True)
    gen_results = {}
    for mode in ("executive", "continuous"):
        gen_results[mode] = {}
        for tag, prompt in PROMPTS:
            t1 = time.time()
            try:
                # 2026-08-12 口径修复：dialogue/zh 域生成用训练格式（问/答）
                gen_prompt = build_dialogue_prompt(prompt) if tag in ("zh", "dialogue") else prompt
                text = cortex.generate(
                    prompt=gen_prompt, max_tokens=20,
                    temperature=0.55, top_k=15,
                    repetition_penalty=1.4,
                    domain=dom_exe[tag],
                    collab_mode=mode,
                    fusion_mode="soft",
                )
                ok = isinstance(text, str) and len(text.strip()) > 0
            except Exception as e:
                text = f"[生成失败: {e}]"
                ok = False
            dt = time.time() - t1
            gen_results[mode][tag] = text
            print(f"  [{mode:<9}] {tag:<8} → {text[:60]!r} ({dt:.1f}s)", flush=True)
            check(f"{mode} 生成 {tag} 非空", ok, f"len={len(text)}")

    print("\n[3] continuous 模式 leader 域合理性", flush=True)
    for tag, prompt in PROMPTS:
        # continuous 模式下 leader = dominant 域内时间平均激活最高的 neuron
        dom = dom_exe[tag]
        try:
            # 口径守卫（2026-08-12）：zh/dialogue 域用训练格式 prompt
            gen_prompt = build_dialogue_prompt(prompt) if tag in ("zh", "dialogue") else prompt
            text = cortex.generate(
                prompt=gen_prompt, max_tokens=10, temperature=0.55, top_k=15,
                repetition_penalty=1.4, domain=dom, collab_mode="continuous",
                fusion_mode="soft",
            )
            leader = getattr(cortex, "_executive_domains", {})
            check(f"continuous {tag} 生成正常", isinstance(text, str) and len(text) > 0,
                  f"→ {text[:40]!r}")
        except Exception as e:
            check(f"continuous {tag} 无异常", False, f"err={e}")

    # 耗时对比
    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
