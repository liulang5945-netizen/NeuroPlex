#!/usr/bin/env python3
"""C25-E 默认装配决策：多次采样统计确认（2026-08-11）。

单次 A/B（temperature 0.55）采样波动大（重复率方向在两次运行间反转：
0.012<0.027 vs 0.039>0.021）→ 本脚本每 prompt 多次采样取均值，
消除随机波动，为 executive vs continuous 默认装配决策提供稳定数据。

评估：非空率 / 平均长度 / 平均重复率 / 质量胜率（均值口径）。

运行：python -u scripts/training/verify_c25_e_ab_stats.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

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

# 混合域 prompt 集（覆盖 5 域；为控制运行时间收敛到 14 条）
PROMPTS = [
    ("zh", "你好"),
    ("zh", "你是谁？"),
    ("zh", "今天天气怎么样？"),
    ("zh", "1+1等于几？"),
    ("zh", "什么是幸福？"),
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("code", "写一个 Python 函数计算斐波那契数列"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("math", "Solve for x: 2x + 5 = 15"),
    ("en", "What is the capital of France?"),
    ("en", "Tell me about the solar system"),
    ("zh", "用中文解释什么是机器学习"),
]

SAMPLES = 3          # 每 prompt 采样次数
MAX_TOKENS = 32
TEMPERATURE = 0.55
TOP_K = 15
REPETITION_PENALTY = 1.4


def repeat_score(text: str) -> float:
    """4-gram 重复率（0=无重复，1=全重复）。"""
    if len(text) < 4:
        return 0.0
    grams = [text[i:i + 4] for i in range(len(text) - 3)]
    uniq = set(grams)
    return 1.0 - len(uniq) / max(len(grams), 1)


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("C25-E 多次采样统计 A/B（executive vs continuous，每 prompt ×%d）" % SAMPLES, flush=True)
    print("=" * 64, flush=True)

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

    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成\n", flush=True)

    print("[0] 判定（按索引存，同 tag 不覆盖）", flush=True)
    dom_map = {}
    for i, (tag, prompt) in enumerate(PROMPTS):
        d, _, _ = cortex._executive_route(prompt)
        dom_map[i] = d
    expect_ok = sum(1 for i, (tag, _) in enumerate(PROMPTS)
                    if (tag == "zh" and dom_map[i] == "zh")
                    or (tag in ("code", "math", "en") and dom_map[i] == tag))
    print(f"  判定正确 {expect_ok}/{len(PROMPTS)}", flush=True)
    check("判定正确率 ≥ 90%", expect_ok / len(PROMPTS) >= 0.9, f"→ {expect_ok}/{len(PROMPTS)}")

    print(f"\n[1] 多次采样生成（{SAMPLES} 次/ prompt）", flush=True)
    # stats[mode][metric] = list per-prompt 均值
    stats = {"executive": {"non_empty": [], "avg_len": [], "avg_rep": []},
             "continuous": {"non_empty": [], "avg_len": [], "avg_rep": []}}
    for mode in ("executive", "continuous"):
        for i, (tag, prompt) in enumerate(PROMPTS):
            texts = []
            for _ in range(SAMPLES):
                text = cortex.generate(
                    prompt=prompt, max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE, top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                    domain=dom_map[i],
                    collab_mode=mode,
                    fusion_mode="soft",
                )
                texts.append(text)
            stats[mode]["non_empty"].append(sum(1 for t in texts if t.strip()) / SAMPLES)
            stats[mode]["avg_len"].append(sum(len(t) for t in texts) / SAMPLES)
            stats[mode]["avg_rep"].append(sum(repeat_score(t) for t in texts) / SAMPLES)

    # 汇总（per-prompt 均值 → 整体均值）
    agg = {}
    for mode in ("executive", "continuous"):
        ne = sum(stats[mode]["non_empty"]) / len(PROMPTS)
        al = sum(stats[mode]["avg_len"]) / len(PROMPTS)
        ar = sum(stats[mode]["avg_rep"]) / len(PROMPTS)
        agg[mode] = {"non_empty": ne, "avg_len": al, "avg_rep": ar}
        print(f"\n  [{mode}]", flush=True)
        print(f"    非空率: {ne:.2f}", flush=True)
        print(f"    平均长度: {al:.1f}", flush=True)
        print(f"    平均重复率: {ar:.3f}", flush=True)

    # 逐 prompt 对比（均值口径）
    better = {"continuous": 0, "executive": 0, "tie": 0}
    for i, (tag, _) in enumerate(PROMPTS):
        se = stats["executive"]["avg_len"][i] * (1 - stats["executive"]["avg_rep"][i]) * stats["executive"]["non_empty"][i]
        sc = stats["continuous"]["avg_len"][i] * (1 - stats["continuous"]["avg_rep"][i]) * stats["continuous"]["non_empty"][i]
        if sc > se * 1.02:      # >2% 视为胜
            better["continuous"] += 1
        elif se > sc * 1.02:
            better["executive"] += 1
        else:
            better["tie"] += 1
    print(f"\n  逐 prompt 质量对比（长度×低重复×非空，均值口径，±2% 容差）: {better}", flush=True)

    # 判定 1：continuous 非空率不劣
    check("continuous 非空率 ≥ executive - 0.05",
          agg["continuous"]["non_empty"] >= agg["executive"]["non_empty"] - 0.05,
          f"{agg['continuous']['non_empty']:.2f} vs {agg['executive']['non_empty']:.2f}")
    # 判定 2：continuous 重复率不劣（均值口径）
    check("continuous 重复率 ≤ executive + 0.02",
          agg["continuous"]["avg_rep"] <= agg["executive"]["avg_rep"] + 0.02,
          f"{agg['continuous']['avg_rep']:.3f} vs {agg['executive']['avg_rep']:.3f}")
    # 判定 3：continuous ≥ executive 的 prompt ≥ 60%
    cont_ok = better["continuous"] + better["tie"]
    check("continuous 质量 ≥ executive 的 prompt ≥ 60%",
          cont_ok / len(PROMPTS) >= 0.6,
          f"{cont_ok}/{len(PROMPTS)}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 64, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
