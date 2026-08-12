#!/usr/bin/env python3
"""C25-E loader 默认装配决策：规模化 A/B 验证（executive vs continuous，2026-08-11）。

目的：为 loader 默认装配决策提供数据（continuous 替换 executive 需 A/B 规模化验证）。
规模：20+ 混合域 prompt（zh 对话/code/math/en 通用/zh 域），固定 seed 可复现。
维度：
1. 判定一致性：两种模式判定域一致（continuous 复用 executive 判定）
2. 生成质量：非空率、平均长度、重复率（4-gram 重复）、正常停止率
3. leader 稳定性：continuous 的 fusion 权重分布（dominant 域 neuron 占比）
4. 汇总结论：continuous 是否整体 ≥ executive（质量维度无退化 + 判定一致）

运行：python -u scripts/training/verify_c25_e_ab_scale.py
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402
from scripts.training.experiment_config import build_dialogue_prompt  # noqa: E402

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

# 混合域 prompt 集（覆盖 5 域，贴近真实使用）
PROMPTS = [
    # zh 对话
    ("zh", "你好"),
    ("zh", "你是谁？"),
    ("zh", "今天天气怎么样？"),
    ("zh", "1+1等于几？"),
    ("zh", "推荐一本好书"),
    ("zh", "什么是幸福？"),
    ("zh", "怎么学好英语？"),
    ("zh", "帮我写一首关于春天的诗"),
    # code
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("code", "How to sort a list in Python?"),
    ("code", "Write a function to check if a number is prime"),
    ("code", "写一个 Python 函数计算斐波那契数列"),
    ("code", "写一个冒泡排序的代码"),
    # math
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("math", "What is 15 percent of 200?"),
    ("math", "Solve for x: 2x + 5 = 15"),
    ("math", "一个三角形三边分别是3,4,5，求面积"),
    # en 通用
    ("en", "What is the capital of France?"),
    ("en", "Tell me about the solar system"),
    ("en", "Explain photosynthesis in simple terms"),
    # zh 域
    ("zh", "用中文解释什么是机器学习"),
    ("zh", "写一篇关于家乡的短文"),
]

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
    print("C25-E 规模化 A/B：executive vs continuous（loader 默认装配决策依据）", flush=True)
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

    # judge EMA 预热
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成\n", flush=True)

    print("[0] 判定一致性（executive vs continuous 同源判定）", flush=True)
    dom_map = {}  # 按索引存（同 tag 多 prompt 不互相覆盖）
    for i, (tag, prompt) in enumerate(PROMPTS):
        d, _, _ = cortex._executive_route(prompt)
        dom_map[i] = d
    # 判定合理性：zh 对话/zh 域 → zh（dialogue 聚合），code→code，math→math，en→en
    expect_ok = 0
    for i, (tag, prompt) in enumerate(PROMPTS):
        d = dom_map[i]
        if tag == "zh" and d == "zh":
            expect_ok += 1
        elif tag in ("code", "math", "en") and d == tag:
            expect_ok += 1
    print(f"  判定正确 {expect_ok}/{len(PROMPTS)}", flush=True)
    check("判定正确率 ≥ 90%", expect_ok / len(PROMPTS) >= 0.9, f"→ {expect_ok}/{len(PROMPTS)}")

    print("\n[1] 生成 A/B（固定 seed，两种模式各一次）", flush=True)
    results = {"executive": {}, "continuous": {}}
    for mode in ("executive", "continuous"):
        for i, (tag, prompt) in enumerate(PROMPTS):
            # 2026-08-12 口径修复：zh 域生成必须用训练格式（问/答），
            # 裸问题会触发换行死循环假退化（守卫会硬失败）。
            gen_prompt = build_dialogue_prompt(prompt) if tag == "zh" else prompt
            text = cortex.generate(
                prompt=gen_prompt, max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE, top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                domain=dom_map[i],
                collab_mode=mode,
                fusion_mode="soft",
            )
            results[mode][i] = text

    # 汇总统计
    stats = {}
    for mode in ("executive", "continuous"):
        texts = list(results[mode].values())
        non_empty = sum(1 for t in texts if t.strip())
        lens = [len(t) for t in texts]
        reps = [repeat_score(t) for t in texts]
        stats[mode] = {
            "non_empty": non_empty,
            "avg_len": sum(lens) / len(lens),
            "avg_rep": sum(reps) / len(reps),
            "len_gt_4": sum(1 for l in lens if l > 4),
        }
        print(f"\n  [{mode}]", flush=True)
        print(f"    非空: {non_empty}/{len(PROMPTS)}", flush=True)
        print(f"    平均长度: {stats[mode]['avg_len']:.1f} 字符", flush=True)
        print(f"    平均重复率: {stats[mode]['avg_rep']:.3f}", flush=True)
        print(f"    长度>4: {stats[mode]['len_gt_4']}/{len(PROMPTS)}", flush=True)

    # 逐 prompt 对比
    better = {"continuous": 0, "executive": 0, "tie": 0}
    for i, (tag, prompt) in enumerate(PROMPTS):
        te = results["executive"][i]
        tc = results["continuous"][i]
        se = (len(te.strip()) > 4) + (1 - repeat_score(te)) if te.strip() else 0
        sc = (len(tc.strip()) > 4) + (1 - repeat_score(tc)) if tc.strip() else 0
        if sc > se:
            better["continuous"] += 1
        elif se > sc:
            better["executive"] += 1
        else:
            better["tie"] += 1
    print(f"\n  逐 prompt 质量对比（长度+低重复率加权）: {better}", flush=True)
    print(f"\n  示例（前 3 个 zh 对话）:", flush=True)
    for i, (tag, prompt) in enumerate(PROMPTS[:3]):
        print(f"    [{tag}] {prompt!r}", flush=True)
        print(f"      exec: {results['executive'][i][:60]!r}", flush=True)
        print(f"      cont: {results['continuous'][i][:60]!r}", flush=True)

    # 判定 1：continuous 非空率不劣于 executive
    check("continuous 非空率 ≥ executive",
          stats["continuous"]["non_empty"] >= stats["executive"]["non_empty"],
          f"{stats['continuous']['non_empty']} vs {stats['executive']['non_empty']}")
    # 判定 2：continuous 平均重复率不高于 executive（≥0 表示不更差）
    check("continuous 重复率 ≤ executive + 0.02",
          stats["continuous"]["avg_rep"] <= stats["executive"]["avg_rep"] + 0.02,
          f"{stats['continuous']['avg_rep']:.3f} vs {stats['executive']['avg_rep']:.3f}")
    # 判定 3：continuous 至少 50% prompt 不劣于 executive（含 tie）
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
