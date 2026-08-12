#!/usr/bin/env python3
"""zh leader 信号 A/B 收益确认（2026-08-11）。

diag_zh_leader/capacity 发现：场共振分 leader 不偏好 134M zh_std0（0/5 当选），
但单 neuron 生成 zh_std0 质量最优（均长 77 vs 69，更连贯）。本脚本扩大样本量化
"强制 leader=zh_std0(134M)" vs "当前机制（场共振分选 leader）"的生成质量差异：

- 12 个 zh prompt（知识问答 8 + 闲聊 4）
- 各 ×3 次采样（temperature 0.55 波动消除，C25-E 增量五教训）
- 指标：平均长度 / 非空率 / 字符级重复率 / 主题词命中率
- 结论方向：zh_std0 显著占优 → 实施 leader 信号改进（所有域受益）；
  接近 → 转向 dialogue 数据扩充重训

运行：python -u scripts/training/verify_zh_leader_ab.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402
from scripts.training.experiment_config import build_dialogue_prompt  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

# 2026-08-12 口径修复：dialogue neuron 必须用训练格式（问/答）评估，
# 裸 prompt 会触发换行死循环假退化（见 plans §2.2）。
_QUESTIONS = [
    "请介绍什么是神经网络",
    "什么是注意力机制",
    "请解释梯度下降的原理",
    "如何缓解过拟合问题",
    "什么是自然语言处理",
    "请介绍 Transformer 架构",
    "什么是词嵌入",
    "深度学习有哪些应用领域",
    "你好，请介绍一下你自己",
    "请用一句话介绍态极",
    "你能帮我做什么",
    "今天天气不错，聊聊吗",
]
PROMPTS = [build_dialogue_prompt(q) for q in _QUESTIONS]

N_SAMPLES = 3
MAX_TOKENS = 40


def quality_metrics(out: str, prompt: str) -> dict:
    """长度 / 字符级重复率（越低越好）/ 主题词命中率。"""
    length = len(out)
    if length == 0:
        return {"len": 0, "dup": 0.0, "hit": 0.0}
    chars = list(out)
    uniq = len(set(chars))
    dup = 1.0 - uniq / max(length, 1)
    # 主题词命中：prompt 中长度 >= 2 的中文词在输出中出现
    stop = set("什么是如何请介绍请你一个一下今天天气不错聊聊能帮做用句话态极的")
    prompt_words = [w for w in prompt if w not in stop and not w.isspace()]
    hit = 0.0
    if prompt_words:
        hit = sum(1 for w in prompt_words if w in out) / len(prompt_words)
    return {"len": length, "dup": dup, "hit": hit}


def run_mode(cortex, prompts, force_nid=None):
    """返回每 prompt 的指标列表。force_nid=None = 当前机制（场共振分 leader）。"""
    per_prompt = []
    for prompt in prompts:
        runs = []
        for _ in range(N_SAMPLES):
            try:
                out = cortex.generate(
                    prompt, max_tokens=MAX_TOKENS, domain="zh",
                    collab_mode="continuous",
                    active_nids=[force_nid] if force_nid else None,
                )
            except Exception:
                out = ""
            runs.append(quality_metrics(out, prompt))
        per_prompt.append(runs)
    return per_prompt


def aggregate(per_prompt):
    lens = [r["len"] for runs in per_prompt for r in runs]
    dups = [r["dup"] for runs in per_prompt for r in runs if r["len"] > 0]
    hits = [r["hit"] for runs in per_prompt for r in runs if r["len"] > 0]
    non_empty = sum(1 for l in lens if l > 0)
    return {
        "avg_len": sum(lens) / len(lens),
        "non_empty": non_empty,
        "dup": sum(dups) / len(dups) if dups else 0.0,
        "hit": sum(hits) / len(hits) if hits else 0.0,
        "n_runs": len(lens),
    }


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("zh leader 信号 A/B（强制 134M vs 当前机制，12 prompt × 3 采样）", flush=True)
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

    print("\n[模式 A] 当前机制（场共振分选 leader）...", flush=True)
    cur = run_mode(cortex, PROMPTS)
    agg_cur = aggregate(cur)

    print("[模式 B] 强制 leader=zh_std0_dialogue（134M）...", flush=True)
    std0 = run_mode(cortex, PROMPTS, force_nid="zh_std0_dialogue")
    agg_std0 = aggregate(std0)

    print("\n" + "=" * 50, flush=True)
    print(f"{'指标':<12}{'当前机制(A)':>14}{'强制134M(B)':>14}{'Δ(B-A)':>12}", flush=True)
    print(f"{'平均长度':<12}{agg_cur['avg_len']:>14.1f}{agg_std0['avg_len']:>14.1f}"
          f"{agg_std0['avg_len'] - agg_cur['avg_len']:>+12.1f}", flush=True)
    print(f"{'非空':<12}{agg_cur['non_empty']:>14d}/{agg_cur['n_runs']}"
          f"{agg_std0['non_empty']:>10d}/{agg_std0['n_runs']}{'':>12}", flush=True)
    print(f"{'字符重复率':<12}{agg_cur['dup']:>14.3f}{agg_std0['dup']:>14.3f}"
          f"{agg_std0['dup'] - agg_cur['dup']:>+12.3f}", flush=True)
    print(f"{'主题词命中':<12}{agg_cur['hit']:>14.3f}{agg_std0['hit']:>14.3f}"
          f"{agg_std0['hit'] - agg_cur['hit']:>+12.3f}", flush=True)

    # 逐 prompt 长度对比（采样均值）
    print("\n逐 prompt 平均长度（3 采样均值）:", flush=True)
    better = worse = tie = 0
    for i, (prompt, runs_c, runs_s) in enumerate(zip(PROMPTS, cur, std0)):
        lc = sum(r["len"] for r in runs_c) / len(runs_c)
        ls = sum(r["len"] for r in runs_s) / len(runs_s)
        if ls > lc + 5:
            better += 1
        elif lc > ls + 5:
            worse += 1
        else:
            tie += 1
        mark = "B胜" if ls > lc + 5 else ("A胜" if lc > ls + 5 else "平")
        print(f"  [{mark}] {prompt[:16]}...  A={lc:.0f}  B={ls:.0f}", flush=True)
    print(f"\n  长度对比: B胜 {better} / A胜 {worse} / 平 {tie}", flush=True)

    # 结论判断
    print("\n" + "=" * 50, flush=True)
    len_gain = agg_std0["avg_len"] - agg_cur["avg_len"]
    dup_gain = agg_cur["dup"] - agg_std0["dup"]  # 正值 = B 重复更少（更好）
    if len_gain >= 8 and dup_gain >= 0.01:
        print(f"→ 强制 134M 显著占优（长度 +{len_gain:.0f}，重复 -{dup_gain:.3f}）"
              "——实施 leader 信号改进", flush=True)
    elif len_gain >= 3 or dup_gain >= 0.005:
        print(f"→ 强制 134M 轻微占优（长度 +{len_gain:.0f}，重复改善 {dup_gain:+.3f}）"
              "——收益边际，需权衡机制改动成本", flush=True)
    else:
        print(f"→ 收益不显著（长度 {len_gain:+.0f}，重复 {dup_gain:+.3f}）"
              "——转向 dialogue 数据扩充重训", flush=True)

    print(f"\n耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
