#!/usr/bin/env python3
"""诊断廿二：800K 成熟检查点上拆分 A2 夜晚的两条更新通路。

arm_wake_only  : experience(learn=True) 但不 consolidate —— 清醒学习贡献
arm_replay_only: experience(learn=False 但保留 episodic 写入需要验证可行性)
arm_full       : 完整 night（与 A2 脚本同口径）

输出打印，不落盘报告。运行：
python -X utf8 -u scripts/archive/diagnostics/_diag_a2_split_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "training"))
sys.path.insert(0, str(REPO))

import _seed_verify_common as common  # noqa: E402
from seed import SeedSleepScheduler  # noqa: E402

CHECKPOINT = REPO / "checkpoints" / "seed_corpus_800k_backup.pt"


def measure(model, judge):
    return common.measure_panel(model, judge)["groups"]


def summarize(tag, base, after):
    print(f"\n=== {tag} ===", flush=True)
    overall_b = sum(g["mean"] for g in base.values()) / len(base)
    overall_a = sum(g["mean"] for g in after.values()) / len(after)
    for name in base:
        print(
            f"  {name}: {base[name]['mean']:.4f} -> {after[name]['mean']:.4f} "
            f"Δ={after[name]['mean'] - base[name]['mean']:+.4f}",
            flush=True,
        )
    print(f"  overall Δ = {overall_a - overall_b:+.4f}", flush=True)


def main() -> None:
    # ---- arm 1: wake-only（experience 学习，不 consolidate）----
    model = common.load_model(str(CHECKPOINT))
    judge = common.calibrated_judge(model)
    base = measure(model, judge)
    scheduler = SeedSleepScheduler(model, judge)
    panel = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(panel, k=6)
    for text in targets:
        scheduler.experience(text, learn=True)
    summarize("arm_wake_only（仅 experience 清醒学习）", base, measure(model, judge))

    # ---- arm 2: full night（与 A2 同口径，独立加载）----
    model2 = common.load_model(str(CHECKPOINT))
    judge2 = common.calibrated_judge(model2)
    base2 = measure(model2, judge2)
    scheduler2 = SeedSleepScheduler(model2, judge2)
    targets2 = scheduler2.select_for_sleep(
        [item[2] for item in common.panel_texts_by_quality(judge2)], k=6
    )
    scheduler2.night(targets2, cycles_per_text=8, learn=True)
    summarize("arm_full（experience + consolidate）", base2, measure(model2, judge2))


if __name__ == "__main__":
    main()
