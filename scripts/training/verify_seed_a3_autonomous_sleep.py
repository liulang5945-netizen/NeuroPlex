#!/usr/bin/env python3
"""阶段 3 门槛 A3（原生版）：自我维持——连续 8 轮自主睡眠。

同判据移植：无外部干预，每轮由器官自己选巩固对象（judge 判定最差的
面板文本优先），机制落在内生 replay。监控面板全体质量均值的漂移。

通过线（原 A3 口径）：
    8 轮累计 |Δ 全体均值| < 0.15，且全程数值有限（无崩溃）。

不写检查点。输出 ``reports/seed_a3_autonomous_sleep_<date>.json``。

运行：python -X utf8 -u scripts/training/verify_seed_a3_autonomous_sleep.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import _seed_verify_common as common  # noqa: E402

from seed import SeedSleepScheduler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 原生版：自我维持")
    parser.add_argument(
        "--checkpoint",
        default=str(common.REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--cycles-per-text", type=int, default=4)
    # 自我维持的夜晚是观察性的（不重复学习面板文本）；max-symbols 只限制
    # 观察窗口，≤0 表示观察整段文本。
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    print("=" * 64, flush=True)
    print("A3 原生版：连续 8 轮自主睡眠（自我维持）", flush=True)
    print("=" * 64, flush=True)

    model = common.load_model(args.checkpoint)
    judge = common.calibrated_judge(model)
    scheduler = SeedSleepScheduler(model, judge)
    print(f"[1/3] 检查点 = {args.checkpoint}（tick={model.tick}）", flush=True)

    baseline = common.measure_panel(model, judge)
    baseline_mean = float(baseline["overall_mean"])
    print(f"  基线全体均值 = {baseline_mean:.4f}", flush=True)

    trajectory = [{"round": 0, "overall_mean": baseline_mean, "delta": 0.0}]
    accepted_total = 0.0
    finite = True
    print("\n[2/3] 自主睡眠循环", flush=True)
    for round_index in range(1, args.rounds + 1):
        # 每轮重新由 judge 选差——巩固对象始终来自自我评估
        panel_texts = [item[2] for item in common.panel_texts_by_quality(judge)]
        targets = scheduler.select_for_sleep(panel_texts, k=args.targets)
        # 自我维持的夜晚是观察性的：情节写入 + 内生回放，清醒预测器不再
        # 重复学习同一面板文本（诊断廿五：观察性夜晚 8 轮漂移 ≈1e-4）。
        night = scheduler.night(
            targets,
            cycles_per_text=args.cycles_per_text,
            learn=True,
            max_symbols=args.max_symbols if args.max_symbols > 0 else None,
            observational=True,
        )
        accepted_total += float(night["accepted"])
        measurement = common.measure_panel(model, judge)
        overall = float(measurement["overall_mean"])
        delta = overall - baseline_mean
        finite = finite and math.isfinite(overall)
        trajectory.append(
            {
                "round": round_index,
                "overall_mean": overall,
                "delta": delta,
                "accepted": float(night["accepted"]),
            }
        )
        print(
            f"  轮 {round_index}: mean={overall:.4f} Δ={delta:+.4f} "
            f"accepted={night['accepted']:.0f}",
            flush=True,
        )

    cumulative = abs(trajectory[-1]["delta"])
    stable = cumulative < 0.15
    a3_pass = finite and stable

    print("\n[3/3] 判定", flush=True)
    print(f"  数值有限 = {'PASS' if finite else 'FAIL'}", flush=True)
    print(
        f"  8 轮累计 |Δ| = {cumulative:.4f} "
        f"→ {'PASS' if stable else 'FAIL'}（线 < 0.15）",
        flush=True,
    )
    print(f"  累计被接受的内生回放 = {accepted_total:.0f}", flush=True)
    print("=" * 64, flush=True)
    print(f"A3 原生版 判定: {'PASS' if a3_pass else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)

    out_path = common.write_report(
        "seed_a3_autonomous_sleep",
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "A3 原生版：连续 8 轮自主睡眠",
            "checkpoint": str(args.checkpoint),
            "rounds": args.rounds,
            "targets": args.targets,
            "cycles_per_text": args.cycles_per_text,
            "max_symbols": args.max_symbols,
            "baseline_overall_mean": baseline_mean,
            "trajectory": trajectory,
            "cumulative_abs_delta": cumulative,
            "accepted_total": accepted_total,
            "criteria": {"finite": finite, "stable": stable},
            "a3_pass": a3_pass,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"报告已写入: {out_path}", flush=True)
    sys.exit(0 if a3_pass else 1)


if __name__ == "__main__":
    main()
