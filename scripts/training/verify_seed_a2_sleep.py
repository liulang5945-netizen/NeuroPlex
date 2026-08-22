#!/usr/bin/env python3
"""阶段 3 门槛 A2（原生版）：改进归因——仅自我评估驱动的巩固。

同判据移植：关闭一切外部监督，巩固目标由 ``SeedJudge`` 自己选出
（判定最差的样本优先），巩固机制完全落在内生 replay
（``SeedSleepScheduler.night`` → ``consolidate``）。

通过线（原 A2 口径）：
    睡眠后 24 条冻结面板的质量不降（全体均值 Δ >= 0），
    且至少一项指标（三组均值之一）改善（Δ > 0），
    且参数确有变化（巩固真实发生过）。

不写检查点。输出 ``reports/seed_a2_sleep_<date>.json``。

运行：python -X utf8 -u scripts/training/verify_seed_a2_sleep.py
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
    parser = argparse.ArgumentParser(description="A2 原生版：改进归因")
    parser.add_argument(
        "--checkpoint",
        default=str(common.REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--cycles-per-text", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    print("=" * 64, flush=True)
    print("A2 原生版：仅自我评估驱动的巩固（judge 选差 → 内生 replay）", flush=True)
    print("=" * 64, flush=True)

    model = common.load_model(args.checkpoint)
    print(f"[1/5] 检查点 = {args.checkpoint}（tick={model.tick}）", flush=True)
    judge = common.calibrated_judge(model)
    print(f"  judge 权重 = {[round(float(w), 4) for w in judge.weights]}", flush=True)

    print("\n[2/5] 基线面板测量（24 条）", flush=True)
    baseline = common.measure_panel(model, judge)
    for name, group in baseline["groups"].items():
        print(f"  {name}: mean={group['mean']:.4f} std={group['std']:.4f}", flush=True)

    # ---- 眼睛驱动手：巩固目标完全由 judge 选出 -----------------------
    scheduler = SeedSleepScheduler(model, judge)
    panel_texts = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(panel_texts, k=args.targets)
    print(f"\n[3/5] judge 选出最差的 {len(targets)} 条作为巩固对象", flush=True)

    before = [tensor.detach().clone() for tensor in model.substrate.parameter_tensors()]
    night = scheduler.night(
        targets, cycles_per_text=args.cycles_per_text, learn=True
    )
    delta_params = common.parameter_delta(model, before)
    print(
        f"  night 报告: cycles={night['cycles']:.0f} accepted={night['accepted']:.0f} "
        f"mean_priority={night['mean_priority']:.4f}",
        flush=True,
    )
    print(f"  参数变化总量 = {delta_params:.6f}", flush=True)

    print("\n[4/5] 睡眠后面板测量", flush=True)
    after = common.measure_panel(model, judge)
    deltas = {}
    for name in baseline["groups"]:
        delta = after["groups"][name]["mean"] - baseline["groups"][name]["mean"]
        deltas[name] = delta
        print(f"  {name}: mean={after['groups'][name]['mean']:.4f} Δ={delta:+.4f}", flush=True)
    overall_delta = after["overall_mean"] - baseline["overall_mean"]
    print(f"  全体均值 Δ = {overall_delta:+.4f}", flush=True)

    finite = all(
        math.isfinite(after["groups"][name]["mean"]) for name in baseline["groups"]
    )
    no_regression = overall_delta >= 0.0
    some_improvement = any(delta > 0.0 for delta in deltas.values())
    consolidation_real = delta_params > 0.0
    a2_pass = finite and no_regression and some_improvement and consolidation_real

    print("\n[5/5] 判定", flush=True)
    print(f"  数值有限 = {'PASS' if finite else 'FAIL'}", flush=True)
    print(f"  质量不降（Δ>=0） = {'PASS' if no_regression else 'FAIL'}（{overall_delta:+.4f}）", flush=True)
    print(f"  至少一项改善 = {'PASS' if some_improvement else 'FAIL'}（{deltas}）", flush=True)
    print(f"  巩固真实发生（参数变化>0） = {'PASS' if consolidation_real else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)
    print(f"A2 原生版 判定: {'PASS' if a2_pass else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)

    out_path = common.write_report(
        "seed_a2_sleep",
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "A2 原生版：仅自我评估驱动的巩固",
            "checkpoint": str(args.checkpoint),
            "targets": args.targets,
            "cycles_per_text": args.cycles_per_text,
            "night_report": night,
            "parameter_delta": delta_params,
            "baseline": {
                name: {"mean": g["mean"], "std": g["std"]}
                for name, g in baseline["groups"].items()
            },
            "after": {
                name: {"mean": g["mean"], "std": g["std"]}
                for name, g in after["groups"].items()
            },
            "group_deltas": deltas,
            "overall_delta": overall_delta,
            "criteria": {
                "finite": finite,
                "no_regression": no_regression,
                "some_improvement": some_improvement,
                "consolidation_real": consolidation_real,
            },
            "a2_pass": a2_pass,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"报告已写入: {out_path}", flush=True)
    sys.exit(0 if a2_pass else 1)


if __name__ == "__main__":
    main()
