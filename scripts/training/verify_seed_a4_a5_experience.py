#!/usr/bin/env python3
"""阶段 3 门槛 A4/A5（原生版）：经验驱动的能力增长与自然饱和。

同判据移植：向有机体喂入它没在面板上见过的新经验（语料中未用于校准的
文档），每批经验通过真实经历（act/settle 情景写入）+ 内生巩固进入模型，
观察 24 条冻结面板三组质量均值与方差。

通过线（原 A4/A5 口径）：
    A5 增长：喂完 10 批后三组均值全部上升（Δ > 0），
             且上升 ≤ 0.30（不爆炸），且全体均值曲线过顶后回落 ≤ 0.15；
    A4 维持：三组 std 保持 ≥ 基线 × 0.95（区分度不倒退）。

不写检查点。输出 ``reports/seed_a4_a5_experience_<date>.json``。

运行：python -X utf8 -u scripts/training/verify_seed_a4_a5_experience.py
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
    parser = argparse.ArgumentParser(description="A4/A5 原生版：经验驱动增长")
    parser.add_argument(
        "--checkpoint",
        default=str(common.REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--texts-per-batch", type=int, default=4)
    parser.add_argument("--cycles-per-text", type=int, default=4)
    # 每条新经验的清醒预算：整篇全文的单遍学习会把已收敛的基底拉向新
    # 文档分布（40 篇全文累计后面板 -0.07）；经历压缩片段保留经验写入与巩固。
    parser.add_argument("--max-symbols", type=int, default=128)
    args = parser.parse_args()

    started = time.time()
    print("=" * 64, flush=True)
    print("A4/A5 原生版：经验驱动的能力增长与自然饱和", flush=True)
    print("=" * 64, flush=True)

    model = common.load_model(args.checkpoint)
    judge = common.calibrated_judge(model)
    scheduler = SeedSleepScheduler(model, judge)
    print(f"[1/4] 检查点 = {args.checkpoint}（tick={model.tick}）", flush=True)

    # 新经验：跳过校准用过的文档，保证对有机体是新的
    new_texts = [
        text.encode("utf-8")
        for text in common.corpus_documents(
            args.batches * args.texts_per_batch, skip=48
        )
    ]
    print(f"  新经验池 = {len(new_texts)} 篇（跳过校准段）", flush=True)

    baseline = common.measure_panel(model, judge)
    print("  基线:", flush=True)
    for name, group in baseline["groups"].items():
        print(f"    {name}: mean={group['mean']:.4f} std={group['std']:.4f}", flush=True)

    trajectory = [
        {
            "batch": 0,
            "group_means": {
                name: float(group["mean"])
                for name, group in baseline["groups"].items()
            },
            "overall_mean": float(baseline["overall_mean"]),
        }
    ]
    finite = True
    print("\n[2/4] 分批喂入新经验（经历 + 内生巩固）", flush=True)
    for batch in range(1, args.batches + 1):
        chunk = new_texts[
            (batch - 1) * args.texts_per_batch : batch * args.texts_per_batch
        ]
        scheduler.night(
            chunk,
            cycles_per_text=args.cycles_per_text,
            learn=True,
            max_symbols=args.max_symbols if args.max_symbols > 0 else None,
        )
        measurement = common.measure_panel(model, judge)
        entry = {
            "batch": batch,
            "group_means": {
                name: float(group["mean"])
                for name, group in measurement["groups"].items()
            },
            "overall_mean": float(measurement["overall_mean"]),
        }
        trajectory.append(entry)
        finite = finite and math.isfinite(entry["overall_mean"])
        print(
            f"  批 {batch}: overall={entry['overall_mean']:.4f} "
            f"（Δ={entry['overall_mean'] - trajectory[0]['overall_mean']:+.4f}）",
            flush=True,
        )

    final = common.measure_panel(model, judge)
    print("\n[3/4] 终态判定", flush=True)
    rises = {}
    bounded = True
    std_holds = True
    for name in baseline["groups"]:
        delta = final["groups"][name]["mean"] - baseline["groups"][name]["mean"]
        rises[name] = delta
        bounded = bounded and delta <= 0.30
        std_ratio = final["groups"][name]["std"] / max(
            baseline["groups"][name]["std"], 1e-9
        )
        std_holds = std_holds and std_ratio >= 0.95
        print(
            f"  {name}: Δmean={delta:+.4f}，std 比={std_ratio:.3f}",
            flush=True,
        )
    all_rise = all(delta > 0.0 for delta in rises.values())
    overall_curve = [entry["overall_mean"] for entry in trajectory]
    peak = max(overall_curve)
    post_peak_drop = peak - overall_curve[-1]
    saturates = post_peak_drop <= 0.15

    a5_pass = finite and all_rise and bounded and saturates
    a4_pass = finite and std_holds

    print("\n[4/4] 判定", flush=True)
    print(f"  A5 三组全部上升 = {'PASS' if all_rise else 'FAIL'}（{rises}）", flush=True)
    print(f"  A5 上升不爆炸（≤0.30） = {'PASS' if bounded else 'FAIL'}", flush=True)
    print(
        f"  A5 过顶回落（{post_peak_drop:.4f} ≤ 0.15） = "
        f"{'PASS' if saturates else 'FAIL'}",
        flush=True,
    )
    print(f"  A4 三组 std 维持（≥ 基线×0.95） = {'PASS' if std_holds else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)
    print(
        f"A4/A5 原生版 判定: A4={'PASS' if a4_pass else 'FAIL'} "
        f"A5={'PASS' if a5_pass else 'FAIL'}",
        flush=True,
    )
    print("=" * 64, flush=True)

    out_path = common.write_report(
        "seed_a4_a5_experience",
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "A4/A5 原生版：经验驱动增长与自然饱和",
            "checkpoint": str(args.checkpoint),
            "batches": args.batches,
            "texts_per_batch": args.texts_per_batch,
            "cycles_per_text": args.cycles_per_text,
            "max_symbols": args.max_symbols,
            "baseline": {
                name: {"mean": g["mean"], "std": g["std"]}
                for name, g in baseline["groups"].items()
            },
            "final": {
                name: {"mean": g["mean"], "std": g["std"]}
                for name, g in final["groups"].items()
            },
            "trajectory": trajectory,
            "group_rises": rises,
            "post_peak_drop": post_peak_drop,
            "criteria": {
                "finite": finite,
                "all_rise": all_rise,
                "bounded": bounded,
                "saturates": saturates,
                "std_holds": std_holds,
            },
            "a4_pass": a4_pass,
            "a5_pass": a5_pass,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"报告已写入: {out_path}", flush=True)
    sys.exit(0 if (a4_pass and a5_pass) else 1)


if __name__ == "__main__":
    main()
