#!/usr/bin/env python3
"""阶段 2 门槛 A1（原生版）：Seed judge 自我评估信度验证。

同判据移植自 ``verify_a1_judge_signal_real.py``（neuroplex judge-NLL 版），
口径不变，信号源替换为原生器官 ``seed/judge.py``：

判据 1（排序信度）：
    用阶段 1 训练语料文档构成已知好/坏对——``score_bytes`` 惊讶度
    （等价于 NLL/loss）低的文档为"好"，高的为"坏"；原生 judge 的
    质量分排序准确率必须 >= 0.7（原通过线）。

判据 2（真实任务区分度）：
    复用 A1 冻结的 24 条真实面板（dialogue/knowledge/unfamiliar 各 8 条），
    每组 8 条的 judge 质量分 std > 0.05；>= 2 组通过 = A1 通过。

约束：不写检查点；不引入外部评分模型；权重由器官局部校准获得。
输出 ``reports/seed_a1_judge_<date>.json``，失败以非零码退出。

运行：python -X utf8 -u scripts/training/verify_seed_a1_judge.py \\
      --checkpoint checkpoints/seed_corpus.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed, SeedJudge  # noqa: E402


def _load_eval_module():
    script = REPO / "scripts" / "training" / "eval_seed_corpus.py"
    spec = importlib.util.spec_from_file_location("eval_seed_corpus", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_corpus_paths() -> list[Path]:
    base = REPO / "data" / "simple_zh"
    return [
        base / "dialogue_extended_clean.jsonl",
        base / "alpaca_zh_sft_clean.jsonl",
        base / "class_a_chinese.jsonl",
    ]


def _load_documents(paths: list[Path], limit: int) -> list[str]:
    documents: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if text:
                    documents.append(text)
                if len(documents) >= limit:
                    return documents
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 同判据原生验证")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--documents", type=int, default=48)
    parser.add_argument("--pairs-per-side", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    started = time.time()
    print("=" * 64, flush=True)
    print("A1 原生版：Seed judge 自我评估信度（排序 + 24 条真实面板）", flush=True)
    print("=" * 64, flush=True)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = Seed.from_checkpoint(checkpoint)
    judge = SeedJudge(model)
    print(f"[1/4] 检查点 = {args.checkpoint}（tick={model.tick}）", flush=True)

    # ---- 判据 1：已知好/坏对的排序信度 -------------------------------
    print(f"\n[2/4] 语料文档 loss 排序 → 构造好/坏对（各 {args.pairs_per_side}）", flush=True)
    documents = _load_documents(_default_corpus_paths(), args.documents)
    scored = []
    for index, text in enumerate(documents):
        data = text.encode("utf-8")
        loss = model.score_bytes(data)["mean_surprise"]
        scored.append((loss, data))
        if (index + 1) % 12 == 0:
            print(f"  已测量 {index + 1}/{len(documents)} 篇", flush=True)
    scored.sort(key=lambda item: item[0])

    good = scored[: args.pairs_per_side]
    bad = scored[-args.pairs_per_side :]
    print(
        f"  loss 范围：好组 [{good[0][0]:.3f}, {good[-1][0]:.3f}]，"
        f"坏组 [{bad[0][0]:.3f}, {bad[-1][0]:.3f}]",
        flush=True,
    )

    # 器官局部校准：以 -loss 为已知质量目标，权重由闭式岭回归学得。
    calibration = [
        (judge.features(data), -loss) for loss, data in scored
    ]
    calibration_accuracy = judge.calibrate(calibration)
    print(f"  校准（局部闭式岭回归）拟合排序准确率 = {calibration_accuracy:.3f}", flush=True)
    print(f"  学得权重 = {[round(float(w), 4) for w in judge.weights]}", flush=True)

    comparable = 0
    agreeing = 0
    for good_loss, good_data in good:
        good_quality = judge.score(good_data)["quality"]
        for bad_loss, bad_data in bad:
            bad_quality = judge.score(bad_data)["quality"]
            comparable += 1
            if good_quality > bad_quality:
                agreeing += 1
    ranking_accuracy = agreeing / comparable
    criterion_1 = ranking_accuracy >= 0.7
    print(
        f"  judge 排序准确率 = {ranking_accuracy:.3f}（{agreeing}/{comparable}）"
        f" → {'PASS' if criterion_1 else 'FAIL'}（线=0.7）",
        flush=True,
    )

    # ---- 判据 2：24 条真实面板区分度 ---------------------------------
    print("\n[3/4] 24 条冻结真实面板（A1 原面板）", flush=True)
    panel = _load_eval_module().PROMPT_PANEL
    groups: dict[str, dict] = {}
    for group_name, prompts in panel.items():
        qualities = []
        details = []
        for text in prompts:
            report = judge.score(text.encode("utf-8"))
            qualities.append(report["quality"])
            details.append(
                {"text": text, "quality": report["quality"],
                 "mean_surprise": report["mean_surprise"],
                 "accuracy": report["accuracy"]}
            )
        mean = sum(qualities) / len(qualities)
        std = (
            sum((value - mean) ** 2 for value in qualities) / len(qualities)
        ) ** 0.5
        groups[group_name] = {
            "details": details,
            "mean": mean,
            "std": std,
            "pass": std > 0.05,
        }
        print(
            f"  {group_name}: mean={mean:.4f} std={std:.4f} "
            f"→ {'PASS' if std > 0.05 else 'FAIL'}（线=std>0.05）",
            flush=True,
        )

    pass_count = sum(1 for group in groups.values() if group["pass"])
    criterion_2 = pass_count >= 2
    a1_pass = criterion_1 and criterion_2

    print("\n[4/4] 汇总", flush=True)
    print(
        f"  判据1 排序信度 = {'PASS' if criterion_1 else 'FAIL'}"
        f"（{ranking_accuracy:.3f} >= 0.7）",
        flush=True,
    )
    print(
        f"  判据2 面板区分度 = {'PASS' if criterion_2 else 'FAIL'}"
        f"（{pass_count}/3 组 std>0.05，需 >= 2）",
        flush=True,
    )
    print("=" * 64, flush=True)
    print(f"A1 原生版 判定: {'PASS' if a1_pass else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)

    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    out_path = reports / f"seed_a1_judge_{time.strftime('%Y%m%d')}.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "A1 原生版：Seed judge 自我评估信度",
        "checkpoint": str(args.checkpoint),
        "judge_weights": [float(weight) for weight in judge.weights],
        "calibration_accuracy": calibration_accuracy,
        "ranking_accuracy": ranking_accuracy,
        "good_loss_range": [good[0][0], good[-1][0]],
        "bad_loss_range": [bad[0][0], bad[-1][0]],
        "groups": groups,
        "pass_count": pass_count,
        "criterion_ranking": criterion_1,
        "criterion_panel": criterion_2,
        "a1_pass": a1_pass,
        "elapsed_seconds": time.time() - started,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"报告已写入: {out_path}", flush=True)
    sys.exit(0 if a1_pass else 1)


if __name__ == "__main__":
    main()
