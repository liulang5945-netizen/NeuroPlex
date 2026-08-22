"""Evaluate Seed on the byte stream with the frozen A1 prompt panel.

阶段 1 评估口径：``score_bytes`` 的 next-byte accuracy/surprise（nats/byte）
换算文本 PPL（``exp(mean_surprise)``）；BOOTSTRAP A1 的 24 条真实面板
（dialogue/knowledge/unfamiliar 三组）逐条测自我惊讶度并采样生成；冻结的
``neuroplex`` judge-NLL 基线数字（``a1_judge_nll_std_real_20260820.json``）
作为同面板对照引用。报告落盘 ``reports/``。

用法::

    python scripts/training/eval_seed_corpus.py \
        --checkpoint checkpoints/seed_corpus.pt \
        --holdout data/simple_zh/class_a_chinese.jsonl --holdout-rows 32
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed  # noqa: E402

BASELINE_REPORT = "a1_judge_nll_std_real_20260820.json"

# Frozen 24-prompt panel from BOOTSTRAP A1 (verify_a1_judge_signal_real.py).
PROMPT_PANEL: Dict[str, tuple] = {
    "dialogue": (
        "你好，请问今天感觉怎么样？",
        "能帮我解释一下你最近在想什么吗？",
        "我有点困惑，你能不能换个方式说一下？",
        "你刚才说的我没明白，再讲一遍好吗？",
        "谢谢你的回答，下次见。",
        "你现在心情如何？会累吗？",
        "我今天遇到了一件不顺心的事，能听我说说吗？",
        "可以推荐一本书给我吗？我想读点轻松的内容。",
    ),
    "knowledge": (
        "水的沸点是多少？为什么高海拔会降低沸点？",
        "请解释一下牛顿第二定律和它的日常应用。",
        "DNA 双螺旋结构是谁发现的？它如何携带遗传信息？",
        "什么是光合作用？它在生态系统中起什么作用？",
        "请简述 HTTPS 与 HTTP 的核心区别。",
        "地球上最大的洋流系统是什么？它如何影响全球气候？",
        "请解释相对论中时间膨胀的概念。",
        "什么是递归？请给出一个递归函数的 Python 示例。",
    ),
    "unfamiliar": (
        "请用古亚述语的楔形文字转写以下句子：他从山上走下来。",
        "解释海洋混合层深度的热焓通量平衡方程。",
        "在连续变量隐形传态协议中，纠缠态的零差测量如何恢复相干性？",
        "紧致单连通黎曼流形上，Yang-Mills 方程的瞬子解如何分类？",
        "贝叶斯神经网络中的 epistemic uncertainty 与 aleatoric uncertainty 有什么区别？",
        "描述超新星遗迹中非热 X 射线辐射的同步辐射模型参数空间。",
        "在范畴论中，adjunction 的 unit 和 counit 满足的三角等式是什么？",
        "请解释 CRISPR-Cas13 系统与 Cas9 在靶标分子类型上的根本差异。",
    ),
}


def _baseline_reference() -> Dict[str, Any]:
    path = PROJECT_ROOT / "reports" / BASELINE_REPORT
    if not path.is_file():
        return {"report": BASELINE_REPORT, "available": False, "groups": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {
        name: {"mean": entry["mean"], "std": entry["std"]}
        for name, entry in payload["groups"].items()
    }
    return {
        "report": BASELINE_REPORT,
        "available": True,
        "metric": "judge NLL (neuroplex frozen baseline, token space)",
        "a1_pass": payload.get("a1_pass"),
        "groups": groups,
    }


def evaluate_seed(
    model: Seed,
    *,
    holdout_bytes: bytes,
    report_path: Optional[Path | str] = None,
    generation_length: int = 64,
) -> Dict[str, Any]:
    """Measure holdout PPL, panel self-surprise and sampled continuations."""

    if not holdout_bytes:
        raise ValueError("holdout_bytes cannot be empty")

    scored = model.score_bytes(holdout_bytes)
    holdout = {
        "observations": scored["observations"],
        "accuracy": scored["accuracy"],
        "mean_surprise": scored["mean_surprise"],
        "byte_ppl": math.exp(scored["mean_surprise"]),
    }

    panel: Dict[str, Any] = {}
    samples = []
    for group, prompts in PROMPT_PANEL.items():
        surprises = []
        for prompt in prompts:
            prompt_bytes = prompt.encode("utf-8")
            prompt_score = model.score_bytes(prompt_bytes)
            surprises.append(prompt_score["mean_surprise"])
            continuation = model.generate(
                prompt_bytes,
                generation_length,
                sample=True,
                stop_at_boundary=True,
            )
            samples.append({
                "group": group,
                "prompt": prompt,
                "continuation": continuation.decode("utf-8", errors="replace"),
                "prompt_surprise": prompt_score["mean_surprise"],
            })
        panel[group] = {
            "surprises": surprises,
            "mean": statistics.fmean(surprises),
            "std": statistics.pstdev(surprises),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "seed-corpus-eval-v1",
        "parameters": model.parameter_count(),
        "holdout": holdout,
        "panel": panel,
        "samples": samples,
        "neuroplex_baseline_reference": _baseline_reference(),
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _holdout_bytes(path: Path, rows: int) -> bytes:
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            text = json.loads(line).get("text", "")
            if text:
                chunks.append(text)
    return "\n".join(chunks).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--holdout",
        default=str(PROJECT_ROOT / "data" / "simple_zh" / "class_a_chinese.jsonl"),
    )
    parser.add_argument("--holdout-rows", type=int, default=32)
    parser.add_argument("--generation-length", type=int, default=64)
    parser.add_argument(
        "--report",
        default=str(
            PROJECT_ROOT
            / "reports"
            / f"seed_corpus_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = Seed.from_checkpoint(checkpoint)
    report = evaluate_seed(
        model,
        holdout_bytes=_holdout_bytes(Path(args.holdout), args.holdout_rows),
        report_path=args.report,
        generation_length=args.generation_length,
    )
    print(
        f"byte_ppl={report['holdout']['byte_ppl']:.3f} "
        f"accuracy={report['holdout']['accuracy']:.3f} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
