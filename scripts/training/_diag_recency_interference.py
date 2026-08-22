#!/usr/bin/env python3
"""诊断：holdout 漂移是否来自近因干扰（局部可塑性遗忘早期内容）。

800K 单遍流式训练中，固定探针 holdout_surprise 在 120K 触底后单调上升。
若这是近因干扰，则同一检查点对「流早期内容」的惊讶度应显著高于
「流晚期内容」（局部突触只保留最近统计）；若两者相近且都高，则是
探针/难度测量问题。

用法::

    python -X utf8 scripts/training/_diag_recency_interference.py \
        --checkpoint checkpoints/seed_corpus.pt --ticks 800000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed  # noqa: E402


def _trainer_module():
    script = REPO / "scripts" / "training" / "train_seed_corpus.py"
    spec = importlib.util.spec_from_file_location("train_seed_corpus", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _window_bytes(symbols: list[int], boundary: int) -> bytes:
    return bytes(value for value in symbols if value != boundary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=str(REPO / "checkpoints" / "seed_corpus.pt")
    )
    parser.add_argument("--ticks", type=int, default=800000)
    parser.add_argument("--window", type=int, default=4000)
    args = parser.parse_args()

    trainer = _trainer_module()
    model = Seed.from_checkpoint(
        torch.load(args.checkpoint, weights_only=False)
    )
    boundary = model.substrate.config.boundary_symbol

    symbols = []
    for value in trainer.iter_corpus_symbols(trainer.DEFAULT_CORPUS):
        symbols.append(value)
        if len(symbols) >= args.ticks:
            break
    if len(symbols) < args.ticks:
        raise SystemExit(f"corpus stream shorter than {args.ticks}")

    mid = args.ticks // 2
    windows = {
        "early": symbols[: args.window],
        "mid": symbols[mid : mid + args.window],
        "late": symbols[args.ticks - args.window : args.ticks],
    }
    print(f"checkpoint tick = {model.tick}")
    for name, window in windows.items():
        data = _window_bytes(window, boundary)
        score = model.score_bytes(data)
        print(
            f"{name:5s} window: surprise={score['mean_surprise']:.4f} "
            f"accuracy={score['accuracy']:.4f} bytes={len(data)}"
        )

    holdout = model.score_bytes(trainer.HOLDOUT_PROBE)
    print(
        f"holdout probe : surprise={holdout['mean_surprise']:.4f} "
        f"accuracy={holdout['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
