"""Train Seed on the raw-byte stream of the simple_zh corpus.

阶段 1 原生数据管线：复用 ``data/simple_zh/`` 既有语料（class_a/b/c、
dialogue_extended、alpaca_zh_sft），但以 raw-byte 流喂入 ``Seed.observe``；
会话边界用 ``boundary_symbol``，对话结构沿用语料里的文本标记（问：/答：），
全程不引入 tokenizer。训练循环为分片流式多 epoch + 周期 ``checkpoint()``
落盘（seed-native-v1 信封），进度曲线逐条写入 ``reports/``。

用法（默认小预算冒烟）::

    python scripts/training/train_seed_corpus.py --smoke

正式训练（放大画像、限量符号数、周期落盘）::

    python scripts/training/train_seed_corpus.py \
        --scale 2 --epochs 1 --max-symbols 400000 \
        --checkpoint checkpoints/seed_corpus.pt \
        --progress reports/seed_corpus_progress.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, SeedConfig  # noqa: E402
from taiji import TaijiConfig  # noqa: E402

DEFAULT_CORPUS = (
    "data/simple_zh/dialogue_extended_clean.jsonl",
    "data/simple_zh/alpaca_zh_sft_clean.jsonl",
    "data/simple_zh/class_a_chinese.jsonl",
)

# Fixed unseen probe: window statistics over the moving stream measure content
# difficulty, not model quality, so every progress entry also scores this
# constant byte string.  A monotone training run must drive its surprise down.
HOLDOUT_PROBE = (
    "水的沸点在标准大气压下是一百摄氏度。"
    "问：你好。\n答：你好，很高兴见到你。"
    "请解释一下牛顿第二定律和它的日常应用。"
).encode("utf-8")


def iter_corpus_symbols(
    paths: Sequence[Path | str],
    *,
    boundary: int = TaijiConfig().boundary_symbol,
) -> Iterator[int]:
    """Stream every corpus document as boundary-separated raw UTF-8 bytes.

    Every jsonl row contributes one session: one ``boundary_symbol`` followed by
    the document's UTF-8 bytes.  The dialogue structure already lives in the
    text (问：/答： markers), so no tokenizer and no structural re-encoding is
    needed -- the model sees exactly the bytes a reader would see.
    """

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                yield boundary
                yield from text.encode("utf-8")


def run_training(
    *,
    corpus_paths: Sequence[Path | str],
    config: SeedConfig,
    epochs: int,
    checkpoint_path: Path | str,
    progress_path: Path | str,
    checkpoint_every: int,
    progress_every: int,
    max_symbols: Optional[int] = None,
    resume_checkpoint: Optional[Path | str] = None,
) -> Dict[str, float]:
    """Stream the corpus through ``Seed.observe`` with periodic persistence."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if checkpoint_every <= 0 or progress_every <= 0:
        raise ValueError("checkpoint/progress intervals must be positive")

    checkpoint_path = Path(checkpoint_path)
    progress_path = Path(progress_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    model = Seed(config, episode_id="seed-corpus")
    if resume_checkpoint is not None:
        model.restore(torch.load(resume_checkpoint, weights_only=False))
    boundary = config.taiji.boundary_symbol

    started = time.perf_counter()
    ticks = 0
    window_ticks = 0
    window_correct = 0
    window_surprise = 0.0

    def _flush(final: bool) -> None:
        if window_ticks <= 0 and not final:
            return
        entry = {
            "epoch": epoch,
            "ticks": ticks,
            "window_ticks": window_ticks,
            "online_accuracy": window_correct / max(1, window_ticks),
            "mean_surprise": window_surprise / max(1, window_ticks),
            "holdout_surprise": model.score_bytes(HOLDOUT_PROBE)[
                "mean_surprise"
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for epoch in range(epochs):
        for symbol in iter_corpus_symbols(corpus_paths, boundary=boundary):
            step = model.observe(symbol, learn=True)
            ticks += 1
            if step.prior_prediction is not None:
                window_ticks += 1
                window_correct += int(step.prior_prediction == symbol)
                window_surprise += float(step.surprise)
            if ticks % progress_every == 0:
                _flush(final=False)
                window_ticks = 0
                window_correct = 0
                window_surprise = 0.0
            if ticks % checkpoint_every == 0:
                torch.save(model.checkpoint(), checkpoint_path)
            if max_symbols is not None and ticks >= max_symbols:
                _flush(final=True)
                torch.save(model.checkpoint(), checkpoint_path)
                return _summary(model, ticks)
    _flush(final=True)
    torch.save(model.checkpoint(), checkpoint_path)
    return _summary(model, ticks)


def _summary(model: Seed, ticks: int) -> Dict[str, float]:
    return {
        "ticks": float(ticks),
        "parameters": float(model.parameter_count()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[str(PROJECT_ROOT / name) for name in DEFAULT_CORPUS],
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument(
        "--progress",
        default=str(
            PROJECT_ROOT / "reports" / "seed_corpus_progress.jsonl"
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny default config and budget for a fast end-to-end run",
    )
    args = parser.parse_args()

    if args.smoke:
        config = SeedConfig()
        max_symbols = 5_000
    else:
        config = SeedConfig(
            taiji=TaijiConfig.training_profile(
                scale=args.scale, seed=args.seed
            )
        )
        max_symbols = args.max_symbols

    summary = run_training(
        corpus_paths=args.corpus,
        config=config,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        progress_path=args.progress,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        max_symbols=max_symbols,
        resume_checkpoint=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
