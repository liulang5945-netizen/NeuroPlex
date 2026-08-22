"""Ablate what makes continued exposure regress: memory feedback vs motor."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from seed import Seed  # noqa: E402
from train_seed_corpus import DEFAULT_CORPUS, iter_corpus_symbols  # noqa: E402

HOLDOUT = (
    "水的沸点在标准大气压下是一百摄氏度。"
    "问：你好。\n答：你好，很高兴见到你。"
    "请解释一下牛顿第二定律和它的日常应用。"
).encode("utf-8")

STEPS = 20_000


def run(name: str, **observe_kwargs) -> None:
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "seed_corpus.pt", weights_only=False
    )
    model = Seed.from_checkpoint(checkpoint)
    before = model.score_bytes(HOLDOUT)["mean_surprise"]
    stream = iter_corpus_symbols([PROJECT_ROOT / DEFAULT_CORPUS[0]])
    for _ in range(STEPS):
        model.observe(next(stream), **observe_kwargs)
    after = model.score_bytes(HOLDOUT)["mean_surprise"]
    print(f"{name:24s} before={before:.4f} after={after:.4f} "
          f"delta={after - before:+.4f}")


def main() -> None:
    run("full (default)", learn=True)
    run("no-memory-feedback", learn=True, use_memory=False)
    run("no-motor-learning", learn=True, learn_motor=False)
    run("no-learning", learn=False)


if __name__ == "__main__":
    main()
