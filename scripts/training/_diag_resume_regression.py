"""Check whether continued training from the 200K checkpoint regresses quality."""
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


def main() -> None:
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "seed_corpus.pt", weights_only=False
    )
    static = Seed.from_checkpoint(checkpoint)
    continued = Seed.from_checkpoint(checkpoint)

    corpus = [PROJECT_ROOT / DEFAULT_CORPUS[0]]
    stream = iter_corpus_symbols(corpus)
    for _ in range(30_000):
        continued.observe(next(stream), learn=True)

    static_score = static.score_bytes(HOLDOUT)
    continued_score = continued.score_bytes(HOLDOUT)
    print(f"static    surprise={static_score['mean_surprise']:.4f} "
          f"acc={static_score['accuracy']:.4f}")
    print(f"continued surprise={continued_score['mean_surprise']:.4f} "
          f"acc={continued_score['accuracy']:.4f}")

    # score the very stream prefix both models just diverged on
    prefix = bytes(
        value for value in list(iter_corpus_symbols(corpus))[:2000]
        if value != 256
    )
    print("on first 2KB of the training stream:")
    for name, model in (("static", static), ("continued", continued)):
        score = model.score_bytes(prefix)
        print(f"  {name}: surprise={score['mean_surprise']:.4f} "
              f"acc={score['accuracy']:.4f}")


if __name__ == "__main__":
    main()
