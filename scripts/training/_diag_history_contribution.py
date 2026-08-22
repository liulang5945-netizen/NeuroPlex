"""Measure how much cortical history contributes to next-byte prediction.

Loads a trained checkpoint and, for a fixed probe stream, compares the
model's distribution with full dynamics against the same distribution after
``reset_dynamics`` (history cleared, synapses kept).  If both agree, the
motor head only learned byte marginals and the fabric contributes nothing;
the gap is the fabric's causal share of prediction quality.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from seed import Seed  # noqa: E402
from train_seed_corpus import DEFAULT_CORPUS, iter_corpus_symbols  # noqa: E402

PROBE_LEN = 4000


def main() -> None:
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "seed_corpus.pt", weights_only=False
    )
    model = Seed.from_checkpoint(checkpoint)
    symbols = list(iter_corpus_symbols(
        [PROJECT_ROOT / DEFAULT_CORPUS[0]]
    ))[200_000:200_000 + PROBE_LEN]

    # Full-history pass: stream the probe with learning off, record surprise.
    model.reset_dynamics(episode_id="probe-full")
    full_surprise = []
    full_top1 = 0
    for symbol in symbols:
        step = model.observe(symbol, learn=False)
        if step.surprise is not None:
            full_surprise.append(step.surprise)
            full_top1 += int(step.prior_prediction == symbol)

    # Reset-history pass: identical stream, but dynamics cleared first.
    model.reset_dynamics(episode_id="probe-reset")
    reset_surprise = []
    reset_top1 = 0
    for symbol in symbols:
        step = model.observe(symbol, learn=False)
        if step.surprise is not None:
            reset_surprise.append(step.surprise)
            reset_top1 += int(step.prior_prediction == symbol)

    n = len(full_surprise)
    full_mean = sum(full_surprise) / n
    reset_mean = sum(reset_surprise) / n
    print(f"n={n}")
    print(f"full-history:  surprise={full_mean:.4f} acc={full_top1 / n:.4f}")
    print(f"reset-history: surprise={reset_mean:.4f} acc={reset_top1 / n:.4f}")
    print(f"fabric share:  {reset_mean - full_mean:+.4f} nats/byte, "
          f"{(full_top1 - reset_top1) / n:+.4f} accuracy")


if __name__ == "__main__":
    main()
