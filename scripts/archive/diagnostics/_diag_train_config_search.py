"""Find the observe configuration that learns monotonically from fresh start.

Arms (each fresh model, same corpus prefix, same budget):
  A: current default            learn=True, use_memory=True,  learn_motor=None
  B: no episodic feedback       learn=True, use_memory=False, learn_motor=None
  C: no motor learning          learn=True, use_memory=True,  learn_motor=False
  D: fabric-only                learn=True, use_memory=False, learn_motor=False

Measured every 20K ticks: holdout surprise (fixed, unseen) and stream-prefix
surprise (content the model has already seen -> re-exposure must not regress).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from seed import Seed, SeedConfig  # noqa: E402
from taiji import TaijiConfig  # noqa: E402
from train_seed_corpus import DEFAULT_CORPUS, iter_corpus_symbols  # noqa: E402

TOTAL = 60_000
WINDOW = 20_000

HOLDOUT = (
    "水的沸点在标准大气压下是一百摄氏度。"
    "问：你好。\n答：你好，很高兴见到你。"
    "请解释一下牛顿第二定律和它的日常应用。"
).encode("utf-8")

ARMS = {
    "A-default": dict(learn=True, use_memory=True),
    "B-no-memory": dict(learn=True, use_memory=False),
    "C-no-motor": dict(learn=True, use_memory=True, learn_motor=False),
    "D-fabric-only": dict(learn=True, use_memory=False, learn_motor=False),
}


def main() -> None:
    config = SeedConfig(taiji=TaijiConfig.training_profile(scale=2, seed=20260822))
    corpus = [PROJECT_ROOT / DEFAULT_CORPUS[0]]
    prefix = [
        value
        for value in list(iter_corpus_symbols(corpus))[:TOTAL + 1]
    ]
    prefix_bytes = bytes(v for v in prefix[:8000] if v != 256)

    results = {}
    for name, kwargs in ARMS.items():
        model = Seed(config, episode_id=f"ablation-{name}")
        rows = []
        for tick in range(1, TOTAL + 1):
            model.observe(prefix[tick - 1], **kwargs)
            if tick % WINDOW == 0:
                holdout = model.score_bytes(HOLDOUT)["mean_surprise"]
                reexpose = model.score_bytes(prefix_bytes)["mean_surprise"]
                rows.append((tick, round(holdout, 4), round(reexpose, 4)))
        results[name] = rows
        print(f"{name}: {rows}", flush=True)

    print()
    for name, rows in results.items():
        holdout_trend = rows[-1][1] - rows[0][1]
        reexpose_trend = rows[-1][2] - rows[0][2]
        print(f"{name}: holdout 40K->60K {holdout_trend:+.4f}, "
              f"re-expose 40K->60K {reexpose_trend:+.4f}")


if __name__ == "__main__":
    main()
