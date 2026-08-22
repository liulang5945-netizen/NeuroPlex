"""诊断十一：A2 最差 6 条的长度、回放选择信号与逐文本 night 行为。

目的：回答"为什么面板最差 6 条触发 47/48 回放接受，而 dialogue_extended
前 6 条只有 3/48"。逐文本测量 writes、accepted、priority 及分量
（familiarity/resonance/novelty/value/recency）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "training"))

import torch  # noqa: E402

from seed import Seed  # noqa: E402

import _seed_verify_common as common  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    model = common.load_model(str(REPO / "checkpoints" / "seed_corpus.pt"))
    judge = common.calibrated_judge(model)
    scheduler = SeedSleepScheduler(model, judge)

    panel = common.panel_texts_by_quality(judge)
    panel_texts = [item[2] for item in panel]
    lengths = [len(t) for t in panel_texts]
    print(
        f"面板 24 条长度: min={min(lengths)} max={max(lengths)} "
        f"mean={sum(lengths) / len(lengths):.0f}",
        flush=True,
    )

    targets = scheduler.select_for_sleep(panel_texts, k=6)
    wc0 = model.substrate.memory.write_count
    print(f"初始 write_count = {wc0}", flush=True)
    for index, text in enumerate(targets):
        quality = judge.score(text)["quality"]
        print(f"worst[{index}] len={len(text)} quality={quality:.3f}", flush=True)

    for index, text in enumerate(targets):
        wc_before = model.substrate.memory.write_count
        scheduler.experience(text, learn=True)
        writes = model.substrate.memory.write_count - wc_before
        result = model.consolidate(cycles=8, learn=True)
        print(
            f"text[{index}]: writes={writes} accepted={result.accepted}/8 "
            f"priority={result.mean_priority:.3f} "
            f"structural={result.structural_events}",
            flush=True,
        )


if __name__ == "__main__":
    main()
