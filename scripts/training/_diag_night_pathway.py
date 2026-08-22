"""诊断九：night 损伤的注入通路定位。

睡眠后测量三组对照：
  a) 原样
  b) consolidation_read_gain=0（慢通路解码注入关闭）
  c) use_memory=False（情景读出注入关闭）
哪一组恢复基线，哪一条就是损伤注入通路。
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed, SeedJudge  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def load_model() -> Seed:
    state = torch.load(REPO / "checkpoints" / "seed_corpus.pt", weights_only=False)
    return Seed.from_checkpoint(state)


def texts(limit: int = 6) -> list[bytes]:
    out = []
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            out.append(json.loads(line)["text"].encode("utf-8"))
    return out


def measure(model: Seed, items: list[bytes], *, use_memory: bool = True) -> float:
    substrate = model.substrate
    total = 0.0
    for text in items:
        substrate.reset_dynamics(episode_id="diag-measure")
        surprise_sum = 0.0
        count = 0
        previous = None
        for symbol in substrate.sensor.symbols(text, include_boundary=True):
            step = substrate.observe(
                int(symbol), learn=False, use_memory=use_memory
            )
            if step.surprise is not None:
                surprise_sum += step.surprise
                count += 1
            previous = step
        del previous
        total += surprise_sum / max(1, count)
    return total / len(items)


def main() -> None:
    torch.manual_seed(7)
    model = load_model()
    items = texts()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    base = measure(model, items)
    base_nomem = measure(model, items, use_memory=False)
    print(f"基线 surprise={base:.3f}（use_memory=False: {base_nomem:.3f}）", flush=True)

    for text in items[:3]:
        scheduler.experience(text, learn=True)
    night = scheduler.night(items[:3], cycles_per_text=8, learn=True)
    print(f"night accepted={night['accepted']:.0f}", flush=True)

    after = measure(model, items)
    after_nomem = measure(model, items, use_memory=False)
    print(f"睡眠后 surprise={after:.3f}（use_memory=False: {after_nomem:.3f}）", flush=True)

    new_config = dataclasses.replace(model.substrate.config, consolidation_read_gain=1e-6)
    model.substrate.config = new_config
    after_noslow = measure(model, items)
    print(f"睡眠后关闭慢通路注入 surprise={after_noslow:.3f}", flush=True)


if __name__ == "__main__":
    main()
