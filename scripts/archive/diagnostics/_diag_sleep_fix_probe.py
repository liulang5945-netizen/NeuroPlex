"""诊断四：修复候选的效果探针。

修改后运行本脚本观察：
  1. experience 之后清醒能力是否保住（surprise 不应暴涨）
  2. 睡眠（learn=True）之后是否至少不破坏
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed, SeedJudge  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def load_model() -> Seed:
    state = torch.load(REPO / "checkpoints" / "seed_corpus.pt", weights_only=False)
    return Seed.from_checkpoint(state)


def probes() -> list[bytes]:
    texts = []
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 6:
                break
            texts.append(json.loads(line)["text"].encode("utf-8"))
    return texts


def measure(model: Seed, texts: list[bytes]) -> float:
    return sum(model.score_bytes(t)["mean_surprise"] for t in texts) / len(texts)


def main() -> None:
    torch.manual_seed(7)
    started = time.time()
    model = load_model()
    texts = probes()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    base = measure(model, texts)
    print(f"基线 surprise={base:.3f}", flush=True)

    for index, text in enumerate(texts[:3]):
        scheduler.experience(text, learn=True)
        print(
            f"experience #{index + 1} 后 surprise="
            f"{measure(model, texts):.3f}（writes={model.substrate.memory.write_count}）",
            flush=True,
        )

    night = scheduler.night(texts[:3], cycles_per_text=4, learn=True)
    after_sleep = measure(model, texts)
    print(
        f"night(learn=True) 后 surprise={after_sleep:.3f} "
        f"accepted={night['accepted']:.0f} [{time.time() - started:.0f}s]",
        flush=True,
    )
    verdict = "HEALTHY" if after_sleep <= base + 0.3 else "DAMAGED"
    print(f"判定: {verdict}", flush=True)


if __name__ == "__main__":
    main()
