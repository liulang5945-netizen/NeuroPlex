"""诊断：成人模型睡眠破坏来源定位。

对 800K 检查点逐一隔离变量，测量睡眠前后固定探针的平均惊讶度：
  1. cycles=1，cue 链开/关
  2. cycles=4 剂量对比
  3. learn=False 对照（应零变化）
探针 = 校准语料前 8 篇 + 面板 3 条，惊讶度来自 score_bytes（只读）。
"""

from __future__ import annotations

import importlib.util
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


def probe_texts() -> list[bytes]:
    texts = []
    base = REPO / "data" / "simple_zh"
    path = base / "dialogue_extended_clean.jsonl"
    import json
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 8:
                break
            texts.append(json.loads(line)["text"].encode("utf-8"))
    texts.append("你好，请问今天感觉怎么样？".encode("utf-8"))
    texts.append("水的沸点是多少度？".encode("utf-8"))
    return texts


def measure(model: Seed, texts: list[bytes]) -> float:
    total = 0.0
    for text in texts:
        total += model.score_bytes(text)["mean_surprise"]
    return total / len(texts)


def variant(
    name: str,
    *,
    cycles: int,
    cue_chain: bool,
    learn: bool,
    texts_count: int = 2,
) -> None:
    started = time.time()
    model = load_model()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)
    probes = probe_texts()
    before = measure(model, probes)
    targets = probes[:texts_count]
    for text in targets:
        scheduler.experience(text, learn=True)
        model.substrate.consolidate(
            cycles=cycles, learn=learn, replay_cue_chain=cue_chain
        )
    after = measure(model, probes)
    print(
        f"{name}: surprise {before:.3f} -> {after:.3f} "
        f"(Δ={after - before:+.3f}) [{time.time() - started:.1f}s]",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(7)
    variant("learn=False 对照（应零变化）", cycles=4, cue_chain=True, learn=False)
    variant("1 周期 + cue 链", cycles=1, cue_chain=True, learn=True)
    variant("1 周期 无 cue 链", cycles=1, cue_chain=False, learn=True)
    variant("4 周期 + cue 链", cycles=4, cue_chain=True, learn=True)
    variant("4 周期 无 cue 链", cycles=4, cue_chain=False, learn=True)


if __name__ == "__main__":
    main()
