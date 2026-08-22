"""诊断六：读出学习率与成人模型单次经历崩塌的关系。

假说：共享稀疏读出行在 0.85 大步长下被上千次写入推到权重帽，
行坍缩到最后上下文方向，召回证据经 memory_read_gain 注入运动预测。
若调低读出率能保住清醒能力，则方向成立。
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402
from seed.judge import SeedJudge  # noqa: E402


def load_model() -> Seed:
    state = torch.load(REPO / "checkpoints" / "seed_corpus.pt", weights_only=False)
    return Seed.from_checkpoint(state)


def probe_text() -> bytes:
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())["text"].encode("utf-8")


def measure(model: Seed, text: bytes) -> float:
    return model.score_bytes(text)["mean_surprise"]


def trial(rate: float, read_gain: float) -> None:
    torch.manual_seed(7)
    model = load_model()
    new_config = dataclasses.replace(
        model.substrate.config,
        episodic_readout_learning_rate=rate,
        cortical_readout_learning_rate=0.3 * rate / 0.85,
        memory_read_gain=read_gain,
    )
    model.substrate.config = new_config
    model.substrate.memory.config = new_config
    text = probe_text()
    base = measure(model, text)
    scheduler = SeedSleepScheduler(model, SeedJudge(model))
    scheduler.experience(text, learn=True)
    after = measure(model, text)
    print(
        f"rate={rate:.2f} gain={read_gain:.1f}: "
        f"base={base:.3f} after={after:.3f} delta={after - base:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    for rate in (0.85, 0.30, 0.10, 0.03):
        trial(rate, 3.0)
    trial(0.85, 1.0)
    trial(0.85, 0.0)
