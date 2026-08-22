"""诊断三：逐子系统指纹，定位 consolidate(learn=False) 改了什么。"""

from __future__ import annotations

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


def probes() -> list[bytes]:
    texts = []
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 4:
                break
            texts.append(json.loads(line)["text"].encode("utf-8"))
    return texts


def fingerprints(model: Seed) -> dict:
    fabric = model.substrate.fabric
    motor = model.substrate.motor
    memory = model.substrate.memory
    def fp(tensors):
        return round(float(sum(t.detach().abs().sum().item() for t in tensors)), 4)
    return {
        "fabric": fp(fabric.parameter_tensors()),
        "motor": fp([motor.synapses.edge_weight, motor.bias]),
        "memory": fp(memory.parameter_tensors()),
        "reward_baseline": round(float(motor.reward_baseline), 6),
        "structural_events": int(fabric.structural_events),
        "memory_writes": int(memory.write_count),
    }


def measure(model: Seed, texts: list[bytes]) -> float:
    return sum(model.score_bytes(t)["mean_surprise"] for t in texts) / len(texts)


def main() -> None:
    torch.manual_seed(7)
    model = load_model()
    texts = probes()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    print("基线:", fingerprints(model), flush=True)
    print(f"基线 surprise={measure(model, texts):.3f}", flush=True)

    scheduler.experience(texts[0], learn=True)
    print("experience(learn=True) 后:", fingerprints(model), flush=True)
    print(f"surprise={measure(model, texts):.3f}", flush=True)

    # 把记忆读出全部清零：若损坏消失则确认读出是注入源
    memory = model.substrate.memory
    for readout in (
        memory.action_readout,
        memory.outcome_readout,
        memory.cortical_readout,
        memory.reward_readout,
        memory.familiarity_readout,
        memory.time_readout,
        memory.episode_readout,
        memory.provenance_readout,
    ):
        readout.edge_weight.zero_()
    print(f"记忆读出清零后 surprise={measure(model, texts):.3f}", flush=True)


if __name__ == "__main__":
    main()
