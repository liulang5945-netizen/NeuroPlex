"""诊断十：A2 剂量下 night 损伤分解（6 文本 × cycles 8）。

测量睡眠前后：
  1. holdout 文本 surprise（use_memory=False 分离记忆注入）
  2. fabric 结构重连事件数
  3. 分系统参数变化量（fabric/motor/memory）
"""

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


def texts(limit: int, skip: int = 0) -> list[bytes]:
    out = []
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            if index >= skip + limit:
                break
            out.append(json.loads(line)["text"].encode("utf-8"))
    return out


def measure(model: Seed, items: list[bytes]) -> tuple[float, float]:
    substrate = model.substrate
    total = 0.0
    total_nomem = 0.0
    for text in items:
        for flag in (True, False):
            substrate.reset_dynamics(episode_id="diag10")
            surprise_sum = 0.0
            count = 0
            for symbol in substrate.sensor.symbols(text, include_boundary=True):
                step = substrate.observe(int(symbol), learn=False, use_memory=flag)
                if step.surprise is not None:
                    surprise_sum += step.surprise
                    count += 1
            value = surprise_sum / max(1, count)
            if flag:
                total += value
            else:
                total_nomem += value
    return total / len(items), total_nomem / len(items)


def fingerprint(model: Seed) -> dict[str, float]:
    substrate = model.substrate
    groups: dict[str, float] = {}
    for tensor in substrate.fabric.parameter_tensors():
        groups["fabric"] = groups.get("fabric", 0.0) + float(tensor.abs().sum().item())
    groups["motor"] = float(substrate.motor.synapses.edge_weight.abs().sum().item()) + float(
        substrate.motor.bias.abs().sum().item()
    )
    for tensor in substrate.memory.parameter_tensors():
        groups["memory"] = groups.get("memory", 0.0) + float(tensor.abs().sum().item())
    return groups


def main() -> None:
    torch.manual_seed(7)
    model = load_model()
    substrate = model.substrate
    targets = texts(6)
    holdout = texts(4, skip=100)
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    base_mem, base_nomem = measure(model, holdout)
    print(f"基线 holdout surprise={base_mem:.3f} (nomem={base_nomem:.3f})", flush=True)
    before = fingerprint(model)
    structural_before = substrate.fabric.structural_events

    night = scheduler.night(targets, cycles_per_text=8, learn=True)
    print(
        f"night accepted={night['accepted']:.0f} "
        f"mean_priority={night['mean_priority']:.3f}",
        flush=True,
    )

    after_mem, after_nomem = measure(model, holdout)
    after = fingerprint(model)
    structural_after = substrate.fabric.structural_events
    print(
        f"睡眠后 holdout surprise={after_mem:.3f} (nomem={after_nomem:.3f})",
        flush=True,
    )
    print(f"结构重连: {structural_before} -> {structural_after}", flush=True)
    for key in sorted(set(before) | set(after)):
        delta = after.get(key, 0.0) - before.get(key, 0.0)
        print(f"  |Δ| {key}: {delta:.3f}", flush=True)


if __name__ == "__main__":
    main()
