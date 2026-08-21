"""N7: falsify whether Taiji state resolves second-order byte ambiguity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Literal, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig


DATA = b"axbcxd" * 4
AMBIGUOUS = ord("x")


def _first_order_accuracy(data: bytes) -> float:
    followers: Dict[int, Counter[int]] = defaultdict(Counter)
    for current, following in zip(data, data[1:]):
        followers[int(current)][int(following)] += 1
    counts = followers[AMBIGUOUS]
    best_count = max(counts.values())
    prediction = min(symbol for symbol, count in counts.items() if count == best_count)
    targets = [int(data[index + 1]) for index, value in enumerate(data[:-1])
               if value == AMBIGUOUS]
    return sum(target == prediction for target in targets) / len(targets)


def _centroid_cosine(groups: Dict[int, list[torch.Tensor]]) -> float:
    ordered = sorted(groups)
    if len(ordered) != 2:
        raise ValueError("N7 requires exactly two ambiguous successor groups")
    left = torch.stack(groups[ordered[0]]).mean(dim=0)
    right = torch.stack(groups[ordered[1]]).mean(dim=0)
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _evaluate(
    model: Taiji,
    *,
    lesion: Literal["none", "trace", "all"] = "none",
) -> Tuple[float, Dict[str, float], list[Dict[str, int]]]:
    model.reset_dynamics(episode_id=f"n7-{lesion}")
    sequence = (model.config.boundary_symbol, *DATA, model.config.boundary_symbol)
    hits = []
    rows = []
    contexts: Dict[int, list[torch.Tensor]] = defaultdict(list)
    fast: Dict[int, list[torch.Tensor]] = defaultdict(list)
    slow: Dict[int, list[torch.Tensor]] = defaultdict(list)
    for index, symbol in enumerate(sequence[:-1]):
        if lesion == "all":
            model.reset_dynamics(episode_id=f"n7-all-{index}")
        elif lesion == "trace":
            for region in model._state.regions:
                region.trace.zero_()
            model._state.motor_context.zero_()
        step = model.observe(symbol, learn=False)
        if symbol != AMBIGUOUS:
            continue
        target = int(sequence[index + 1])
        hit = int(step.predicted_symbol == target)
        hits.append(hit)
        rows.append({
            "target": target,
            "prediction": step.predicted_symbol,
            "hit": hit,
        })
        state = model.snapshot()
        contexts[target].append(state.motor_context)
        fast[target].append(torch.cat([region.activity for region in state.regions]))
        slow[target].append(torch.cat([region.trace for region in state.regions]))
    diagnostics = {
        "motor_context_centroid_cosine": _centroid_cosine(contexts),
        "fast_activity_centroid_cosine": _centroid_cosine(fast),
        "slow_trace_centroid_cosine": _centroid_cosine(slow),
    }
    return sum(hits) / len(hits), diagnostics, rows


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> Dict[str, object]:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )
    model = Taiji(config, episode_id="n7-train")
    model.learn_bytes(DATA, epochs=epochs)
    learned = model.checkpoint()

    full, diagnostics, rows = _evaluate(
        Taiji.from_checkpoint(learned), lesion="none"
    )
    trace_lesion, trace_diagnostics, _ = _evaluate(
        Taiji.from_checkpoint(learned), lesion="trace"
    )
    state_lesion, _, _ = _evaluate(
        Taiji.from_checkpoint(learned), lesion="all"
    )
    first_order = _first_order_accuracy(DATA)
    strongest_causal_control = max(first_order, state_lesion)
    checks = {
        "ambiguous_accuracy": full >= 0.75,
        "beats_first_order_by_20pp": full >= first_order + 0.20,
        "beats_full_state_lesion_by_20pp": full >= state_lesion + 0.20,
    }
    return {
        "benchmark": "taiji_n7_second_order_context",
        "seed": seed,
        "epochs": epochs,
        "stream": DATA.decode("ascii"),
        "ambiguous_symbol": chr(AMBIGUOUS),
        "metrics": {
            "full_accuracy": full,
            "first_order_accuracy": first_order,
            "trace_lesion_accuracy": trace_lesion,
            "state_lesion_accuracy": state_lesion,
            "gain_over_strongest_causal_control": full - strongest_causal_control,
            **diagnostics,
            "trace_lesion_motor_context_centroid_cosine": (
                trace_diagnostics["motor_context_centroid_cosine"]
            ),
        },
        "ambiguous_predictions": rows,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(epochs=args.epochs, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
