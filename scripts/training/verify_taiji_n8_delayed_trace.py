"""N8: test whether slow trace controls actions after shared distractors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, Literal, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig


DATA = b"a1234xbc1234xd" * 4
PROBE = ord("x")


def _first_order_accuracy(data: bytes) -> float:
    followers: Counter[int] = Counter()
    targets = []
    for current, following in zip(data, data[1:]):
        if current == PROBE:
            followers[int(following)] += 1
            targets.append(int(following))
    best_count = max(followers.values())
    prediction = min(symbol for symbol, count in followers.items()
                     if count == best_count)
    return sum(target == prediction for target in targets) / len(targets)


def _centroid_cosine(groups: Dict[int, list[torch.Tensor]]) -> float:
    labels = sorted(groups)
    left = torch.stack(groups[labels[0]]).mean(dim=0)
    right = torch.stack(groups[labels[1]]).mean(dim=0)
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _intervene(
    model: Taiji,
    mode: Literal["no_trace", "trace_only", "all"],
    index: int,
) -> None:
    if mode == "all":
        model.reset_dynamics(episode_id=f"n8-all-{index}")
        return
    for region in model._state.regions:
        if mode == "no_trace":
            region.trace.zero_()
        else:
            region.membrane.zero_()
            region.activity.zero_()
            region.prediction.zero_()
            region.error.zero_()
            region.threshold.fill_(model.config.threshold_base)
            region.inhibition.zero_()
    model._state.motor_context.zero_()


def _evaluate(
    model: Taiji,
    *,
    mode: Literal["full", "no_trace", "trace_only", "all"],
) -> Tuple[float, list[Dict[str, int]], Dict[str, float]]:
    model.reset_dynamics(episode_id=f"n8-{mode}")
    sequence = (model.config.boundary_symbol, *DATA, model.config.boundary_symbol)
    hits = []
    rows = []
    pre_fast: Dict[int, list[torch.Tensor]] = defaultdict(list)
    pre_trace: Dict[int, list[torch.Tensor]] = defaultdict(list)
    post_context: Dict[int, list[torch.Tensor]] = defaultdict(list)

    for index, symbol in enumerate(sequence[:-1]):
        if symbol != PROBE:
            model.observe(symbol, learn=False)
            continue

        target = int(sequence[index + 1])
        state = model.snapshot()
        pre_fast[target].append(torch.cat([
            *(region.membrane for region in state.regions),
            *(region.activity for region in state.regions),
        ]))
        pre_trace[target].append(torch.cat([
            region.trace for region in state.regions
        ]))
        if mode != "full":
            _intervene(model, mode, index)
        step = model.observe(symbol, learn=False)
        hit = int(step.predicted_symbol == target)
        hits.append(hit)
        rows.append({
            "target": target,
            "prediction": step.predicted_symbol,
            "hit": hit,
        })
        post_context[target].append(model.snapshot().motor_context)

    diagnostics = {
        "pre_probe_fast_centroid_cosine": _centroid_cosine(pre_fast),
        "pre_probe_trace_centroid_cosine": _centroid_cosine(pre_trace),
        "post_probe_motor_context_centroid_cosine": _centroid_cosine(post_context),
    }
    return sum(hits) / len(hits), rows, diagnostics


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> Dict[str, object]:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )
    model = Taiji(config, episode_id="n8-train")
    model.learn_bytes(DATA, epochs=epochs)
    learned = model.checkpoint()

    full, rows, diagnostics = _evaluate(
        Taiji.from_checkpoint(learned), mode="full"
    )
    without_trace, no_trace_rows, _ = _evaluate(
        Taiji.from_checkpoint(learned), mode="no_trace"
    )
    trace_only, trace_only_rows, _ = _evaluate(
        Taiji.from_checkpoint(learned), mode="trace_only"
    )
    without_state, all_rows, _ = _evaluate(
        Taiji.from_checkpoint(learned), mode="all"
    )
    first_order = _first_order_accuracy(DATA)
    checks = {
        "full_delayed_accuracy": full >= 0.75,
        "beats_first_order_by_20pp": full >= first_order + 0.20,
        "trace_is_necessary_by_20pp": full >= without_trace + 0.20,
        "trace_only_is_sufficient": trace_only >= 0.75,
        "trace_only_beats_all_state_lesion_by_20pp": (
            trace_only >= without_state + 0.20
        ),
    }
    return {
        "benchmark": "taiji_n8_delayed_trace",
        "seed": seed,
        "epochs": epochs,
        "stream": DATA.decode("ascii"),
        "cue_to_probe_distractors": "1234",
        "probe": chr(PROBE),
        "metrics": {
            "full_accuracy": full,
            "first_order_accuracy": first_order,
            "no_trace_accuracy": without_trace,
            "trace_only_accuracy": trace_only,
            "all_state_lesion_accuracy": without_state,
            "trace_necessity_gap": full - without_trace,
            "trace_sufficiency_gap": trace_only - without_state,
            **diagnostics,
        },
        "predictions": {
            "full": rows,
            "no_trace": no_trace_rows,
            "trace_only": trace_only_rows,
            "all_state_lesion": all_rows,
        },
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
