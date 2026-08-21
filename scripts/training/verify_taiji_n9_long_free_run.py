"""N9: verify 128 exact, bounded Taiji actions without teacher forcing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig


DATA = b"abcd" * 4
PROMPT = b"a"
EXPECTED = b"bcda" * 32


def _instrumented_free_run(model: Taiji) -> Dict[str, object]:
    model.reset_dynamics(episode_id="n9-instrumented")
    step = model.observe(model.config.boundary_symbol, learn=False)
    for symbol in PROMPT:
        step = model.observe(symbol, learn=False)

    generated = bytearray()
    invalid_actions = 0
    max_membrane_norm = 0.0
    max_trace_norm = 0.0
    threshold_min = float("inf")
    threshold_max = float("-inf")
    finite = True
    for _ in range(len(EXPECTED)):
        symbol = step.predicted_symbol
        if not 0 <= symbol <= 255:
            invalid_actions += 1
            symbol = 0
        generated.append(symbol)
        step = model.observe(symbol, learn=False)
        state = model.snapshot()
        for region in state.regions:
            max_membrane_norm = max(
                max_membrane_norm, float(region.membrane.norm().item())
            )
            max_trace_norm = max(max_trace_norm, float(region.trace.norm().item()))
            threshold_min = min(threshold_min, float(region.threshold.min().item()))
            threshold_max = max(threshold_max, float(region.threshold.max().item()))
            finite = finite and bool(torch.isfinite(region.membrane).all())
            finite = finite and bool(torch.isfinite(region.trace).all())
            finite = finite and bool(torch.isfinite(region.threshold).all())

    output = bytes(generated)
    errors = [
        index for index, (actual, expected) in enumerate(zip(output, EXPECTED))
        if actual != expected
    ]
    return {
        "generated": output,
        "accuracy": 1.0 - len(errors) / len(EXPECTED),
        "first_error_index": None if not errors else errors[0],
        "invalid_actions": invalid_actions,
        "max_membrane_norm": max_membrane_norm,
        "max_trace_norm": max_trace_norm,
        "threshold_min": threshold_min,
        "threshold_max": threshold_max,
        "finite_state": finite,
        "final_tick": model.tick,
    }


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> Dict[str, object]:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )
    model = Taiji(config, episode_id="n9-train")
    model.learn_bytes(DATA, epochs=epochs, include_boundary=False)
    learned = model.checkpoint()

    instrumented = _instrumented_free_run(Taiji.from_checkpoint(learned))
    api_generated = Taiji.from_checkpoint(learned).generate(PROMPT, len(EXPECTED))
    generated = instrumented.pop("generated")
    checks = {
        "exact_128_step_cycle": generated == EXPECTED,
        "no_first_error": instrumented["first_error_index"] is None,
        "all_four_actions_remain_present": set(generated) == set(b"abcd"),
        "no_invalid_or_boundary_action": instrumented["invalid_actions"] == 0,
        "membrane_bound_holds": (
            instrumented["max_membrane_norm"] <= config.max_membrane_norm + 1e-6
        ),
        "trace_bound_holds": (
            instrumented["max_trace_norm"] <= config.max_trace_norm + 1e-6
        ),
        "threshold_bounds_hold": (
            instrumented["threshold_min"] >= config.threshold_min - 1e-6
            and instrumented["threshold_max"] <= config.threshold_max + 1e-6
        ),
        "all_state_is_finite": instrumented["finite_state"],
        "public_generate_matches_instrumented_loop": api_generated == generated,
    }
    return {
        "benchmark": "taiji_n9_long_free_run",
        "seed": seed,
        "epochs": epochs,
        "training_stream": DATA.decode("ascii"),
        "training_includes_boundary": False,
        "prompt": PROMPT.decode("ascii"),
        "free_run_steps": len(EXPECTED),
        "metrics": {
            **instrumented,
            "generated_head": generated[:32].decode("ascii"),
            "generated_tail": generated[-32:].decode("ascii"),
            "distinct_actions": sorted(chr(symbol) for symbol in set(generated)),
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
