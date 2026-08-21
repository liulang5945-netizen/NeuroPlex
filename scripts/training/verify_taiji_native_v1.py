"""Verify the first complete native Taiji byte-stream architecture."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import taiji
from taiji import Taiji, TaijiConfig


def _native_import_contract() -> bool:
    package = Path(taiji.__file__).resolve().parent
    imported = set()
    attributes = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Attribute):
                attributes.add(node.attr)
    return (
        not any(module.startswith(("neuroplex", "transformers")) for module in imported)
        and not {"backward", "MultiheadAttention", "TransformerEncoder"} & attributes
    )


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> Dict[str, object]:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )
    data = b"abcdabcdabcdabcd"
    model = Taiji(config, episode_id="native-v1")
    initial_parameters = [tensor.clone() for tensor in model.parameter_tensors()]
    before = model.score_bytes(data)

    started = time.perf_counter()
    training = model.learn_bytes(data, epochs=epochs)
    elapsed = time.perf_counter() - started
    after = model.score_bytes(data)
    generated = model.generate(b"a", 8)
    changed_tensors = sum(
        not torch.equal(before_tensor, after_tensor)
        for before_tensor, after_tensor in zip(
            initial_parameters, model.parameter_tensors()
        )
    )

    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(checkpoint)
    next_left = model.observe(ord("!"), learn=True)
    next_right = restored.observe(ord("!"), learn=True)
    exact_next_step = (
        next_left.predicted_symbol == next_right.predicted_symbol
        and torch.equal(next_left.probabilities, next_right.probabilities)
        and all(
            torch.equal(left, right)
            for left, right in zip(
                model.parameter_tensors(), restored.parameter_tensors()
            )
        )
    )

    active_parameters = model.parameter_count(active_only=True)
    dense_storage = model.parameter_count(active_only=False)
    checks = {
        "native_namespace": Path(taiji.__file__).resolve().parent.name == "taiji",
        "no_legacy_or_transformer_dependency": _native_import_contract(),
        "no_autograd_parameters": all(
            tensor.requires_grad is False for tensor in model.parameter_tensors()
        ),
        "online_learning_reduces_surprise": (
            after["mean_surprise"] <= before["mean_surprise"] * 0.30
        ),
        "teacher_forced_accuracy": after["accuracy"] >= 0.75,
        "free_generation_prefix": generated.startswith(b"bcda"),
        "local_parameters_changed": changed_tensors >= 3,
        "checkpoint_exact_next_step": exact_next_step,
    }
    return {
        "benchmark": "taiji_native_v1",
        "seed": seed,
        "epochs": epochs,
        "architecture": {
            "sensor": "raw-byte-one-hot",
            "regions": list(config.region_sizes),
            "learning": "local-predictive-and-motor-delta",
            "sequence_window": None,
            "active_parameters": active_parameters,
            "dense_tensor_storage": dense_storage,
            "structural_sparsity": 1.0 - active_parameters / dense_storage,
        },
        "metrics": {
            "before": before,
            "training": training,
            "after": after,
            "surprise_reduction": (
                1.0 - after["mean_surprise"] / before["mean_surprise"]
            ),
            "generated_hex": generated.hex(),
            "generated_text": generated.decode("latin-1"),
            "changed_parameter_tensors": changed_tensors,
            "training_seconds": elapsed,
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
