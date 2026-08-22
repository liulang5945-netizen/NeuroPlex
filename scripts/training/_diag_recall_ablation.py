"""Ablation: unit-test recall protocol vs gate/repeats/binding."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "taiji_native"))

from taiji import Taiji, TaijiConfig  # noqa: E402
from test_episodic_field import (  # noqa: E402
    _record_balanced_one_shot_episodes,
    _recall_accuracy,
)


def run(**overrides) -> float:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
        seed=23,
    )
    for key, value in overrides.items():
        object.__setattr__(config, key, value)
    model = Taiji(config)
    mapping = _record_balanced_one_shot_episodes(model)
    checkpoint = model.checkpoint()
    recalled, confidences = _recall_accuracy(checkpoint, mapping, use_memory=True)
    print(f"{overrides}: recalled={recalled:.3f} min_conf={min(confidences):.3f}")
    return recalled


if __name__ == "__main__":
    run()
    run(memory_identity_gate_sigma=0.0)
    run(memory_identity_gate_sigma=0.25)
    run(memory_identity_gate_sigma=0.5)
    run(memory_identity_gate_sigma=2.0)
    run(episodic_write_repeats=1)
    run(episodic_write_repeats=8)
    run(memory_meta_dim=32)
    run(memory_identity_gate_sigma=0.0, memory_meta_dim=32)
