"""Run the unit-test recall protocol against the isolated HEAD package."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_scratch"))

from taiji_head import Taiji, TaijiConfig  # noqa: E402

CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = (ord("+"), ord("-"))


def main() -> None:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_context_dim=32,
        memory_iterations=3,
        seed=23,
    )
    model = Taiji(config)
    mapping = {
        cue: ACTIONS[index % len(ACTIONS)]
        for index, cue in enumerate(CUES)
    }
    for index, (cue, action) in enumerate(mapping.items()):
        model.reset_dynamics(episode_id=f"store-{index}")
        model.observe(256, learn=False, learn_motor=False)
        model.observe(cue, learn=False, learn_motor=False)
        model.act((action,), sample=False)
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(OUTCOMES[index % len(OUTCOMES)], learn=False, learn_motor=False)

    checkpoint = model.checkpoint()
    correct = 0
    confidences = []
    for index, (cue, expected) in enumerate(mapping.items()):
        restored = Taiji.from_checkpoint(checkpoint)
        restored.reset_dynamics(episode_id=f"recall-{index}")
        restored.observe(256, learn=False, learn_motor=False)
        step = restored.observe(cue, learn=False, learn_motor=False)
        decision = restored.act(ACTIONS, sample=False)
        correct += int(decision.action_symbol == expected)
        confidences.append(step.memory_recall.confidence)
    print(f"HEAD package recalled={correct/len(mapping):.3f} min_conf={min(confidences):.3f}")


if __name__ == "__main__":
    main()
