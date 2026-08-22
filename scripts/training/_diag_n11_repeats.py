"""Check N11 active-environment collapse vs episodic_write_repeats."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "taiji_native"))

from taiji import Taiji, TaijiConfig  # noqa: E402
from test_active_environment import (  # noqa: E402
    ACTIONS,
    CUES,
    BinaryCueEnvironment,
)


def run(repeats: int) -> None:
    model = Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        episodic_write_repeats=repeats,
        seed=7,
    ))
    environment = BinaryCueEnvironment()
    successes = []
    for trial in range(200):
        cue = CUES[trial % len(CUES)]
        model.reset_dynamics(episode_id=f"n11-{trial}")
        model.observe(model.config.boundary_symbol, learn=True, learn_motor=False)
        model.observe(cue, learn=True, learn_motor=False)
        action = model.act(ACTIONS, sample=True).action_symbol
        outcome = environment.outcome(cue, action)
        model.settle_action(outcome.reward, learn=True)
        model.observe(outcome.sensation, learn=True, learn_motor=False)
        successes.append(outcome.reward > 0.0)
    print(f"repeats={repeats}: final={sum(successes[-40:]) / 40:.3f}")


if __name__ == "__main__":
    run(1)
    run(2)
    run(4)
