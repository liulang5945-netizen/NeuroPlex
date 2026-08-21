"""N11: verify reward-modulated action learning in an active environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import EnvironmentOutcome, Taiji, TaijiConfig


CUES = (ord("L"), ord("R"))
ACTIONS = (ord("0"), ord("1"))


class BinaryCueEnvironment:
    """Two-cue world; action changes both reward and next sensation."""

    def __init__(self) -> None:
        self.trial = 0
        self.current_cue: int | None = None

    def reset(self) -> Tuple[int, Sequence[int]]:
        self.current_cue = CUES[self.trial % len(CUES)]
        self.trial += 1
        return self.current_cue, ACTIONS

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        if self.current_cue is None:
            raise RuntimeError("environment must be reset before step")
        correct = ACTIONS[CUES.index(self.current_cue)]
        success = int(action_symbol) == correct
        self.current_cue = None
        return EnvironmentOutcome(
            sensation=ord("+") if success else ord("-"),
            reward=1.0 if success else -1.0,
            terminal=True,
        )


def _config(seed: int) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )


def _interact(
    model: Taiji,
    *,
    interactions: int,
    learn_action: bool,
) -> Dict[str, object]:
    environment = BinaryCueEnvironment()
    initial_motor = (
        model.motor.synapses.edge_weight.clone(),
        model.motor.bias.clone(),
    )
    successes = []
    rewards = []
    modulations = []
    decisions = []
    for trial in range(interactions):
        cue, available = environment.reset()
        model.reset_dynamics(episode_id=f"n11-{trial}")
        model.observe(
            model.config.boundary_symbol, learn=True, learn_motor=False
        )
        model.observe(cue, learn=True, learn_motor=False)
        decision = model.act(available, sample=True)
        outcome = environment.step(decision.action_symbol)
        settled = model.settle_action(
            outcome.reward,
            learn=learn_action,
            learn_memory=False,
        )
        model.observe(outcome.sensation, learn=True, learn_motor=False)

        successes.append(outcome.reward > 0.0)
        rewards.append(outcome.reward)
        modulations.append(settled.reward_prediction_error)
        if trial < 5 or trial >= interactions - 5:
            decisions.append({
                "trial": trial,
                "cue": chr(cue),
                "action": chr(decision.action_symbol),
                "sensation": chr(outcome.sensation),
                "reward": outcome.reward,
                "success": outcome.reward > 0.0,
            })

    motor_changed = (
        not torch.equal(initial_motor[0], model.motor.synapses.edge_weight)
        or not torch.equal(initial_motor[1], model.motor.bias)
    )
    window = min(40, interactions)
    return {
        "first_window_accuracy": sum(successes[:window]) / window,
        "final_window_accuracy": sum(successes[-window:]) / window,
        "overall_accuracy": sum(successes) / interactions,
        "mean_reward": sum(rewards) / interactions,
        "final_reward_baseline": model.motor.reward_baseline,
        "reward_updates": model.motor.reward_updates,
        "mean_abs_reward_prediction_error": (
            sum(abs(value) for value in modulations) / interactions
        ),
        "motor_changed": motor_changed,
        "pending_action_cleared": model.snapshot().pending_action is None,
        "pending_experience_cleared": (
            model.snapshot().pending_experience is None
        ),
        "decision_samples": decisions,
    }


def _deterministic_accuracy(model: Taiji) -> tuple[float, list[Dict[str, object]]]:
    environment = BinaryCueEnvironment()
    rows = []
    for cue in CUES:
        environment.current_cue = cue
        model.reset_dynamics(episode_id=f"n11-eval-{cue}")
        model.observe(model.config.boundary_symbol, learn=False)
        model.observe(cue, learn=False)
        decision = model.act(ACTIONS, sample=False)
        outcome = environment.step(decision.action_symbol)
        model.settle_action(
            outcome.reward,
            learn=False,
            learn_memory=False,
        )
        model.observe(
            outcome.sensation,
            learn=False,
            learn_motor=False,
            use_memory=False,
        )
        rows.append({
            "cue": chr(cue),
            "action": chr(decision.action_symbol),
            "reward": outcome.reward,
            "success": outcome.reward > 0.0,
        })
    return sum(bool(row["success"]) for row in rows) / len(rows), rows


def _actions_change_sensation() -> bool:
    environment = BinaryCueEnvironment()
    for cue in CUES:
        outcomes = []
        for action in ACTIONS:
            environment.current_cue = cue
            outcomes.append(environment.step(action))
        if outcomes[0].sensation == outcomes[1].sensation:
            return False
        if outcomes[0].reward == outcomes[1].reward:
            return False
    return True


def run_benchmark(*, interactions: int = 200, seed: int = 7) -> Dict[str, object]:
    learned_model = Taiji(_config(seed), episode_id="n11-learned")
    lesion_model = Taiji(_config(seed), episode_id="n11-lesion")
    learned = _interact(
        learned_model, interactions=interactions, learn_action=True
    )
    lesion = _interact(
        lesion_model, interactions=interactions, learn_action=False
    )
    deterministic, deterministic_rows = _deterministic_accuracy(learned_model)
    random_baseline = 1.0 / len(ACTIONS)
    checks = {
        "environment_transition_depends_on_action": _actions_change_sensation(),
        "final_success_at_least_90pct": learned["final_window_accuracy"] >= 0.90,
        "beats_random_by_35pp": (
            learned["final_window_accuracy"] >= random_baseline + 0.35
        ),
        "beats_action_learning_lesion_by_25pp": (
            learned["final_window_accuracy"]
            >= lesion["final_window_accuracy"] + 0.25
        ),
        "deterministic_policy_solves_both_cues": deterministic == 1.0,
        "reward_updates_are_local_and_counted": (
            learned["reward_updates"] == interactions
            and lesion["reward_updates"] == 0
            and learned["motor_changed"]
            and not lesion["motor_changed"]
        ),
        "all_pending_actions_are_settled": (
            learned["pending_action_cleared"]
            and lesion["pending_action_cleared"]
            and learned["pending_experience_cleared"]
            and lesion["pending_experience_cleared"]
        ),
    }
    return {
        "benchmark": "taiji_n11_active_environment",
        "seed": seed,
        "interactions": interactions,
        "environment": {
            "cues": [chr(value) for value in CUES],
            "actions": [chr(value) for value in ACTIONS],
            "success_sensation": "+",
            "failure_sensation": "-",
            "reward": {"success": 1.0, "failure": -1.0},
            "teacher_action_label_exposed_to_taiji": False,
            "episodic_memory_learning_enabled": False,
            "reason": "N11 isolates motor reward causality; M5 tests memory separately",
        },
        "metrics": {
            "learned": learned,
            "action_learning_lesion": lesion,
            "random_policy_accuracy": random_baseline,
            "gain_over_random": learned["final_window_accuracy"] - random_baseline,
            "gain_over_action_lesion": (
                learned["final_window_accuracy"]
                - lesion["final_window_accuracy"]
            ),
            "deterministic_accuracy": deterministic,
            "deterministic_decisions": deterministic_rows,
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(interactions=args.interactions, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
