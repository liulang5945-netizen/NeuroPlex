"""M5: falsify native distributed episodic-field memory and its lesions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Dict, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig


CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = tuple(ord(value) for value in "+-!?")
PROVENANCE = ("experienced", "imagined", "replayed", "external")


def _config(seed: int) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_context_dim=32,
        memory_iterations=3,
        seed=seed,
    )


def _episodes() -> Dict[int, Dict[str, object]]:
    return {
        cue: {
            "action": ACTIONS[index % len(ACTIONS)],
            "outcome": OUTCOMES[index % len(OUTCOMES)],
            "provenance": PROVENANCE[index % len(PROVENANCE)],
            "episode_id": f"m5-store-{index}",
            "prefix_length": index,
            "tick": index + 2,
        }
        for index, cue in enumerate(CUES)
    }


def _present_cue(
    model: Taiji,
    cue: int,
    prefix_length: int,
    *,
    use_memory: bool,
) -> object:
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        learn_motor=False,
        use_memory=use_memory,
    )
    for _ in range(prefix_length):
        model.observe(
            ord("."),
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
    return model.observe(
        cue,
        learn=False,
        learn_motor=False,
        use_memory=use_memory,
    )


def _store(model: Taiji, episodes: Mapping[int, Mapping[str, object]]) -> None:
    for cue, event in episodes.items():
        model.reset_dynamics(episode_id=str(event["episode_id"]))
        _present_cue(
            model,
            cue,
            int(event["prefix_length"]),
            use_memory=True,
        )
        action = int(event["action"])
        model.act((action,), sample=False)
        model.settle_action(
            1.0,
            learn=False,
            learn_memory=True,
            provenance=str(event["provenance"]),
        )
        model.observe(
            int(event["outcome"]),
            learn=False,
            learn_motor=False,
        )


def _evaluate(
    checkpoint: Mapping[str, object],
    episodes: Mapping[int, Mapping[str, object]],
    *,
    use_memory: bool,
) -> Dict[str, object]:
    rows = []
    action_correct = 0
    outcome_correct = 0
    provenance_correct = 0
    episode_correct = 0
    time_cosines = []
    for index, (cue, event) in enumerate(episodes.items()):
        model = Taiji.from_checkpoint(checkpoint)
        model.reset_dynamics(episode_id=f"m5-query-{index}")
        step = _present_cue(
            model,
            cue,
            int(event["prefix_length"]),
            use_memory=use_memory,
        )
        recall = step.memory_recall
        decision = model.act(ACTIONS, sample=False)
        expected_action = int(event["action"])
        expected_outcome = int(event["outcome"])
        expected_provenance = PROVENANCE.index(str(event["provenance"]))
        action_correct += int(decision.action_symbol == expected_action)
        outcome_correct += int(
            int(recall.outcome_probabilities.argmax().item()) == expected_outcome
        )
        provenance_correct += int(
            int(recall.provenance_probabilities.argmax().item())
            == expected_provenance
        )

        episode_candidates = torch.stack([
            model.memory._episode_code(str(candidate["episode_id"]))
            for candidate in episodes.values()
        ])
        episode_scores = torch.nn.functional.cosine_similarity(
            recall.episode_code.unsqueeze(0), episode_candidates, dim=1
        )
        episode_prediction = int(episode_scores.argmax().item())
        episode_correct += int(episode_prediction == index)
        expected_time = model.memory._time_code(int(event["tick"]))
        time_cosine = float(torch.nn.functional.cosine_similarity(
            recall.time_code, expected_time, dim=0
        ).item()) if recall.time_code.norm() > 0 else 0.0
        time_cosines.append(time_cosine)
        rows.append({
            "cue": chr(cue),
            "expected_action": chr(expected_action),
            "action": chr(decision.action_symbol),
            "expected_outcome": chr(expected_outcome),
            "recalled_outcome": int(recall.outcome_probabilities.argmax().item()),
            "expected_provenance": str(event["provenance"]),
            "recalled_provenance": PROVENANCE[
                int(recall.provenance_probabilities.argmax().item())
            ],
            "episode_prediction": episode_prediction,
            "confidence": recall.confidence,
            "expected_reward": recall.expected_reward,
            "feedback_norm": float(recall.cortical_feedback.norm().item()),
            "time_cosine": time_cosine,
        })
    count = len(episodes)
    return {
        "action_accuracy": action_correct / count,
        "outcome_accuracy": outcome_correct / count,
        "provenance_accuracy": provenance_correct / count,
        "episode_identity_accuracy": episode_correct / count,
        "mean_time_code_cosine": sum(time_cosines) / count,
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / count,
        "mean_feedback_norm": sum(float(row["feedback_norm"]) for row in rows) / count,
        "rows": rows,
    }


def _feedback_is_causal(
    checkpoint: Mapping[str, object],
    episodes: Mapping[int, Mapping[str, object]],
) -> bool:
    cue, event = next(iter(episodes.items()))
    full = Taiji.from_checkpoint(checkpoint)
    lesion = Taiji.from_checkpoint(checkpoint)
    for model, enabled in ((full, True), (lesion, False)):
        model.reset_dynamics(episode_id=f"m5-feedback-{enabled}")
        _present_cue(
            model,
            cue,
            int(event["prefix_length"]),
            use_memory=enabled,
        )
    full.observe(ord("~"), learn=False, learn_motor=False, use_memory=False)
    lesion.observe(ord("~"), learn=False, learn_motor=False, use_memory=False)
    return not torch.equal(
        full.snapshot().regions[0].membrane,
        lesion.snapshot().regions[0].membrane,
    )


def _checkpoint_transaction_is_exact(
    checkpoint: Mapping[str, object],
) -> bool:
    original = Taiji.from_checkpoint(checkpoint)
    original.reset_dynamics(episode_id="m5-transaction")
    _present_cue(original, ord("Z"), 2, use_memory=True)
    original.act((ord("1"),), sample=False)
    original.settle_action(
        1.0,
        learn=False,
        learn_memory=True,
        provenance="external",
    )
    restored = Taiji.from_checkpoint(original.checkpoint())
    left = original.observe(ord("!"), learn=False, learn_motor=False)
    right = restored.observe(ord("!"), learn=False, learn_motor=False)
    return (
        left.memory_write_strength == right.memory_write_strength
        and left.memory_write_strength > 0.0
        and all(
            torch.equal(a, b)
            for a, b in zip(
                original.parameter_tensors(), restored.parameter_tensors()
            )
        )
    )


def run_benchmark(*, seed: int = 23) -> Dict[str, object]:
    model = Taiji(_config(seed), episode_id="m5-bootstrap")
    episodes = _episodes()
    topology_before = model.memory.association.pre_index.clone()
    edge_count_before = model.memory.association.edge_count
    _store(model, episodes)
    checkpoint = model.checkpoint()
    full = _evaluate(checkpoint, episodes, use_memory=True)
    trace_only = _evaluate(checkpoint, episodes, use_memory=False)
    recurrent_lesion_checkpoint = deepcopy(checkpoint)
    recurrent_lesion_checkpoint["memory"]["association"][
        "edge_weight"
    ].zero_()
    recurrent_lesion = _evaluate(
        recurrent_lesion_checkpoint,
        episodes,
        use_memory=True,
    )

    state = model.snapshot().memory
    dynamic_scalars = (
        state.activity.numel()
        + state.trace.numel()
        + state.threshold.numel()
        + state.cortical_feedback.numel()
    )
    no_slots = not {"events", "keys", "values", "slots"} & (
        set(vars(model.memory)) | set(model.memory.to_payload())
    )
    checks = {
        "one_shot_action_recall_at_least_87_5pct": (
            full["action_accuracy"] >= 0.875
        ),
        "beats_equal_width_trace_only_by_37_5pp": (
            full["action_accuracy"] >= trace_only["action_accuracy"] + 0.375
        ),
        "recurrent_completion_is_causally_necessary": (
            full["action_accuracy"]
            >= recurrent_lesion["action_accuracy"] + 0.25
        ),
        "recalls_outcome_and_provenance": (
            full["outcome_accuracy"] >= 0.75
            and full["provenance_accuracy"] >= 0.75
        ),
        "recalls_time_and_episode_codes": (
            full["mean_time_code_cosine"] >= 0.50
            and full["episode_identity_accuracy"] >= 0.75
        ),
        "recalled_state_feeds_next_fabric_tick": _feedback_is_causal(
            checkpoint, episodes
        ),
        "action_outcome_transaction_checkpoint_exact": (
            _checkpoint_transaction_is_exact(checkpoint)
        ),
        "fixed_topology_no_event_slots": (
            no_slots
            and model.memory.association.edge_count == edge_count_before
            and torch.equal(
                model.memory.association.pre_index, topology_before
            )
        ),
    }
    return {
        "benchmark": "taiji_m5_native_episodic_field",
        "seed": seed,
        "episodes": len(episodes),
        "protocol": {
            "write_exposures_per_episode": 1,
            "write_action_affordance_count": 1,
            "query_action_affordance_count": len(ACTIONS),
            "reward_per_write": 1.0,
            "fabric_learning_enabled": False,
            "motor_learning_enabled": False,
            "claim_boundary": "one-shot associative recall, not action discovery",
        },
        "architecture": {
            "checkpoint_format": model.CHECKPOINT_FORMAT,
            "state_version": model.STATE_VERSION,
            "memory_units": model.config.memory_units,
            "memory_context_dim": model.config.memory_context_dim,
            "memory_iterations": model.config.memory_iterations,
            "memory_association_edges": model.memory.association.edge_count,
            "event_slots_allocated": 0,
            "trace_only_dynamic_scalars": dynamic_scalars,
            "full_field_dynamic_scalars": dynamic_scalars,
            "stored_event_count_metadata": model.memory.write_count,
        },
        "metrics": {
            "full_field": full,
            "equal_width_trace_only": trace_only,
            "recurrent_association_lesion": recurrent_lesion,
            "action_gain_over_trace_only": (
                full["action_accuracy"] - trace_only["action_accuracy"]
            ),
            "action_gain_over_recurrent_lesion": (
                full["action_accuracy"]
                - recurrent_lesion["action_accuracy"]
            ),
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
