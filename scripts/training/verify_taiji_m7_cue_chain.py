"""M7: falsify cue-conditioned replay transfer into cortical action choice."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    FILLER,
    PROVENANCE,
    _config,
    _sleep,
)


CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = tuple(ord(value) for value in "01")
OUTCOMES = tuple(ord(value) for value in "+-")


def _episodes() -> Dict[int, Dict[str, object]]:
    return {
        cue: {
            "action": ACTIONS[index % len(ACTIONS)],
            "outcome": OUTCOMES[index % len(OUTCOMES)],
            "provenance": PROVENANCE[index % len(PROVENANCE)],
            "episode_id": f"m7-store-{index}",
            "prefix_length": 0,
        }
        for index, cue in enumerate(CUES)
    }


def _pretrain_corpus() -> bytes:
    """Uniform cue/action/outcome marginals with no conditional relation."""

    return bytes((FILLER,)).join(
        bytes((cue, action, outcome))
        for cue in CUES
        for action in ACTIONS
        for outcome in OUTCOMES
    )


def _present_cue(
    model: Taiji,
    cue: int,
    prefix_length: int,
    *,
    use_memory: bool,
) -> object:
    step = model.observe(
        model.config.boundary_symbol,
        learn=False,
        learn_motor=False,
        use_memory=use_memory,
    )
    for _ in range(prefix_length):
        step = model.observe(
            FILLER,
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
    """Write each demonstrated causal chain once; cortex and motor stay frozen."""

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


def _restricted_metrics(
    evidence: torch.Tensor,
    candidates: Sequence[int],
    expected: int,
) -> Dict[str, object]:
    selector = torch.tensor(tuple(candidates), dtype=torch.long)
    probabilities = torch.softmax(evidence[selector], dim=0)
    target = tuple(candidates).index(int(expected))
    rival = 1 - target
    prediction = int(candidates[int(probabilities.argmax().item())])
    return {
        "prediction": prediction,
        "true_probability": float(probabilities[target].item()),
        "margin": float((probabilities[target] - probabilities[rival]).item()),
    }


def _evaluate_cue_actions(
    checkpoint: Mapping[str, object],
    episodes: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    rows = []
    behavior_correct = 0
    cortical_correct = 0
    for cue, event in episodes.items():
        model = Taiji.from_checkpoint(deepcopy(checkpoint))
        model.reset_dynamics(episode_id=f"m7-cue-probe-{cue}")
        step = _present_cue(
            model,
            cue,
            int(event["prefix_length"]),
            use_memory=False,
        )
        expected = int(event["action"])
        decision = model.act(ACTIONS, sample=False)
        snapshot = model.snapshot()
        trace = snapshot.regions[0].trace
        cortical_evidence = model.fabric.consolidated_decode(0, trace)
        cortical = _restricted_metrics(
            cortical_evidence,
            ACTIONS,
            expected,
        )
        fast_evidence = (
            model.motor.synapses.forward(snapshot.motor_context)
            + model.motor.bias
        )
        fast = _restricted_metrics(fast_evidence, ACTIONS, expected)
        behavior_correct += int(decision.action_symbol == expected)
        cortical_correct += int(cortical["prediction"] == expected)
        rows.append({
            "cue": chr(cue),
            "expected_action": chr(expected),
            "behavior_action": chr(decision.action_symbol),
            "cortical_action": chr(int(cortical["prediction"])),
            "cortical_margin": cortical["margin"],
            "cortical_evidence_norm": float(cortical_evidence.norm().item()),
            "fast_margin": fast["margin"],
            "fast_evidence_norm": float(fast_evidence.norm().item()),
            "episodic_confidence": step.memory_recall.confidence,
            "episodic_feedback_norm": float(
                step.memory_recall.cortical_feedback.norm().item()
            ),
        })
    count = len(rows)
    return {
        "behavior_accuracy": behavior_correct / count,
        "cortical_accuracy": cortical_correct / count,
        "mean_cortical_margin": sum(
            float(row["cortical_margin"]) for row in rows
        )
        / count,
        "rows": rows,
    }


def _evaluate_action_outcomes(
    checkpoint: Mapping[str, object],
    episodes: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    pairs = {
        int(event["action"]): int(event["outcome"])
        for event in episodes.values()
    }
    rows = []
    correct = 0
    for action, outcome in sorted(pairs.items()):
        model = Taiji.from_checkpoint(deepcopy(checkpoint))
        model.reset_dynamics(episode_id=f"m7-outcome-probe-{action}")
        for _ in range(int(model.config.replay_burst_repeats)):
            model.observe(
                action,
                learn=False,
                learn_motor=False,
                use_memory=False,
            )
        trace = model.snapshot().regions[0].trace
        metrics = _restricted_metrics(
            model.fabric.consolidated_decode(0, trace),
            OUTCOMES,
            outcome,
        )
        correct += int(metrics["prediction"] == outcome)
        rows.append({
            "action": chr(action),
            "expected_outcome": chr(outcome),
            "predicted_outcome": chr(int(metrics["prediction"])),
            "margin": metrics["margin"],
        })
    return {"accuracy": correct / len(rows), "rows": rows}


def _readback_closed(rows: Sequence[Mapping[str, object]]) -> bool:
    return all(
        float(row["episodic_confidence"]) == 0.0
        and float(row["episodic_feedback_norm"]) == 0.0
        for row in rows
    )


def _cue_probes(
    checkpoint: Mapping[str, object],
    episodes: Mapping[int, Mapping[str, object]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    probes = {}
    for cue, event in episodes.items():
        cortical_model = Taiji.from_checkpoint(deepcopy(checkpoint))
        cortical_model.reset_dynamics(episode_id=f"m7-cortical-probe-{cue}")
        _present_cue(
            cortical_model,
            cue,
            int(event["prefix_length"]),
            use_memory=False,
        )
        cortical_state = cortical_model.snapshot()

        memory_model = Taiji.from_checkpoint(deepcopy(checkpoint))
        memory_model.reset_dynamics(episode_id=f"m7-memory-probe-{cue}")
        memory_step = _present_cue(
            memory_model,
            cue,
            int(event["prefix_length"]),
            use_memory=True,
        )
        probes[cue] = {
            "region_trace": cortical_state.regions[0].trace.detach().clone(),
            "region_opponent_trace": cortical_model.fabric.opponent_trace(
                0, cortical_state.regions[0].trace
            ).detach().clone(),
            "cortical_context": cortical_model.fabric.cortical_context(
                cortical_state.regions
            ).detach().clone(),
            "memory_activity": memory_model.snapshot().memory.activity.detach().clone(),
            "memory_action": torch.tensor(
                int(memory_step.memory_recall.action_probabilities.argmax().item())
            ),
            "memory_confidence": torch.tensor(memory_step.memory_recall.confidence),
            "memory_cortical_projection": (
                memory_step.memory_recall.cortical_feedback.detach().clone()
            ),
            "episode_code": memory_model.memory._episode_code(
                str(event["episode_id"])
            ).detach().clone(),
        }
    return probes


@contextmanager
def _capture_replays():
    original = EpisodicField.replay
    records = []

    def replay(self, previous, *, tick, generator):
        next_state, event = original(
            self, previous, tick=tick, generator=generator
        )
        if event.accepted:
            records.append({
                "action": int(event.action_probabilities.argmax().item()),
                "pattern": event.pattern.detach().cpu().clone(),
                "projection": event.cortical_projection.detach().cpu().clone(),
                "reciprocal_projection": self.cue_encoder.backproject(
                    event.pattern
                ).detach().cpu().clone(),
                "episode_code": event.episode_code.detach().cpu().clone(),
                "confidence": float(event.familiarity * event.resonance),
            })
        return next_state, event

    EpisodicField.replay = replay
    try:
        yield records
    finally:
        EpisodicField.replay = original


def _nearest(
    value: torch.Tensor,
    probes: Mapping[int, Mapping[str, torch.Tensor]],
    key: str,
) -> int:
    return max(
        probes,
        key=lambda cue: float(
            torch.nn.functional.cosine_similarity(
                value, probes[cue][key], dim=0
            ).item()
        ),
    )


def _wake_jointness(
    probes: Mapping[int, Mapping[str, torch.Tensor]],
    episodes: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    action_matches = 0
    projection_matches = 0
    projection_counts = Counter()
    for cue, probe in probes.items():
        action = int(probe["memory_action"].item())
        action_matches += int(action == int(episodes[cue]["action"]))
        confidence = max(1e-8, float(probe["memory_confidence"].item()))
        nearest = _nearest(
            probe["memory_cortical_projection"] / confidence,
            probes,
            "cortical_context",
        )
        projection_counts[chr(nearest)] += 1
        projection_matches += int(nearest == cue)
    count = len(probes)
    return {
        "cue_recall_action_accuracy": action_matches / count,
        "cue_recall_cortical_identity_accuracy": projection_matches / count,
        "cue_recall_cortical_identity_counts": dict(sorted(projection_counts.items())),
    }


def _replay_jointness(
    records: Sequence[Mapping[str, object]],
    probes: Mapping[int, Mapping[str, torch.Tensor]],
    episodes: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    action_counts = Counter()
    pattern_counts = Counter()
    cortical_counts = Counter()
    reciprocal_counts = Counter()
    reinstated_trace_counts = Counter()
    reinstated_opponent_counts = Counter()
    episode_counts = Counter()
    pattern_action_matches = 0
    cortical_action_matches = 0
    reciprocal_action_matches = 0
    reinstated_trace_action_matches = 0
    reinstated_opponent_action_matches = 0
    episode_action_matches = 0
    pattern_cortical_matches = 0
    for record in records:
        action = int(record["action"])
        action_counts[chr(action)] += 1
        pattern_cue = _nearest(record["pattern"], probes, "memory_activity")
        confidence = max(1e-8, float(record["confidence"]))
        projection = record["projection"] / confidence
        cortical_cue = _nearest(projection, probes, "cortical_context")
        trace_offset = projection.numel() // 2
        region_size = next(iter(probes.values()))["region_trace"].numel()
        reinstated_trace_cue = _nearest(
            projection[trace_offset:trace_offset + region_size],
            probes,
            "region_trace",
        )
        first_probe = next(iter(probes.values()))
        baseline = first_probe["region_trace"] - first_probe["region_opponent_trace"]
        reinstated_opponent_cue = _nearest(
            projection[trace_offset:trace_offset + region_size] - baseline,
            probes,
            "region_opponent_trace",
        )
        reciprocal_cue = _nearest(
            record["reciprocal_projection"],
            probes,
            "cortical_context",
        )
        episode_cue = _nearest(
            record["episode_code"] / confidence,
            probes,
            "episode_code",
        )
        pattern_counts[chr(pattern_cue)] += 1
        cortical_counts[chr(cortical_cue)] += 1
        reciprocal_counts[chr(reciprocal_cue)] += 1
        reinstated_trace_counts[chr(reinstated_trace_cue)] += 1
        reinstated_opponent_counts[chr(reinstated_opponent_cue)] += 1
        episode_counts[chr(episode_cue)] += 1
        pattern_action_matches += int(
            action == int(episodes[pattern_cue]["action"])
        )
        cortical_action_matches += int(
            action == int(episodes[cortical_cue]["action"])
        )
        reciprocal_action_matches += int(
            action == int(episodes[reciprocal_cue]["action"])
        )
        reinstated_trace_action_matches += int(
            action == int(episodes[reinstated_trace_cue]["action"])
        )
        reinstated_opponent_action_matches += int(
            action == int(episodes[reinstated_opponent_cue]["action"])
        )
        episode_action_matches += int(
            action == int(episodes[episode_cue]["action"])
        )
        pattern_cortical_matches += int(pattern_cue == cortical_cue)
    count = max(1, len(records))
    return {
        "accepted_replays": len(records),
        "action_mode_counts": dict(sorted(action_counts.items())),
        "nearest_pattern_cue_counts": dict(sorted(pattern_counts.items())),
        "nearest_cortical_cue_counts": dict(sorted(cortical_counts.items())),
        "nearest_reciprocal_cue_counts": dict(sorted(reciprocal_counts.items())),
        "nearest_reinstated_trace_cue_counts": dict(
            sorted(reinstated_trace_counts.items())
        ),
        "nearest_reinstated_opponent_cue_counts": dict(
            sorted(reinstated_opponent_counts.items())
        ),
        "nearest_episode_cue_counts": dict(sorted(episode_counts.items())),
        "action_matches_pattern_cue_rate": pattern_action_matches / count,
        "action_matches_cortical_cue_rate": cortical_action_matches / count,
        "action_matches_reciprocal_cue_rate": reciprocal_action_matches / count,
        "action_matches_reinstated_trace_cue_rate": (
            reinstated_trace_action_matches / count
        ),
        "action_matches_reinstated_opponent_cue_rate": (
            reinstated_opponent_action_matches / count
        ),
        "action_matches_episode_cue_rate": episode_action_matches / count,
        "pattern_matches_cortical_cue_rate": pattern_cortical_matches / count,
    }


def run_benchmark(*, seed: int = 29, cycles: int = 96) -> Dict[str, object]:
    model = Taiji(_config(seed), episode_id="m7-bootstrap")
    pretrain = model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    stored = model.checkpoint()

    probes = _cue_probes(stored, episodes)
    with _capture_replays() as replay_records:
        full = _sleep(stored, cycles=cycles, learn=True, tag="m7-full")
    control = _sleep(stored, cycles=cycles, learn=False, tag="m7-control")
    content = _sleep(
        stored,
        cycles=cycles,
        learn=True,
        lesion=("action_readout", "outcome_readout", "cortical_readout"),
        tag="m7-content-lesion",
    )
    order = _sleep(
        stored,
        cycles=cycles,
        learn=True,
        tag="m7-order-lesion",
        replay_cue_chain=False,
    )

    full_cues = _evaluate_cue_actions(full["checkpoint"], episodes)
    control_cues = _evaluate_cue_actions(control["checkpoint"], episodes)
    content_cues = _evaluate_cue_actions(content["checkpoint"], episodes)
    order_cues = _evaluate_cue_actions(order["checkpoint"], episodes)
    outcome_leg = _evaluate_action_outcomes(full["checkpoint"], episodes)
    chance = 1.0 / len(ACTIONS)
    checks = {
        "m6_outcome_leg_is_preserved": outcome_leg["accuracy"] == 1.0,
        "cue_action_behavior_above_chance": (
            full_cues["behavior_accuracy"] > chance
        ),
        "cue_action_behavior_beats_no_replay": (
            full_cues["behavior_accuracy"]
            > control_cues["behavior_accuracy"]
        ),
        "cue_action_is_present_in_slow_cortex": (
            full_cues["cortical_accuracy"] > chance
            and full_cues["mean_cortical_margin"] > 0.0
        ),
        "engram_content_is_causally_necessary": (
            full_cues["behavior_accuracy"] > content_cues["behavior_accuracy"]
        ),
        "cue_before_action_order_is_causally_necessary": (
            full_cues["behavior_accuracy"] > order_cues["behavior_accuracy"]
            and full_cues["cortical_accuracy"] > order_cues["cortical_accuracy"]
        ),
        "evaluation_has_no_episodic_readback": (
            _readback_closed(full_cues["rows"])
            and _readback_closed(control_cues["rows"])
        ),
    }
    return {
        "benchmark": "taiji-m7-cue-conditioned-chain",
        "status": "pass" if all(checks.values()) else "fail",
        "seed": seed,
        "cycles": cycles,
        "episodes": len(episodes),
        "protocol": {
            "cue_count": len(CUES),
            "action_count": len(ACTIONS),
            "outcome_count": len(OUTCOMES),
            "chance_accuracy": chance,
            "one_shot_writes": True,
            "uniform_pretraining_marginals": True,
            "external_replay_list": False,
            "teacher_action_during_sleep": False,
            "episodic_readback_during_evaluation": False,
            "pretrain_online_accuracy": pretrain["online_accuracy"],
            "claim_boundary": (
                "cue-conditioned action and action-conditioned outcome after "
                "episodic lesion; not planning or open-world policy learning"
            ),
        },
        "metrics": {
            "full_cue_action": full_cues,
            "no_replay_cue_action": control_cues,
            "content_lesion_cue_action": content_cues,
            "order_lesion_cue_action": order_cues,
            "full_action_outcome": outcome_leg,
            "wake_jointness": _wake_jointness(probes, episodes),
            "replay_jointness": _replay_jointness(
                replay_records,
                probes,
                episodes,
            ),
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cycles", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m7_baseline_20260822.json",
    )
    args = parser.parse_args()
    report = run_benchmark(seed=args.seed, cycles=args.cycles)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
