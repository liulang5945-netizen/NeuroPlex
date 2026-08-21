"""M6: falsify endogenous replay consolidating field content into the cortex."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig


CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = tuple(ord(value) for value in "0123")
OUTCOMES = tuple(ord(value) for value in "+-!?")
FILLER = ord(".")
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
    """Eight one-shot episodes carrying a deterministic action to outcome map."""

    return {
        cue: {
            "action": ACTIONS[index % len(ACTIONS)],
            "outcome": OUTCOMES[index % len(OUTCOMES)],
            "provenance": "experienced",
            "episode_id": f"m6-store-{index}",
            "prefix_length": index,
        }
        for index, cue in enumerate(CUES)
    }


def _contingency(episodes: Mapping[int, Mapping[str, object]]) -> Dict[int, int]:
    pairs: Dict[int, int] = {}
    for event in episodes.values():
        pairs[int(event["action"])] = int(event["outcome"])
    return pairs


def _pretrain_corpus() -> bytes:
    """Every cue/action/outcome triple exactly once: marginals only, no pairing."""

    return bytes((FILLER,)).join(
        bytes((cue, action, outcome))
        for cue in CUES
        for action in ACTIONS
        for outcome in OUTCOMES
    )


def _present_cue(model: Taiji, cue: int, prefix_length: int) -> None:
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    for _ in range(prefix_length):
        model.observe(FILLER, learn=False, learn_motor=False)
    model.observe(cue, learn=False, learn_motor=False)


def _store(model: Taiji, episodes: Mapping[int, Mapping[str, object]]) -> None:
    """One-shot writes only: the cortex and motor are frozen throughout."""

    for cue, event in episodes.items():
        model.reset_dynamics(episode_id=str(event["episode_id"]))
        _present_cue(model, cue, int(event["prefix_length"]))
        model.act((int(event["action"]),), sample=False)
        model.settle_action(
            1.0,
            learn=False,
            learn_memory=True,
            provenance=str(event["provenance"]),
        )
        model.observe(int(event["outcome"]), learn=False, learn_motor=False)


def _sleep(
    checkpoint: Mapping[str, object],
    *,
    cycles: int,
    learn: bool,
    lesion: Sequence[str] = (),
    tag: str,
) -> Dict[str, object]:
    """Run one consolidation arm and audit what it was allowed to touch."""

    payload = deepcopy(checkpoint)
    for name in lesion:
        payload["memory"][name]["edge_weight"].zero_()
    model = Taiji.from_checkpoint(payload)
    model.reset_dynamics(episode_id=f"m6-sleep-{tag}")

    fabric_count = len(model.fabric.parameter_tensors())
    before = tuple(tensor.clone() for tensor in model.parameter_tensors())
    write_count_before = model.memory.write_count
    topology_before = model.memory.association.pre_index.clone()
    edge_count_before = model.memory.association.edge_count

    summary = model.consolidate(cycles=cycles, learn=learn)

    after = model.parameter_tensors()
    cortex_changed = any(
        not torch.equal(left, right)
        for left, right in zip(before[:fabric_count], after[:fabric_count])
    )
    non_cortex_intact = all(
        torch.equal(left, right)
        for left, right in zip(before[fabric_count:], after[fabric_count:])
    )
    return {
        "checkpoint": model.checkpoint(),
        "cortex_changed": cortex_changed,
        "non_cortex_intact": non_cortex_intact,
        "field_write_count_stable": model.memory.write_count == write_count_before,
        "field_topology_stable": (
            model.memory.association.edge_count == edge_count_before
            and torch.equal(model.memory.association.pre_index, topology_before)
        ),
        "summary": {
            "cycles": summary.cycles,
            "accepted": summary.accepted,
            "acceptance_rate": summary.accepted / max(1, summary.cycles),
            "mean_priority": summary.mean_priority,
            "mean_novelty": summary.mean_novelty,
            "mean_value": summary.mean_value,
            "mean_confidence": summary.mean_confidence,
            "mean_error_norm": summary.mean_error_norm,
            "replayed_probability": summary.replayed_probability,
            "structural_events": summary.structural_events,
        },
    }


def _evaluate_contingency(
    checkpoint: Mapping[str, object],
    pairs: Mapping[int, int],
) -> Dict[str, object]:
    """Probe action -> outcome with episodic action and readback fully closed.

    Two properties of the substrate dictate this protocol.

    First, the read has to happen on the channel that sleep actually writes.
    ``consolidate`` only touches ``fabric.decoders`` and ``fabric.transitions``
    -- the motor bank and the episodic field come out of sleep bit-identical,
    which ``sleep_only_touches_cortex`` asserts.  Reading ``step.probabilities``
    therefore reads a surface no amount of consolidation can move, so the
    prediction is taken from ``decoders[0]``, whose whole job is to predict the
    next lower-level activity: exactly the action -> outcome contingency.

    Second, the read has to happen on the same basis the write happened on.
    Region 0 is sparse, and a single byte after a reset leaves only 4-9 of its
    64 units active, whereas the settled traces that replay writes on carry
    ~35.  Because each decoder row draws a fixed fan-in of 16 of those 64
    units, a 4-unit trace can miss a given row's support outright -- measured
    fan-in energy overlap ranged from 50% down to a hard 0%, making some pairs
    unlearnable as a matter of arithmetic rather than of consolidation.  The
    action is therefore presented for the same number of passes the replay
    burst uses, which brings the probe to 21-34 active units.  Only the action
    is ever presented; the outcome is never shown, so nothing leaks.
    """

    index_of = {symbol: position for position, symbol in enumerate(OUTCOMES)}
    selector = torch.tensor(OUTCOMES, dtype=torch.long)
    rows = []
    correct = 0
    probability_sum = 0.0
    margin_sum = 0.0
    surprise_sum = 0.0
    for action in sorted(pairs):
        outcome = pairs[action]
        model = Taiji.from_checkpoint(checkpoint)
        model.reset_dynamics(episode_id=f"m6-probe-{action}")
        for _ in range(int(model.config.replay_burst_repeats)):
            step = model.observe(
                action,
                learn=False,
                learn_motor=False,
                use_memory=False,
            )
        trace = model.snapshot().regions[0].trace
        evidence = model.fabric.decoders[0].forward(trace).detach()
        restricted = torch.softmax(evidence[selector], dim=0)
        target = index_of[outcome]
        prediction = OUTCOMES[int(restricted.argmax().item())]
        true_probability = float(restricted[target].item())
        rivals = torch.cat([restricted[:target], restricted[target + 1 :]])
        margin = true_probability - float(rivals.max().item())
        correct += int(prediction == outcome)
        probability_sum += true_probability
        margin_sum += margin
        surprise_sum += -math.log(max(true_probability, 1e-12))
        rows.append({
            "action": chr(action),
            "expected_outcome": chr(outcome),
            "predicted_outcome": chr(prediction),
            "true_probability": true_probability,
            "margin": margin,
            "episodic_feedback_norm": float(
                step.memory_recall.cortical_feedback.norm().item()
            ),
            "episodic_confidence": step.memory_recall.confidence,
        })
    count = len(pairs)
    return {
        "contingency_accuracy": correct / count,
        "mean_true_probability": probability_sum / count,
        "mean_margin": margin_sum / count,
        "mean_surprise": surprise_sum / count,
        "rows": rows,
    }


def _episodic_readback_is_closed(rows: Sequence[Mapping[str, object]]) -> bool:
    return all(
        float(row["episodic_feedback_norm"]) == 0.0
        and float(row["episodic_confidence"]) == 0.0
        for row in rows
    )


def _rejects_unsettled_state(checkpoint: Mapping[str, object]) -> bool:
    model = Taiji.from_checkpoint(checkpoint)
    model.reset_dynamics(episode_id="m6-guard")
    model.act(ACTIONS, sample=False)
    try:
        model.consolidate(cycles=1)
    except RuntimeError:
        pass
    else:
        return False
    model.settle_action(0.0, learn=False, learn_memory=False)
    try:
        model.consolidate(cycles=1)
    except RuntimeError:
        return True
    return False


def _sleep_needs_a_written_field(seed: int) -> bool:
    model = Taiji(_config(seed), episode_id="m6-empty")
    try:
        model.consolidate(cycles=1)
    except RuntimeError:
        return True
    return False


def run_benchmark(*, seed: int = 29, cycles: int = 96) -> Dict[str, object]:
    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    corpus = _pretrain_corpus()
    pretrain = model.learn_bytes(corpus, epochs=6)
    episodes = _episodes()
    pairs = _contingency(episodes)
    _store(model, episodes)
    stored = model.checkpoint()

    full = _sleep(stored, cycles=cycles, learn=True, tag="full")
    control = _sleep(stored, cycles=cycles, learn=False, tag="control")
    content_lesion = _sleep(
        stored,
        cycles=cycles,
        learn=True,
        lesion=("action_readout", "outcome_readout"),
        tag="content-lesion",
    )
    association_lesion = _sleep(
        stored,
        cycles=cycles,
        learn=True,
        lesion=("association",),
        tag="association-lesion",
    )

    baseline_metrics = _evaluate_contingency(stored, pairs)
    full_metrics = _evaluate_contingency(full["checkpoint"], pairs)
    control_metrics = _evaluate_contingency(control["checkpoint"], pairs)
    content_metrics = _evaluate_contingency(content_lesion["checkpoint"], pairs)
    association_metrics = _evaluate_contingency(
        association_lesion["checkpoint"], pairs
    )

    chance = 1.0 / len(OUTCOMES)
    no_slots = not {"events", "keys", "values", "slots"} & (
        set(vars(model.memory)) | set(model.memory.to_payload())
    )
    checks = {
        "sleep_reactivates_own_engrams": (
            full["summary"]["accepted"] > 0
            and full["summary"]["replayed_probability"] > 0.0
        ),
        "consolidation_beats_no_replay_control": (
            full_metrics["contingency_accuracy"]
            > control_metrics["contingency_accuracy"]
            and full_metrics["mean_margin"] > control_metrics["mean_margin"]
        ),
        "consolidation_beats_chance": (
            full_metrics["contingency_accuracy"] > chance
        ),
        "engram_content_is_causally_necessary": (
            full_metrics["contingency_accuracy"]
            > content_metrics["contingency_accuracy"]
        ),
        "recurrent_completion_is_causally_necessary": (
            full_metrics["contingency_accuracy"]
            > association_metrics["contingency_accuracy"]
        ),
        "no_replay_control_writes_nothing": (
            not control["cortex_changed"]
            and control_metrics["contingency_accuracy"]
            == baseline_metrics["contingency_accuracy"]
        ),
        "sleep_only_touches_cortex": (
            full["cortex_changed"]
            and full["non_cortex_intact"]
            and full["field_write_count_stable"]
            and full["field_topology_stable"]
        ),
        "evaluation_has_no_episodic_readback": (
            _episodic_readback_is_closed(full_metrics["rows"])
            and _episodic_readback_is_closed(control_metrics["rows"])
        ),
        "sleep_requires_settled_state_and_written_field": (
            _rejects_unsettled_state(stored)
            and _sleep_needs_a_written_field(seed)
        ),
        "fixed_topology_no_event_slots": no_slots,
    }
    return {
        "benchmark": "taiji_m6_endogenous_replay",
        "seed": seed,
        "episodes": len(episodes),
        "contingency_pairs": len(pairs),
        "protocol": {
            "pretrain_corpus_bytes": len(corpus),
            "pretrain_epochs": 6,
            "pretrain_online_accuracy": pretrain["online_accuracy"],
            "pretrain_mean_surprise": pretrain["mean_surprise"],
            "pretrain_pairing_is_uniform": True,
            "write_exposures_per_episode": 1,
            "reward_per_write": 1.0,
            "cortex_learning_during_storage": False,
            "motor_learning_during_storage": False,
            "motor_learning_during_sleep": False,
            "consolidation_cycles": cycles,
            "external_replay_list": False,
            "teacher_target": False,
            "memory_to_fabric_weight_copy": False,
            "probe_chance_accuracy": chance,
            "claim_boundary": (
                "action-outcome contingency transfer, not cue-conditioned "
                "policy transfer: the replayed sequence carries no cue byte"
            ),
        },
        "architecture": {
            "checkpoint_format": model.CHECKPOINT_FORMAT,
            "state_version": model.STATE_VERSION,
            "memory_units": model.config.memory_units,
            "memory_association_edges": model.memory.association.edge_count,
            "event_slots_allocated": 0,
            "stored_event_count_metadata": model.memory.write_count,
            "replay_seed_gain": model.config.replay_seed_gain,
            "replay_noise_scale": model.config.replay_noise_scale,
            "replay_value_weight": model.config.replay_value_weight,
            "replay_priority_threshold": model.config.replay_priority_threshold,
            "replay_learning_scale": model.config.replay_learning_scale,
        },
        "metrics": {
            "before_sleep": baseline_metrics,
            "full_replay": full_metrics,
            "no_replay_control": control_metrics,
            "engram_content_lesion": content_metrics,
            "recurrent_association_lesion": association_metrics,
            "accuracy_gain_over_control": (
                full_metrics["contingency_accuracy"]
                - control_metrics["contingency_accuracy"]
            ),
            "margin_gain_over_control": (
                full_metrics["mean_margin"] - control_metrics["mean_margin"]
            ),
            "surprise_drop_over_control": (
                control_metrics["mean_surprise"] - full_metrics["mean_surprise"]
            ),
        },
        "consolidation": {
            "full_replay": full["summary"],
            "no_replay_control": control["summary"],
            "engram_content_lesion": content_lesion["summary"],
            "recurrent_association_lesion": association_lesion["summary"],
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cycles", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(seed=args.seed, cycles=args.cycles)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
