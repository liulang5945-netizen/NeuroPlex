from copy import deepcopy

import pytest
import torch

from taiji import Taiji, TaijiConfig


CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = (ord("+"), ord("-"))


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_context_dim=32,
        memory_iterations=3,
        seed=23,
    )


def _record_balanced_one_shot_episodes(model: Taiji) -> dict[int, int]:
    mapping = {
        cue: ACTIONS[index % len(ACTIONS)]
        for index, cue in enumerate(CUES)
    }
    for index, (cue, action) in enumerate(mapping.items()):
        model.reset_dynamics(episode_id=f"store-{index}")
        model.observe(256, learn=False, learn_motor=False)
        model.observe(cue, learn=False, learn_motor=False)
        assert model.act((action,), sample=False).action_symbol == action
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(
            OUTCOMES[index % len(OUTCOMES)],
            learn=False,
            learn_motor=False,
        )
    return mapping


def _recall_accuracy(
    checkpoint: dict,
    mapping: dict[int, int],
    *,
    use_memory: bool,
) -> tuple[float, list[float]]:
    correct = 0
    confidences = []
    for index, (cue, expected) in enumerate(mapping.items()):
        model = Taiji.from_checkpoint(checkpoint)
        model.reset_dynamics(episode_id=f"recall-{index}")
        model.observe(
            256,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        step = model.observe(
            cue,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        decision = model.act(ACTIONS, sample=False)
        correct += int(decision.action_symbol == expected)
        confidences.append(step.memory_recall.confidence)
    return correct / len(mapping), confidences


def test_episodic_field_beats_equal_width_trace_only_and_read_lesion() -> None:
    model = Taiji(_config())
    mapping = _record_balanced_one_shot_episodes(model)
    checkpoint = model.checkpoint()

    recalled, confidences = _recall_accuracy(
        checkpoint, mapping, use_memory=True
    )
    trace_only, lesion_confidences = _recall_accuracy(
        checkpoint, mapping, use_memory=False
    )
    recurrent_lesion_checkpoint = deepcopy(checkpoint)
    recurrent_lesion_checkpoint["memory"]["association"][
        "edge_weight"
    ].zero_()
    recurrent_lesion, _ = _recall_accuracy(
        recurrent_lesion_checkpoint,
        mapping,
        use_memory=True,
    )

    assert recalled >= 0.875
    assert recalled - trace_only >= 0.375
    assert recalled - recurrent_lesion >= 0.25
    assert min(confidences) > 0.0
    assert max(lesion_confidences) == 0.0


def test_action_outcome_experience_is_atomic_and_checkpoint_exact() -> None:
    original = Taiji(_config(), episode_id="atomic-event")
    original.observe(256, learn=False, learn_motor=False)
    original.observe(ord("Q"), learn=False, learn_motor=False)
    original.act((ord("1"),), sample=False)
    original.settle_action(1.0, learn=False, learn_memory=True)
    restored = Taiji.from_checkpoint(original.checkpoint())

    with pytest.raises(RuntimeError, match="pending experience"):
        original.reset_dynamics(episode_id="illegal-reset")
    with pytest.raises(RuntimeError, match="pending experience"):
        original.act(ACTIONS, sample=False)

    left = original.observe(ord("+"), learn=False, learn_motor=False)
    right = restored.observe(ord("+"), learn=False, learn_motor=False)

    assert left.memory_write_strength == right.memory_write_strength
    assert left.memory_write_strength > 0.0
    for a, b in zip(original.parameter_tensors(), restored.parameter_tensors()):
        assert torch.equal(a, b)


def test_one_event_recalls_action_outcome_value_without_allocating_a_slot() -> None:
    model = Taiji(_config())
    topology_before = model.memory.association.pre_index.clone()
    edge_count_before = model.memory.association.edge_count
    model.observe(256, learn=False, learn_motor=False)
    model.observe(ord("Q"), learn=False, learn_motor=False)
    model.act((ord("1"),), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=True)
    model.observe(ord("+"), learn=False, learn_motor=False)

    assert model.memory.write_count == 1
    assert model.memory.association.edge_count == edge_count_before
    assert torch.equal(model.memory.association.pre_index, topology_before)
    assert not {
        "slots",
        "keys",
        "values",
        "events",
    } & set(model.memory.to_payload())

    model.reset_dynamics(episode_id="one-event-recall")
    model.observe(256, learn=False, learn_motor=False)
    recall = model.observe(
        ord("Q"), learn=False, learn_motor=False
    ).memory_recall

    assert recall.used_long_term is True
    assert recall.confidence > 0.0
    assert recall.action_evidence[ord("1")] > recall.action_evidence[ord("0")]
    assert recall.outcome_probabilities[ord("+")] > recall.outcome_probabilities[
        ord("-")
    ]
    assert recall.expected_reward > 0.0
    assert int(recall.provenance_probabilities.argmax().item()) == 0
    stored_time = model.memory._time_code(2)
    stored_episode = model.memory._episode_code("episode-0")
    other_episode = model.memory._episode_code("other-episode")
    assert torch.cosine_similarity(recall.time_code, stored_time, dim=0) > 0.5
    assert torch.cosine_similarity(
        recall.episode_code, stored_episode, dim=0
    ) > torch.cosine_similarity(
        recall.episode_code, other_episode, dim=0
    )


def test_recalled_cortical_state_is_fed_back_on_the_next_causal_tick() -> None:
    trained = Taiji(_config())
    _record_balanced_one_shot_episodes(trained)
    checkpoint = trained.checkpoint()
    full = Taiji.from_checkpoint(checkpoint)
    lesioned = Taiji.from_checkpoint(checkpoint)
    for model, enabled in ((full, True), (lesioned, False)):
        model.reset_dynamics(episode_id=f"feedback-{enabled}")
        model.observe(256, learn=False, learn_motor=False, use_memory=enabled)
        cue_step = model.observe(
            CUES[0], learn=False, learn_motor=False, use_memory=enabled
        )
        if enabled:
            assert cue_step.memory_recall.cortical_feedback.norm() > 0.0
        else:
            assert cue_step.memory_recall.cortical_feedback.norm() == 0.0

    full_probe = full.observe(ord("?"), learn=False, learn_motor=False)
    lesion_probe = lesioned.observe(ord("?"), learn=False, learn_motor=False)

    assert full_probe.activity_rates != lesion_probe.activity_rates
    assert not torch.equal(
        full.snapshot().regions[0].membrane,
        lesioned.snapshot().regions[0].membrane,
    )
