import pytest
import torch

from taiji import EnvironmentOutcome, Taiji, TaijiConfig


CUES = (ord("L"), ord("R"))
ACTIONS = (ord("0"), ord("1"))


class BinaryCueEnvironment:
    def outcome(self, cue: int, action: int) -> EnvironmentOutcome:
        correct = ACTIONS[CUES.index(cue)]
        success = action == correct
        return EnvironmentOutcome(
            sensation=ord("+") if success else ord("-"),
            reward=1.0 if success else -1.0,
            terminal=True,
        )


def _model() -> Taiji:
    return Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
    ))


def _run_interactions(model: Taiji, *, learn_action: bool) -> list[bool]:
    environment = BinaryCueEnvironment()
    successes = []
    for trial in range(200):
        cue = CUES[trial % len(CUES)]
        model.reset_dynamics(episode_id=f"n11-{trial}")
        model.observe(
            model.config.boundary_symbol, learn=True, learn_motor=False
        )
        model.observe(cue, learn=True, learn_motor=False)
        action = model.act(ACTIONS, sample=True).action_symbol
        outcome = environment.outcome(cue, action)
        model.settle_action(outcome.reward, learn=learn_action)
        model.observe(outcome.sensation, learn=True, learn_motor=False)
        successes.append(outcome.reward > 0.0)
    return successes


def test_pending_action_is_atomic_and_checkpointed() -> None:
    original = _model()
    original.observe(256, learn=True, learn_motor=False)
    original.observe(ord("L"), learn=True, learn_motor=False)
    decision = original.act(ACTIONS, sample=False)
    restored = Taiji.from_checkpoint(original.checkpoint())

    with pytest.raises(RuntimeError, match="pending action"):
        original.observe(ord("+"), learn=True, learn_motor=False)

    left = original.settle_action(1.0, learn=True)
    right = restored.settle_action(1.0, learn=True)

    assert left.action_symbol == decision.action_symbol
    assert left == right
    for a, b in zip(original.parameter_tensors(), restored.parameter_tensors()):
        assert torch.equal(a, b)


def test_reward_modulated_motor_learning_beats_random_and_action_lesion() -> None:
    learned = _run_interactions(_model(), learn_action=True)
    lesioned = _run_interactions(_model(), learn_action=False)
    learned_final = sum(learned[-40:]) / 40
    lesion_final = sum(lesioned[-40:]) / 40
    random_baseline = 0.5

    assert learned_final >= 0.90
    assert learned_final - lesion_final >= 0.25
    assert learned_final - random_baseline >= 0.35
