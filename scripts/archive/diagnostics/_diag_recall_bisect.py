"""Bisect: which working-tree mechanism costs recall vs HEAD."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from taiji import Taiji, TaijiConfig  # noqa: E402
from taiji import memory as memory_module  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from taiji.sparse import bound_norm  # noqa: E402

CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = (ord("+"), ord("-"))

_original_write = EpisodicField.write
_original_replay = EpisodicField.replay


def _no_gate_write(self, cortical_context, *, action_symbol, reward,
                    outcome_symbol, tick, episode_id, provenance, threshold):
    """Working-tree write with the identity gate neutralised.

    Setting the gate sigma would still threshold at the mean; this patch keeps
    every unit, matching HEAD's ungated event pattern exactly.
    """

    with object.__setattr__ if False else _patched_sigma(self, -1e9):
        return _original_write(
            self, cortical_context, action_symbol=action_symbol, reward=reward,
            outcome_symbol=outcome_symbol, tick=tick, episode_id=episode_id,
            provenance=provenance, threshold=threshold,
        )


class _patched_sigma:
    def __init__(self, field, sigma):
        self.field = field
        self.sigma = sigma

    def __enter__(self):
        config = self.field.config
        object.__setattr__(config, "memory_identity_gate_sigma", self.sigma)
        return self

    def __exit__(self, *exc):
        object.__setattr__(self.field.config, "memory_identity_gate_sigma", 1.0)


def _no_binding_replay(self, previous, *, tick, generator):
    """Working-tree replay without the action binding loop."""

    import math

    if self.write_count <= 0:
        raise RuntimeError("episodic replay requires at least one write")
    units = self.config.memory_units
    self_clock = self._time_code(tick)
    time_drive = self._normalize_drive(self.time_encoder.forward(self_clock))
    noise = torch.randn(units, generator=generator, dtype=torch.float32).to(
        self.device
    )
    value_weight = float(self.config.replay_value_weight)
    seed_drive = (
        float(self.config.replay_seed_gain)
        * (value_weight * self.reward_code + (1.0 - value_weight) * time_drive)
        + float(self.config.replay_noise_scale) * self._normalize_drive(noise)
    )
    adapted = previous.threshold + float(self.config.replay_fatigue_gain) * (
        previous.trace - previous.trace.mean()
    )
    activity, inhibition = self._activate(seed_drive, adapted)
    for _ in range(self.config.memory_iterations):
        recurrent = self.association.forward(activity)
        activity, inhibition = self._activate(
            seed_drive + self.config.memory_recurrent_gain * recurrent,
            adapted,
        )
    # Skip the binding loop, then reuse the stock tail via a direct tail call:
    # emulate by temporarily restoring the original replay is not possible, so
    # compute the rest inline (mirrors working-tree replay after the loop).
    recurrent_support = self.association.forward(activity)
    completion_error = activity - recurrent_support
    novelty = float(
        torch.clamp(
            completion_error.norm() / activity.norm().clamp_min(1e-8),
            min=0.0, max=1.0,
        ).item()
    )
    context = self.readout_receptors.forward(activity)
    familiarity = float(self.familiarity_readout.forward(context)[0].item())
    familiarity_confidence = 1.0 - math.exp(-max(0.0, familiarity))
    resonance = 1.0 - math.exp(-float(recurrent_support.norm().item()))
    confidence = familiarity_confidence * resonance
    raw_expected_reward = float(self.reward_readout.forward(context)[0].item())
    expected_reward = confidence * raw_expected_reward
    value = math.tanh(abs(raw_expected_reward))
    time_code = confidence * self.time_readout.forward(context)
    episode_code = confidence * self.episode_readout.forward(context)
    cortical_projection = confidence * self.cortical_readout.forward(context)
    action_probabilities = torch.softmax(
        self.action_readout.forward(context), dim=0
    )
    outcome_probabilities = torch.softmax(
        self.outcome_readout.forward(context), dim=0
    )
    provenance_probabilities = torch.softmax(
        self.provenance_readout.forward(context), dim=0
    )
    clock_norm = self_clock.norm().clamp_min(1e-8)
    recalled_norm = time_code.norm().clamp_min(1e-8)
    recency = 0.5 * (
        1.0
        + float(
            (torch.dot(time_code, self_clock) / (recalled_norm * clock_norm)).item()
        )
    )
    selection = value_weight * value + (1.0 - value_weight) * novelty
    priority = familiarity_confidence * resonance * selection * recency
    accepted = priority >= float(self.config.replay_priority_threshold)
    trace = bound_norm(
        self.config.memory_trace_decay * previous.trace
        + (1.0 - self.config.memory_trace_decay) * activity,
        self.config.max_trace_norm,
    )
    from taiji.memory import EpisodicReplay
    from taiji.state import MemoryState

    next_state = MemoryState(
        activity=activity,
        trace=trace,
        cortical_feedback=cortical_projection.detach().clone(),
        threshold=previous.threshold.detach().clone(),
        inhibition=float(inhibition),
        last_confidence=float(confidence),
    )
    event = EpisodicReplay(
        pattern=activity.detach().clone(),
        cortical_projection=cortical_projection.detach().clone(),
        action_probabilities=action_probabilities.detach().clone(),
        outcome_probabilities=outcome_probabilities.detach().clone(),
        time_code=time_code.detach().clone(),
        episode_code=episode_code.detach().clone(),
        provenance_probabilities=provenance_probabilities.detach().clone(),
        novelty=float(novelty),
        value=float(value),
        familiarity=float(familiarity_confidence),
        resonance=float(resonance),
        priority=float(priority),
        expected_reward=float(expected_reward),
        accepted=bool(accepted),
    )
    return next_state, event


def run(label, *, gate=False, repeats=None, meta_dim=None, fan_in=None) -> None:
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
    if repeats is not None:
        object.__setattr__(config, "episodic_write_repeats", repeats)
    if meta_dim is not None:
        object.__setattr__(config, "memory_meta_dim", meta_dim)
    if fan_in is not None:
        object.__setattr__(config, "memory_readout_fan_in", fan_in)
    if gate:
        EpisodicField.write = _original_write
    else:
        EpisodicField.write = _no_gate_write
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
    for index, (cue, expected) in enumerate(mapping.items()):
        restored = Taiji.from_checkpoint(checkpoint)
        restored.reset_dynamics(episode_id=f"recall-{index}")
        restored.observe(256, learn=False, learn_motor=False)
        restored.observe(cue, learn=False, learn_motor=False)
        decision = restored.act(ACTIONS, sample=False)
        correct += int(decision.action_symbol == expected)
    print(f"{label}: recalled={correct/len(mapping):.3f}")


if __name__ == "__main__":
    run("current (gate, repeats=4)")
    run("no gate, repeats=4", gate=False)
    run("gate, repeats=1", gate=True, repeats=1)
    run("no gate, repeats=1", gate=False, repeats=1)
    run("HEAD-equiv (no gate, repeats=1, meta=32)", gate=False, repeats=1, meta_dim=32)
    run("dense-48 (gate, repeats=4)", gate=True, repeats=4, fan_in=48)
    run("dense-48 no gate, repeats=1", gate=False, repeats=1, fan_in=48)
    run("dense-48 no gate, repeats=4", gate=False, repeats=4, fan_in=48)
    run("dense-48 gate, repeats=1", gate=True, repeats=1, fan_in=48)
    EpisodicField.write = _original_write
