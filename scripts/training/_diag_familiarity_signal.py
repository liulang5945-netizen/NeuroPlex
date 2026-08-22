"""诊断五：familiarity 信号在一次性写入与重复经历两个场景下的取值分布。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from taiji import Taiji, TaijiConfig  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402


# 拦截 write，记录写入前的 cue_familiarity 与 novelty
_original_write = EpisodicField.write


def patched_write(self, cortical_context, **kwargs):
    threshold = kwargs["threshold"]
    cue_pattern = self._cue_pattern(cortical_context, threshold)
    completion = self.association.forward(cue_pattern)
    context_probe = self.readout_receptors.forward(cue_pattern)
    fam = float(self.familiarity_readout.forward(context_probe)[0].item())
    result = _original_write(self, cortical_context, **kwargs)
    print(
        f"write#{self.write_count} fam={fam:.3f} novelty={result.novelty:.3f}",
        flush=True,
    )
    return result


EpisodicField.write = patched_write


def one_shot_scenario() -> None:
    print("=== 一次性写入（episodic field 场景）===", flush=True)
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
        seed=23,
    )
    model = Taiji(config)
    for index, cue in enumerate("ABCDEFGH"):
        model.reset_dynamics(episode_id=f"store-{index}")
        model.observe(256, learn=False, learn_motor=False)
        model.observe(ord(cue), learn=False, learn_motor=False)
        model.act((ord("0"),), sample=False)
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(ord("+"), learn=False, learn_motor=False)


def repeat_scenario() -> None:
    print("=== 重复经历（崩塌场景）===", flush=True)
    config = TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=8,
        motor_fan_in=12,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=12,
        memory_meta_dim=12,
        memory_iterations=2,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=6,
        seed=31,
    )
    stream = bytes((ord("a") + i % 6) for i in range(12))
    model = Taiji(config, episode_id="waking")
    for _ in range(3):
        model.reset_dynamics(episode_id="waking")
        for symbol in stream:
            model.observe(symbol, learn=True, learn_motor=False)
    for trial in range(3):
        model.reset_dynamics(episode_id=f"lived-{trial}")
        model.observe(model.config.boundary_symbol, learn=False)
        for symbol in stream:
            model.observe(int(symbol), learn=False)
            probabilities = model.snapshot().motor_probabilities
            candidates = [
                int(i) for i in torch.argsort(probabilities, descending=True)[:8]
            ]
            model.act(candidates, sample=False)
            model.settle_action(-3.0, learn=False, learn_memory=True)
        model.observe(model.config.boundary_symbol, learn=False)


if __name__ == "__main__":
    one_shot_scenario()
    repeat_scenario()
