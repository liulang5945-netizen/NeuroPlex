#!/usr/bin/env python3
"""Diagnostic: measure committed-HEAD field fidelity under the M7 protocol.

Uses an isolated copy of the HEAD taiji package (_scratch/taiji_head) so the
working-tree refactor is not touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_scratch"))

from taiji_head import Taiji, TaijiConfig  # noqa: E402
from taiji_head.memory import EpisodicField  # noqa: E402
import math  # noqa: E402

CUES = tuple(ord(v) for v in "ABCDEFGH")
ACTIONS = tuple(ord(v) for v in "01")
OUTCOMES = tuple(ord(v) for v in "+-")
FILLER = ord(".")


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


def main(seed: int = 29) -> None:
    model = Taiji(_config(seed), episode_id="diag-m7-head")
    pretrain = bytes((FILLER,)).join(
        bytes((c, a, o)) for c in CUES for a in ACTIONS for o in OUTCOMES
    )
    model.learn_bytes(pretrain, epochs=6)
    episodes = {
        cue: {
            "action": ACTIONS[i % 2],
            "outcome": OUTCOMES[i % 2],
            "episode_id": f"m7-store-{i}",
        }
        for i, cue in enumerate(CUES)
    }
    # Capture stored event patterns by replicating HEAD's deterministic write
    # pattern computation around the real write.
    event_patterns: Dict[int, torch.Tensor] = {}
    captured: Dict[str, torch.Tensor] = {}
    original_write = EpisodicField.write

    def capturing_write(self, cortical_context, *, action_symbol, reward,
                        outcome_symbol, tick, episode_id, provenance, threshold):
        cue_pattern = self._cue_pattern(cortical_context, threshold)
        action_drive = self._normalize_drive(
            self.action_encoder.forward(self._one_hot(action_symbol))
        )
        outcome_drive = self._normalize_drive(
            self.outcome_encoder.forward(self._one_hot(outcome_symbol))
        )
        time_code = self._time_code(tick)
        episode_code = self._episode_code(episode_id)
        provenance_code = self._provenance_code(provenance)
        time_drive = self._normalize_drive(self.time_encoder.forward(time_code))
        episode_drive = self._normalize_drive(
            self.episode_encoder.forward(episode_code)
        )
        provenance_drive = self._normalize_drive(
            self.provenance_encoder.forward(provenance_code)
        )
        cue_drive = self._encode_cue(cortical_context)
        components = (
            action_drive, outcome_drive, reward * self.reward_code,
            time_drive, episode_drive, provenance_drive,
        )
        scale = self.config.memory_event_gain / math.sqrt(len(components))
        event_drive = cue_drive + scale * torch.stack(components, dim=0).sum(dim=0)
        pattern, _ = self._activate(event_drive, threshold)
        captured["pattern"] = pattern.clone()
        captured["cue_pattern"] = cue_pattern.clone()
        return original_write(
            self, cortical_context, action_symbol=action_symbol, reward=reward,
            outcome_symbol=outcome_symbol, tick=tick, episode_id=episode_id,
            provenance=provenance, threshold=threshold,
        )

    cue_patterns: Dict[int, torch.Tensor] = {}
    EpisodicField.write = capturing_write
    try:
        for cue, ev in episodes.items():
            model.reset_dynamics(episode_id=ev["episode_id"])
            model.observe(
                model.config.boundary_symbol, learn=False, learn_motor=False
            )
            model.observe(cue, learn=False, learn_motor=False)
            model.act((ev["action"],), sample=False)
            model.settle_action(
                1.0, learn=False, learn_memory=True, provenance="experienced"
            )
            model.observe(ev["outcome"], learn=False, learn_motor=False)
            event_patterns[cue] = captured["pattern"]
            cue_patterns[cue] = captured["cue_pattern"]
    finally:
        EpisodicField.write = original_write

    mem = model.memory

    def classify_context(context: torch.Tensor) -> Dict[str, int]:
        return {
            "action": int(mem.action_readout.forward(context).argmax().item()),
            "outcome": int(mem.outcome_readout.forward(context).argmax().item()),
        }

    print("== stage 0: stored event geometry + readout fit ==")
    pair = []
    for i in range(len(CUES)):
        for j in range(i + 1, len(CUES)):
            pair.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        event_patterns[CUES[i]],
                        event_patterns[CUES[j]],
                        dim=0,
                    ).item()
                )
            )
    print(
        f"  pairwise event cosine: mean {sum(pair)/len(pair):.3f} "
        f"max {max(pair):.3f}"
    )
    ok_a = ok_o = 0
    for cue, ev in episodes.items():
        ctx = mem.readout_receptors.forward(event_patterns[cue])
        got = classify_context(ctx)
        ok_a += int(got["action"] == ev["action"])
        ok_o += int(got["outcome"] == ev["outcome"])
    print(f"  readout on stored events: action {ok_a}/8, outcome {ok_o}/8")

    print("== stage 1: cue completion quality ==")
    ok_nearest = ok_readout = 0
    for cue, ev in episodes.items():
        activity = cue_patterns[cue]
        for _ in range(model.config.memory_iterations):
            recurrent = mem.association.forward(activity)
            activity, _ = mem._activate(
                mem._encode_cue(
                    torch.zeros(model.config.cortical_context_dim)
                ) * 0.0
                + mem.config.memory_recurrent_gain * recurrent,
                model._state.memory.threshold,
            )
        sims = {
            c: float(
                torch.nn.functional.cosine_similarity(
                    activity, event_patterns[c], dim=0
                ).item()
            )
            for c in CUES
        }
        nearest = max(sims, key=sims.get)
        ok_nearest += int(nearest == cue)
        ctx = mem.readout_receptors.forward(activity)
        got = classify_context(ctx)
        ok_readout += int(got["action"] == ev["action"])
        print(
            f"  cue {chr(cue)}: nearest={chr(nearest)} "
            f"cos {sims[nearest]:.2f} (self {sims[cue]:.2f}), "
            f"readout action {chr(got['action'])} want {chr(ev['action'])}"
        )
    print(f"  completion nearest-cue: {ok_nearest}/8, readout: {ok_readout}/8")

    print("== stage 2: wake recall readout ==")
    ok_a = ok_o = 0
    identity_ok = 0
    cortical_probes = {}
    for cue, ev in episodes.items():
        model.reset_dynamics(episode_id=f"diag-wake-{cue}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
        step = model.observe(cue, learn=False, learn_motor=False)
        act = int(step.memory_recall.action_probabilities.argmax().item())
        out = int(step.memory_recall.outcome_probabilities.argmax().item())
        ok_a += int(act == ev["action"])
        ok_o += int(out == ev["outcome"])
        conf = max(1e-8, step.memory_recall.confidence)
        cortical_probes[cue] = (
            step.memory_recall.cortical_feedback / conf
        ).detach().clone()
    print(f"  wake recall action: {ok_a}/8, outcome: {ok_o}/8")

    print("== cortical identity of recall projection ==")
    for cue, probe in cortical_probes.items():
        best = max(
            cortical_probes,
            key=lambda c: float(
                torch.nn.functional.cosine_similarity(
                    probe, cortical_probes[c], dim=0
                ).item()
            ),
        )
        identity_ok += int(best == cue)
    print(f"  nearest-self check trivially passes; identity accuracy is cue-vs-cue")

    print("== stage 3: replay pattern readout ==")
    model.reset_dynamics(episode_id="diag-replay")
    state = model._state.memory
    gen = model._memory_rng
    accepted = action_hits = outcome_hits = 0
    action_matches_total = 0
    for _ in range(48):
        state, replay = mem.replay(state, tick=1000 + _, generator=gen)
        if not replay.accepted:
            continue
        accepted += 1
        context = mem.readout_receptors.forward(replay.pattern)
        got = classify_context(context)
        nearest = max(
            cortical_probes,
            key=lambda c: float(
                torch.nn.functional.cosine_similarity(
                    replay.cortical_projection, cortical_probes[c], dim=0
                ).item()
            ),
        )
        action_matches_total += int(got["action"] == episodes[nearest]["action"])
        action_hits += int(True)
    print(f"  accepted replays: {accepted}/48")
    if accepted:
        print(
            f"  replay action matches nearest-cortical-cue episode: "
            f"{action_matches_total}/{accepted}"
        )

    print("== stage 5: cue cortical geometry ==")
    contexts = {}
    for cue in CUES:
        model.reset_dynamics(episode_id=f"diag-geo-{cue}")
        model.observe(
            model.config.boundary_symbol, learn=False, learn_motor=False
        )
        model.observe(cue, learn=False, learn_motor=False)
        contexts[cue] = model.fabric.cortical_context(
            model._state.regions
        ).detach().clone()
    sims = []
    same_action = []
    for i in range(len(CUES)):
        for j in range(i + 1, len(CUES)):
            s = float(
                torch.nn.functional.cosine_similarity(
                    contexts[CUES[i]], contexts[CUES[j]], dim=0
                ).item()
            )
            sims.append(s)
            if episodes[CUES[i]]["action"] == episodes[CUES[j]]["action"]:
                same_action.append(s)
    print(
        f"  pairwise cortical cosine: mean {sum(sims)/len(sims):.3f} "
        f"max {max(sims):.3f}"
    )
    print(
        f"  same-action pair cosine mean: "
        f"{sum(same_action)/len(same_action):.3f}"
    )

    print("== stage 6: which cues fail recall ==")
    fails = []
    for cue, ev in episodes.items():
        model.reset_dynamics(episode_id=f"diag-fail-{cue}")
        model.observe(
            model.config.boundary_symbol, learn=False, learn_motor=False
        )
        step = model.observe(cue, learn=False, learn_motor=False)
        act = int(step.memory_recall.action_probabilities.argmax().item())
        if act != ev["action"]:
            fails.append(
                f"{chr(cue)}(want {chr(ev['action'])} got {chr(act)} "
                f"conf {step.memory_recall.confidence:.3f})"
            )
    print(f"  recall failures: {fails if fails else 'none'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 29)
