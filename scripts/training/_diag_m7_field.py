#!/usr/bin/env python3
"""Diagnostic: decompose M7 field failures into write/recall/replay stages.

Measures, under the exact M7 storage protocol (8 cues, 2 actions, 2 outcomes,
prefix_length=0):
  1. Write-time readout fit: does one-shot write leave the action/outcome
     readouts able to classify the event pattern they were trained on?
  2. Wake recall readout: cue -> recurrent completion -> readout accuracy.
  3. Replay pattern readout: regenerated pattern -> readout accuracy, and how
     close regenerated patterns sit to stored event patterns.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from taiji import Taiji  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from taiji.sparse import bound_norm  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    FILLER,
    PROVENANCE,
    _config,
)

CUES = tuple(ord(v) for v in "ABCDEFGH")
ACTIONS = tuple(ord(v) for v in "01")
OUTCOMES = tuple(ord(v) for v in "+-")


def main(seed: int = 29, **overrides) -> None:
    config = _config(seed)
    model = Taiji(config, episode_id="diag-m7")
    # Diagnostic overrides bypass frozen/validation on purpose.
    for key, value in overrides.items():
        object.__setattr__(model.config, key, value)
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

    # Capture the event pattern each write constructs by duplicating the
    # deterministic pattern computation around the real write.
    event_patterns: Dict[int, torch.Tensor] = {}
    pending_cue: Dict[str, int] = {}
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
        event_pattern, _ = self._activate(event_drive, threshold)
        captured["pattern"] = event_pattern.clone()
        return original_write(
            self, cortical_context, action_symbol=action_symbol, reward=reward,
            outcome_symbol=outcome_symbol, tick=tick, episode_id=episode_id,
            provenance=provenance, threshold=threshold,
        )

    EpisodicField.write = capturing_write
    captured: Dict[str, torch.Tensor] = {}
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
    finally:
        EpisodicField.write = original_write

    mem = model.memory

    def classify(pattern: torch.Tensor) -> Dict[str, object]:
        context = mem.readout_receptors.forward(pattern)
        action = int(mem.action_readout.forward(context).argmax().item())
        outcome = int(mem.outcome_readout.forward(context).argmax().item())
        return {"action": action, "outcome": outcome}

    print("== stage 1: readout on stored event patterns ==")
    ok_a = ok_o = 0
    for cue, ev in episodes.items():
        got = classify(event_patterns[cue])
        ok_a += int(got["action"] == ev["action"])
        ok_o += int(got["outcome"] == ev["outcome"])
    print(f"  event-pattern action readout: {ok_a}/8, outcome: {ok_o}/8")

    print("== stage 2: wake recall readout ==")
    ok_a = ok_o = 0
    for cue, ev in episodes.items():
        model.reset_dynamics(episode_id=f"diag-wake-{cue}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
        step = model.observe(cue, learn=False, learn_motor=False)
        act = int(step.memory_recall.action_probabilities.argmax().item())
        out = int(step.memory_recall.outcome_probabilities.argmax().item())
        ok_a += int(act == ev["action"])
        ok_o += int(out == ev["outcome"])
    print(f"  wake recall action: {ok_a}/8, outcome: {ok_o}/8")

    print("== stage 3: replay pattern readout + geometry ==")
    model.reset_dynamics(episode_id="diag-replay")
    state = model._state.memory
    gen = model._memory_rng
    action_hits = outcome_hits = nearest_hits = accepted = 0
    sims = []
    for _ in range(48):
        state, replay = mem.replay(state, tick=1000 + _, generator=gen)
        if not replay.accepted:
            continue
        accepted += 1
        got = classify(replay.pattern)
        best_cue, best_sim = None, -2.0
        for cue, pat in event_patterns.items():
            s = float(
                torch.nn.functional.cosine_similarity(
                    replay.pattern, pat, dim=0
                ).item()
            )
            if s > best_sim:
                best_cue, best_sim = cue, s
        sims.append(best_sim)
        nearest_hits += int(best_cue is not None)
        action_hits += int(got["action"] == episodes[best_cue]["action"])
        outcome_hits += int(got["outcome"] == episodes[best_cue]["outcome"])
    print(
        f"  accepted replays: {accepted}/48; "
        f"mean cosine to nearest stored event: "
        f"{(sum(sims) / len(sims)) if sims else float('nan'):.3f}"
    )
    if accepted:
        print(
            f"  replay readout vs nearest event: action {action_hits}/{accepted}, "
            f"outcome {outcome_hits}/{accepted}"
        )

    print("== stage 4: pattern norms ==")
    norms = [float(p.norm().item()) for p in event_patterns.values()]
    print(f"  event pattern norms: min {min(norms):.3f} max {max(norms):.3f}")
    pair_sims = []
    cues = list(event_patterns)
    for i in range(len(cues)):
        for j in range(i + 1, len(cues)):
            pair_sims.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        event_patterns[cues[i]], event_patterns[cues[j]], dim=0
                    ).item()
                )
            )
    print(
        f"  pairwise event cosine: mean {sum(pair_sims)/len(pair_sims):.3f} "
        f"max {max(pair_sims):.3f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--binding", type=float, default=None,
                        help="override memory_action_binding_gain")
    parser.add_argument("--repeats", type=int, default=None,
                        help="override episodic_write_repeats")
    args = parser.parse_args()
    ov = {}
    if args.binding is not None:
        ov["memory_action_binding_gain"] = args.binding
    if args.repeats is not None:
        ov["episodic_write_repeats"] = args.repeats
    print(f"overrides: {ov}")
    main(args.seed, **ov)
