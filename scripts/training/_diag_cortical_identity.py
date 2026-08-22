"""Diagnose cortical projection identity collapse under the M7 protocol."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from taiji import Taiji  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    FILLER,
    _config,
)

CUES = tuple(ord(v) for v in "ABCDEFGH")
ACTIONS = tuple(ord(v) for v in "01")
OUTCOMES = tuple(ord(v) for v in "+-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.30)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    config = dataclasses.replace(
        _config(29),
        cortical_readout_learning_rate=args.rate,
        cortical_readout_repeats=args.repeats,
    )
    print(f"rate={args.rate} repeats={args.repeats}")
    model = Taiji(config, episode_id="diag-cortical")
    pretrain = bytes((FILLER,)).join(
        bytes((c, a, o)) for c in CUES for a in ACTIONS for o in OUTCOMES
    )
    model.learn_bytes(pretrain, epochs=6)

    captured: Dict[str, torch.Tensor] = {}
    original_write = EpisodicField.write
    cortical_targets: Dict[int, torch.Tensor] = {}
    write_contexts: Dict[int, torch.Tensor] = {}
    current: Dict[str, int] = {"cue": 0}

    def capturing_write(self, cortical_context, *, action_symbol, reward,
                        outcome_symbol, tick, episode_id, provenance, threshold):
        captured["ctx"] = cortical_context.detach().clone()
        return original_write(
            self, cortical_context, action_symbol=action_symbol, reward=reward,
            outcome_symbol=outcome_symbol, tick=tick, episode_id=episode_id,
            provenance=provenance, threshold=threshold,
        )

    mem_probe = model.memory
    original_local = mem_probe.cortical_readout.local_update

    def capturing_local(error, trace, **kwargs):
        write_contexts[current["cue"]] = trace.detach().clone()
        return original_local(error, trace, **kwargs)

    mem_probe.cortical_readout.local_update = capturing_local

    EpisodicField.write = capturing_write
    try:
        for i, cue in enumerate(CUES):
            current["cue"] = cue
            model.reset_dynamics(episode_id=f"m7-store-{i}")
            model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
            step = model.observe(cue, learn=False, learn_motor=False)
            model.act((ACTIONS[i % 2],), sample=False)
            model.settle_action(1.0, learn=False, learn_memory=True,
                                provenance="experienced")
            model.observe(OUTCOMES[i % 2], learn=False, learn_motor=False)
            cortical_targets[cue] = captured["ctx"].clone()
    finally:
        EpisodicField.write = original_write
        mem_probe.cortical_readout.local_update = original_local

    mem = model.memory
    print("== write-time receptor context discriminability ==")
    sims = []
    for i in range(len(CUES)):
        for j in range(i + 1, len(CUES)):
            sims.append(float(torch.nn.functional.cosine_similarity(
                write_contexts[CUES[i]], write_contexts[CUES[j]], dim=0
            ).item()))
    print(
        f"  pairwise write context cosine: mean {sum(sims)/len(sims):.3f} "
        f"max {max(sims):.3f}"
    )
    # can the cortical readout even fit its own training points?
    print("== direct fit on write contexts ==")
    ok_w = 0
    for cue in CUES:
        out = mem.cortical_readout.forward(write_contexts[cue])
        sims_to_targets = {
            c: float(torch.nn.functional.cosine_similarity(
                out, cortical_targets[c], dim=0).item())
            for c in CUES
        }
        nearest = max(sims_to_targets, key=sims_to_targets.get)
        ok_w += int(nearest == cue)
        print(
            f"  cue {chr(cue)}: nearest={chr(nearest)} "
            f"cos={sims_to_targets[nearest]:.3f} self={sims_to_targets[cue]:.3f}"
        )
    print(f"  write-context fit identity: {ok_w}/8")

    print("== cue cortical targets geometry ==")
    sims = []
    for i in range(len(CUES)):
        for j in range(i + 1, len(CUES)):
            sims.append(float(torch.nn.functional.cosine_similarity(
                cortical_targets[CUES[i]], cortical_targets[CUES[j]], dim=0
            ).item()))
    print(f"  pairwise target cosine: mean {sum(sims)/len(sims):.3f} max {max(sims):.3f}")

    print("== wake cortical identity + recall context geometry ==")
    recall_contexts: Dict[int, torch.Tensor] = {}
    ok = 0
    for cue in CUES:
        model.reset_dynamics(episode_id=f"diag-r-{cue}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
        step = model.observe(cue, learn=False, learn_motor=False)
        recall_contexts[cue] = mem.readout_receptors.forward(
            model._state.memory.activity
        ).clone()
        conf = max(1e-8, step.memory_recall.confidence)
        proj = step.memory_recall.cortical_feedback / conf
        sims_to_targets = {
            c: float(torch.nn.functional.cosine_similarity(
                proj, cortical_targets[c], dim=0).item())
            for c in CUES
        }
        nearest = max(sims_to_targets, key=sims_to_targets.get)
        ok += int(nearest == cue)
        print(
            f"  cue {chr(cue)}: nearest={chr(nearest)} "
            f"cos={sims_to_targets[nearest]:.3f} self={sims_to_targets[cue]:.3f} "
            f"conf={conf:.3f}"
        )
    print(f"  wake cortical identity: {ok}/8")
    sims = []
    for i in range(len(CUES)):
        for j in range(i + 1, len(CUES)):
            sims.append(float(torch.nn.functional.cosine_similarity(
                recall_contexts[CUES[i]], recall_contexts[CUES[j]], dim=0
            ).item()))
    print(
        f"  pairwise recall context cosine: mean {sum(sims)/len(sims):.3f} "
        f"max {max(sims):.3f}"
    )
    # direct fit probe: how well does cortical_readout map each recall
    # context back to any stored target (upper bound of current weights)?
    print("== direct readout fit on recall contexts ==")
    ok_fit = 0
    for cue in CUES:
        out = mem.cortical_readout.forward(recall_contexts[cue])
        sims_to_targets = {
            c: float(torch.nn.functional.cosine_similarity(
                out, cortical_targets[c], dim=0).item())
            for c in CUES
        }
        nearest = max(sims_to_targets, key=sims_to_targets.get)
        ok_fit += int(nearest == cue)
    print(f"  direct-fit identity: {ok_fit}/8")


    print("== replay projection identity ==")
    model.reset_dynamics(episode_id="diag-replay")
    state = model._state.memory
    gen = model._memory_rng
    ok = accepted = 0
    for k in range(24):
        state, replay = mem.replay(state, tick=2000 + k, generator=gen)
        if not replay.accepted:
            continue
        accepted += 1
        conf = max(1e-8, replay.familiarity * replay.resonance)
        proj = replay.cortical_projection / conf
        sims_to_targets = {
            c: float(torch.nn.functional.cosine_similarity(
                proj, cortical_targets[c], dim=0).item())
            for c in CUES
        }
        nearest = max(sims_to_targets, key=sims_to_targets.get)
        ok += int(nearest is not None)
    print(f"  accepted {accepted}")
    # projection spread
    projections = []
    state = model._state.memory
    for k in range(16):
        state, replay = mem.replay(state, tick=3000 + k, generator=gen)
        projections.append(replay.cortical_projection)
    spread = []
    for i in range(len(projections)):
        for j in range(i + 1, len(projections)):
            spread.append(float(torch.nn.functional.cosine_similarity(
                projections[i], projections[j], dim=0).item()))
    print(f"  pairwise replay projection cosine: mean {sum(spread)/len(spread):.3f}")
    norms = [float(p.norm().item()) for p in projections]
    print(f"  projection norms: min {min(norms):.4f} max {max(norms):.4f}")


if __name__ == "__main__":
    main()
