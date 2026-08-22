"""Diagnose why the consolidated cue->action decode is ~1e-3 at evaluation."""
from __future__ import annotations

import sys
from pathlib import Path
from copy import deepcopy

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from taiji import Taiji  # noqa: E402
from verify_taiji_m7_cue_chain import (  # noqa: E402
    ACTIONS,
    CUES,
    _episodes,
    _pretrain_corpus,
    _present_cue,
    _store,
)
from verify_taiji_m6_endogenous_replay import _config, _sleep  # noqa: E402


def main() -> None:
    model = Taiji(_config(29), episode_id="diag-decode")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    stored = model.checkpoint()

    result = _sleep(stored, cycles=96, learn=True, tag="diag-decode")
    checkpoint = result["checkpoint"]

    decoder = None
    probe = Taiji.from_checkpoint(deepcopy(checkpoint))
    decoder = probe.fabric.consolidation_decoders[0]
    norms = decoder.edge_weight.norm(dim=1)
    print(
        f"decoder[0] row norms: nonzero={int((norms > 1e-9).sum())} "
        f"max={float(norms.max()):.4f} mean_nz="
        f"{float(norms[norms > 1e-9].mean() if (norms > 1e-9).any() else 0.0):.4f}"
    )

    print("== eval-basis decode per cue ==")
    for cue, event in episodes.items():
        m = Taiji.from_checkpoint(deepcopy(checkpoint))
        m.reset_dynamics(episode_id=f"probe-{cue}")
        _present_cue(m, cue, 0, use_memory=False)
        trace = m.snapshot().regions[0].trace
        evidence = m.fabric.consolidated_decode(0, trace)
        expected = int(event["action"])
        pair = torch.tensor(list(ACTIONS))
        logits = evidence[pair]
        margin = float((logits[0] - logits[1]).item())
        correct = (0 if margin > 0 else 1) == ACTIONS.index(expected)
        print(
            f"  cue {chr(cue)}: expected {chr(expected)} "
            f"margin(0-1)={margin:+.5f} norm={float(evidence.norm()):.5f} "
            f"trace_norm={float(trace.norm()):.3f} ok={correct}"
        )

    print("== replay-basis decode (sleep cue reinstatement) ==")
    from taiji.memory import EpisodicField  # noqa: E402
    from taiji.sparse import bound_norm  # noqa: E402
    from taiji.state import RegionState  # noqa: E402

    m = Taiji.from_checkpoint(deepcopy(checkpoint))
    m.reset_dynamics(episode_id="diag-replay-basis")
    mem = m.memory
    state = m._state.memory
    gen = m._memory_rng
    for k in range(12):
        state, replay = mem.replay(state, tick=5000 + k, generator=gen)
        if not replay.accepted:
            continue
        confidence = max(1e-8, float(replay.familiarity * replay.resonance))
        reinstated = replay.cortical_projection / confidence
        fast_offset = 0
        trace_offset = sum(m.config.region_sizes)
        trace = bound_norm(
            reinstated[
                trace_offset + fast_offset:trace_offset + fast_offset
                + m.config.region_sizes[0]
            ],
            m.config.max_trace_norm,
        )
        evidence = m.fabric.consolidated_decode(0, trace)
        pair = torch.tensor(list(ACTIONS))
        logits = evidence[pair]
        margin = float((logits[0] - logits[1]).item())
        action = int(replay.action_probabilities.argmax().item())
        print(
            f"  replay {k}: proposed {chr(action)} margin(0-1)={margin:+.5f} "
            f"norm={float(evidence.norm()):.5f} trace_norm={float(trace.norm()):.3f}"
        )


if __name__ == "__main__":
    main()
