"""Reconcile the fast-probe margin with the act() decision on cue B."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from taiji import Taiji  # noqa: E402
from verify_taiji_m7_cue_chain import (  # noqa: E402
    ACTIONS,
    _episodes,
    _pretrain_corpus,
    _present_cue,
    _store,
)
from verify_taiji_m6_endogenous_replay import _config, _sleep  # noqa: E402


def main() -> None:
    model = Taiji(_config(29), episode_id="diag-motor")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    stored = model.checkpoint()
    result = _sleep(stored, cycles=96, learn=True, tag="diag-motor")
    checkpoint = result["checkpoint"]

    for cue in (ord("A"), ord("B")):
        m = Taiji.from_checkpoint(deepcopy(checkpoint))
        m.reset_dynamics(episode_id=f"probe-{cue}")
        step = _present_cue(m, cue, 0, use_memory=False)
        expected = int(episodes[cue]["action"])
        decision = m.act(tuple(ACTIONS), sample=False)
        snap = m.snapshot()
        context = snap.motor_context
        fast = m.motor.synapses.forward(context) + m.motor.bias
        slow = m.fabric.consolidated_decode(0, snap.regions[0].trace)
        pair = torch.tensor(list(ACTIONS))
        print(f"cue {chr(cue)} expected {chr(expected)} "
              f"decided {chr(decision.action_symbol)}")
        print(f"  state.motor_probabilities[01] = "
              f"{[float(m._state.motor_probabilities[a]) for a in ACTIONS]}")
        print(f"  fast logits[01] = {[float(fast[a]) for a in ACTIONS]}")
        print(f"  slow logits[01] = {[float(slow[a]) for a in ACTIONS]}")
        print(f"  bias[01] = {[float(m.motor.bias[a]) for a in ACTIONS]}")
        combined = fast + float(m.config.consolidation_read_gain) * slow
        print(f"  combined logits[01] = {[float(combined[a]) for a in ACTIONS]}")
        probs = torch.softmax(combined / m.config.motor_temperature, dim=0)
        print(f"  combined probs[01] = {[float(probs[a]) for a in ACTIONS]}")


if __name__ == "__main__":
    main()
