"""Throwaway diagnostic: which candidate set-point fix actually moves behaviour?

The probe softmaxes only over the four outcome bytes, so chance is 0.25 and the
entropy trap of the full 257-way readout does not apply here.  Yet measured
true probabilities sit at 0.2509 with margins near 1e-3, which means the four
raw logits are nearly identical.  This measures the raw evidence directly: the
absolute scale, the pre-existing row bias sleep has to overcome, and how much
one consolidation bout actually moves it.

It now runs that measurement once per arm of ``_diag_m6_write_basis``.  Both
candidate fixes for the ``clear_dynamics`` set-point asymmetry restore the write
basis and collapse rewiring from 311 to 16, so basis fidelity cannot choose
between them: ``reset-threshold`` lands exactly on probe scale but discards the
waking-learned set point, while ``freeze-homeostasis`` preserves it and
overshoots (ratio 1.13-1.43), risking a write/read scale mismatch.  Only the
evidence the probe actually reads can arbitrate.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from taiji import Taiji  # noqa: E402
from _diag_m6_write_basis import ARMS, arm_patch  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    OUTCOMES,
    _config,
    _contingency,
    _episodes,
    _pretrain_corpus,
    _store,
)


def _settled_trace(checkpoint, action: int) -> torch.Tensor:
    probe = Taiji.from_checkpoint(deepcopy(checkpoint))
    probe.reset_dynamics(episode_id=f"probe-{action}")
    for _ in range(int(probe.config.replay_burst_repeats)):
        probe.observe(int(action), learn=False, learn_motor=False, use_memory=False)
    return probe.snapshot().regions[0].trace.detach().clone()


def _logit_matrix(checkpoint, pairs) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows are probe actions, columns are the four outcome bytes."""

    selector = torch.tensor(OUTCOMES, dtype=torch.long)
    model = Taiji.from_checkpoint(deepcopy(checkpoint))
    decoder = model.fabric.decoders[0]
    logits = []
    norms = []
    for action in sorted(pairs):
        trace = _settled_trace(checkpoint, int(action))
        evidence = decoder.forward(trace).detach()
        logits.append(evidence[selector].clone())
        norms.append(float(trace.norm()))
    return torch.stack(logits), torch.tensor(norms)


def _report(tag: str, logits: torch.Tensor, norms: torch.Tensor, pairs) -> None:
    actions = sorted(pairs)
    print(f"\n{tag}")
    print(f"  whole-alphabet-free logits, rows=probe action, cols={[chr(o) for o in OUTCOMES]}")
    for row, action in enumerate(actions):
        target = OUTCOMES.index(pairs[action])
        cells = "  ".join(
            f"{'*' if col == target else ' '}{float(logits[row, col]):+.5f}"
            for col in range(logits.shape[1])
        )
        print(f"   '{chr(action)}'->'{chr(pairs[action])}' {cells}   |trace|={float(norms[row]):.4f}")
    spread = float(logits.max() - logits.min())
    print(f"  logit spread across the whole matrix = {spread:.5f}")
    print(f"  column means (per-outcome row bias)  = "
          + "  ".join(f"{chr(OUTCOMES[c])}:{float(logits[:, c].mean()):+.5f}"
                      for c in range(logits.shape[1])))

    # Every one of the four outcome bytes is read out of the SAME decoder row
    # across all four probe actions, so that row's column mean is its response
    # to any trace at all -- a marginal frequency prior with no contingency in
    # it.  Pretraining sets it, replay has to outrun it, and it is measured at
    # two orders of magnitude above what one bout writes.  Subtracting it costs
    # nothing here and says whether the contingency is present but buried, or
    # simply absent.
    centred = logits - logits.mean(dim=0, keepdim=True)
    raw_hits = 0
    centred_hits = 0
    for row, action in enumerate(actions):
        target = OUTCOMES.index(pairs[action])
        raw_hits += int(int(logits[row].argmax()) == target)
        centred_hits += int(int(centred[row].argmax()) == target)
    print(f"  argmax accuracy: raw={raw_hits}/{len(actions)}"
          f"   column-centred={centred_hits}/{len(actions)}")


def _arbitrate(checkpoint, before, pairs, *, cycles: int, arm: str) -> int:
    """Runs one bout under one candidate fix and reports what behaviour did.

    Only ``consolidate`` runs inside the patch.  The probe that reads the result
    must run outside it, or ``freeze-homeostasis`` would pin the probe's own set
    point too and the arbitration would be measuring its own intervention rather
    than what the bout wrote.
    """

    slept = Taiji.from_checkpoint(deepcopy(checkpoint))
    with arm_patch(arm):
        summary = slept.consolidate(cycles=cycles, learn=True)
    after, norms_after = _logit_matrix(slept.checkpoint(), pairs)
    _report(
        f"arm={arm} AFTER {cycles} cycles "
        f"(accepted={summary.accepted}, rewires={summary.structural_events})",
        after,
        norms_after,
        pairs,
    )

    delta = after - before
    print(f"\narm={arm}: did sleep move the TRUE cell more than its rivals?")
    actions = sorted(pairs)
    wins = 0
    for row, action in enumerate(actions):
        target = OUTCOMES.index(pairs[action])
        true_move = float(delta[row, target])
        rival_move = float(
            torch.cat([delta[row, :target], delta[row, target + 1 :]]).max()
        )
        won = true_move > rival_move
        wins += int(won)
        print(
            f"   '{chr(action)}'->'{chr(pairs[action])}'  true{true_move:+.6f}"
            f"  best_rival{rival_move:+.6f}  {'WINS ' if won else 'loses'}"
        )
    print(f"  mean |delta| = {float(delta.abs().mean()):.6f}   wins={wins}/{len(actions)}")

    # The logit is an inner product of a row's weights with the settled trace.
    # |trace| barely moves across sleep, so any collapse has to come from the
    # weights.  Decay is far too small to explain it: 1e-5 scaled by a 0.45
    # learn_scale over ~768 writes retains 99.7%.  So the question is whether
    # the row still points at the units the trace occupies -- that is, whether
    # rewiring is trading away contacts that were carrying the signal.
    print(f"arm={arm}: is the TRUE row still aimed at the trace?")
    old_decoder = Taiji.from_checkpoint(deepcopy(checkpoint)).fabric.decoders[0]
    new_decoder = slept.fabric.decoders[0]
    for action in sorted(pairs):
        outcome = int(pairs[action])
        trace = _settled_trace(checkpoint, int(action))
        activity = trace.abs()
        old_support = old_decoder.pre_index[outcome].long()
        new_support = new_decoder.pre_index[outcome].long()
        whole = float((trace**2).sum())
        old_captured = float((trace[old_support] ** 2).sum()) / max(1e-12, whole)
        new_captured = float((trace[new_support] ** 2).sum()) / max(1e-12, whole)
        kept = len(set(old_support.tolist()) & set(new_support.tolist()))
        old_weights = old_decoder.edge_weight[outcome]
        new_weights = new_decoder.edge_weight[outcome]
        old_contribution = float((old_weights * trace[old_support]).sum())
        new_contribution = float((new_weights * trace[new_support]).sum())
        print(
            f"   '{chr(action)}'->'{chr(outcome)}'"
            f"  captured {old_captured:.4f}->{new_captured:.4f}"
            f"  kept={kept}/{old_support.numel()}"
            f"  |w| {float(old_weights.abs().sum()):.4f}->{float(new_weights.abs().sum()):.4f}"
            f"  logit {old_contribution:+.6f}->{new_contribution:+.6f}"
        )
        # A zero-weight donor contributes nothing, so if captured energy rose
        # while the logit fell, the row swapped away weight that was working.
        moved_in = sorted(set(new_support.tolist()) - set(old_support.tolist()))
        if moved_in:
            gained = float(sum(float(activity[u]) for u in moved_in))
            print(f"      grew {len(moved_in)} contacts carrying activity {gained:.4f}")
    return wins


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 96

    config = _config(seed)
    model = Taiji(config)
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    checkpoint = deepcopy(model.checkpoint())
    pairs = _contingency(episodes)

    before, norms_before = _logit_matrix(checkpoint, pairs)
    _report(f"seed={seed} BEFORE sleep", before, norms_before, pairs)

    tally = {arm: _arbitrate(checkpoint, before, pairs, cycles=cycles, arm=arm)
             for arm in ARMS}
    print(f"\nseed={seed} {cycles} cycles -- TRUE-cell wins by arm")
    for arm in ARMS:
        print(f"   {arm:<20} {tally[arm]}/{len(pairs)}")


if __name__ == "__main__":
    main()
