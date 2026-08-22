"""Historical Native v6: why was the write basis weak and unstable?

ANSWERED, and the fix is shipped.  The defect was a path asymmetry in the
homeostatic set point.  The probe reads from ``reset_dynamics``, which rebuilds
state via ``initial_state`` and so pins every threshold at ``threshold_base``
(0.02).  Consolidation starts each replay from ``clear_dynamics``, which
deliberately *preserves* the adapted set point -- correct in itself, since sleep
must not discard what waking learned.  But replay then let homeostasis keep
integrating over a degenerate input: one symbol driven for sixteen ticks with no
waking traffic to balance it.  A unit the engram drives gains
``rate * (1 - target)`` per tick while a silent one sheds only ``rate * target``,
a 7:1 ratchet on exactly the units carrying the memory, measured at a 0.43 peak
set point -- twenty-one times base.  Since

    activity = tanh(relu(membrane - threshold - inhibition))

that subtracts straight off the drive, and ``local_update`` is linear in |trace|,
so the write all but vanished, ``captured`` became arbitrary on a near-null
trace, and one decoder row churned through 118 rewires without terminating.

``fabric.step`` now takes ``adapt_homeostasis``, and ``consolidate`` passes
False: the set point is read during replay, never written.  The arms here are
kept as regression controls -- ``adapt-homeostasis`` reinstates the defect,
``reset-threshold`` is the alternative fix that was rejected because it discards
the waking set point and lost to freezing on every behavioural column.

Two earlier candidates were measured and killed, and should not be revisited:

  * the episodic feedback term -- zeroing ``replay.cortical_projection`` inside the
    burst moved |basis| by under 12% and left the ratio to probe at 0.040, and the
    measured inhibition is 0.0002-0.0021, far too small to suppress anything;
  * a *progressive* within-bout ratchet -- |basis| quartiles for '0'->'+' ran
    0.0075 / 0.0033 / 0.0040 / 0.0055, which is not monotone, and Q1 was already
    1/15 of the probe.  The basis was not ground down over the bout; it was born
    weak, because the inflated set point is inherited across replays through
    ``clear_dynamics`` rather than accumulated within one.

Usage: python scripts/training/_diag_m6_write_basis.py [seed] [cycles]
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from taiji import Taiji  # noqa: E402
from taiji.fabric import RegionState, TaijiFabric  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    _config,
    _contingency,
    _episodes,
    _pretrain_corpus,
    _store,
)

ARMS = ("shipped", "adapt-homeostasis", "reset-threshold")


class BasisTally:
    """Records the eligibility basis every consolidation write actually lands on."""

    def __init__(self, *, arm: str) -> None:
        self.arm = arm
        self.bases: dict[int, list[torch.Tensor]] = {}
        self.thresholds: dict[int, list[torch.Tensor]] = {}

    def record(self, symbol: int, previous) -> None:
        self.bases.setdefault(symbol, []).append(previous[0].trace.detach().clone())
        self.thresholds.setdefault(symbol, []).append(
            previous[0].threshold.detach().clone()
        )


def _run(checkpoint, cycles: int, *, arm: str) -> tuple[BasisTally, object]:
    tally = BasisTally(arm=arm)
    with arm_patch(arm, tally):
        model = Taiji.from_checkpoint(deepcopy(checkpoint))
        summary = model.consolidate(cycles=cycles, learn=True)
    return tally, summary


@contextmanager
def arm_patch(arm: str, tally: BasisTally | None = None):
    """Applies one candidate fix as a monkey-patch and reverts it on the way out.

    Both fixes live here rather than in the model so that a bout can be measured
    under either one without committing anything on a guess, and so the
    behavioural arbiter and the basis diagnostic run *the same* intervention
    instead of two drifting copies of it.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    original_step = TaijiFabric.step
    original_clear = TaijiFabric.clear_dynamics

    def instrumented_step(
        self,
        sensory_activity,
        previous,
        *,
        learn: bool,
        episodic_feedback=None,
        learn_scale: float = 1.0,
        restructure: bool = False,
        adapt_homeostasis: bool = True,
    ):
        # ``restructure`` is true on exactly the first write repeat of each
        # accepted replay, and every later repeat is driven from the same held
        # ``settled`` state.  Recording only that tick keeps one sample per
        # replay; recording all eight would stack eight identical copies of one
        # basis and report a reproducibility that is an artefact of the hold.
        if tally is not None and learn and restructure:
            tally.record(int(sensory_activity.argmax()), previous)
        # ``adapt-homeostasis`` reinstates the pre-fix behaviour so the defect
        # this bout used to suffer stays measurable as a regression control.
        return original_step(
            self,
            sensory_activity,
            previous,
            learn=learn,
            episodic_feedback=episodic_feedback,
            learn_scale=learn_scale,
            restructure=restructure,
            adapt_homeostasis=(arm == "adapt-homeostasis"),
        )

    def instrumented_clear(self, regions):
        cleared = original_clear(self, regions)
        if arm != "reset-threshold":
            return cleared
        # Hand the burst the set point the probe reads from, so the only thing
        # separating the two measurements is removed.
        return tuple(
            RegionState(
                membrane=region.membrane,
                activity=region.activity,
                trace=region.trace,
                prediction=region.prediction,
                error=region.error,
                threshold=torch.full_like(
                    region.threshold, float(self.config.threshold_base)
                ),
                inhibition=region.inhibition,
            )
            for region in cleared
        )

    TaijiFabric.step = instrumented_step
    TaijiFabric.clear_dynamics = instrumented_clear
    try:
        yield
    finally:
        TaijiFabric.step = original_step
        TaijiFabric.clear_dynamics = original_clear


def _quartiles(values: list[float]) -> list[float]:
    """Mean of each successive quarter, so a monotone drift shows as a trend."""
    n = len(values)
    out = []
    for q in range(4):
        lo = (n * q) // 4
        hi = max(lo + 1, (n * (q + 1)) // 4)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def _describe(tag: str, tally: BasisTally, summary, pairs, probe_norms) -> None:
    print(f"\n{tag}   accepted={summary.accepted} rewires={summary.structural_events}")
    print(
        f"  {'pair':>10}  {'n':>4}  {'|basis|':>18}  {'cos':>14}"
        f"  {'probe':>7}  {'ratio':>6}  {'thr in->out':>14}  {'thr max':>8}"
        f"  {'|basis| Q1..Q4':>34}"
    )
    for action in sorted(pairs):
        outcome = int(pairs[action])
        traces = tally.bases.get(outcome, [])
        if not traces:
            print(f"  {chr(action)!r}->{chr(outcome)!r}: never written")
            continue
        stack = torch.stack(traces)
        norms = stack.norm(dim=1)
        unit = stack / norms.clamp_min(1e-12).unsqueeze(1)
        if stack.shape[0] >= 2:
            gram = unit @ unit.t()
            offdiag = gram[~torch.eye(gram.shape[0], dtype=torch.bool)]
            cos_mean, cos_min = float(offdiag.mean()), float(offdiag.min())
        else:
            cos_mean = cos_min = float("nan")
        probe = probe_norms.get(outcome, float("nan"))
        mean_norm = float(norms.mean())
        thresholds = tally.thresholds[outcome]
        peak = max(float(t.max()) for t in thresholds)
        # Separates a set point inherited from pretraining from one the bout
        # itself manufactured: if first and last agree, waking left it that way.
        first = float(thresholds[0].mean())
        last = float(thresholds[-1].mean())
        quarts = _quartiles([float(v) for v in norms])
        print(
            f"  {chr(action)!r}->{chr(outcome)!r}  {stack.shape[0]:>4}"
            f"  {mean_norm:>6.4f} [{float(norms.min()):.4f},{float(norms.max()):.4f}]"
            f"  {cos_mean:>6.4f}/{cos_min:>6.4f}"
            f"  {probe:>7.4f}  {mean_norm / probe if probe else float('nan'):>6.3f}"
            f"  {first:>6.4f}->{last:>6.4f}  {peak:>8.4f}"
            f"  {' '.join(f'{v:7.4f}' for v in quarts)}"
        )


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 384

    config = _config(seed)
    model = Taiji(config)
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    checkpoint = deepcopy(model.checkpoint())
    pairs = _contingency(episodes)

    # The probe basis is the target every write has to land on, so its norm sets
    # the scale the replayed bases are judged against.
    probe_norms: dict[int, float] = {}
    for action in sorted(pairs):
        probe = Taiji.from_checkpoint(deepcopy(checkpoint))
        probe.reset_dynamics(episode_id=f"scale-{action}")
        for _ in range(int(probe.config.replay_burst_repeats)):
            probe.observe(int(action), learn=False, learn_motor=False, use_memory=False)
        trace = probe.snapshot().regions[0].trace
        probe_norms[int(pairs[action])] = float(trace.norm())

    for arm in ARMS:
        tally, summary = _run(checkpoint, cycles, arm=arm)
        _describe(f"seed={seed} {cycles} cycles -- {arm}", tally, summary, pairs, probe_norms)


if __name__ == "__main__":
    main()
