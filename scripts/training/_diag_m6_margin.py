"""Throwaway diagnostic: why does a well rehearsed pair still lose its margin?

Coverage is no longer the constraint -- every pair now gets 8-33% of the
rehearsals -- yet three pairs still read back wrong with margins within 0.002 of
zero.  Section 6.5 asks for a quantitative attribution before any mechanism
change: is the true cell simply underdosed, or is the same burst that teaches it
also lifting its competitor?

The write is linear in the basis it lands on, so the attribution can be read off
directly.  Four bases are frozen up front by probing the pre-sleep checkpoint
exactly the way ``_evaluate_contingency`` probes the post-sleep one.  Then, at
every replay entry, ``decoders[0]`` is evaluated on all four frozen bases, which
costs four sparse matvecs.  Differencing consecutive snapshots attributes the
whole change to the one accepted replay that happened in between, so each burst
can be charged for what it did to every basis, not just its own.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from taiji import Taiji  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    ACTIONS,
    OUTCOMES,
    _config,
    _contingency,
    _episodes,
    _evaluate_contingency,
    _pretrain_corpus,
    _store,
)

SEEDS = (11, 17, 29, 43, 61)
SELECTOR = torch.tensor(OUTCOMES, dtype=torch.long)


def _probe_basis(checkpoint, action: int) -> torch.Tensor:
    """Reproduce the evaluation probe's settled region-0 trace for one action."""

    model = Taiji.from_checkpoint(deepcopy(checkpoint))
    model.reset_dynamics(episode_id=f"m6-basis-{action}")
    for _ in range(int(model.config.replay_burst_repeats)):
        model.observe(action, learn=False, learn_motor=False, use_memory=False)
    return model.snapshot().regions[0].trace.detach().clone()


def _evidence(model: Taiji, bases) -> torch.Tensor:
    """Rows are probe bases, columns are the four outcome rows of decoder 0."""

    decoder = model.fabric.decoders[0]
    return torch.stack([
        decoder.forward(basis).detach()[SELECTOR] for basis in bases
    ])


def _margins(evidence: torch.Tensor, pairs, actions) -> torch.Tensor:
    """True-cell minus best-rival, in logit space, one entry per basis."""

    out = torch.zeros(len(actions))
    for row, action in enumerate(actions):
        target = OUTCOMES.index(pairs[action])
        values = evidence[row]
        rivals = torch.cat([values[:target], values[target + 1:]])
        out[row] = values[target] - rivals.max()
    return out


@contextmanager
def attributing(model: Taiji, bases, pairs, actions, ledger, counts):
    """Charge each accepted replay with its effect on every probe basis."""

    original = EpisodicField.replay
    state = {"pair": None, "margins": _margins(_evidence(model, bases), pairs, actions)}

    def instrumented(self, memory_state, *, tick, generator):
        current = _margins(_evidence(model, bases), pairs, actions)
        if state["pair"] is not None:
            ledger[state["pair"]] += current - state["margins"]
            counts[state["pair"]] += 1
        state["margins"] = current
        next_state, replay = original(self, memory_state, tick=tick, generator=generator)
        if replay.accepted:
            state["pair"] = (
                int(replay.action_probabilities.argmax().item()),
                int(replay.outcome_probabilities.argmax().item()),
            )
        else:
            state["pair"] = None
        return next_state, replay

    EpisodicField.replay = instrumented
    try:
        yield
    finally:
        EpisodicField.replay = original
        if state["pair"] is not None:
            final = _margins(_evidence(model, bases), pairs, actions)
            ledger[state["pair"]] += final - state["margins"]
            counts[state["pair"]] += 1


def _support_overlap(model: Taiji) -> str:
    """How many of the 16 contacts the four outcome rows hold in common."""

    decoder = model.fabric.decoders[0]
    supports = [set(decoder.pre_index[row].tolist()) for row in OUTCOMES]
    cells = []
    for left in range(len(OUTCOMES)):
        cells.append(" ".join(
            f"{len(supports[left] & supports[right]):3d}"
            for right in range(len(OUTCOMES))
        ))
    return "\n".join(f"    {chr(OUTCOMES[i])} {row}" for i, row in enumerate(cells))


def run(seed: int, cycles: int) -> None:
    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    pairs = _contingency(episodes)
    _store(model, episodes)
    stored = model.checkpoint()

    actions = sorted(pairs)
    bases = [_probe_basis(stored, action) for action in actions]

    sleeper = Taiji.from_checkpoint(deepcopy(stored))
    sleeper.reset_dynamics(episode_id="m6-sleep-full")

    ledger = defaultdict(lambda: torch.zeros(len(actions)))
    counts: Counter = Counter()
    with attributing(sleeper, bases, pairs, actions, ledger, counts):
        summary = sleeper.consolidate(cycles=cycles, learn=True)

    metrics = _evaluate_contingency(sleeper.checkpoint(), pairs)

    print(f"\n=== seed {seed}  cycles {cycles}  accepted {summary.accepted} ===")

    print("  basis cosine overlap (pre-sleep probe traces)")
    for left, action in enumerate(actions):
        row = " ".join(
            f"{torch.nn.functional.cosine_similarity(bases[left], bases[right], dim=0).item():5.2f}"
            for right in range(len(actions))
        )
        print(f"    {chr(action)} {row}")

    print("  outcome-row support overlap after sleep (of 16)")
    print(_support_overlap(sleeper))

    print("  per-rehearsal margin delta x1e4: rows = burst pair, cols = probe basis")
    print("    burst    n  " + "  ".join(f"{chr(a):>8}" for a in actions))
    for pair in sorted(ledger, key=lambda p: -counts[p]):
        action, outcome = pair
        n = counts[pair]
        mean = ledger[pair] / max(1, n)
        tag = f"{chr(action)}->{chr(outcome)}"
        true_pair = pairs.get(action) == outcome
        cells = "  ".join(f"{value * 1e4:8.2f}" for value in mean.tolist())
        print(f"    {tag:<7} {n:4d}  {cells}{'' if true_pair else '   [mis]'}")

    print("  read-back")
    for row in metrics["rows"]:
        ok = "ok" if row["predicted_outcome"] == row["expected_outcome"] else "WRONG"
        print(f"    {row['action']}->{row['expected_outcome']} "
              f"margin={row['margin']:+.5f}  {ok}")


def sweep(seed: int) -> None:
    """Does the basis correlation grow with the burst?  Measured: no.

    The hypothesis was that a single byte drives only the ~4 of 64 region-0 units
    whose fan-in samples that sensory unit -- near disjoint across the four
    actions -- and that every later tick spreads activity through the shared
    transition matrix until the four bases converge on common modes.  The sweep
    falsifies it.  At one single tick seed 11 already sits at max cosine 0.321
    and only reaches 0.369 by tick 8; seed 61 goes 0.258 -> 0.275 and falls back
    to 0.224 by tick 12; the all-correct control seed 17 *decreases* monotonically
    from 0.200 to 0.139.  Recurrent spread is therefore not the source and a
    shorter burst is not a fix: the correlation is already present in the very
    first tick, which only bottom-up drive and the carried-over per-unit
    thresholds can explain.  See ``origin`` for the decomposition.
    """

    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    stored = model.checkpoint()
    actions = sorted(_contingency(episodes))

    print(f"\n=== seed {seed}: basis density and correlation vs burst length ===")
    print(f"    {'ticks':>5} {'active':>16} {'max cos':>8} {'mean cos':>9}")
    for repeats in (1, 2, 3, 4, 6, 8, 12):
        traces = []
        for action in actions:
            probe = Taiji.from_checkpoint(deepcopy(stored))
            probe.reset_dynamics(episode_id=f"m6-sweep-{action}")
            for _ in range(repeats):
                probe.observe(action, learn=False, learn_motor=False, use_memory=False)
            traces.append(probe.snapshot().regions[0].trace.detach().clone())
        active = [int((trace.abs() > 1e-6).sum().item()) for trace in traces]
        cosines = [
            torch.nn.functional.cosine_similarity(
                traces[left], traces[right], dim=0
            ).item()
            for left in range(len(actions))
            for right in range(left + 1, len(actions))
        ]
        counts = "/".join(str(value) for value in active)
        print(f"    {repeats:5d} {counts:>16} {max(cosines):8.3f} "
              f"{sum(cosines) / len(cosines):9.3f}")


def origin(seed: int) -> None:
    """Split each basis into the four-action common mode and its residual.

    The sweep shows the correlation is already there at tick 1, so it cannot come
    from recurrent mixing.  With fan-in 16 over 257 sensory units, two different
    bytes should share about 0.25 of their ~4 driven units, which would put the
    tick-1 cosine near zero -- yet it measures 0.2-0.32.  The excess can only be
    units that respond to *every* action, so decompose the settled basis into the
    mean across actions and what is left.  If the common mode carries a large
    share of the energy and the residuals are near orthogonal, the interference is
    a shared-substrate problem (promiscuous, low-threshold units) rather than a
    per-pair geometry problem, and the lever is whatever sets those thresholds.
    """

    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    _store(model, episodes)
    stored = model.checkpoint()
    actions = sorted(_contingency(episodes))

    bases = torch.stack([_probe_basis(stored, action) for action in actions])
    common = bases.mean(dim=0)
    residual = bases - common

    total = float((bases**2).sum().item())
    shared = float((common**2).sum().item()) * len(actions)
    print(f"\n=== seed {seed}: common mode vs residual ===")
    print(f"    energy in common mode: {shared / total:6.1%}")

    def _cosines(stack: torch.Tensor) -> list[float]:
        return [
            torch.nn.functional.cosine_similarity(
                stack[left], stack[right], dim=0
            ).item()
            for left in range(len(actions))
            for right in range(left + 1, len(actions))
        ]

    raw = _cosines(bases)
    stripped = _cosines(residual)
    print(f"    pairwise cosine  raw: max {max(raw):.3f} mean "
          f"{sum(raw) / len(raw):.3f}")
    print(f"    pairwise cosine  residual: max {max(stripped):.3f} mean "
          f"{sum(stripped) / len(stripped):.3f}")

    counts = (bases.abs() > 1e-6).sum(dim=0)
    promiscuous = int((counts == len(actions)).sum().item())
    touched = int((counts > 0).sum().item())
    print(f"    units driven by all {len(actions)} actions: {promiscuous} "
          f"of {touched} touched ({bases.shape[1]} total)")

    checkpoint = Taiji.from_checkpoint(deepcopy(stored))
    thresholds = checkpoint.snapshot().regions[0].threshold
    base = float(checkpoint.config.threshold_base)
    if promiscuous:
        selected = thresholds[counts == len(actions)]
        print(f"    threshold on those units: mean {selected.mean() / base:5.2f}x "
              f"base, min {selected.min() / base:5.2f}x")
    quiet = thresholds[counts == 0]
    if quiet.numel():
        print(f"    threshold on never-driven units: mean "
              f"{quiet.mean() / base:5.2f}x base")
    energy = bases.abs().sum(dim=0)
    order = torch.argsort(energy, descending=True)[:6]
    top = ", ".join(
        f"u{int(i)}:{int(counts[i])}a/{float(thresholds[i] / base):.1f}x"
        for i in order
    )
    print(f"    strongest units (actions/threshold): {top}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        for seed in [int(a) for a in sys.argv[2:]] or list(SEEDS):
            sweep(seed)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "origin":
        for seed in [int(a) for a in sys.argv[2:]] or list(SEEDS):
            origin(seed)
        return
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    seeds = [int(a) for a in sys.argv[2:]] or list(SEEDS)
    for seed in seeds:
        run(seed, cycles)


if __name__ == "__main__":
    main()
