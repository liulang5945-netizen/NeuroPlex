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


def _stream_baseline(checkpoint, ticks: int) -> torch.Tensor:
    """Per-unit long-run mean of the region-0 trace over one waking stream.

    This is the quantity an adaptive baseline inside ``fabric.step`` converges
    to, so it is the honest stand-in for the mechanism under test.  The oracle
    baseline below (the mean over the four probe bases) is not implementable
    online -- it needs all four probes at once -- and only bounds what any
    common-mode removal could achieve at read time.
    """

    model = Taiji.from_checkpoint(deepcopy(checkpoint))
    model.reset_dynamics(episode_id="m6-baseline-stream")
    corpus = _pretrain_corpus()
    total = torch.zeros_like(model.snapshot().regions[0].trace)
    for index in range(ticks):
        model.observe(
            corpus[index % len(corpus)],
            learn=False,
            learn_motor=False,
            use_memory=False,
        )
        total += model.snapshot().regions[0].trace
    return total / float(ticks)


def _positive_count(margins: torch.Tensor) -> int:
    return int((margins > 0.0).sum().item())


def locus(seed: int, cycles: int) -> dict:
    """Offline locus check: can common-mode removal on the basis flip all 4 pairs?

    Section 6.6 fixed this as the precondition before touching ``fabric.step``.
    The margin is ``(w_true - w_rival) . x``, linear in the basis ``x``, so
    positive rescaling cannot change a sign: the renormalisation the real
    mechanism would apply is invisible to this test, and so is the softmax,
    being monotone.  Sign counting on the raw logit difference is therefore
    exactly the end-to-end criterion, and the whole check reduces to a scan of
    one scalar.

    The scan has a distinguished point.  Writing the basis as
    ``b_p = a_p u + d_p`` with ``u`` the four-action mean, subtracting the mean
    at gain exactly 1 leaves ``d_p`` alone -- the only part of the basis that
    carries probe identity.  Gain 1 is thus the honest ceiling of any
    common-mode removal: beyond it the mean is being *added back with the
    opposite sign*, which buys margin from a probe-independent direction rather
    than from information, and a pair rescued there is rescued by a bias, not by
    a better read-out.  The verdict below is therefore taken at gain <= 1, and
    the residual margins are reported separately so a failure can be charged to
    the right cause.
    """

    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    pairs = _contingency(episodes)
    _store(model, episodes)
    stored = model.checkpoint()

    sleeper = Taiji.from_checkpoint(deepcopy(stored))
    sleeper.reset_dynamics(episode_id="m6-sleep-full")
    sleeper.consolidate(cycles=cycles, learn=True)
    rested = sleeper.checkpoint()

    actions = sorted(pairs)
    bases = torch.stack([_probe_basis(rested, action) for action in actions])
    common = bases.mean(dim=0)
    decoder = sleeper.fabric.decoders[0]

    def _margins_at(baseline: torch.Tensor, gain: float) -> torch.Tensor:
        adjusted = bases - gain * baseline
        evidence = torch.stack([
            decoder.forward(row).detach()[SELECTOR] for row in adjusted
        ])
        return _margins(evidence, pairs, actions)

    raw = _margins_at(common, 0.0)
    residual = _margins_at(common, 1.0)
    stream = _stream_baseline(rested, ticks=256)

    def _cells(margins: torch.Tensor) -> str:
        return "  ".join(f"{v * 1e4:+8.2f}" for v in margins.tolist())

    print(f"\n=== seed {seed}: offline locus check (cycles {cycles}) ===")
    print(f"    raw basis       {_cells(raw)}   {_positive_count(raw)}/{len(actions)}")
    print(f"    pure residual   {_cells(residual)}   "
          f"{_positive_count(residual)}/{len(actions)}   <- ceiling of any common-mode removal")

    reachable = _positive_count(raw)
    for kind, baseline in (("oracle", common), ("stream", stream)):
        for gain in (0.25, 0.5, 0.75, 1.0):
            hits = _positive_count(_margins_at(baseline, gain))
            reachable = max(reachable, hits)
        window = torch.linspace(0.0, 1.0, 41)
        curve = " ".join(
            str(_positive_count(_margins_at(baseline, float(g)))) for g in window[::4]
        )
        print(f"    {kind:<7} positives over gain 0.0..1.0: {curve}")

    delta_all = bases - common
    normed = delta_all / delta_all.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosines = normed @ normed.t()
    offdiag = [
        float(cosines[i, j].item())
        for i in range(len(actions)) for j in range(i + 1, len(actions))
    ]
    print(f"    residual pairwise cosine  min {min(offdiag):+.3f}"
          f"  mean {sum(offdiag) / len(offdiag):+.3f}  max {max(offdiag):+.3f}")

    projections = _replay_projections(deepcopy(stored), cycles=cycles)
    if projections:
        mean_projection = torch.stack(projections).mean(dim=0)
        write = torch.stack([
            _write_basis(rested, action, mean_projection) for action in actions
        ])
        alignment = []
        for row in range(len(actions)):
            probe = bases[row]
            other = write[row]
            denominator = (probe.norm() * other.norm()).clamp_min(1e-12)
            alignment.append(float((torch.dot(probe, other) / denominator).item()))
        print(f"    write-vs-probe basis cosine  "
              + "  ".join(f"{value:+.3f}" for value in alignment))
        write_margins = _margins(
            torch.stack([decoder.forward(row).detach()[SELECTOR] for row in write]),
            pairs, actions,
        )
        print(f"    margins on write basis  {_cells(write_margins)}"
              f"   {_positive_count(write_margins)}/{len(actions)}")

    print("    per-pair attribution (true row vs best rival at gain 0)")
    for row, action in enumerate(actions):
        target = OUTCOMES.index(pairs[action])
        evidence = decoder.forward(bases[row]).detach()[SELECTOR]
        masked = evidence.clone()
        masked[target] = float("-inf")
        rival = int(masked.argmax().item())
        true_row = _row_vector(decoder, OUTCOMES[target])
        rival_row = _row_vector(decoder, OUTCOMES[rival])
        weight_diff = true_row - rival_row
        delta = bases[row] - common
        common_term = float(torch.dot(weight_diff, common).item())
        residual_term = float(torch.dot(weight_diff, delta).item())
        verdict = "residual WRONG" if residual_term <= 0.0 else "residual ok"
        print(f"      {chr(action)}->{chr(pairs[action])} vs {chr(OUTCOMES[rival])}"
              f"  common {common_term * 1e4:+9.2f}  residual {residual_term * 1e4:+9.2f}"
              f"   {verdict}")
        if residual_term <= 0.0:
            true_term = float(torch.dot(true_row, delta).item())
            rival_term = float(torch.dot(rival_row, delta).item())
            support_true = set(decoder.pre_index[OUTCOMES[target]].tolist())
            support_rival = set(decoder.pre_index[OUTCOMES[rival]].tolist())
            shared = len(support_true & support_rival)
            peak = int(delta.abs().argmax().item())
            seen = "yes" if peak in support_true else "NO"
            print(f"        root cause: true row reads {true_term * 1e4:+9.2f},"
                  f" rival reads {rival_term * 1e4:+9.2f},"
                  f" shared contacts {shared}/{len(support_true)},"
                  f" peak residual unit in true fan-in: {seen}")

    return {
        "seed": seed,
        "raw_positive": _positive_count(raw),
        "residual_positive": _positive_count(residual),
        "reachable_positive": reachable,
        "pairs": len(actions),
    }


def _replay_projections(checkpoint, cycles: int) -> list:
    """Collect the cortical projections consolidation actually replays with."""

    model = Taiji.from_checkpoint(checkpoint)
    model.reset_dynamics(episode_id="m6-projection-probe")
    memory_state = model.snapshot().memory
    tick = int(model.snapshot().tick)
    collected = []
    for _ in range(int(cycles)):
        memory_state, replay = model.memory.replay(
            memory_state, tick=tick, generator=model._memory_rng
        )
        tick += 1
        if replay.accepted:
            collected.append(replay.cortical_projection.detach().clone())
    return collected


def _write_basis(checkpoint, action: int, projection: torch.Tensor) -> torch.Tensor:
    """The settled trace a *replay burst* writes onto, feedback included.

    ``_probe_basis`` reproduces the evaluation probe, which arrives with the
    action alone and ``use_memory=False``.  Consolidation drives the identical
    burst but passes ``episodic_feedback=replay.cortical_projection``, and that
    feedback enters ``drive`` with a nonzero gain.  If the two bases differ, the
    write lands on one vector and the probe interrogates another -- a mismatch no
    amount of common-mode removal at read time can repair.
    """

    model = Taiji.from_checkpoint(deepcopy(checkpoint))
    model.reset_dynamics(episode_id=f"m6-write-{action}")
    regions = model.fabric.clear_dynamics(model.snapshot().regions)
    for _ in range(int(model.config.replay_burst_repeats)):
        regions, _rates, _errors = model.fabric.step(
            model.sensor.encode(action),
            regions,
            learn=False,
            episodic_feedback=projection,
            learn_scale=0.0,
            adapt_homeostasis=False,
        )
    return regions[0].trace.detach().clone()


def _row_vector(decoder, row: int) -> torch.Tensor:
    """Densify one decoder row so margins can be attributed by inner product."""

    dense = torch.zeros(decoder.in_features)
    dense.scatter_add_(0, decoder.pre_index[row], decoder.edge_weight[row])
    return dense


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "locus":
        cycles = 96
        seeds = [int(a) for a in sys.argv[2:]] or list(SEEDS)
        results = [locus(seed, cycles) for seed in seeds]
        print("\n=== locus verdict (criterion: 4/4 positive at gain <= 1) ===")
        passing = 0
        for row in results:
            ok = row["reachable_positive"] == row["pairs"]
            passing += int(ok)
            print(f"    seed {row['seed']:3d}  raw {row['raw_positive']}/{row['pairs']}"
                  f"  residual {row['residual_positive']}/{row['pairs']}"
                  f"  best {row['reachable_positive']}/{row['pairs']}"
                  f"  {'PASS' if ok else 'fail'}")
        print(f"    seeds reaching 4/4: {passing}/{len(results)}")
        return
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
