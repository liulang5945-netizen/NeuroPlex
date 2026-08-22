"""Historical Native v6: did replay coverage explain which pair failed?

The plan's next step assumes ``priority`` lets a few engrams monopolise
rehearsal, and that the starved pair is the one the probe cannot read back.  An
earlier version of this script measured a mild 35/19/15/11 split and appeared to
contradict that -- but it was not measuring the same thing the benchmark runs:
it drove ``memory.replay`` in a hand-rolled loop with a freshly seeded
generator, pretrained for 3 epochs instead of 6, and hardcoded seed 29 with 96
ticks while the failure was characterised on seed 11 at 384.

This version instruments the real ``consolidate`` -- same RNG stream, same
pretraining, same arm the verifier scores -- and joins the rehearsal counts
against the per-pair margins from the same run, so coverage and outcome are
measured on one model instead of two.
"""

from __future__ import annotations

import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from taiji import Taiji  # noqa: E402
from taiji.memory import EpisodicField  # noqa: E402
from verify_taiji_m6_endogenous_replay import (  # noqa: E402
    _config,
    _contingency,
    _episodes,
    _evaluate_contingency,
    _pretrain_corpus,
    _sleep,
    _store,
)

SEEDS = (11, 17, 29, 43, 61)


@contextmanager
def recording(counts: Counter):
    """Count the pair each accepted replay actually rehearses, in situ."""

    original = EpisodicField.replay

    def instrumented(self, state, *, tick, generator):
        next_state, replay = original(self, state, tick=tick, generator=generator)
        if replay.accepted:
            action = int(replay.action_probabilities.argmax().item())
            outcome = int(replay.outcome_probabilities.argmax().item())
            counts[(action, outcome)] += 1
        return next_state, replay

    EpisodicField.replay = instrumented
    try:
        yield
    finally:
        EpisodicField.replay = original


def run(seed: int, cycles: int) -> None:
    model = Taiji(_config(seed), episode_id="m6-bootstrap")
    model.learn_bytes(_pretrain_corpus(), epochs=6)
    episodes = _episodes()
    pairs = _contingency(episodes)
    _store(model, episodes)
    stored = model.checkpoint()

    counts: Counter = Counter()
    with recording(counts):
        full = _sleep(stored, cycles=cycles, learn=True, tag="full")

    metrics = _evaluate_contingency(full["checkpoint"], pairs)
    accepted = int(full["summary"]["accepted"])
    total = sum(counts.values())

    rehearsed: Counter = Counter()
    wrong = 0
    for (action, outcome), n in counts.items():
        if pairs.get(action) == outcome:
            rehearsed[action] += n
        else:
            wrong += n

    print(f"\n=== seed {seed}  cycles {cycles} ===")
    print(f"accepted={accepted}  recorded={total}  mis-rehearsed={wrong}")
    print(f"accuracy={metrics['contingency_accuracy']:.2f}  "
          f"mean_margin={metrics['mean_margin']:+.5f}")
    print(f"{'pair':>6} {'rehearsals':>11} {'share':>7} {'margin':>10}  read-back")
    for row in metrics["rows"]:
        action = ord(str(row["action"]))
        n = rehearsed.get(action, 0)
        share = n / total if total else 0.0
        ok = "ok" if row["predicted_outcome"] == row["expected_outcome"] else "WRONG"
        print(f"{row['action']}->{row['expected_outcome']:>3} {n:11d} {share:6.1%} "
              f"{row['margin']:+10.5f}  {ok}")

    covered = sorted(action for action in pairs if rehearsed.get(action, 0) > 0)
    starved = min(pairs, key=lambda a: rehearsed.get(a, 0))
    losers = [ord(str(r["action"])) for r in metrics["rows"]
              if r["predicted_outcome"] != r["expected_outcome"]]
    print(f"covered {len(covered)}/{len(pairs)}  "
          f"starved={chr(starved)}({rehearsed.get(starved, 0)})  "
          f"losers={[chr(a) for a in losers] or 'none'}  "
          f"starved_is_loser={starved in losers}")


def main() -> None:
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    seeds = [int(a) for a in sys.argv[2:]] or list(SEEDS)
    for seed in seeds:
        run(seed, cycles)


if __name__ == "__main__":
    main()
