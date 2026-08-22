"""Native sleep scheduling: the judge drives consolidation.

Phase 3 wiring ("the eye drives the hand"): the self-evaluation organ picks
*what* to sleep on -- patterns the organism handles worst, ranked by judge
quality -- and the substrate's own endogenous replay decides *how* to
consolidate.  No Python replay list exists: the scheduler only creates real
experiences (``act`` + ``settle_action`` + next ``observe`` writes one
episodic engram each) and then hands the field to ``consolidate``, whose
internal priority gate accepts or rejects each spontaneous reactivation.

Pure observational training never writes episodes (there is no act/settle),
which is exactly why waking corpus runs replay nothing.  The reward of each
created experience is the organism's own judge quality for the text, so the
field's value axis is populated by self-assessment rather than by an
environment teacher.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .judge import SeedJudge
from .model import Seed

# The organism acts by predicting the next byte among its own strongest
# candidates; the restriction keeps the policy honest (it can only afford what
# it already believes) while leaving the full alphabet to the readouts.
ACTION_FANOUT = 8


class SeedSleepScheduler:
    """Selects consolidation targets by judge quality and sleeps endogenously."""

    def __init__(self, seed: Seed, judge: SeedJudge) -> None:
        self.seed = seed
        self.judge = judge

    def select_for_sleep(self, texts: Sequence[bytes], *, k: int) -> List[bytes]:
        """Return the ``k`` texts the organism handles worst.

        Lower judge quality means the organism predicts the text worse, so
        those patterns are consolidated first.  This is the explicit
        judge-to-sleep wire: selection reads only the self-evaluation organ.
        """

        if k < 0:
            raise ValueError("k cannot be negative")
        scored = [(self.judge.score(text)["quality"], text) for text in texts]
        scored.sort(key=lambda item: item[0])
        return [text for _quality, text in scored[: int(k)]]

    def experience(self, text: bytes, *, learn: bool = True) -> Dict[str, float]:
        """Live the text once so the episodic field holds a real engram.

        Each tick observes the true byte, acts on the organism's own top
        candidates and settles with the text's judge quality as reward.  The
        next observation binds each settled action to its outcome, producing
        one endogenous episodic write per tick.  Motor and fabric weights are
        never touched here -- only the episodic write path is exercised, and
        with ``learn=False`` even that write is withheld, so waking
        competence is untouched either way.
        """

        if not text:
            raise ValueError("cannot experience empty text")

        quality = self.judge.score(text)["quality"]
        boundary = self.seed.substrate.config.boundary_symbol
        self.seed.reset_dynamics(episode_id="sleep-experience")
        self.seed.observe(boundary, learn=False)
        actions = 0
        for symbol in text:
            self.seed.observe(int(symbol), learn=False)
            probabilities = self.seed.snapshot().motor_probabilities
            candidates = top_candidates(probabilities, ACTION_FANOUT)
            self.seed.act(candidates, sample=False)
            self.seed.settle_action(
                quality,
                learn=False,
                learn_memory=bool(learn),
                provenance="experienced",
            )
            actions += 1
        # Close the chain: the final pending experience needs its outcome.
        self.seed.observe(boundary, learn=False)
        return {"reward": quality, "actions": float(actions)}

    def night(
        self,
        texts: Sequence[bytes],
        *,
        cycles_per_text: int,
        learn: bool,
    ) -> Dict[str, float]:
        """Experience the selected texts, then sleep on what the field holds.

        Consolidation is entirely endogenous: ``consolidate`` reactivates
        engrams from the field's own value axis and its priority gate decides
        what is worth replaying.  With ``learn=False`` the bout replays
        without touching any weight.
        """

        if cycles_per_text <= 0:
            raise ValueError("cycles_per_text must be positive")
        if not texts:
            raise ValueError("night requires at least one text")

        total_cycles = 0
        accepted = 0
        priority_sum = 0.0
        confidence_sum = 0.0
        error_sum = 0.0
        for text in texts:
            self.experience(text, learn=bool(learn))
            if self.seed.substrate.memory.write_count <= 0:
                continue
            result = self.seed.consolidate(
                cycles=int(cycles_per_text), learn=bool(learn)
            )
            total_cycles += result.cycles
            accepted += result.accepted
            priority_sum += result.mean_priority * result.cycles
            confidence_sum += result.mean_confidence * result.cycles
            error_sum += result.mean_error_norm * result.cycles
        attempts = max(1, total_cycles)
        return {
            "texts": float(len(texts)),
            "cycles": float(total_cycles),
            "accepted": float(accepted),
            "mean_priority": priority_sum / attempts,
            "mean_confidence": confidence_sum / attempts,
            "mean_error_norm": error_sum / attempts,
        }


def top_candidates(probabilities: torch.Tensor, count: int) -> List[int]:
    """Return the ``count`` strongest byte candidates from a policy vector."""

    _values, indices = torch.sort(probabilities.detach(), descending=True)
    return [int(index) for index in indices[: int(count)]]
