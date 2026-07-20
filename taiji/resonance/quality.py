"""Quality filtering for resonance neurons.

Experiment 9 demonstrated that weak neurons dilute the strong ones in
resonance. Quality filtering ensures only neurons with sufficient PPL
participate in resonance, preventing the "weakest link" problem.

Uses both static and adaptive thresholds:
- Static: PPL < 100 for all domains (baseline)
- Adaptive: PPL < best_neuron_ppl * 2 (domain-aware)
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger("Taiji.QualityFilter")


class QualityFilter:
    """Filter neurons by PPL quality before resonance participation.

    Only neurons with PPL below the threshold participate in resonance.
    Neurons above the threshold continue training but don't dilute the ensemble.
    """

    def __init__(self, ppl_threshold: float = 100.0):
        """Args:
            ppl_threshold: static threshold. Neurons with PPL >= this
                           are excluded from resonance.
        """
        self.ppl_threshold = ppl_threshold
        self.neuron_ppls: Dict[str, float] = {}

    def set_ppls(self, neuron_ppls: Dict[str, float]) -> None:
        """Update per-neuron PPL records (called after each evaluation).

        Args:
            neuron_ppls: {neuron_id: ppl_value}
        """
        self.neuron_ppls.update(neuron_ppls)

    def filter(self, neuron_ids: list[str]) -> list[str]:
        """Filter out low-quality neurons.

        Args:
            neuron_ids: list of candidate neuron IDs.

        Returns:
            list of neuron IDs that passed the quality threshold.
        """
        filtered = []
        excluded = []
        for nid in neuron_ids:
            ppl = self.neuron_ppls.get(nid, float("inf"))
            if ppl < self.ppl_threshold:
                filtered.append(nid)
            else:
                excluded.append((nid, ppl))
        # 大规模时聚合输出，避免 N=1000 时刷屏
        if excluded:
            summary = ", ".join(f"{n}({p:.1f})" for n, p in excluded[:10])
            suffix = f" ... +{len(excluded)-10} more" if len(excluded) > 10 else ""
            logger.info(
                "[QualityFilter] excluded %d neurons (threshold=%.1f): %s%s",
                len(excluded), self.ppl_threshold, summary, suffix,
            )
        return filtered

    def adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on the best neuron's PPL.

        threshold = best_ppl * 2

        This is more lenient for hard domains (math has higher natural PPL)
        and stricter for easy domains (code has lower natural PPL).
        """
        if not self.neuron_ppls:
            return self.ppl_threshold
        best = min(self.neuron_ppls.values())
        return best * 2.0

    def filter_adaptive(self, neuron_ids: list[str]) -> list[str]:
        """Filter using adaptive threshold (domain-aware).

        Args:
            neuron_ids: list of candidate neuron IDs.

        Returns:
            list of neuron IDs that passed the adaptive threshold.
        """
        threshold = self.adaptive_threshold()
        filtered = []
        excluded = []
        for nid in neuron_ids:
            ppl = self.neuron_ppls.get(nid, float("inf"))
            if ppl < threshold:
                filtered.append(nid)
            else:
                excluded.append((nid, ppl))
        if excluded:
            summary = ", ".join(f"{n}({p:.1f})" for n, p in excluded[:10])
            suffix = f" ... +{len(excluded)-10} more" if len(excluded) > 10 else ""
            logger.info(
                "[QualityFilter:adaptive] excluded %d neurons (threshold=%.1f): %s%s",
                len(excluded), threshold, summary, suffix,
            )
        return filtered
