"""Seed's native self-evaluation organ.

The judge is the organism's "eye": it reads the self-referential signals the
substrate already produces while it consumes text -- per-tick surprise,
regional prediction-error accumulation and episodic recall confidence -- and
folds them into a single sortable quality score.  Higher quality means the
organism predicts the text better.

The combination weights are local parameters of the organ.  ``calibrate``
fits them with a closed-form ridge solve from known good/bad pairs; no
external reward model, teacher signal or legacy judge head participates.

Scoring is strictly read-only: the organ snapshots the substrate, runs a
no-learning pass over the text, and restores the snapshot before returning.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from .model import Seed

# Feature order used everywhere: surprise, accumulated regional error,
# episodic recall confidence, next-byte accuracy.
FEATURE_NAMES = (
    "mean_surprise",
    "mean_error_norm",
    "mean_confidence",
    "accuracy",
)

# Direction of quality: surprise and error accumulation hurt, recall
# confidence and prediction accuracy help.  These are only the birth
# weights; calibration replaces them with locally learned values.
DEFAULT_WEIGHTS = (-1.0, -0.25, 0.25, 0.5)


class SeedJudge:
    """Ranks text by how well the organism predicts it with its own state."""

    def __init__(self, seed: Seed) -> None:
        self.seed = seed
        self.weights = torch.tensor(
            DEFAULT_WEIGHTS, dtype=torch.float64, device=seed.device
        )

    @torch.no_grad()
    def score(self, text: bytes) -> Dict[str, float]:
        """Return the self-referential quality report for ``text``.

        Higher ``quality`` means the organism handles the text better.  The
        pass never learns and the substrate is restored to its prior state.
        """

        if not text:
            raise ValueError("cannot judge empty text")

        substrate = self.seed.substrate
        checkpoint = substrate.checkpoint()
        surprise_sum = 0.0
        error_sum = 0.0
        confidence_sum = 0.0
        correct = 0
        observations = 0
        try:
            substrate.reset_dynamics(episode_id="judge")
            step = substrate.observe(
                substrate.config.boundary_symbol, learn=False
            )
            for symbol in text:
                step = substrate.observe(int(symbol), learn=False)
                if step.prior_prediction is None:
                    continue
                observations += 1
                surprise_sum += float(step.surprise)
                error_sum += sum(step.local_error_norms) / max(
                    1, len(step.local_error_norms)
                )
                confidence_sum += float(step.memory_recall.confidence)
                correct += int(step.prior_prediction == int(symbol))
        finally:
            substrate.restore(checkpoint)

        if observations == 0:
            raise ValueError("text produced no predictive observations")

        features = torch.tensor(
            [
                surprise_sum / observations,
                error_sum / observations,
                confidence_sum / observations,
                correct / observations,
            ],
            dtype=torch.float64,
            device=self.seed.device,
        )
        quality = float(torch.dot(features, self.weights))
        return {
            "quality": quality,
            "mean_surprise": float(features[0]),
            "mean_error_norm": float(features[1]),
            "mean_confidence": float(features[2]),
            "accuracy": float(features[3]),
            "observations": float(observations),
        }

    def features(self, text: bytes) -> torch.Tensor:
        """Expose the raw feature vector in ``FEATURE_NAMES`` order."""

        report = self.score(text)
        return torch.tensor(
            [report[name] for name in FEATURE_NAMES],
            dtype=torch.float64,
            device=self.seed.device,
        )

    def calibrate(
        self,
        pairs: Sequence[Tuple[torch.Tensor, float]],
        *,
        ridge: float = 1e-3,
    ) -> float:
        """Learn the combination weights from known-quality pairs.

        Each pair is ``(feature_vector, target_quality)`` with the feature
        vector in ``FEATURE_NAMES`` order.  The solve is a local closed-form
        ridge regression; nothing outside this organ changes.  Returns the
        pairwise ranking accuracy of the fitted weights over the pairs.
        """

        if not pairs:
            raise ValueError("calibration requires at least one pair")
        if ridge <= 0:
            raise ValueError("ridge must be positive")

        features = torch.stack(
            [torch.as_tensor(pair[0], dtype=torch.float64) for pair in pairs]
        ).to(self.seed.device)
        targets = torch.tensor(
            [float(pair[1]) for pair in pairs],
            dtype=torch.float64,
            device=self.seed.device,
        )
        gram = features.T @ features + ridge * torch.eye(
            features.shape[1], dtype=torch.float64, device=self.seed.device
        )
        self.weights = torch.linalg.solve(gram, features.T @ targets)
        return self.ranking_accuracy(features, targets)

    @staticmethod
    def ranking_accuracy(
        features: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Fraction of comparable pairs whose score order matches the target."""

        scores = features @ torch.linalg.solve(
            features.T @ features
            + 1e-3 * torch.eye(
                features.shape[1], dtype=torch.float64, device=features.device
            ),
            features.T @ targets,
        )
        comparable = 0
        agreeing = 0
        for i in range(features.shape[0]):
            for j in range(i + 1, features.shape[0]):
                difference = targets[i] - targets[j]
                if difference == 0.0:
                    continue
                comparable += 1
                if (scores[i] - scores[j]) * difference > 0.0:
                    agreeing += 1
        if comparable == 0:
            return 1.0
        return agreeing / comparable
