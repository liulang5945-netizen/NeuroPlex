"""Seed model boundary built on the native Taiji substrate."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from taiji import (
    Taiji,
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
)

from .config import SeedConfig


class Seed:
    """The model-level organism; Taiji is its computational substrate.

    This first Seed runtime intentionally adds no second learning algorithm.
    Sensation, persistent dynamics, local plasticity, episodic completion,
    action and replay all cross the single ``substrate`` boundary.  Future
    organs and population coordination belong here and may consume Taiji's
    public state/action contracts, but may not reach around them to install a
    Transformer, teacher logits or an external event K/V store.
    """

    CHECKPOINT_FORMAT = "seed-native-v1"

    def __init__(
        self,
        config: Optional[SeedConfig] = None,
        *,
        device: torch.device | str = "cpu",
        episode_id: str = "episode-0",
    ) -> None:
        self.config = config or SeedConfig()
        self.substrate = Taiji(
            self.config.taiji,
            device=device,
            episode_id=episode_id,
        )

    @property
    def device(self) -> torch.device:
        return self.substrate.device

    @property
    def tick(self) -> int:
        return self.substrate.tick

    def snapshot(self) -> TaijiState:
        return self.substrate.snapshot()

    def reset_dynamics(self, *, episode_id: Optional[str] = None) -> None:
        self.substrate.reset_dynamics(episode_id=episode_id)

    def observe(
        self,
        symbol: int,
        *,
        learn: bool = True,
        learn_motor: Optional[bool] = None,
        use_memory: bool = True,
    ) -> TaijiStep:
        return self.substrate.observe(
            symbol,
            learn=learn,
            learn_motor=learn_motor,
            use_memory=use_memory,
        )

    def act(
        self,
        available_actions: Sequence[int],
        *,
        sample: bool = True,
    ) -> TaijiDecision:
        return self.substrate.act(available_actions, sample=sample)

    def settle_action(
        self,
        reward: float,
        *,
        learn: bool = True,
        learn_memory: bool = True,
        provenance: str = "experienced",
    ) -> TaijiOutcome:
        return self.substrate.settle_action(
            reward,
            learn=learn,
            learn_memory=learn_memory,
            provenance=provenance,
        )

    def consolidate(
        self,
        *,
        cycles: int = 1,
        learn: bool = True,
    ) -> TaijiConsolidation:
        return self.substrate.consolidate(cycles=cycles, learn=learn)

    def learn_bytes(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        include_boundary: bool = True,
    ) -> Dict[str, float]:
        return self.substrate.learn_bytes(
            data,
            epochs=epochs,
            include_boundary=include_boundary,
        )

    def score_bytes(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
    ) -> Dict[str, float]:
        return self.substrate.score_bytes(data, include_boundary=include_boundary)

    def generate(
        self,
        prompt: bytes,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
    ) -> bytes:
        return self.substrate.generate(
            prompt,
            length,
            stop_at_boundary=stop_at_boundary,
            sample=sample,
            reset=reset,
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        return self.substrate.parameter_count(active_only=active_only)

    def dense_equivalent_parameter_count(self) -> int:
        return self.substrate.dense_equivalent_parameter_count()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "substrate": self.substrate.checkpoint(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported Seed checkpoint format")
        actual = SeedConfig.from_dict(checkpoint["config"])
        if actual != self.config:
            raise ValueError("checkpoint configuration does not match Seed")
        self.substrate.restore(checkpoint["substrate"])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> "Seed":
        if checkpoint.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported Seed checkpoint format")
        config = SeedConfig.from_dict(checkpoint["config"])
        model = cls(config, device=device)
        model.restore(checkpoint)
        return model
