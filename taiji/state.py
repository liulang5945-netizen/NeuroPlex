"""Persistent native state for the Taiji predictive fabric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


@dataclass
class RegionState:
    membrane: torch.Tensor
    activity: torch.Tensor
    trace: torch.Tensor
    prediction: torch.Tensor
    error: torch.Tensor
    threshold: torch.Tensor
    inhibition: float

    def clone(self) -> "RegionState":
        return RegionState(
            membrane=self.membrane.detach().clone(),
            activity=self.activity.detach().clone(),
            trace=self.trace.detach().clone(),
            prediction=self.prediction.detach().clone(),
            error=self.error.detach().clone(),
            threshold=self.threshold.detach().clone(),
            inhibition=float(self.inhibition),
        )

    def to_payload(self) -> Dict[str, Any]:
        cloned = self.clone()
        return {
            "membrane": cloned.membrane.cpu(),
            "activity": cloned.activity.cpu(),
            "trace": cloned.trace.cpu(),
            "prediction": cloned.prediction.cpu(),
            "error": cloned.error.cpu(),
            "threshold": cloned.threshold.cpu(),
            "inhibition": cloned.inhibition,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> "RegionState":
        return cls(
            membrane=payload["membrane"].detach().to(device).clone(),
            activity=payload["activity"].detach().to(device).clone(),
            trace=payload["trace"].detach().to(device).clone(),
            prediction=payload["prediction"].detach().to(device).clone(),
            error=payload["error"].detach().to(device).clone(),
            threshold=payload["threshold"].detach().to(device).clone(),
            inhibition=float(payload["inhibition"]),
        )


@dataclass
class TaijiState:
    version: int
    tick: int
    episode_id: str
    regions: Tuple[RegionState, ...]
    motor_context: torch.Tensor
    motor_probabilities: torch.Tensor
    last_symbol: Optional[int]

    def clone(self) -> "TaijiState":
        return TaijiState(
            version=int(self.version),
            tick=int(self.tick),
            episode_id=str(self.episode_id),
            regions=tuple(region.clone() for region in self.regions),
            motor_context=self.motor_context.detach().clone(),
            motor_probabilities=self.motor_probabilities.detach().clone(),
            last_symbol=None if self.last_symbol is None else int(self.last_symbol),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "tick": int(self.tick),
            "episode_id": str(self.episode_id),
            "regions": [region.to_payload() for region in self.regions],
            "motor_context": self.motor_context.detach().cpu().clone(),
            "motor_probabilities": self.motor_probabilities.detach().cpu().clone(),
            "last_symbol": self.last_symbol,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> "TaijiState":
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            episode_id=str(payload["episode_id"]),
            regions=tuple(
                RegionState.from_payload(region, device=device)
                for region in payload["regions"]
            ),
            motor_context=payload["motor_context"].detach().to(device).clone(),
            motor_probabilities=(
                payload["motor_probabilities"].detach().to(device).clone()
            ),
            last_symbol=(
                None if payload.get("last_symbol") is None
                else int(payload["last_symbol"])
            ),
        )


@dataclass(frozen=True)
class TaijiStep:
    tick: int
    observed_symbol: int
    predicted_symbol: int
    probabilities: torch.Tensor
    prior_prediction: Optional[int]
    prior_probability: Optional[float]
    surprise: Optional[float]
    activity_rates: Tuple[float, ...]
    local_error_norms: Tuple[float, ...]
