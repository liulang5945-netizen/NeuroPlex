"""Typed events used by the Taiji substrate boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional

import torch


class EventKind(str, Enum):
    SENSORY = "sensory"
    PEER = "peer"
    PREDICTION = "prediction"
    REWARD = "reward"
    GOAL = "goal"
    MOTOR = "motor"
    CONTROL = "control"
    MEMORY = "memory"


class EventMode(str, Enum):
    REAL = "real"
    IMAGINED = "imagined"
    REPLAY = "replay"


@dataclass(frozen=True)
class TaijiEvent:
    """One causally labelled event.

    ``tick`` is logical time.  ``mode`` keeps real observations separate from
    internal simulation and replay.  The runtime validates vector dimensions.
    """

    tick: int
    episode_id: str
    source: str
    kind: EventKind
    value: torch.Tensor
    target: Optional[str] = None
    salience: float = 1.0
    reliability: float = 1.0
    mode: EventMode = EventMode.REAL

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("event tick cannot be negative")
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if not self.source:
            raise ValueError("source cannot be empty")
        if self.value.dim() != 1:
            raise ValueError(f"event value must be rank-1, got {tuple(self.value.shape)}")
        if not bool(torch.isfinite(self.value).all()):
            raise ValueError("event value contains NaN or Inf")
        if self.salience < 0.0:
            raise ValueError("salience cannot be negative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")

    def at_tick(self, tick: int) -> "TaijiEvent":
        return replace(self, tick=int(tick), value=self.value.detach().clone())

    def to_payload(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "episode_id": self.episode_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "value": self.value.detach().cpu().clone(),
            "salience": float(self.salience),
            "reliability": float(self.reliability),
            "mode": self.mode.value,
        }

    @classmethod
    def from_payload(
        cls, payload: Dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> "TaijiEvent":
        return cls(
            tick=int(payload["tick"]),
            episode_id=str(payload["episode_id"]),
            source=str(payload["source"]),
            target=payload.get("target"),
            kind=EventKind(payload["kind"]),
            value=payload["value"].detach().to(device=device).clone(),
            salience=float(payload.get("salience", 1.0)),
            reliability=float(payload.get("reliability", 1.0)),
            mode=EventMode(payload.get("mode", EventMode.REAL.value)),
        )


def dominant_mode(events: list[TaijiEvent]) -> EventMode:
    """Return the safest provenance label for a derived event."""

    if any(event.mode is EventMode.REAL for event in events):
        return EventMode.REAL
    if any(event.mode is EventMode.REPLAY for event in events):
        return EventMode.REPLAY
    return EventMode.IMAGINED

