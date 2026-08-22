"""Top-level configuration for the Seed model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from taiji import TaijiConfig


@dataclass(frozen=True)
class SeedConfig:
    """Seed owns model-level composition; Taiji owns substrate dynamics."""

    taiji: TaijiConfig = field(default_factory=TaijiConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"taiji": self.taiji.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedConfig":
        return cls(taiji=TaijiConfig.from_dict(dict(payload["taiji"])))
