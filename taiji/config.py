"""Configuration contract for the native Taiji architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TaijiConfig:
    """Shape and dynamics of one Taiji predictive fabric.

    Sizes are implementation parameters, not model identities.  The default
    sensor and motor alphabet is raw bytes plus one boundary action.
    """

    alphabet_size: int = 257
    boundary_symbol: int = 256
    region_sizes: Tuple[int, ...] = (128, 96, 64)
    synapse_fan_in: int = 24
    motor_fan_in: int = 48

    membrane_decay: float = 0.65
    trace_decay: float = 0.82
    inhibition_decay: float = 0.80
    inhibition_gain: float = 0.55
    bottom_up_gain: float = 1.00
    recurrent_gain: float = 0.55
    top_down_gain: float = 0.30

    threshold_base: float = 0.02
    threshold_min: float = -0.20
    threshold_max: float = 1.50
    homeostasis_rate: float = 0.015
    target_activity: float = 0.12

    predictive_learning_rate: float = 0.025
    transition_learning_rate: float = 0.012
    motor_learning_rate: float = 0.10
    bias_learning_rate: float = 0.025
    synapse_decay: float = 1e-5

    weight_init_scale: float = 0.45
    max_weight_norm: float = 2.5
    max_membrane_norm: float = 8.0
    max_trace_norm: float = 5.0
    motor_context_norm: float = 4.0
    motor_temperature: float = 0.75
    seed: int = 20260821

    def __post_init__(self) -> None:
        if self.alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2")
        if not 0 <= self.boundary_symbol < self.alphabet_size:
            raise ValueError("boundary_symbol must be inside the alphabet")
        if not self.region_sizes or any(size <= 1 for size in self.region_sizes):
            raise ValueError("region_sizes must contain dimensions greater than 1")
        if self.synapse_fan_in <= 0 or self.motor_fan_in <= 0:
            raise ValueError("fan-in values must be positive")
        for name in (
            "membrane_decay",
            "trace_decay",
            "inhibition_decay",
            "target_activity",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.threshold_min >= self.threshold_max:
            raise ValueError("threshold_min must be smaller than threshold_max")
        for name in (
            "homeostasis_rate",
            "predictive_learning_rate",
            "transition_learning_rate",
            "motor_learning_rate",
            "bias_learning_rate",
            "weight_init_scale",
            "max_weight_norm",
            "max_membrane_norm",
            "max_trace_norm",
            "motor_context_norm",
            "motor_temperature",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.synapse_decay < 0.0:
            raise ValueError("synapse_decay cannot be negative")

    @property
    def motor_context_dim(self) -> int:
        return sum(self.region_sizes)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["region_sizes"] = list(self.region_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaijiConfig":
        values = dict(payload)
        values["region_sizes"] = tuple(values["region_sizes"])
        return cls(**values)
