"""Configuration contract for the Taiji-0 dynamics kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TaijiConfig:
    """Small deterministic configuration used to falsify the substrate contract.

    The defaults are deliberately tiny.  They are a dynamics experiment, not a
    model-size target and not a language-quality configuration.
    """

    event_dim: int = 32
    state_dim: int = 64
    field_dim: int = 128
    dendritic_branches: int = 4
    fast_memory_slots: int = 16
    cell_ids: Tuple[str, ...] = ("cell_0", "cell_1", "cell_2")

    active_budget: int = 2
    activation_threshold: float = 0.0

    dendrite_alpha: float = 0.40
    apical_alpha: float = 0.20
    soma_alpha: float = 0.35
    idle_state_decay: float = 0.08
    prediction_decay: float = 0.20
    error_decay: float = 0.30

    field_fast_decay: float = 0.55
    field_working_decay: float = 0.90
    field_context_decay: float = 0.985
    field_inhibit_decay: float = 0.85
    field_working_write: float = 0.45
    field_context_write: float = 0.08
    field_write_scale: float = 0.50

    priority_event_gain: float = 1.00
    priority_error_gain: float = 0.35
    priority_field_gain: float = 0.15
    priority_goal_gain: float = 0.75
    priority_memory_gain: float = 0.50

    memory_recall_threshold: float = 0.60
    memory_temperature: float = 12.0
    memory_merge_threshold: float = 0.95

    energy_capacity: float = 1.0
    energy_recovery: float = 0.08
    firing_cost: float = 0.30
    minimum_firing_energy: float = 0.25
    threshold_base: float = 0.05
    threshold_adaptation: float = 0.08
    threshold_decay: float = 0.15
    refractory_ticks: int = 1

    phase_velocity: float = 0.35
    peer_event_scale: float = 0.35
    axon_event_threshold: float = 0.01

    max_state_norm: float = 4.0
    max_field_norm: float = 8.0
    max_output_norm: float = 2.0
    seed: int = 20260821

    def __post_init__(self) -> None:
        positive_ints = {
            "event_dim": self.event_dim,
            "state_dim": self.state_dim,
            "field_dim": self.field_dim,
            "dendritic_branches": self.dendritic_branches,
            "fast_memory_slots": self.fast_memory_slots,
            "active_budget": self.active_budget,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not self.cell_ids or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must be non-empty and unique")
        if self.active_budget > len(self.cell_ids):
            raise ValueError("active_budget cannot exceed the cell population")
        for name in (
            "dendrite_alpha",
            "apical_alpha",
            "soma_alpha",
            "idle_state_decay",
            "prediction_decay",
            "error_decay",
            "field_fast_decay",
            "field_working_decay",
            "field_context_decay",
            "field_inhibit_decay",
            "threshold_decay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.energy_capacity <= 0 or self.firing_cost <= 0:
            raise ValueError("energy_capacity and firing_cost must be positive")
        if not 0.0 <= self.minimum_firing_energy <= self.energy_capacity:
            raise ValueError("minimum_firing_energy must fit within energy_capacity")
        if self.refractory_ticks < 0:
            raise ValueError("refractory_ticks cannot be negative")
        for name in ("memory_recall_threshold", "memory_merge_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.memory_temperature <= 0:
            raise ValueError("memory_temperature must be positive")
        for name in ("max_state_norm", "max_field_norm", "max_output_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cell_ids"] = list(self.cell_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaijiConfig":
        values = dict(payload)
        values["cell_ids"] = tuple(values.get("cell_ids", cls.cell_ids))
        return cls(**values)
