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

    memory_units: int = 192
    memory_fan_in: int = 32
    memory_readout_fan_in: int = 48
    memory_meta_dim: int = 48
    memory_iterations: int = 3
    memory_time_dim: int = 8
    memory_episode_dim: int = 16

    membrane_decay: float = 0.65
    trace_decay: float = 0.82
    inhibition_decay: float = 0.80
    inhibition_gain: float = 0.55
    lateral_fan_in: int = 16
    lateral_learning_rate: float = 0.02
    lateral_seed_offset: int = 977
    consolidation_seed_offset: int = 1951
    bottom_up_gain: float = 1.00
    recurrent_gain: float = 0.55
    top_down_gain: float = 0.30

    threshold_base: float = 0.02
    threshold_min: float = -0.20
    threshold_max: float = 1.50
    homeostasis_rate: float = 0.015
    target_activity: float = 0.12
    cortical_baseline_rate: float = 0.00390625
    consolidation_read_gain: float = 1.00

    predictive_learning_rate: float = 0.025
    transition_learning_rate: float = 0.012
    motor_learning_rate: float = 0.10
    bias_learning_rate: float = 0.025
    reward_baseline_rate: float = 0.05
    episodic_learning_rate: float = 0.60
    episodic_readout_learning_rate: float = 0.85
    episodic_write_repeats: int = 2
    cortical_readout_learning_rate: float = 0.30
    cortical_readout_repeats: int = 8
    synapse_decay: float = 1e-5

    weight_init_scale: float = 0.45
    max_weight_norm: float = 2.5
    max_membrane_norm: float = 8.0
    max_trace_norm: float = 5.0
    motor_context_norm: float = 4.0
    motor_temperature: float = 0.75
    memory_trace_decay: float = 0.72
    memory_inhibition_gain: float = 0.75
    memory_recurrent_gain: float = 1.35
    memory_event_gain: float = 0.80
    memory_action_binding_gain: float = 2.00
    memory_read_gain: float = 3.00
    memory_feedback_gain: float = 0.25
    memory_novelty_gain: float = 0.70
    memory_reward_gain: float = 0.30

    replay_seed_gain: float = 0.65
    replay_noise_scale: float = 0.75
    replay_value_weight: float = 0.60
    replay_priority_threshold: float = 0.05
    replay_fatigue_gain: float = 1.20
    replay_learning_scale: float = 0.45
    replay_burst_repeats: int = 8
    replay_write_repeats: int = 8
    replay_winner_resource_retention: float = 0.90
    structural_turnover_ratio: float = 0.25
    structural_capture_target: float = 0.90
    structural_error_threshold: float = 0.35
    seed: int = 20260821

    def __post_init__(self) -> None:
        if self.alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2")
        if not 0 <= self.boundary_symbol < self.alphabet_size:
            raise ValueError("boundary_symbol must be inside the alphabet")
        if not self.region_sizes or any(size <= 1 for size in self.region_sizes):
            raise ValueError("region_sizes must contain dimensions greater than 1")
        if (
            self.synapse_fan_in <= 0
            or self.motor_fan_in <= 0
            or self.memory_fan_in <= 0
            or self.lateral_fan_in <= 0
        ):
            raise ValueError("fan-in values must be positive")
        if self.lateral_seed_offset <= 0:
            raise ValueError(
                "lateral_seed_offset must be positive so the lateral bank draws "
                "from a stream distinct from the feedforward projections"
            )
        if self.consolidation_seed_offset <= 0:
            raise ValueError(
                "consolidation_seed_offset must select a positive independent "
                "random stream"
            )
        if self.consolidation_seed_offset == self.lateral_seed_offset:
            raise ValueError(
                "consolidation and lateral banks require distinct random streams"
            )
        if self.motor_fan_in > 2 * sum(self.region_sizes):
            raise ValueError(
                "motor_fan_in cannot exceed the available cortical state"
            )
        if self.memory_units <= 1:
            raise ValueError("memory_units must be greater than 1")
        if not 0 < self.memory_readout_fan_in <= self.memory_meta_dim:
            raise ValueError(
                "memory_readout_fan_in must be in [1, memory_meta_dim]"
            )
        if not 0 < self.memory_meta_dim <= self.memory_units:
            raise ValueError("memory_meta_dim must be in [1, memory_units]")
        if self.memory_iterations <= 0:
            raise ValueError("memory_iterations must be positive")
        if self.episodic_write_repeats <= 0:
            raise ValueError("episodic_write_repeats must be positive")
        if self.cortical_readout_repeats <= 0:
            raise ValueError("cortical_readout_repeats must be positive")
        if self.memory_time_dim < 2 or self.memory_time_dim % 2:
            raise ValueError("memory_time_dim must be a positive even dimension")
        if self.memory_episode_dim <= 0:
            raise ValueError("memory_episode_dim must be positive")
        for name in (
            "membrane_decay",
            "trace_decay",
            "inhibition_decay",
            "target_activity",
            "memory_trace_decay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.threshold_min >= self.threshold_max:
            raise ValueError("threshold_min must be smaller than threshold_max")
        for name in (
            "homeostasis_rate",
            "inhibition_gain",
            "predictive_learning_rate",
            "transition_learning_rate",
            "lateral_learning_rate",
            "motor_learning_rate",
            "bias_learning_rate",
            "episodic_learning_rate",
            "episodic_readout_learning_rate",
            "cortical_readout_learning_rate",
            "weight_init_scale",
            "max_weight_norm",
            "max_membrane_norm",
            "max_trace_norm",
            "motor_context_norm",
            "motor_temperature",
            "memory_inhibition_gain",
            "memory_recurrent_gain",
            "memory_event_gain",
            "memory_action_binding_gain",
            "memory_read_gain",
            "replay_seed_gain",
            "replay_learning_scale",
            "consolidation_read_gain",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.synapse_decay < 0.0:
            raise ValueError("synapse_decay cannot be negative")
        if self.memory_feedback_gain < 0.0:
            raise ValueError("memory_feedback_gain cannot be negative")
        if not 0.0 < self.reward_baseline_rate <= 1.0:
            raise ValueError("reward_baseline_rate must be in (0, 1]")
        if not 0.0 < self.cortical_baseline_rate <= 1.0:
            raise ValueError("cortical_baseline_rate must be in (0, 1]")
        if not 0.0 <= self.memory_novelty_gain <= 1.0:
            raise ValueError("memory_novelty_gain must be in [0, 1]")
        if not 0.0 <= self.memory_reward_gain <= 1.0:
            raise ValueError("memory_reward_gain must be in [0, 1]")
        if self.memory_novelty_gain + self.memory_reward_gain <= 0.0:
            raise ValueError("at least one episodic write gate must be active")
        if self.replay_noise_scale < 0.0:
            raise ValueError("replay_noise_scale cannot be negative")
        if self.replay_fatigue_gain < 0.0:
            raise ValueError("replay_fatigue_gain cannot be negative")
        if not 0.0 <= self.replay_value_weight <= 1.0:
            raise ValueError("replay_value_weight must be in [0, 1]")
        if not 0.0 <= self.replay_priority_threshold < 1.0:
            raise ValueError("replay_priority_threshold must be in [0, 1)")
        if self.replay_burst_repeats <= 0:
            raise ValueError("replay_burst_repeats must be positive")
        if self.replay_write_repeats <= 0:
            raise ValueError("replay_write_repeats must be positive")
        if not 0.0 < self.replay_winner_resource_retention <= 1.0:
            raise ValueError(
                "replay_winner_resource_retention must be in (0, 1]"
            )
        if not 0.0 <= self.structural_turnover_ratio <= 1.0:
            raise ValueError("structural_turnover_ratio must be in [0, 1]")
        if not 0.0 < self.structural_capture_target <= 1.0:
            raise ValueError("structural_capture_target must be in (0, 1]")
        if self.structural_error_threshold < 0.0:
            raise ValueError("structural_error_threshold must be non-negative")

    @property
    def cortical_context_dim(self) -> int:
        return 2 * sum(self.region_sizes)

    @property
    def motor_context_dim(self) -> int:
        return self.motor_fan_in

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["region_sizes"] = list(self.region_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaijiConfig":
        values = dict(payload)
        values["region_sizes"] = tuple(values["region_sizes"])
        return cls(**values)
