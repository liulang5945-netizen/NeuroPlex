"""Hierarchical predictive-error dynamics of the native Taiji fabric."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from .config import TaijiConfig
from .sparse import SparseSynapses, bound_norm
from .state import RegionState


class TaijiFabric:
    """A hierarchy of recurrent regions with reciprocal predictive synapses.

    A decoder predicts the region below from the previous local trace.  The
    same physical synapses carry bottom-up prediction error through their
    transpose.  Recurrent synapses predict the region's own next activity.
    No sequence matrix, content-addressed attention, or global router exists.
    """

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        lower_sizes = (config.alphabet_size, *config.region_sizes[:-1])
        self.decoders = tuple(
            SparseSynapses(
                lower_size,
                region_size,
                config.synapse_fan_in,
                generator=generator,
                init_scale=config.weight_init_scale,
                max_weight_norm=config.max_weight_norm,
                device=self.device,
            )
            for lower_size, region_size in zip(lower_sizes, config.region_sizes)
        )
        self.transitions = tuple(
            SparseSynapses(
                region_size,
                region_size,
                config.synapse_fan_in,
                generator=generator,
                init_scale=config.weight_init_scale,
                max_weight_norm=config.max_weight_norm,
                device=self.device,
                allow_self=False,
            )
            for region_size in config.region_sizes
        )

    def initial_state(self) -> Tuple[RegionState, ...]:
        lower_sizes = (self.config.alphabet_size, *self.config.region_sizes[:-1])
        states = []
        for lower_size, region_size in zip(lower_sizes, self.config.region_sizes):
            zero = torch.zeros(region_size, device=self.device)
            states.append(RegionState(
                membrane=zero.clone(),
                activity=zero.clone(),
                trace=zero.clone(),
                prediction=torch.zeros(lower_size, device=self.device),
                error=torch.zeros(lower_size, device=self.device),
                threshold=torch.full(
                    (region_size,), self.config.threshold_base, device=self.device
                ),
                inhibition=0.0,
            ))
        return tuple(states)

    @torch.no_grad()
    def step(
        self,
        sensory_activity: torch.Tensor,
        previous: Sequence[RegionState],
        *,
        learn: bool,
    ) -> Tuple[Tuple[RegionState, ...], Tuple[float, ...], Tuple[float, ...]]:
        if sensory_activity.shape != (self.config.alphabet_size,):
            raise ValueError("sensory activity does not match the receptor population")
        if len(previous) != len(self.config.region_sizes):
            raise ValueError("region state count does not match the architecture")

        lower_activity = sensory_activity.to(self.device)
        next_states = []
        activity_rates = []
        error_norms = []

        for index, (region_size, decoder, transition) in enumerate(zip(
            self.config.region_sizes, self.decoders, self.transitions
        )):
            old = previous[index]
            lower_prediction = decoder.forward(old.trace)
            lower_error = lower_activity - lower_prediction
            recurrent_prediction = transition.forward(old.trace)
            bottom_up = decoder.backproject(lower_error)

            if index + 1 < len(previous):
                top_down = self.decoders[index + 1].forward(previous[index + 1].trace)
            else:
                top_down = torch.zeros(region_size, device=self.device)

            drive = (
                self.config.bottom_up_gain * bottom_up
                + self.config.recurrent_gain * recurrent_prediction
                + self.config.top_down_gain * top_down
            )
            membrane = bound_norm(
                self.config.membrane_decay * old.membrane + drive,
                self.config.max_membrane_norm,
            )
            positive_drive = torch.relu(membrane - old.threshold)
            inhibition = (
                self.config.inhibition_decay * old.inhibition
                + (1.0 - self.config.inhibition_decay)
                * self.config.inhibition_gain
                * float(positive_drive.mean().item())
            )
            activity = torch.tanh(torch.relu(
                membrane - old.threshold - float(inhibition)
            ))
            active_indicator = (activity > 1e-6).to(activity.dtype)
            threshold = torch.clamp(
                old.threshold
                + self.config.homeostasis_rate
                * (active_indicator - self.config.target_activity),
                min=self.config.threshold_min,
                max=self.config.threshold_max,
            )
            trace = bound_norm(
                self.config.trace_decay * old.trace
                + (1.0 - self.config.trace_decay) * activity,
                self.config.max_trace_norm,
            )
            state_error = activity - recurrent_prediction

            if learn:
                decoder.local_update(
                    lower_error,
                    old.trace,
                    learning_rate=self.config.predictive_learning_rate,
                    weight_decay=self.config.synapse_decay,
                )
                transition.local_update(
                    state_error,
                    old.trace,
                    learning_rate=self.config.transition_learning_rate,
                    weight_decay=self.config.synapse_decay,
                )

            next_states.append(RegionState(
                membrane=membrane,
                activity=activity,
                trace=trace,
                prediction=lower_prediction,
                error=lower_error,
                threshold=threshold,
                inhibition=float(inhibition),
            ))
            activity_rates.append(float(active_indicator.mean().item()))
            error_norms.append(float(lower_error.norm().item()))
            lower_activity = activity

        return tuple(next_states), tuple(activity_rates), tuple(error_norms)

    def cortical_context(self, regions: Sequence[RegionState]) -> torch.Tensor:
        """Expose time-separated fast activity and slow trace to an organ."""

        return torch.cat([
            *(region.activity for region in regions),
            *(region.trace for region in regions),
        ], dim=0)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "decoders": [decoder.to_payload() for decoder in self.decoders],
            "transitions": [transition.to_payload() for transition in self.transitions],
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if len(payload["decoders"]) != len(self.decoders):
            raise ValueError("decoder count does not match architecture")
        if len(payload["transitions"]) != len(self.transitions):
            raise ValueError("transition count does not match architecture")
        for synapses, state in zip(self.decoders, payload["decoders"]):
            synapses.load_payload(state)
        for synapses, state in zip(self.transitions, payload["transitions"]):
            synapses.load_payload(state)

    def parameter_tensors(self) -> Tuple[torch.Tensor, ...]:
        return tuple(
            synapses.weight
            for synapses in (*self.decoders, *self.transitions)
        )

    def active_edge_count(self) -> int:
        return sum(
            int(synapses.mask.sum().item())
            for synapses in (*self.decoders, *self.transitions)
        )
