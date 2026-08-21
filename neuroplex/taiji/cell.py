"""Persistent multi-compartment cell used by the Taiji-0 kernel."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TaijiConfig
from .events import EventMode
from .state import CellProposal, FieldWrite, TaijiCellState, bound_vector


class TaijiCell(nn.Module):
    """A locally stateful cell with basal, apical, somatic and axonal paths."""

    def __init__(self, cell_id: str, config: TaijiConfig):
        super().__init__()
        self.cell_id = cell_id
        self.config = config
        branches = config.dendritic_branches
        state_dim = config.state_dim

        self.event_to_dendrites = nn.Linear(
            config.event_dim, branches * state_dim, bias=False
        )
        self.field_to_dendrites = nn.Linear(
            config.field_dim, branches * state_dim, bias=False
        )
        self.field_to_apical = nn.Linear(config.field_dim, state_dim, bias=False)
        self.soma_to_prediction = nn.Linear(state_dim, state_dim, bias=False)
        self.apical_to_prediction = nn.Linear(state_dim, state_dim, bias=False)

        self.basal_to_soma = nn.Linear(state_dim, state_dim, bias=False)
        self.apical_to_soma = nn.Linear(state_dim, state_dim, bias=False)
        self.error_to_soma = nn.Linear(state_dim, state_dim, bias=False)
        self.recurrent_to_soma = nn.Linear(state_dim, state_dim, bias=False)

        self.event_prediction = nn.Linear(state_dim, config.event_dim, bias=False)
        self.axon_projection = nn.Linear(state_dim, config.event_dim, bias=False)
        self.field_projection = nn.Linear(state_dim, config.field_dim, bias=False)
        self.branch_mix_logits = nn.Parameter(torch.zeros(branches))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.55)
        nn.init.orthogonal_(self.recurrent_to_soma.weight, gain=0.70)
        nn.init.zeros_(self.branch_mix_logits)

    def initial_state(
        self,
        *,
        phase_offset: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> TaijiCellState:
        cfg = self.config
        dev = torch.device(device)
        return TaijiCellState(
            dendrites=torch.zeros(cfg.dendritic_branches, cfg.state_dim, device=dev),
            apical=torch.zeros(cfg.state_dim, device=dev),
            soma=torch.zeros(cfg.state_dim, device=dev),
            prediction=torch.zeros(cfg.event_dim, device=dev),
            error=torch.zeros(cfg.state_dim, device=dev),
            phase=torch.tensor(
                [math.cos(phase_offset), math.sin(phase_offset)], device=dev
            ),
            energy=float(cfg.energy_capacity),
            threshold=float(cfg.threshold_base),
            refractory=0,
            eligibility=torch.zeros(cfg.state_dim, cfg.event_dim, device=dev),
            memory_keys=torch.zeros(cfg.fast_memory_slots, cfg.event_dim, device=dev),
            memory_values=torch.zeros(cfg.fast_memory_slots, cfg.state_dim, device=dev),
            memory_usage=torch.zeros(cfg.fast_memory_slots, device=dev),
        )

    def _advance_phase(self, phase: torch.Tensor) -> torch.Tensor:
        angle = self.config.phase_velocity
        rotation = phase.new_tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        advanced = rotation @ phase
        return advanced / advanced.norm().clamp_min(1e-8)

    def _idle_state(self, state: TaijiCellState) -> TaijiCellState:
        cfg = self.config
        retain = 1.0 - cfg.idle_state_decay
        threshold = cfg.threshold_base + (
            state.threshold - cfg.threshold_base
        ) * (1.0 - cfg.threshold_decay)
        return TaijiCellState(
            dendrites=state.dendrites * retain,
            apical=state.apical * retain,
            soma=bound_vector(state.soma * retain, cfg.max_state_norm),
            prediction=state.prediction * (1.0 - cfg.prediction_decay),
            error=state.error * (1.0 - cfg.error_decay),
            phase=self._advance_phase(state.phase),
            energy=min(cfg.energy_capacity, state.energy + cfg.energy_recovery),
            threshold=float(max(cfg.threshold_base, threshold)),
            refractory=max(0, state.refractory - 1),
            eligibility=state.eligibility * 0.95,
            memory_keys=state.memory_keys,
            memory_values=state.memory_values,
            memory_usage=state.memory_usage,
        )

    def propose(
        self,
        state: TaijiCellState,
        event_input: torch.Tensor,
        field_input: torch.Tensor,
        *,
        goal_signal: float = 0.0,
        mode: EventMode = EventMode.REAL,
    ) -> CellProposal:
        """Compute active and idle proposals without mutating runtime state."""

        cfg = self.config
        if event_input.shape != (cfg.event_dim,):
            raise ValueError(f"event_input must have shape ({cfg.event_dim},)")
        if field_input.shape != (cfg.field_dim,):
            raise ValueError(f"field_input must have shape ({cfg.field_dim},)")

        idle = self._idle_state(state)
        branch_drive = (
            self.event_to_dendrites(event_input)
            + 0.25 * self.field_to_dendrites(field_input)
        ).view(cfg.dendritic_branches, cfg.state_dim)
        dendrites = (
            (1.0 - cfg.dendrite_alpha) * idle.dendrites
            + cfg.dendrite_alpha * torch.tanh(branch_drive)
        )
        branch_weights = torch.softmax(self.branch_mix_logits, dim=0).unsqueeze(-1)
        basal = (branch_weights * dendrites).sum(dim=0)

        apical = (
            (1.0 - cfg.apical_alpha) * idle.apical
            + cfg.apical_alpha * torch.tanh(self.field_to_apical(field_input))
        )
        predicted_basal = torch.tanh(
            self.soma_to_prediction(idle.soma)
            + self.apical_to_prediction(apical)
        )
        local_error = basal - predicted_basal
        candidate = torch.tanh(
            self.basal_to_soma(basal)
            + self.apical_to_soma(apical)
            + self.error_to_soma(local_error)
            + self.recurrent_to_soma(idle.soma)
        )
        soma = bound_vector(
            (1.0 - cfg.soma_alpha) * idle.soma + cfg.soma_alpha * candidate,
            cfg.max_state_norm,
        )
        predicted_event = torch.tanh(self.event_prediction(soma))
        prediction = (
            (1.0 - cfg.prediction_decay) * idle.prediction
            + cfg.prediction_decay * predicted_event
        )
        axon = bound_vector(
            torch.tanh(self.axon_projection(soma)), cfg.max_output_norm
        )
        field_vector = bound_vector(
            torch.tanh(self.field_projection(soma)), cfg.max_field_norm
        )

        event_novelty = (event_input - state.prediction).pow(2).mean().sqrt()
        error_strength = local_error.pow(2).mean().sqrt()
        field_strength = field_input.pow(2).mean().sqrt()
        phase_gate = 0.9 + 0.1 * float(idle.phase[0].item())
        priority_tensor = (
            cfg.priority_event_gain * event_novelty
            + cfg.priority_error_gain * error_strength
            + cfg.priority_field_gain * field_strength
            + cfg.priority_goal_gain * float(goal_signal)
            - idle.threshold
        )
        priority = float(priority_tensor.item()) * phase_gate
        if state.refractory > 0 or idle.energy < cfg.minimum_firing_energy:
            priority = float("-inf")

        active = TaijiCellState(
            dendrites=dendrites,
            apical=apical,
            soma=soma,
            prediction=prediction,
            error=local_error,
            phase=idle.phase,
            energy=max(0.0, idle.energy - cfg.firing_cost),
            threshold=float(idle.threshold + cfg.threshold_adaptation),
            refractory=cfg.refractory_ticks,
            eligibility=idle.eligibility,
            memory_keys=idle.memory_keys,
            memory_values=idle.memory_values,
            memory_usage=idle.memory_usage,
        )
        write_scale = cfg.field_write_scale * max(active.energy, 0.05) * phase_gate
        field_write = FieldWrite(
            source=self.cell_id,
            excite=field_vector,
            inhibit=torch.zeros_like(field_vector),
            scale=write_scale,
        )
        return CellProposal(
            cell_id=self.cell_id,
            priority=priority,
            active_state=active,
            idle_state=idle,
            field_write=field_write,
            axon=axon,
            mode=mode,
        )

