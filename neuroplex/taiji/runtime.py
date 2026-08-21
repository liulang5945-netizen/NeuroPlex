"""Deterministic two-phase runtime for the Taiji-0 substrate."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .cell import TaijiCell
from .config import TaijiConfig
from .events import EventKind, EventMode, TaijiEvent, dominant_mode
from .field import TaijiField
from .state import (
    AssociationLearningResult,
    TaijiState,
    TaijiStepResult,
    bound_vector,
)


class TaijiRuntime(nn.Module):
    """Own model parameters and the complete mutable cognition state."""

    CHECKPOINT_FORMAT = "taiji-dynamics-v1"
    STATE_VERSION = 1

    def __init__(
        self,
        config: Optional[TaijiConfig] = None,
        *,
        device: torch.device | str = "cpu",
        episode_id: str = "episode-0",
    ):
        super().__init__()
        self.config = config or TaijiConfig()
        self.device = torch.device(device)
        self.field_operator = TaijiField(self.config, device=self.device)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.cells = nn.ModuleDict({
                cell_id: TaijiCell(cell_id, self.config)
                for cell_id in sorted(self.config.cell_ids)
            })
        self.to(self.device)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.config.seed)
        self._state = self._initial_state(episode_id=episode_id)

    def _initial_state(self, *, episode_id: str) -> TaijiState:
        ordered_ids = sorted(self.config.cell_ids)
        count = len(ordered_ids)
        cell_states = {
            cell_id: self.cells[cell_id].initial_state(
                phase_offset=(2.0 * math.pi * index / count),
                device=self.device,
            )
            for index, cell_id in enumerate(ordered_ids)
        }
        return TaijiState(
            version=self.STATE_VERSION,
            tick=0,
            episode_id=episode_id,
            field=self.field_operator.initial_state(),
            cells=cell_states,
            pending_events={},
            last_output=torch.zeros(self.config.event_dim, device=self.device),
        )

    @property
    def tick(self) -> int:
        return self._state.tick

    def snapshot(self) -> TaijiState:
        return self._state.clone()

    def reset(self, *, episode_id: str = "episode-0") -> None:
        """Explicitly reset all transient and contextual runtime state."""

        self._state = self._initial_state(episode_id=episode_id)

    def reset_dynamics(
        self,
        *,
        preserve_fast_memory: bool = True,
        episode_id: Optional[str] = None,
    ) -> None:
        """Start a clean episode while optionally retaining learned associations."""

        previous = self._state
        fresh = self._initial_state(episode_id=episode_id or previous.episode_id)
        if preserve_fast_memory:
            for cell_id in sorted(self.cells.keys()):
                old = previous.cells[cell_id]
                new = fresh.cells[cell_id]
                new.memory_keys = old.memory_keys.detach().clone()
                new.memory_values = old.memory_values.detach().clone()
                new.memory_usage = old.memory_usage.detach().clone()
        self._state = fresh

    def make_event(
        self,
        value: torch.Tensor,
        *,
        kind: EventKind = EventKind.SENSORY,
        source: str = "sensor",
        target: Optional[str] = None,
        salience: float = 1.0,
        reliability: float = 1.0,
        mode: EventMode = EventMode.REAL,
        delay: int = 0,
    ) -> TaijiEvent:
        if delay < 0:
            raise ValueError("delay cannot be negative")
        return TaijiEvent(
            tick=self.tick + delay,
            episode_id=self._state.episode_id,
            source=source,
            target=target,
            kind=kind,
            value=value.detach().to(device=self.device, dtype=torch.float32).clone(),
            salience=salience,
            reliability=reliability,
            mode=mode,
        )

    def schedule(self, event: TaijiEvent) -> None:
        if event.tick < self.tick:
            raise ValueError(f"cannot schedule stale event at tick {event.tick} < {self.tick}")
        self._validate_event(event)
        self._state.pending_events.setdefault(event.tick, []).append(event)

    def _validate_event(self, event: TaijiEvent) -> None:
        if event.value.shape != (self.config.event_dim,):
            raise ValueError(
                f"event value shape {tuple(event.value.shape)} does not match "
                f"({self.config.event_dim},)"
            )
        if event.target is not None and event.target not in self.cells:
            raise KeyError(f"unknown event target: {event.target}")

    @staticmethod
    def _event_sort_key(event: TaijiEvent) -> Tuple[str, str, str, str]:
        return (
            event.target or "",
            event.source,
            event.kind.value,
            event.mode.value,
        )

    def _collect_events(self, external_events: Iterable[TaijiEvent]) -> List[TaijiEvent]:
        due = list(self._state.pending_events.pop(self.tick, []))
        for event in external_events:
            self._validate_event(event)
            if event.tick < self.tick:
                raise ValueError(
                    f"stale external event at tick {event.tick}; runtime is at {self.tick}"
                )
            if event.tick > self.tick:
                self._state.pending_events.setdefault(event.tick, []).append(event)
            else:
                due.append(event)
        return sorted(due, key=self._event_sort_key)

    def _aggregate_for_cell(
        self, cell_id: str, events: Sequence[TaijiEvent]
    ) -> Tuple[torch.Tensor, float, EventMode]:
        value = torch.zeros(self.config.event_dim, device=self.device)
        goal_signal = 0.0
        relevant: List[TaijiEvent] = []
        for event in events:
            if event.target is not None and event.target != cell_id:
                continue
            if event.kind is EventKind.PEER and event.source == cell_id:
                continue
            scale = float(event.salience) * float(event.reliability)
            if event.kind is EventKind.PEER:
                scale *= self.config.peer_event_scale
            value = value + event.value.to(self.device) * scale
            if event.kind is EventKind.GOAL:
                goal_signal += scale * float(event.value.pow(2).mean().sqrt().item())
            relevant.append(event)
        value = bound_vector(value, self.config.max_output_norm)
        mode = dominant_mode(relevant) if relevant else EventMode.IMAGINED
        return value, goal_signal, mode

    @torch.no_grad()
    def step(
        self,
        events: Iterable[TaijiEvent] = (),
        *,
        evaluation_order: Optional[Sequence[str]] = None,
    ) -> TaijiStepResult:
        """Advance one tick using an immutable proposal snapshot and atomic commit."""

        due_events = self._collect_events(events)
        snapshot = self._state.clone()
        cell_ids = list(evaluation_order or sorted(self.cells.keys()))
        if len(cell_ids) != len(self.cells) or set(cell_ids) != set(self.cells.keys()):
            raise ValueError("evaluation_order must be a permutation of all cell ids")

        field_input = self.field_operator.effective(snapshot.field)
        proposals = {}
        for cell_id in cell_ids:
            event_input, goal_signal, mode = self._aggregate_for_cell(
                cell_id, due_events
            )
            proposals[cell_id] = self.cells[cell_id].propose(
                snapshot.cells[cell_id],
                event_input,
                field_input,
                goal_signal=goal_signal,
                mode=mode,
            )

        candidates = [
            proposal for proposal in proposals.values()
            if math.isfinite(proposal.priority)
            and proposal.priority > self.config.activation_threshold
        ]
        candidates.sort(key=lambda proposal: (-proposal.priority, proposal.cell_id))
        selected_ids = tuple(
            proposal.cell_id for proposal in candidates[: self.config.active_budget]
        )
        selected = set(selected_ids)

        next_cells = {
            cell_id: (
                proposals[cell_id].active_state
                if cell_id in selected
                else proposals[cell_id].idle_state
            )
            for cell_id in sorted(self.cells.keys())
        }
        writes = [proposals[cell_id].field_write for cell_id in selected_ids]
        next_field = self.field_operator.advance(snapshot.field, writes)

        if selected_ids:
            output = torch.stack([
                proposals[cell_id].axon for cell_id in selected_ids
            ]).mean(dim=0)
            output = bound_vector(output, self.config.max_output_norm)
        else:
            output = torch.zeros(self.config.event_dim, device=self.device)

        emitted: List[TaijiEvent] = []
        next_pending = {
            tick: list(items)
            for tick, items in self._state.pending_events.items()
        }
        for cell_id in selected_ids:
            proposal = proposals[cell_id]
            if float(proposal.axon.norm().item()) < self.config.axon_event_threshold:
                continue
            event = TaijiEvent(
                tick=snapshot.tick + 1,
                episode_id=snapshot.episode_id,
                source=cell_id,
                target=None,
                kind=EventKind.PEER,
                value=proposal.axon.detach().clone(),
                salience=min(1.0, max(0.0, proposal.priority)),
                reliability=1.0,
                mode=proposal.mode,
            )
            emitted.append(event)
            next_pending.setdefault(event.tick, []).append(event)
        for tick in next_pending:
            next_pending[tick] = sorted(next_pending[tick], key=self._event_sort_key)

        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=snapshot.tick + 1,
            episode_id=snapshot.episode_id,
            field=next_field,
            cells=next_cells,
            pending_events=next_pending,
            last_output=output,
        )
        return TaijiStepResult(
            tick=snapshot.tick,
            active_cell_ids=selected_ids,
            priorities={
                cell_id: float(proposals[cell_id].priority)
                for cell_id in sorted(proposals)
            },
            output=output.detach().clone(),
            field=next_field.clone(),
            emitted_events=tuple(emitted),
            memory_confidences={
                cell_id: float(proposals[cell_id].memory_confidence)
                for cell_id in sorted(proposals)
            },
        )

    @torch.no_grad()
    def learn_association(
        self,
        cue: TaijiEvent,
        observed_outcome: TaijiEvent,
        *,
        reward: float = 1.0,
    ) -> AssociationLearningResult:
        """Learn one real cue/outcome pair in cue-active cells only.

        The cue and outcome both pass through normal ticks.  The outcome acts as
        a broadcast third factor, while the eligibility set is the cells that
        fired for the cue.  Model parameters are never touched.
        """

        if not 0.0 < reward <= 1.0:
            raise ValueError("reward must be in (0, 1] for positive association")
        self._validate_event(cue)
        self._validate_event(observed_outcome)
        if cue.tick != self.tick:
            raise ValueError("cue must be presented at the current runtime tick")
        if observed_outcome.tick != cue.tick + 1:
            raise ValueError("observed outcome must immediately follow the cue")
        if observed_outcome.mode is not EventMode.REAL:
            raise ValueError("an imagined or replayed event cannot be a real outcome")

        cue_result = self.step([cue])
        outcome_result = self.step([observed_outcome])
        slots: Dict[str, int] = {}
        next_cells = dict(self._state.cells)
        for cell_id in cue_result.active_cell_ids:
            updated, slot = self.cells[cell_id].store_fast_association(
                self._state.cells[cell_id],
                cue.value.to(self.device),
                observed_outcome.value.to(self.device),
                strength=float(reward),
            )
            next_cells[cell_id] = updated
            slots[cell_id] = slot
        self._state.cells = next_cells
        return AssociationLearningResult(
            cue_tick=cue_result.tick,
            outcome_tick=outcome_result.tick,
            active_cell_ids=cue_result.active_cell_ids,
            written_slots=slots,
            reward=float(reward),
        )

    def readout(self) -> torch.Tensor:
        """Return a diagnostic state readout without changing cognition."""

        mean_soma = torch.stack([
            self._state.cells[cell_id].soma for cell_id in sorted(self.cells.keys())
        ]).mean(dim=0)
        return torch.cat([
            self.field_operator.effective(self._state.field),
            mean_soma,
            self._state.last_output,
        ])

    def checkpoint(self) -> Dict[str, Any]:
        """Capture parameters, cognition state, queues and RNG in one payload."""

        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "module_state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "taiji_state": self._state.to_payload(),
            "rng_state": self._rng.get_state().clone(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')}")
        expected = self.config.to_dict()
        actual = TaijiConfig.from_dict(checkpoint["config"]).to_dict()
        if actual != expected:
            raise ValueError("checkpoint config does not match this runtime")
        self.load_state_dict(checkpoint["module_state"], strict=True)
        state = TaijiState.from_payload(checkpoint["taiji_state"], device=self.device)
        if state.version != self.STATE_VERSION:
            raise ValueError(f"unsupported Taiji state version: {state.version}")
        if set(state.cells) != set(self.cells.keys()):
            raise ValueError("checkpoint cell population does not match runtime")
        self._state = state
        self._rng.set_state(checkpoint["rng_state"].detach().cpu())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> "TaijiRuntime":
        config = TaijiConfig.from_dict(checkpoint["config"])
        runtime = cls(config, device=device)
        runtime.restore(checkpoint)
        return runtime
