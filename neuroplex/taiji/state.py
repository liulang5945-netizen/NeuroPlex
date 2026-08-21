"""Runtime state objects for the Taiji-0 kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import torch

from .events import EventMode, TaijiEvent


def bound_vector(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Bound a vector without erasing magnitude below the limit."""

    norm = value.norm()
    scale = torch.clamp(
        torch.as_tensor(max_norm, device=value.device, dtype=value.dtype)
        / norm.clamp_min(1e-8),
        max=1.0,
    )
    return value * scale


@dataclass
class TaijiCellState:
    dendrites: torch.Tensor
    apical: torch.Tensor
    soma: torch.Tensor
    prediction: torch.Tensor
    error: torch.Tensor
    phase: torch.Tensor
    energy: float
    threshold: float
    refractory: int
    eligibility: torch.Tensor
    memory_keys: torch.Tensor
    memory_values: torch.Tensor
    memory_usage: torch.Tensor

    def clone(self) -> "TaijiCellState":
        return TaijiCellState(
            dendrites=self.dendrites.detach().clone(),
            apical=self.apical.detach().clone(),
            soma=self.soma.detach().clone(),
            prediction=self.prediction.detach().clone(),
            error=self.error.detach().clone(),
            phase=self.phase.detach().clone(),
            energy=float(self.energy),
            threshold=float(self.threshold),
            refractory=int(self.refractory),
            eligibility=self.eligibility.detach().clone(),
            memory_keys=self.memory_keys.detach().clone(),
            memory_values=self.memory_values.detach().clone(),
            memory_usage=self.memory_usage.detach().clone(),
        )

    def to_payload(self) -> Dict[str, Any]:
        cloned = self.clone()
        return {
            "dendrites": cloned.dendrites.cpu(),
            "apical": cloned.apical.cpu(),
            "soma": cloned.soma.cpu(),
            "prediction": cloned.prediction.cpu(),
            "error": cloned.error.cpu(),
            "phase": cloned.phase.cpu(),
            "energy": cloned.energy,
            "threshold": cloned.threshold,
            "refractory": cloned.refractory,
            "eligibility": cloned.eligibility.cpu(),
            "memory_keys": cloned.memory_keys.cpu(),
            "memory_values": cloned.memory_values.cpu(),
            "memory_usage": cloned.memory_usage.cpu(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> "TaijiCellState":
        tensor_names = (
            "dendrites",
            "apical",
            "soma",
            "prediction",
            "error",
            "phase",
            "eligibility",
            "memory_keys",
            "memory_values",
            "memory_usage",
        )
        tensors = {
            name: payload[name].detach().to(device=device).clone()
            for name in tensor_names
        }
        return cls(
            **tensors,
            energy=float(payload["energy"]),
            threshold=float(payload["threshold"]),
            refractory=int(payload["refractory"]),
        )


@dataclass
class TaijiFieldState:
    fast: torch.Tensor
    working: torch.Tensor
    context: torch.Tensor
    inhibit: torch.Tensor

    def clone(self) -> "TaijiFieldState":
        return TaijiFieldState(
            fast=self.fast.detach().clone(),
            working=self.working.detach().clone(),
            context=self.context.detach().clone(),
            inhibit=self.inhibit.detach().clone(),
        )

    def to_payload(self) -> Dict[str, torch.Tensor]:
        cloned = self.clone()
        return {
            "fast": cloned.fast.cpu(),
            "working": cloned.working.cpu(),
            "context": cloned.context.cpu(),
            "inhibit": cloned.inhibit.cpu(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, torch.Tensor], *, device: torch.device | str = "cpu"
    ) -> "TaijiFieldState":
        return cls(**{
            name: payload[name].detach().to(device=device).clone()
            for name in ("fast", "working", "context", "inhibit")
        })


@dataclass(frozen=True)
class FieldWrite:
    source: str
    excite: torch.Tensor
    inhibit: torch.Tensor
    scale: float = 1.0


@dataclass
class CellProposal:
    cell_id: str
    priority: float
    active_state: TaijiCellState
    idle_state: TaijiCellState
    field_write: FieldWrite
    axon: torch.Tensor
    mode: EventMode


@dataclass
class TaijiState:
    version: int
    tick: int
    episode_id: str
    field: TaijiFieldState
    cells: Dict[str, TaijiCellState]
    pending_events: Dict[int, List[TaijiEvent]]
    last_output: torch.Tensor

    def clone(self) -> "TaijiState":
        return TaijiState(
            version=int(self.version),
            tick=int(self.tick),
            episode_id=str(self.episode_id),
            field=self.field.clone(),
            cells={cell_id: state.clone() for cell_id, state in self.cells.items()},
            pending_events={
                int(tick): [TaijiEvent.from_payload(event.to_payload(), device=event.value.device)
                            for event in events]
                for tick, events in self.pending_events.items()
            },
            last_output=self.last_output.detach().clone(),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "tick": int(self.tick),
            "episode_id": str(self.episode_id),
            "field": self.field.to_payload(),
            "cells": {cell_id: state.to_payload()
                      for cell_id, state in self.cells.items()},
            "pending_events": {
                int(tick): [event.to_payload() for event in events]
                for tick, events in self.pending_events.items()
            },
            "last_output": self.last_output.detach().cpu().clone(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> "TaijiState":
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            episode_id=str(payload["episode_id"]),
            field=TaijiFieldState.from_payload(payload["field"], device=device),
            cells={
                str(cell_id): TaijiCellState.from_payload(state, device=device)
                for cell_id, state in payload["cells"].items()
            },
            pending_events={
                int(tick): [TaijiEvent.from_payload(event, device=device) for event in events]
                for tick, events in payload.get("pending_events", {}).items()
            },
            last_output=payload["last_output"].detach().to(device=device).clone(),
        )


@dataclass(frozen=True)
class TaijiStepResult:
    tick: int
    active_cell_ids: Tuple[str, ...]
    priorities: Dict[str, float]
    output: torch.Tensor
    field: TaijiFieldState
    emitted_events: Tuple[TaijiEvent, ...]

