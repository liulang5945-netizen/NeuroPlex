"""Persistent multi-timescale field dynamics for Taiji-0."""

from __future__ import annotations

from typing import Iterable

import torch

from .config import TaijiConfig
from .state import FieldWrite, TaijiFieldState, bound_vector


class TaijiField:
    """Pure field transition operator.

    The operator never mutates its input state.  A runtime advances it once per
    logical tick after all cells have produced proposals from the same snapshot.
    """

    def __init__(self, config: TaijiConfig, *, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def initial_state(self) -> TaijiFieldState:
        zero = torch.zeros(self.config.field_dim, device=self.device)
        return TaijiFieldState(
            fast=zero.clone(),
            working=zero.clone(),
            context=zero.clone(),
            inhibit=zero.clone(),
        )

    def effective(self, state: TaijiFieldState) -> torch.Tensor:
        raw = state.fast + state.working + state.context
        gate = torch.exp(-state.inhibit.clamp_min(0.0))
        return bound_vector(raw * gate, self.config.max_field_norm)

    def advance(
        self,
        state: TaijiFieldState,
        writes: Iterable[FieldWrite] = (),
    ) -> TaijiFieldState:
        """Decay the old field and atomically apply a tick's selected writes."""

        ordered = sorted(writes, key=lambda write: write.source)
        if ordered:
            for write in ordered:
                if write.excite.shape != (self.config.field_dim,):
                    raise ValueError(
                        f"field excite shape {tuple(write.excite.shape)} does not match "
                        f"({self.config.field_dim},)"
                    )
                if write.inhibit.shape != (self.config.field_dim,):
                    raise ValueError(
                        f"field inhibit shape {tuple(write.inhibit.shape)} does not match "
                        f"({self.config.field_dim},)"
                    )
            excite = torch.stack([
                write.excite.to(self.device) * float(write.scale)
                for write in ordered
            ]).sum(dim=0)
            inhibit = torch.stack([
                write.inhibit.to(self.device).abs() * float(write.scale)
                for write in ordered
            ]).sum(dim=0)
        else:
            excite = torch.zeros(self.config.field_dim, device=self.device)
            inhibit = torch.zeros_like(excite)

        fast = state.fast.to(self.device) * self.config.field_fast_decay + excite
        working = (
            state.working.to(self.device) * self.config.field_working_decay
            + excite * self.config.field_working_write
        )
        context = (
            state.context.to(self.device) * self.config.field_context_decay
            + excite * self.config.field_context_write
        )
        inhibit_state = (
            state.inhibit.to(self.device) * self.config.field_inhibit_decay
            + inhibit
        ).clamp_min(0.0)

        return TaijiFieldState(
            fast=bound_vector(fast, self.config.max_field_norm),
            working=bound_vector(working, self.config.max_field_norm),
            context=bound_vector(context, self.config.max_field_norm),
            inhibit=bound_vector(inhibit_state, self.config.max_field_norm),
        )

    def reset(self) -> TaijiFieldState:
        """Explicitly clear all field timescales."""

        return self.initial_state()

