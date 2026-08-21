"""Explicit sparse local synapses used by Taiji regions and organs."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


def bound_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Return ``value`` with a bounded whole-vector norm."""

    norm = value.norm()
    scale = torch.clamp(value.new_tensor(limit) / norm.clamp_min(1e-8), max=1.0)
    return value * scale


class SparseSynapses:
    """A masked projection with an entirely local delta update.

    Each postsynaptic unit owns a fixed fan-in mask.  A learning update uses
    only that unit's error and the presynaptic trace on existing edges.
    ``weight`` is deliberately not an autograd parameter.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        fan_in: int,
        *,
        generator: torch.Generator,
        init_scale: float,
        max_weight_norm: float,
        device: torch.device | str = "cpu",
        allow_self: bool = True,
    ) -> None:
        if out_features <= 0 or in_features <= 0:
            raise ValueError("synapse dimensions must be positive")
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.fan_in = min(int(fan_in), self.in_features)
        self.max_weight_norm = float(max_weight_norm)
        self.device = torch.device(device)

        mask = torch.zeros(out_features, in_features, dtype=torch.bool)
        for post in range(out_features):
            candidates = torch.arange(in_features)
            if not allow_self and out_features == in_features and in_features > 1:
                candidates = candidates[candidates != post]
            count = min(self.fan_in, int(candidates.numel()))
            order = torch.randperm(int(candidates.numel()), generator=generator)[:count]
            mask[post, candidates[order]] = True
        normal = torch.randn(out_features, in_features, generator=generator)
        weight = normal * mask * (float(init_scale) / max(1, self.fan_in) ** 0.5)
        self.mask = mask.to(self.device)
        self.weight = weight.to(self.device, dtype=torch.float32)
        self._bound_rows()

    def forward(self, presynaptic: torch.Tensor) -> torch.Tensor:
        if presynaptic.shape != (self.in_features,):
            raise ValueError(
                f"presynaptic shape must be ({self.in_features},), "
                f"got {tuple(presynaptic.shape)}"
            )
        return self.weight @ presynaptic.to(self.device)

    def backproject(self, postsynaptic_error: torch.Tensor) -> torch.Tensor:
        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError(
                f"error shape must be ({self.out_features},), "
                f"got {tuple(postsynaptic_error.shape)}"
            )
        return self.weight.T @ postsynaptic_error.to(self.device)

    @torch.no_grad()
    def local_update(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Apply error × eligibility only on physically present edges."""

        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError("postsynaptic error dimension mismatch")
        if presynaptic_trace.shape != (self.in_features,):
            raise ValueError("presynaptic trace dimension mismatch")
        scale = max(1.0, float((presynaptic_trace != 0).sum().item()) ** 0.5)
        delta = torch.outer(postsynaptic_error, presynaptic_trace) / scale
        if weight_decay:
            self.weight.mul_(1.0 - float(weight_decay))
        self.weight.add_(float(learning_rate) * delta * self.mask)
        self.weight.mul_(self.mask)
        self._bound_rows()

    @torch.no_grad()
    def _bound_rows(self) -> None:
        norms = self.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scales = torch.clamp(self.max_weight_norm / norms, max=1.0)
        self.weight.mul_(scales)
        self.weight.mul_(self.mask)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "fan_in": self.fan_in,
            "max_weight_norm": self.max_weight_norm,
            "mask": self.mask.detach().cpu().clone(),
            "weight": self.weight.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        expected = (self.out_features, self.in_features)
        if (int(payload["out_features"]), int(payload["in_features"])) != expected:
            raise ValueError("synapse payload shape does not match architecture")
        mask = payload["mask"].detach().to(device=self.device, dtype=torch.bool)
        weight = payload["weight"].detach().to(device=self.device, dtype=torch.float32)
        if mask.shape != expected or weight.shape != expected:
            raise ValueError("synapse tensor shape does not match architecture")
        if not torch.equal(mask, self.mask):
            raise ValueError("synapse structural mask does not match architecture")
        self.weight = weight.clone()
        self.weight.mul_(self.mask)
        self._bound_rows()
