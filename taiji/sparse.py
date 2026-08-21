"""Compressed fixed-fan-in synapses used by Taiji regions and organs."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


def bound_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Return ``value`` with a bounded whole-vector norm."""

    norm = value.norm()
    scale = torch.clamp(value.new_tensor(limit) / norm.clamp_min(1e-8), max=1.0)
    return value * scale


class SparseSynapses:
    """A fixed-fan-in projection that stores and executes only real edges.

    ``pre_index[post, local_edge]`` and ``edge_weight[post, local_edge]``
    describe every physical synapse. The postsynaptic index is implicit in the
    row, avoiding a second index per edge. Forward and local plasticity are
    compressed gather operations; reciprocal backprojection is edge scatter.
    No structural mask, dense weight matrix or dense outer update is retained.
    """

    STORAGE_FORMAT = "fixed-fan-in-v1"

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

        selected_by_post = []
        for post in range(self.out_features):
            candidates = torch.arange(self.in_features)
            if (
                not allow_self
                and self.out_features == self.in_features
                and self.in_features > 1
            ):
                candidates = candidates[candidates != post]
            count = min(self.fan_in, int(candidates.numel()))
            order = torch.randperm(
                int(candidates.numel()), generator=generator
            )[:count]
            selected_by_post.append(candidates[order].to(torch.long))

        row_fan_in = int(selected_by_post[0].numel())
        if any(int(selected.numel()) != row_fan_in for selected in selected_by_post):
            raise ValueError("fixed-fan-in topology produced ragged rows")
        self.row_fan_in = row_fan_in
        pre_index = torch.stack(selected_by_post)

        # Native v2 initialized one dense normal row after topology creation.
        # Drawing one transient row at a time preserves that configured RNG
        # stream without retaining an out_features × in_features matrix.
        edge_weight = torch.stack([
            torch.randn(self.in_features, generator=generator)[selected]
            for selected in selected_by_post
        ]) * (
            float(init_scale) / max(1, self.fan_in) ** 0.5
        )

        self.pre_index = pre_index.to(self.device, dtype=torch.int32)
        self.edge_weight = edge_weight.to(self.device, dtype=torch.float32)
        self._bound_rows()

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    @property
    def dense_equivalent_count(self) -> int:
        return self.out_features * self.in_features

    def forward(self, presynaptic: torch.Tensor) -> torch.Tensor:
        if presynaptic.shape != (self.in_features,):
            raise ValueError(
                f"presynaptic shape must be ({self.in_features},), "
                f"got {tuple(presynaptic.shape)}"
            )
        presynaptic = presynaptic.to(self.device)
        return (self.edge_weight * presynaptic[self.pre_index]).sum(dim=1)

    def backproject(self, postsynaptic_error: torch.Tensor) -> torch.Tensor:
        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError(
                f"error shape must be ({self.out_features},), "
                f"got {tuple(postsynaptic_error.shape)}"
            )
        postsynaptic_error = postsynaptic_error.to(self.device)
        projected = torch.zeros(
            self.in_features,
            device=self.device,
            dtype=self.edge_weight.dtype,
        )
        projected.scatter_add_(
            0,
            self.pre_index.flatten(),
            (self.edge_weight * postsynaptic_error.unsqueeze(1)).flatten(),
        )
        return projected

    @torch.no_grad()
    def local_update(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Apply error × eligibility on existing edges only."""

        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError("postsynaptic error dimension mismatch")
        if presynaptic_trace.shape != (self.in_features,):
            raise ValueError("presynaptic trace dimension mismatch")
        postsynaptic_error = postsynaptic_error.to(self.device)
        presynaptic_trace = presynaptic_trace.to(self.device)
        scale = max(1.0, float((presynaptic_trace != 0).sum().item()) ** 0.5)
        if weight_decay:
            self.edge_weight.mul_(1.0 - float(weight_decay))
        self.edge_weight.add_(
            float(learning_rate)
            * postsynaptic_error.unsqueeze(1)
            * presynaptic_trace[self.pre_index]
            / scale
        )
        self._bound_rows()

    @torch.no_grad()
    def _bound_rows(self) -> None:
        norms = self.edge_weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scales = torch.clamp(self.max_weight_norm / norms, max=1.0)
        self.edge_weight.mul_(scales)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "storage": self.STORAGE_FORMAT,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "fan_in": self.fan_in,
            "row_fan_in": self.row_fan_in,
            "max_weight_norm": self.max_weight_norm,
            "pre_index": self.pre_index.detach().cpu().clone(),
            "edge_weight": self.edge_weight.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("storage") != self.STORAGE_FORMAT:
            raise ValueError("unsupported synapse storage format")
        expected = (
            self.out_features,
            self.in_features,
            self.fan_in,
            self.row_fan_in,
        )
        actual = (
            int(payload["out_features"]),
            int(payload["in_features"]),
            int(payload["fan_in"]),
            int(payload["row_fan_in"]),
        )
        if actual != expected:
            raise ValueError("synapse payload shape does not match architecture")
        pre_index = payload["pre_index"].detach().to(
            device=self.device, dtype=torch.int32
        )
        edge_weight = payload["edge_weight"].detach().to(
            device=self.device, dtype=torch.float32
        )
        if not torch.equal(pre_index, self.pre_index):
            raise ValueError("synapse presynaptic topology does not match architecture")
        if edge_weight.shape != self.edge_weight.shape:
            raise ValueError("synapse edge weights do not match architecture")
        self.edge_weight = edge_weight.clone()
        self._bound_rows()
