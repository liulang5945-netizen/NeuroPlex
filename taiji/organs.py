"""Native raw-byte sensory and motor organs for Taiji."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .config import TaijiConfig
from .sparse import SparseSynapses


class ByteSensor:
    """Map each raw byte and the boundary marker to one receptor population."""

    def __init__(self, config: TaijiConfig, *, device: torch.device | str = "cpu"):
        if config.alphabet_size < 257:
            raise ValueError("ByteSensor requires 256 bytes plus one boundary symbol")
        self.config = config
        self.device = torch.device(device)

    def encode(self, symbol: int) -> torch.Tensor:
        if not 0 <= int(symbol) < self.config.alphabet_size:
            raise ValueError(f"symbol {symbol} is outside the sensor alphabet")
        value = torch.zeros(self.config.alphabet_size, device=self.device)
        value[int(symbol)] = 1.0
        return value

    def symbols(self, data: bytes, *, include_boundary: bool = True) -> tuple[int, ...]:
        body = tuple(int(value) for value in data)
        if not include_boundary:
            return body
        boundary = self.config.boundary_symbol
        return (boundary, *body, boundary)


class ByteMotor:
    """A single action organ trained by local outcome error."""

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)

    def probabilities(self, context: torch.Tensor) -> torch.Tensor:
        evidence = self.synapses.forward(context) + self.bias
        evidence = evidence / float(self.config.motor_temperature)
        return torch.softmax(evidence, dim=0)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
    ) -> torch.Tensor:
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(observed_symbol)] = 1.0
        error = target - predicted.to(self.device)
        self.synapses.local_update(
            error,
            context,
            learning_rate=self.config.motor_learning_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)
        return error

    def to_payload(self) -> Dict[str, Any]:
        return {
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("motor bias shape does not match architecture")
        self.bias = bias
