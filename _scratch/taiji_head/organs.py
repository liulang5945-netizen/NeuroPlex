"""Native raw-byte sensory and motor organs for Taiji."""

from __future__ import annotations

import math
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


class SparseReceptorBank:
    """Fold every cortical signal into a shared, bounded motor evidence space.

    Each input coordinate has exactly one fixed excitatory or inhibitory edge.
    Inputs are assigned evenly across receptor channels, so sparse compression
    never makes a cortical coordinate invisible to the action population.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: torch.Generator,
        context_norm: float,
        device: torch.device | str = "cpu",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("receptor dimensions must be positive")
        if out_features > in_features:
            raise ValueError("receptor count cannot exceed cortical inputs")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.context_norm = float(context_norm)
        self.device = torch.device(device)

        order = torch.randperm(self.in_features, generator=generator)
        channel = torch.empty(self.in_features, dtype=torch.long)
        channel[order] = torch.arange(self.in_features) % self.out_features
        polarity = (
            torch.randint(0, 2, (self.in_features,), generator=generator) * 2 - 1
        ).to(torch.float32)
        counts = torch.bincount(channel, minlength=self.out_features).to(torch.float32)

        self.channel = channel.to(self.device)
        self.polarity = polarity.to(self.device)
        self.channel_scale = counts.rsqrt().to(self.device)

    def forward(self, cortical_state: torch.Tensor) -> torch.Tensor:
        if cortical_state.shape != (self.in_features,):
            raise ValueError(
                f"cortical state shape must be ({self.in_features},), "
                f"got {tuple(cortical_state.shape)}"
            )
        context = torch.zeros(self.out_features, device=self.device)
        context.scatter_add_(
            0,
            self.channel,
            cortical_state.to(self.device) * self.polarity,
        )
        context.mul_(self.channel_scale)
        norm = context.norm()
        if float(norm.item()) < 1e-8:
            return context
        return context * (self.context_norm / norm)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "context_norm": self.context_norm,
            "channel": self.channel.detach().cpu().clone(),
            "polarity": self.polarity.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        expected = (self.in_features, self.out_features, self.context_norm)
        actual = (
            int(payload["in_features"]),
            int(payload["out_features"]),
            float(payload["context_norm"]),
        )
        if actual != expected:
            raise ValueError("receptor payload does not match architecture")
        channel = payload["channel"].detach().to(self.device, dtype=torch.long)
        polarity = payload["polarity"].detach().to(
            self.device, dtype=torch.float32
        )
        if not torch.equal(channel, self.channel):
            raise ValueError("receptor channel map does not match architecture")
        if not torch.equal(polarity, self.polarity):
            raise ValueError("receptor polarity does not match architecture")


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
        self.receptors = SparseReceptorBank(
            config.cortical_context_dim,
            config.motor_context_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_context_dim,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)
        self.reward_baseline = 0.0
        self.reward_updates = 0

    def encode_context(self, cortical_state: torch.Tensor) -> torch.Tensor:
        return self.receptors.forward(cortical_state)

    def probabilities(
        self,
        context: torch.Tensor,
        *,
        episodic_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        evidence = self.synapses.forward(context) + self.bias
        if episodic_evidence is not None:
            if episodic_evidence.shape != (self.config.alphabet_size,):
                raise ValueError("episodic evidence dimension mismatch")
            evidence = evidence + episodic_evidence.to(self.device)
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
        self._apply_error(context, error)
        return error

    @torch.no_grad()
    def learn_reward(
        self,
        context: torch.Tensor,
        policy_probabilities: torch.Tensor,
        action_symbol: int,
        reward: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply a local three-factor action × eligibility × reward update."""

        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if policy_probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("policy probability dimension mismatch")
        if not 0 <= int(action_symbol) < self.config.alphabet_size:
            raise ValueError("action is outside the motor alphabet")
        modulation = reward - self.reward_baseline
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(action_symbol)] = 1.0
        error = modulation * (
            target - policy_probabilities.to(self.device)
        )
        self._apply_error(context, error)
        self.reward_baseline += self.config.reward_baseline_rate * modulation
        self.reward_updates += 1
        return error, modulation

    @torch.no_grad()
    def _apply_error(
        self,
        context: torch.Tensor,
        error: torch.Tensor,
    ) -> None:
        self.synapses.local_update(
            error,
            context,
            learning_rate=self.config.motor_learning_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "receptors": self.receptors.to_payload(),
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
            "reward_baseline": self.reward_baseline,
            "reward_updates": self.reward_updates,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.receptors.load_payload(payload["receptors"])
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("motor bias shape does not match architecture")
        self.bias = bias
        self.reward_baseline = float(payload["reward_baseline"])
        self.reward_updates = int(payload["reward_updates"])
