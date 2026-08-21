"""End-to-end native Taiji byte-stream learner and generator."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from .config import TaijiConfig
from .fabric import TaijiFabric
from .organs import ByteMotor, ByteSensor
from .state import TaijiState, TaijiStep


class Taiji:
    """Complete sensor → predictive fabric → motor execution path.

    Learning occurs online at local predictive and motor synapses.  The class
    intentionally exposes no loss.backward() or optimizer contract.
    """

    CHECKPOINT_FORMAT = "taiji-native-v2"
    STATE_VERSION = 2

    def __init__(
        self,
        config: Optional[TaijiConfig] = None,
        *,
        device: torch.device | str = "cpu",
        episode_id: str = "episode-0",
    ) -> None:
        self.config = config or TaijiConfig()
        self.device = torch.device(device)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.config.seed)
        self.sensor = ByteSensor(self.config, device=self.device)
        self.fabric = TaijiFabric(
            self.config, generator=self._rng, device=self.device
        )
        self.motor = ByteMotor(
            self.config, generator=self._rng, device=self.device
        )
        self._state = self._initial_state(episode_id)

    def _initial_state(self, episode_id: str) -> TaijiState:
        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        uniform = torch.full(
            (self.config.alphabet_size,),
            1.0 / self.config.alphabet_size,
            device=self.device,
        )
        return TaijiState(
            version=self.STATE_VERSION,
            tick=0,
            episode_id=episode_id,
            regions=self.fabric.initial_state(),
            motor_context=torch.zeros(
                self.config.motor_context_dim, device=self.device
            ),
            motor_probabilities=uniform,
            last_symbol=None,
        )

    @property
    def tick(self) -> int:
        return self._state.tick

    def snapshot(self) -> TaijiState:
        return self._state.clone()

    def reset_dynamics(self, *, episode_id: Optional[str] = None) -> None:
        """Clear activity while preserving all learned synapses."""

        self._state = self._initial_state(episode_id or self._state.episode_id)

    @torch.no_grad()
    def observe(self, symbol: int, *, learn: bool = True) -> TaijiStep:
        symbol = int(symbol)
        sensory = self.sensor.encode(symbol)
        previous = self._state

        prior_prediction: Optional[int] = None
        prior_probability: Optional[float] = None
        surprise: Optional[float] = None
        if previous.last_symbol is not None:
            prior_prediction = int(previous.motor_probabilities.argmax().item())
            prior_probability = float(previous.motor_probabilities[symbol].item())
            surprise = -math.log(max(prior_probability, 1e-12))
            if learn:
                self.motor.learn(
                    previous.motor_context,
                    previous.motor_probabilities,
                    symbol,
                )

        regions, activity_rates, error_norms = self.fabric.step(
            sensory, previous.regions, learn=learn
        )
        cortical_state = self.fabric.cortical_context(regions)
        context = self.motor.encode_context(cortical_state)
        probabilities = self.motor.probabilities(context)
        predicted_symbol = int(probabilities.argmax().item())
        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=previous.tick + 1,
            episode_id=previous.episode_id,
            regions=regions,
            motor_context=context,
            motor_probabilities=probabilities,
            last_symbol=symbol,
        )
        return TaijiStep(
            tick=previous.tick,
            observed_symbol=symbol,
            predicted_symbol=predicted_symbol,
            probabilities=probabilities.detach().clone(),
            prior_prediction=prior_prediction,
            prior_probability=prior_probability,
            surprise=surprise,
            activity_rates=activity_rates,
            local_error_norms=error_norms,
        )

    def learn_bytes(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        include_boundary: bool = True,
    ) -> Dict[str, float]:
        """Develop on a byte stream using only online local updates."""

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        observations = 0
        correct = 0
        surprise_sum = 0.0
        for epoch in range(epochs):
            self.reset_dynamics(episode_id=f"learn-{epoch}")
            for symbol in self.sensor.symbols(data, include_boundary=include_boundary):
                step = self.observe(symbol, learn=True)
                if step.prior_prediction is not None:
                    observations += 1
                    correct += int(step.prior_prediction == symbol)
                    surprise_sum += float(step.surprise)
        return {
            "observations": float(observations),
            "online_accuracy": correct / max(1, observations),
            "mean_surprise": surprise_sum / max(1, observations),
        }

    def score_bytes(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
    ) -> Dict[str, float]:
        """Evaluate without changing learned parameters or persistent state."""

        checkpoint = self.checkpoint()
        self.reset_dynamics(episode_id="evaluation")
        observations = 0
        correct = 0
        surprise_sum = 0.0
        try:
            for symbol in self.sensor.symbols(data, include_boundary=include_boundary):
                step = self.observe(symbol, learn=False)
                if step.prior_prediction is not None:
                    observations += 1
                    correct += int(step.prior_prediction == symbol)
                    surprise_sum += float(step.surprise)
            return {
                "observations": float(observations),
                "accuracy": correct / max(1, observations),
                "mean_surprise": surprise_sum / max(1, observations),
            }
        finally:
            self.restore(checkpoint)

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
    ) -> bytes:
        if length < 0:
            raise ValueError("length cannot be negative")
        if reset:
            self.reset_dynamics(episode_id="generation")
        step = self.observe(self.config.boundary_symbol, learn=False)
        for symbol in prompt:
            step = self.observe(int(symbol), learn=False)

        generated = bytearray()
        for _ in range(length):
            if sample:
                next_symbol = int(torch.multinomial(
                    step.probabilities.detach().cpu(), 1, generator=self._rng
                ).item())
            else:
                next_symbol = step.predicted_symbol
            if next_symbol == self.config.boundary_symbol and stop_at_boundary:
                break
            if not 0 <= next_symbol <= 255:
                next_symbol = 0
            generated.append(next_symbol)
            step = self.observe(next_symbol, learn=False)
        return bytes(generated)

    def parameter_tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            *self.fabric.parameter_tensors(),
            self.motor.synapses.weight,
            self.motor.bias,
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        if not active_only:
            return sum(tensor.numel() for tensor in self.parameter_tensors())
        return (
            self.fabric.active_edge_count()
            + int(self.motor.synapses.mask.sum().item())
            + self.motor.bias.numel()
        )

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "fabric": self.fabric.to_payload(),
            "motor": self.motor.to_payload(),
            "state": self._state.to_payload(),
            "rng_state": self._rng.get_state().clone(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported Taiji checkpoint format")
        actual = TaijiConfig.from_dict(dict(checkpoint["config"]))
        if actual != self.config:
            raise ValueError("checkpoint configuration does not match architecture")
        self.fabric.load_payload(checkpoint["fabric"])
        self.motor.load_payload(checkpoint["motor"])
        state = TaijiState.from_payload(checkpoint["state"], device=self.device)
        if state.version != self.STATE_VERSION:
            raise ValueError("unsupported Taiji state version")
        if len(state.regions) != len(self.config.region_sizes):
            raise ValueError("checkpoint region state does not match architecture")
        self._state = state
        self._rng.set_state(checkpoint["rng_state"].detach().cpu())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> "Taiji":
        config = TaijiConfig.from_dict(dict(checkpoint["config"]))
        model = cls(config, device=device)
        model.restore(checkpoint)
        return model
