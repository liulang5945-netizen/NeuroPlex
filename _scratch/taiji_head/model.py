"""End-to-end native Taiji byte-stream learner and generator."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from .config import TaijiConfig
from .fabric import TaijiFabric
from .memory import EpisodicField
from .organs import ByteMotor, ByteSensor
from .state import (
    PendingAction,
    PendingExperience,
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
)


class Taiji:
    """Complete sensor → predictive fabric ↔ episodic field → motor path.

    Learning occurs online at local predictive, memory and motor synapses.
    The class intentionally exposes no loss.backward() or optimizer contract.
    """

    CHECKPOINT_FORMAT = "taiji-native-v7"
    STATE_VERSION = 5

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
        self._memory_rng = torch.Generator(device="cpu")
        self._memory_rng.set_state(self._rng.get_state().clone())
        self.memory = EpisodicField(
            self.config, generator=self._memory_rng, device=self.device
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
            memory=self.memory.initial_state(),
            motor_context=torch.zeros(
                self.config.motor_context_dim, device=self.device
            ),
            motor_probabilities=uniform,
            last_symbol=None,
            pending_action=None,
            pending_experience=None,
        )

    @property
    def tick(self) -> int:
        return self._state.tick

    def snapshot(self) -> TaijiState:
        return self._state.clone()

    def reset_dynamics(self, *, episode_id: Optional[str] = None) -> None:
        """Clear activity while preserving all learned synapses."""

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before reset")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must observe its outcome before reset")
        self._state = self._initial_state(episode_id or self._state.episode_id)

    @torch.no_grad()
    def observe(
        self,
        symbol: int,
        *,
        learn: bool = True,
        learn_motor: Optional[bool] = None,
        use_memory: bool = True,
    ) -> TaijiStep:
        """Advance one sensation tick.

        ``learn_motor=False`` keeps local fabric learning active without
        treating an externally caused sensation as the correct motor action.
        """

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before observation")
        symbol = int(symbol)
        sensory = self.sensor.encode(symbol)
        previous = self._state
        motor_learning = learn if learn_motor is None else bool(learn_motor)
        memory_write_strength = 0.0
        if previous.pending_experience is not None:
            pending_experience = previous.pending_experience
            if pending_experience.learn_memory:
                memory_write = self.memory.write(
                    pending_experience.cortical_context,
                    action_symbol=pending_experience.action_symbol,
                    reward=pending_experience.reward,
                    outcome_symbol=symbol,
                    tick=pending_experience.tick,
                    episode_id=pending_experience.episode_id,
                    provenance=pending_experience.provenance,
                    threshold=previous.memory.threshold,
                )
                memory_write_strength = memory_write.strength

        prior_prediction: Optional[int] = None
        prior_probability: Optional[float] = None
        surprise: Optional[float] = None
        if previous.last_symbol is not None:
            prior_prediction = int(previous.motor_probabilities.argmax().item())
            prior_probability = float(previous.motor_probabilities[symbol].item())
            surprise = -math.log(max(prior_probability, 1e-12))
            if motor_learning:
                self.motor.learn(
                    previous.motor_context,
                    previous.motor_probabilities,
                    symbol,
                )

        regions, activity_rates, error_norms = self.fabric.step(
            sensory,
            previous.regions,
            learn=learn,
            episodic_feedback=previous.memory.cortical_feedback,
        )
        cortical_state = self.fabric.cortical_context(regions)
        memory_state, memory_recall = self.memory.recall(
            cortical_state,
            previous.memory,
            use_long_term=use_memory,
        )
        context = self.motor.encode_context(cortical_state)
        episodic_evidence = (
            self.config.memory_read_gain
            * memory_recall.confidence
            * memory_recall.action_evidence
        )
        probabilities = self.motor.probabilities(
            context,
            episodic_evidence=episodic_evidence,
        )
        predicted_symbol = int(probabilities.argmax().item())
        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=previous.tick + 1,
            episode_id=previous.episode_id,
            regions=regions,
            memory=memory_state,
            motor_context=context,
            motor_probabilities=probabilities,
            last_symbol=symbol,
            pending_action=None,
            pending_experience=None,
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
            memory_recall=memory_recall,
            memory_write_strength=float(memory_write_strength),
        )

    @torch.no_grad()
    def act(
        self,
        available_actions: Sequence[int],
        *,
        sample: bool = True,
    ) -> TaijiDecision:
        """Select one afforded action and preserve its local eligibility."""

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before acting again")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must observe its outcome before acting")
        actions = tuple(int(value) for value in available_actions)
        if not actions:
            raise ValueError("available_actions cannot be empty")
        if len(set(actions)) != len(actions):
            raise ValueError("available_actions cannot contain duplicates")
        if any(not 0 <= value < self.config.alphabet_size for value in actions):
            raise ValueError("available action is outside the motor alphabet")

        indices = torch.tensor(actions, device=self.device, dtype=torch.long)
        restricted = self._state.motor_probabilities[indices]
        restricted = restricted / restricted.sum().clamp_min(1e-12)
        policy = torch.zeros(
            self.config.alphabet_size, device=self.device
        )
        policy[indices] = restricted
        if sample:
            local_index = int(torch.multinomial(
                restricted.detach().cpu(), 1, generator=self._rng
            ).item())
        else:
            local_index = int(restricted.argmax().item())
        action_symbol = actions[local_index]
        pending = PendingAction(
            tick=self._state.tick,
            action_symbol=action_symbol,
            available_actions=actions,
            context=self._state.motor_context.detach().clone(),
            policy_probabilities=policy.detach().clone(),
        )
        self._state.pending_action = pending
        return TaijiDecision(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            available_actions=pending.available_actions,
            policy_probabilities=pending.policy_probabilities.detach().clone(),
        )

    @torch.no_grad()
    def settle_action(
        self,
        reward: float,
        *,
        learn: bool = True,
        learn_memory: Optional[bool] = None,
        provenance: str = "experienced",
    ) -> TaijiOutcome:
        """Consume the pending action with a scalar environment outcome."""

        pending = self._state.pending_action
        if pending is None:
            raise RuntimeError("no pending action to settle")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if provenance not in self.memory.PROVENANCE_KINDS:
            raise ValueError(f"unsupported episodic provenance: {provenance}")
        modulation = reward - self.motor.reward_baseline
        error_norm = 0.0
        if learn:
            error, modulation = self.motor.learn_reward(
                pending.context,
                pending.policy_probabilities,
                pending.action_symbol,
                reward,
            )
            error_norm = float(error.norm().item())
        self._state.pending_action = None
        self._state.pending_experience = PendingExperience(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            reward=reward,
            cortical_context=self.fabric.cortical_context(
                self._state.regions
            ).detach().clone(),
            episode_id=self._state.episode_id,
            provenance=provenance,
            learn_memory=(bool(learn) if learn_memory is None else bool(learn_memory)),
        )
        return TaijiOutcome(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            reward=reward,
            reward_prediction_error=float(modulation),
            learning_error_norm=error_norm,
        )

    @torch.no_grad()
    def consolidate(self, *, cycles: int = 1, learn: bool = True) -> TaijiConsolidation:
        """Sleep on what the field already holds, with no external input at all.

        Each cycle asks the episodic field to spontaneously regenerate one engram
        from its own value axis, clock, residual trace and noise.  The field's own
        priority gate decides whether that reactivation is worth anything; only
        accepted ones are replayed through the very same predictive fabric: the
        engram's action is driven until the fabric settles, then its outcome is
        presented once as the single writing tick.  The fabric learns from its
        ordinary local prediction error, scaled by how strongly the field vouched
        for the engram.

        No external replay list, no teacher target, and no weight is ever copied
        from the field into the fabric: the only channel is the episodic feedback
        gain that waking observation already uses.  The field itself is left
        unchanged, so a self generated pattern cannot reinforce itself.
        """

        if cycles <= 0:
            raise ValueError("consolidation cycles must be positive")
        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before consolidation")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must be observed before consolidation")
        if self.memory.write_count <= 0:
            raise RuntimeError("consolidation requires at least one episodic write")

        replayed_index = self.memory.PROVENANCE_KINDS.index("replayed")
        state = self._state
        regions = state.regions
        memory_state = state.memory
        tick = int(state.tick)
        structural_before = int(self.fabric.structural_events)
        winner_resource = torch.ones(
            self.config.alphabet_size, device=self.device
        )

        accepted = 0
        priority_sum = 0.0
        novelty_sum = 0.0
        value_sum = 0.0
        confidence_sum = 0.0
        replayed_sum = 0.0
        error_sum = 0.0
        error_count = 0

        for _ in range(int(cycles)):
            memory_state, replay = self.memory.replay(
                memory_state,
                tick=tick,
                generator=self._memory_rng,
            )
            tick += 1
            priority_sum += replay.priority
            novelty_sum += replay.novelty
            value_sum += replay.value
            confidence_sum += memory_state.last_confidence
            replayed_sum += float(replay.provenance_probabilities[replayed_index].item())
            if not replay.accepted:
                continue
            accepted += 1
            threshold = float(self.config.replay_priority_threshold)
            endorsement = (
                min(1.0, replay.priority / threshold) if threshold > 0.0 else 1.0
            )
            learn_scale = (
                float(self.config.replay_learning_scale) * endorsement
                if learn
                else 0.0
            )
            # The engram is read at its mode, not sampled.  The readouts are
            # softmaxes over the whole 257 byte alphabet, so a correct but
            # low-margin reactivation still looks nearly uniform: measured peak
            # mass is ~0.03 and only ~9% of the mass sits on the task vocabulary.
            # Sampling that distribution discards the ordering that carries all
            # of the information -- the mode recovers the true action/outcome
            # pair on every accepted tick, while multinomial draws recovered it
            # on none.  Stochastic exploration belongs in the seed, which is
            # already noise driven; the read-out stage must stay faithful.
            burst = tuple(
                int(probabilities.argmax().item())
                for probabilities in (
                    replay.action_probabilities,
                    replay.outcome_probabilities,
                )
            )
            # A replayed contingency is only worth anything if waking can read it
            # back, and waking arrives with the action alone.  That fixes the
            # shape of the burst completely.
            #
            # The eligibility carrier is the previous tick's slow trace, so the
            # basis a write lands on is whatever the fabric was holding one tick
            # earlier.  If the burst alternated action and outcome, that basis
            # would carry the preceding outcome's residue -- the trace decays at
            # 0.82, so the carryover dominates -- and no probe could reproduce it
            # without being shown the outcome, which would be leakage.  The
            # action is therefore driven on its own from a cleared state, and the
            # outcome is presented exactly once, at the end, as the only writing
            # tick.  The basis is then a pure function of the action.
            #
            # It also has to be the *settled* basis rather than the first
            # transient.  Region 0 is sparse: one tick after a clear leaves 4-9
            # of its 64 units active, while each decoder row draws a fixed fan-in
            # of 16 of those 64.  A 4-unit trace can miss a row's support
            # outright -- measured fan-in energy overlap ran from 50% down to a
            # hard 0%, which makes the corresponding pair unlearnable as
            # arithmetic, not as a matter of dose.  Driving the action until the
            # trace settles brings it to 21-34 active units, which every row's
            # support intersects.
            action_symbol, outcome_symbol = burst
            winner_gain = float(winner_resource[action_symbol].item())
            winner_resource[action_symbol].mul_(
                float(self.config.replay_winner_resource_retention)
            )
            regions = self.fabric.clear_dynamics(regions)
            for _ in range(int(self.config.replay_burst_repeats)):
                regions, _rates, error_norms = self.fabric.step(
                    self.sensor.encode(action_symbol),
                    regions,
                    learn=False,
                    episodic_feedback=replay.cortical_projection,
                    learn_scale=learn_scale,
                    use_consolidated=False,
                    # ``clear_dynamics`` hands the burst the set point waking
                    # left, which is correct -- sleep must not discard what
                    # waking learned.  But a sixteen-tick burst of one symbol is
                    # not the varied traffic homeostasis integrates over, and
                    # letting it adapt inflated the set point on exactly the
                    # engram's units until the write basis collapsed.  The
                    # threshold is therefore read, not written, for the whole
                    # replay.  See the note in ``fabric.step``.
                    adapt_homeostasis=False,
                )
                tick += 1
                error_sum += sum(error_norms) / len(error_norms)
                error_count += 1
            # The settled state is now held fixed while the outcome is presented
            # repeatedly.  Advancing the fabric between writes would move the
            # eligibility trace off the one basis the probe can reproduce, so
            # every write is driven from the *same* captured state instead.
            #
            # This is not a disguised learning-rate multiplier.  Each step
            # recomputes ``lower_error`` from the current weights, so with the
            # basis pinned the row update becomes an error driven fixed point
            # iteration -- w <- w + lr * (target - w.t) * t / scale -- which
            # converges geometrically onto the contingency instead of drifting
            # linearly.  Raising the rate would overshoot the row-norm bound and
            # amplify whatever noise the first error happened to carry; iterating
            # a contracting map cannot.
            #
            # Recovering these writes matters because sleep is on a far smaller
            # plasticity budget than waking: pretraining lands ~3000 updates at
            # full rate, while a replay pass accepts a few hundred at the gated
            # rate.  Discarding all but one tick of each burst threw away the
            # only lever that does not distort the dynamics.
            settled = regions
            outcome_activity = self.sensor.encode(outcome_symbol)
            for repeat in range(int(self.config.replay_write_repeats)):
                regions, _rates, error_norms = self.fabric.step(
                    outcome_activity,
                    settled,
                    learn=learn,
                    episodic_feedback=replay.cortical_projection,
                    learn_scale=learn_scale,
                    consolidation_learn_scale=learn_scale * winner_gain,
                    use_consolidated=False,
                    # Rewire once, on the opening write, then spend the rest of
                    # the burst growing what was just recruited.  Restructuring
                    # on every repeat would leave the final contact stranded at
                    # the zero weight it opens with, and each swap would discard
                    # the partner the previous one had only begun to train.
                    restructure=learn and repeat == 0,
                    adapt_homeostasis=False,
                )
                tick += 1
                error_sum += sum(error_norms) / len(error_norms)
                error_count += 1

        cortical_state = self.fabric.cortical_context(regions)
        context = self.motor.encode_context(cortical_state)
        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=tick,
            episode_id=state.episode_id,
            regions=regions,
            memory=memory_state,
            motor_context=context,
            motor_probabilities=self.motor.probabilities(context),
            last_symbol=None,
            pending_action=None,
            pending_experience=None,
        )
        attempts = float(int(cycles))
        return TaijiConsolidation(
            cycles=int(cycles),
            accepted=accepted,
            mean_priority=priority_sum / attempts,
            mean_novelty=novelty_sum / attempts,
            mean_value=value_sum / attempts,
            mean_confidence=confidence_sum / attempts,
            mean_error_norm=(error_sum / error_count) if error_count else 0.0,
            replayed_probability=replayed_sum / attempts,
            structural_events=int(self.fabric.structural_events) - structural_before,
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
            self.motor.synapses.edge_weight,
            self.motor.bias,
            *self.memory.parameter_tensors(),
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        active = (
            self.fabric.active_edge_count()
            + self.motor.synapses.edge_count
            + self.motor.bias.numel()
            + self.memory.active_edge_count()
        )
        if active_only:
            return active
        return active

    def dense_equivalent_parameter_count(self) -> int:
        """Return the learned scalar count a dense implementation would store."""

        return (
            self.fabric.dense_equivalent_edge_count()
            + self.motor.synapses.dense_equivalent_count
            + self.motor.bias.numel()
            + self.memory.dense_equivalent_edge_count()
        )

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "fabric": self.fabric.to_payload(),
            "motor": self.motor.to_payload(),
            "memory": self.memory.to_payload(),
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
        self.memory.load_payload(checkpoint["memory"])
        state = TaijiState.from_payload(checkpoint["state"], device=self.device)
        if state.version != self.STATE_VERSION:
            raise ValueError("unsupported Taiji state version")
        if len(state.regions) != len(self.config.region_sizes):
            raise ValueError("checkpoint region state does not match architecture")
        memory_shape = (self.config.memory_units,)
        if (
            state.memory.activity.shape != memory_shape
            or state.memory.trace.shape != memory_shape
            or state.memory.threshold.shape != memory_shape
        ):
            raise ValueError("checkpoint memory state does not match architecture")
        if state.memory.cortical_feedback.shape != (
            self.config.cortical_context_dim,
        ):
            raise ValueError("checkpoint memory feedback does not match architecture")
        if state.motor_context.shape != (self.config.motor_context_dim,):
            raise ValueError("checkpoint motor context does not match architecture")
        if state.motor_probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("checkpoint motor probabilities do not match architecture")
        if state.pending_action is not None:
            pending = state.pending_action
            if pending.context.shape != (self.config.motor_context_dim,):
                raise ValueError("checkpoint pending context does not match architecture")
            if pending.policy_probabilities.shape != (self.config.alphabet_size,):
                raise ValueError("checkpoint pending policy does not match architecture")
            if pending.action_symbol not in pending.available_actions:
                raise ValueError("checkpoint pending action is not afforded")
        if state.pending_experience is not None:
            experience = state.pending_experience
            if experience.cortical_context.shape != (
                self.config.cortical_context_dim,
            ):
                raise ValueError(
                    "checkpoint pending experience does not match architecture"
                )
            if not 0 <= experience.action_symbol < self.config.alphabet_size:
                raise ValueError("checkpoint pending experience action is invalid")
            if not math.isfinite(experience.reward):
                raise ValueError("checkpoint pending experience reward is invalid")
            if not experience.episode_id:
                raise ValueError("checkpoint pending experience episode is invalid")
            if experience.provenance not in self.memory.PROVENANCE_KINDS:
                raise ValueError("checkpoint pending experience provenance is invalid")
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
