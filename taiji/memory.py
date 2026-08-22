"""Distributed episodic field for the native Taiji substrate.

The field does not allocate one key/value slot per event.  A cortical cue,
executed action, scalar reward and resulting sensation excite overlapping
engram populations.  Existing recurrent edges learn cue-to-event completion;
local readout edges recover action, outcome and value evidence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from .config import TaijiConfig
from .organs import SparseReceptorBank
from .sparse import SparseSynapses, bound_norm
from .state import MemoryRecall, MemoryState


@dataclass(frozen=True)
class EpisodicWrite:
    strength: float
    novelty: float
    salience: float
    recurrent_error_norm: float


@dataclass(frozen=True, eq=False)
class EpisodicReplay:
    """One spontaneously reactivated engram and its selection signals.

    Nothing here is retrieved from a stored list.  The pattern is regenerated
    by the field's own recurrent completion from an endogenous seed, and every
    scalar is measured on that regenerated pattern.
    """

    pattern: torch.Tensor
    cortical_projection: torch.Tensor
    action_probabilities: torch.Tensor
    outcome_probabilities: torch.Tensor
    time_code: torch.Tensor
    episode_code: torch.Tensor
    provenance_probabilities: torch.Tensor
    novelty: float
    value: float
    familiarity: float
    resonance: float
    priority: float
    expected_reward: float
    accepted: bool


class EpisodicField:
    """Sparse autoassociative population with local three-factor writes."""

    PROVENANCE_KINDS = ("experienced", "imagined", "replayed", "external")

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        units = config.memory_units
        alphabet = config.alphabet_size

        # Fixed convergent pathways create overlapping population codes.  They
        # are not event-specific keys and are never changed by experience.
        self.cue_encoder = SparseSynapses(
            units,
            config.cortical_context_dim,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.action_encoder = SparseSynapses(
            units,
            alphabet,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.outcome_encoder = SparseSynapses(
            units,
            alphabet,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.time_encoder = SparseSynapses(
            units,
            config.memory_time_dim,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.episode_encoder = SparseSynapses(
            units,
            config.memory_episode_dim,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.provenance_encoder = SparseSynapses(
            units,
            len(self.PROVENANCE_KINDS),
            len(self.PROVENANCE_KINDS),
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        reward_code = (
            torch.randint(0, 2, (units,), generator=generator) * 2 - 1
        ).to(torch.float32)
        self.reward_code = (
            reward_code * float(config.weight_init_scale)
        ).to(self.device)

        # All plastic pathways begin blank.  Fixed fan-in topology determines
        # which physically existing synapses may participate in an engram.
        self.association = SparseSynapses(
            units,
            units,
            config.memory_fan_in,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=False,
        )
        self.association.edge_weight.zero_()

        # Every readout decodes the field through one fixed receptor context.
        # The receptors pool the population into a normalized low-dimensional
        # basis, which is what lets a one-shot write train its readouts in a
        # handful of local updates: decoding the raw 192-unit pattern directly
        # spreads each readout row's fixed fan-in over support the engram may
        # not use, and measured readout fit collapsed to chance on exactly the
        # patterns it had just been trained on.
        self.readout_receptors = SparseReceptorBank(
            units,
            config.memory_meta_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        self.action_readout = self._blank_readout(alphabet, generator)
        self.outcome_readout = self._blank_readout(alphabet, generator)
        self.reward_readout = self._blank_readout(1, generator)
        self.familiarity_readout = self._blank_readout(1, generator)
        self.cortical_readout = self._blank_readout(
            config.cortical_context_dim, generator
        )
        self.time_readout = self._blank_readout(
            config.memory_time_dim, generator
        )
        self.episode_readout = self._blank_readout(
            config.memory_episode_dim, generator
        )
        self.provenance_readout = self._blank_readout(
            len(self.PROVENANCE_KINDS), generator
        )
        self.write_count = 0

    def _blank_readout(
        self, out_features: int, generator: torch.Generator
    ) -> SparseSynapses:
        readout = SparseSynapses(
            out_features,
            self.config.memory_meta_dim,
            min(
                self.config.memory_readout_fan_in,
                self.config.memory_meta_dim,
            ),
            generator=generator,
            init_scale=self.config.weight_init_scale,
            max_weight_norm=self.config.max_weight_norm,
            device=self.device,
        )
        readout.edge_weight.zero_()
        return readout

    def initial_state(self) -> MemoryState:
        zero = torch.zeros(self.config.memory_units, device=self.device)
        return MemoryState(
            activity=zero.clone(),
            trace=zero.clone(),
            cortical_feedback=torch.zeros(
                self.config.cortical_context_dim, device=self.device
            ),
            threshold=torch.full(
                (self.config.memory_units,),
                self.config.threshold_base,
                device=self.device,
            ),
            inhibition=0.0,
            last_confidence=0.0,
        )

    def _one_hot(self, symbol: int) -> torch.Tensor:
        if not 0 <= int(symbol) < self.config.alphabet_size:
            raise ValueError("episodic symbol is outside the alphabet")
        encoded = torch.zeros(self.config.alphabet_size, device=self.device)
        encoded[int(symbol)] = 1.0
        return encoded

    def _time_code(self, tick: int) -> torch.Tensor:
        if int(tick) < 0:
            raise ValueError("episodic tick cannot be negative")
        pairs = self.config.memory_time_dim // 2
        tick_value = float(tick)
        values = []
        for index in range(pairs):
            period = float(2 ** index)
            values.extend((
                math.sin(tick_value / period),
                math.cos(tick_value / period),
            ))
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _episode_code(self, episode_id: str) -> torch.Tensor:
        if not episode_id:
            raise ValueError("episodic episode_id cannot be empty")
        raw = hashlib.shake_256(episode_id.encode("utf-8")).digest(
            self.config.memory_episode_dim
        )
        values = [1.0 if value >= 128 else -1.0 for value in raw]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _provenance_code(self, provenance: str) -> torch.Tensor:
        try:
            index = self.PROVENANCE_KINDS.index(str(provenance))
        except ValueError as exc:
            raise ValueError(
                f"unsupported episodic provenance: {provenance}"
            ) from exc
        code = torch.zeros(len(self.PROVENANCE_KINDS), device=self.device)
        code[index] = 1.0
        return code

    def _activate(
        self,
        drive: torch.Tensor,
        threshold: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        positive = torch.relu(drive - threshold)
        inhibition = (
            self.config.memory_inhibition_gain
            * float(positive.mean().item())
        )
        activity = torch.tanh(torch.relu(drive - threshold - inhibition))
        activity = bound_norm(activity, self.config.max_trace_norm)
        return activity, inhibition

    def _normalize_drive(self, drive: torch.Tensor) -> torch.Tensor:
        """Divisive population normalization without selecting global winners."""

        rms = drive.square().mean().sqrt()
        if float(rms.item()) < 1e-8:
            return drive
        return drive * (float(self.config.weight_init_scale) / rms)

    def _encode_cue(self, cortical_context: torch.Tensor) -> torch.Tensor:
        return self._normalize_drive(
            self.cue_encoder.forward(cortical_context.to(self.device))
        )

    def _cue_pattern(
        self, cortical_context: torch.Tensor, threshold: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.config.cortical_context_dim,)
        if cortical_context.shape != expected:
            raise ValueError(
                f"episodic cortical context must be {expected}, "
                f"got {tuple(cortical_context.shape)}"
            )
        drive = self._encode_cue(cortical_context)
        activity, _ = self._activate(drive, threshold)
        return activity

    @torch.no_grad()
    def recall(
        self,
        cortical_context: torch.Tensor,
        previous: MemoryState,
        *,
        use_long_term: bool = True,
    ) -> Tuple[MemoryState, MemoryRecall]:
        """Complete a distributed cue and expose recalled causal evidence."""

        cue_drive = self._encode_cue(cortical_context)
        drive = cue_drive + (1.0 - self.config.memory_trace_decay) * previous.trace
        activity, inhibition = self._activate(drive, previous.threshold)
        long_term = bool(use_long_term and self.write_count > 0)
        if long_term:
            for _ in range(self.config.memory_iterations):
                recurrent = self.association.forward(activity)
                activity, inhibition = self._activate(
                    cue_drive
                    + self.config.memory_recurrent_gain * recurrent
                    + (1.0 - self.config.memory_trace_decay) * previous.trace,
                    previous.threshold,
                )

        active = (activity > 1e-6).to(activity.dtype)
        threshold = torch.clamp(
            previous.threshold
            + self.config.homeostasis_rate
            * (active - self.config.target_activity),
            min=self.config.threshold_min,
            max=self.config.threshold_max,
        )
        trace = bound_norm(
            self.config.memory_trace_decay * previous.trace
            + (1.0 - self.config.memory_trace_decay) * activity,
            self.config.max_trace_norm,
        )

        zeros = torch.zeros(self.config.alphabet_size, device=self.device)
        if long_term:
            recurrent_support = self.association.forward(activity)
            context = self.readout_receptors.forward(activity)
            action_evidence = self.action_readout.forward(context)
            outcome_evidence = self.outcome_readout.forward(context)
            time_code = self.time_readout.forward(context)
            episode_code = self.episode_readout.forward(context)
            provenance_evidence = self.provenance_readout.forward(context)
            raw_expected_reward = float(
                self.reward_readout.forward(context)[0].item()
            )
            familiarity = float(
                self.familiarity_readout.forward(context)[0].item()
            )
            familiarity_confidence = 1.0 - math.exp(-max(0.0, familiarity))
            resonance_confidence = 1.0 - math.exp(
                -float(recurrent_support.norm().item())
            )
            confidence = familiarity_confidence * resonance_confidence
            cortical_feedback = (
                confidence * self.cortical_readout.forward(context)
                * float(self.config.max_membrane_norm)
            )
            time_code = confidence * time_code
            episode_code = confidence * episode_code
            expected_reward = confidence * raw_expected_reward
            action_probabilities = torch.softmax(
                confidence * action_evidence, dim=0
            )
            outcome_probabilities = torch.softmax(
                confidence * outcome_evidence, dim=0
            )
            provenance_probabilities = torch.softmax(
                confidence * provenance_evidence, dim=0
            )
        else:
            action_evidence = zeros.clone()
            action_probabilities = torch.full_like(
                zeros, 1.0 / self.config.alphabet_size
            )
            outcome_probabilities = action_probabilities.clone()
            cortical_feedback = torch.zeros(
                self.config.cortical_context_dim, device=self.device
            )
            time_code = torch.zeros(
                self.config.memory_time_dim, device=self.device
            )
            episode_code = torch.zeros(
                self.config.memory_episode_dim, device=self.device
            )
            provenance_probabilities = torch.full(
                (len(self.PROVENANCE_KINDS),),
                1.0 / len(self.PROVENANCE_KINDS),
                device=self.device,
            )
            expected_reward = 0.0
            confidence = 0.0

        next_state = MemoryState(
            activity=activity,
            trace=trace,
            cortical_feedback=cortical_feedback.detach().clone(),
            threshold=threshold,
            inhibition=float(inhibition),
            last_confidence=float(confidence),
        )
        recall = MemoryRecall(
            action_evidence=action_evidence.detach().clone(),
            action_probabilities=action_probabilities.detach().clone(),
            outcome_probabilities=outcome_probabilities.detach().clone(),
            cortical_feedback=cortical_feedback.detach().clone(),
            time_code=time_code.detach().clone(),
            episode_code=episode_code.detach().clone(),
            provenance_probabilities=provenance_probabilities.detach().clone(),
            expected_reward=float(expected_reward),
            confidence=float(confidence),
            used_long_term=long_term,
        )
        return next_state, recall

    @torch.no_grad()
    def write(
        self,
        cortical_context: torch.Tensor,
        *,
        action_symbol: int,
        reward: float,
        outcome_symbol: int,
        tick: int,
        episode_id: str,
        provenance: str,
        threshold: torch.Tensor,
    ) -> EpisodicWrite:
        """Bind one real action transition into overlapping local synapses."""

        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("episodic reward must be finite")
        cue_pattern = self._cue_pattern(cortical_context, threshold)
        action_drive = self._normalize_drive(
            self.action_encoder.forward(self._one_hot(action_symbol))
        )
        outcome_drive = self._normalize_drive(
            self.outcome_encoder.forward(self._one_hot(outcome_symbol))
        )
        time_code = self._time_code(tick)
        episode_code = self._episode_code(episode_id)
        provenance_code = self._provenance_code(provenance)
        time_drive = self._normalize_drive(self.time_encoder.forward(time_code))
        episode_drive = self._normalize_drive(
            self.episode_encoder.forward(episode_code)
        )
        provenance_drive = self._normalize_drive(
            self.provenance_encoder.forward(provenance_code)
        )
        cue_drive = self._encode_cue(cortical_context)
        event_components = (
            action_drive,
            outcome_drive,
            reward * self.reward_code,
            time_drive,
            episode_drive,
            provenance_drive,
        )
        event_scale = self.config.memory_event_gain / math.sqrt(
            len(event_components)
        )
        event_drive = cue_drive + event_scale * torch.stack(
            event_components, dim=0
        ).sum(dim=0)
        event_pattern, _ = self._activate(event_drive, threshold)

        completion_before = self.association.forward(cue_pattern)
        recurrent_error = event_pattern - completion_before
        novelty = float(
            torch.clamp(
                recurrent_error.norm() / event_pattern.norm().clamp_min(1e-8),
                min=0.0,
                max=1.0,
            ).item()
        )
        salience = math.tanh(abs(reward))
        strength = min(
            1.0,
            self.config.memory_novelty_gain * novelty
            + self.config.memory_reward_gain * salience,
        )

        association_rate = self.config.episodic_learning_rate * strength
        for _ in range(int(self.config.episodic_write_repeats)):
            self.association.local_update(
                event_pattern - self.association.forward(cue_pattern),
                cue_pattern,
                learning_rate=association_rate,
                weight_decay=self.config.synapse_decay,
            )
            self.association.local_update(
                event_pattern - self.association.forward(event_pattern),
                event_pattern,
                learning_rate=0.5 * association_rate,
                weight_decay=self.config.synapse_decay,
            )

        context = self.readout_receptors.forward(event_pattern)
        readout_rate = self.config.episodic_readout_learning_rate * strength
        # The cortical readout regresses a high-dimensional value, unlike the
        # softmax readouts whose error is a probability residual bounded by one.
        # At the shared one-shot rate the delta step saturates every row
        # against the weight cap before any fit exists, and clipped rows all
        # collapse onto the last context direction, erasing cue identity on
        # the very patterns they were just trained on.  A unit-scale target,
        # a sub-stable rate for the receptor norm, and extra repeats let the
        # regression converge instead of oscillating.
        cortical_rate = (
            self.config.cortical_readout_learning_rate * strength
        )
        cortical_scale = float(self.config.max_membrane_norm)
        # The episodic action projection is value-bearing: the field records
        # which action occurred, while reward valence determines whether its
        # recalled evidence should invite or suppress repeating that action.
        action_target = self._one_hot(action_symbol)
        outcome_target = self._one_hot(outcome_symbol)
        cortical_target = (
            bound_norm(
                cortical_context.to(self.device), self.config.max_membrane_norm
            )
            / cortical_scale
        )
        for _ in range(int(self.config.episodic_write_repeats)):
            action_policy = torch.softmax(
                self.action_readout.forward(context), dim=0
            )
            action_error = reward * (action_target - action_policy)
            self.action_readout.local_update(
                action_error,
                context,
                learning_rate=readout_rate,
                weight_decay=self.config.synapse_decay,
            )
            outcome_error = outcome_target - torch.softmax(
                self.outcome_readout.forward(context), dim=0
            )
            self.outcome_readout.local_update(
                outcome_error,
                context,
                learning_rate=readout_rate,
                weight_decay=self.config.synapse_decay,
            )
        for _ in range(int(self.config.cortical_readout_repeats)):
            self.cortical_readout.local_update(
                cortical_target - self.cortical_readout.forward(context),
                context,
                learning_rate=cortical_rate,
                weight_decay=self.config.synapse_decay,
            )
        reward_error = torch.tensor(
            [
                reward
                - float(self.reward_readout.forward(context)[0].item())
            ],
            device=self.device,
        )
        self.reward_readout.local_update(
            reward_error,
            context,
            learning_rate=readout_rate,
            weight_decay=self.config.synapse_decay,
        )
        familiarity = float(
            self.familiarity_readout.forward(context)[0].item()
        )
        familiarity_error = torch.tensor(
            [1.0 - (1.0 - math.exp(-max(0.0, familiarity)))],
            device=self.device,
        )
        self.familiarity_readout.local_update(
            familiarity_error,
            context,
            learning_rate=readout_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.time_readout.local_update(
            time_code - self.time_readout.forward(context),
            context,
            learning_rate=readout_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.episode_readout.local_update(
            episode_code - self.episode_readout.forward(context),
            context,
            learning_rate=readout_rate,
            weight_decay=self.config.synapse_decay,
        )
        provenance_error = provenance_code - torch.softmax(
            self.provenance_readout.forward(context), dim=0
        )
        self.provenance_readout.local_update(
            provenance_error,
            context,
            learning_rate=readout_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.write_count += 1
        return EpisodicWrite(
            strength=float(strength),
            novelty=float(novelty),
            salience=float(salience),
            recurrent_error_norm=float(recurrent_error.norm().item()),
        )

    @torch.no_grad()
    def replay(
        self,
        previous: MemoryState,
        *,
        tick: int,
        generator: torch.Generator,
    ) -> Tuple[MemoryState, EpisodicReplay]:
        """Spontaneously regenerate one engram without any external cue.

        No stored event list is consulted.  The seed is built only from field
        internal quantities: the fixed value axis, the field's own multi-scale
        clock and endogenous noise.  Recurrent completion then pulls that seed
        onto whichever engram its own edges support, and every selection signal
        is measured on the regenerated pattern.  The association matrix is
        deliberately left untouched, so a self generated pattern can never
        reinforce itself into a false memory.

        The residual trace enters as fatigue rather than as drive.  Awake, an
        external cue decides what is recalled and the trace merely binds
        successive cues, so it belongs in the drive.  Asleep there is no cue, so
        the trace becomes the only thing deciding what regenerates, and adding it
        to the drive makes the field return to whatever it just rehearsed until
        one engram monopolises the bout.  Subtracting it instead -- by raising
        the activation threshold on the units that just fired -- is the spike
        frequency adaptation real cortex shows, and it lets a consolidated trace
        withdraw from the competition so its neighbours get their turn.

        The offset is deliberately zero mean across the population: units that
        just fired are held back by exactly as much as the silent ones are
        released.  Cortical homeostasis conserves total population activity and
        adaptation only redistributes which cells carry it, so a one sided
        suppression would be the wrong mechanism as well as the wrong result --
        it dims the whole bout, and since resonance and familiarity are read off
        the magnitude of the regenerated pattern, it would drag priority under
        the acceptance gate for a reason unrelated to which engram won.  Made
        conservative, fatigue moves selection without touching expression, and
        coverage equalises out of the field's own dynamics, with no replay list,
        no quota and no per engram counter.

        Two homeostatic invariants keep sleep from corrupting waking.  The
        excitability threshold is carried through unchanged, because a set point
        earned by real experience must not be recalibrated against self
        generated activity; fatigue is a transient offset read on top of it and
        is never written back.  And the byte readouts are decoded at their native
        scale: confidence gates how much a dream is allowed to teach, never what
        was dreamt, so a hesitant reactivation stays a sharp guess rather than
        degenerating into a uniform one.
        """

        if self.write_count <= 0:
            raise RuntimeError("episodic replay requires at least one write")

        units = self.config.memory_units
        self_clock = self._time_code(tick)
        time_drive = self._normalize_drive(self.time_encoder.forward(self_clock))
        noise = torch.randn(
            units, generator=generator, dtype=torch.float32
        ).to(self.device)
        value_weight = float(self.config.replay_value_weight)
        seed_drive = (
            float(self.config.replay_seed_gain)
            * (value_weight * self.reward_code + (1.0 - value_weight) * time_drive)
            + float(self.config.replay_noise_scale) * self._normalize_drive(noise)
        )
        adapted = previous.threshold + float(self.config.replay_fatigue_gain) * (
            previous.trace - previous.trace.mean()
        )
        activity, inhibition = self._activate(seed_drive, adapted)
        for _ in range(self.config.memory_iterations):
            recurrent = self.association.forward(activity)
            activity, inhibition = self._activate(
                seed_drive + self.config.memory_recurrent_gain * recurrent,
                adapted,
            )

        # Close the causal binding loop before anything leaves the field.  The
        # first spontaneous completion proposes an action from the field's own
        # learned readout; its fixed encoder then projects that proposal back
        # into the same population.  A second recurrent completion must settle
        # cue, action and outcome around that common constraint, so independent
        # readout heads cannot assemble a cue from one engram and an action from
        # another.  This is endogenous attractor refinement, not a teacher: no
        # external symbol or replay list enters the loop.
        proposed_action = int(
            self.action_readout.forward(
                self.readout_receptors.forward(activity)
            ).argmax().item()
        )
        binding_drive = self._normalize_drive(
            self.action_encoder.forward(self._one_hot(proposed_action))
        )
        bound_seed = (
            seed_drive
            + float(self.config.memory_action_binding_gain) * binding_drive
        )
        for _ in range(self.config.memory_iterations):
            recurrent = self.association.forward(activity)
            activity, inhibition = self._activate(
                bound_seed + self.config.memory_recurrent_gain * recurrent,
                adapted,
            )

        recurrent_support = self.association.forward(activity)
        completion_error = activity - recurrent_support
        novelty = float(
            torch.clamp(
                completion_error.norm() / activity.norm().clamp_min(1e-8),
                min=0.0,
                max=1.0,
            ).item()
        )

        context = self.readout_receptors.forward(activity)
        familiarity = float(
            self.familiarity_readout.forward(context)[0].item()
        )
        familiarity_confidence = 1.0 - math.exp(-max(0.0, familiarity))
        resonance = 1.0 - math.exp(-float(recurrent_support.norm().item()))
        confidence = familiarity_confidence * resonance
        raw_expected_reward = float(
            self.reward_readout.forward(context)[0].item()
        )
        expected_reward = confidence * raw_expected_reward
        value = math.tanh(abs(raw_expected_reward))

        time_code = confidence * self.time_readout.forward(context)
        episode_code = confidence * self.episode_readout.forward(context)
        cortical_projection = (
            confidence
            * self.cortical_readout.forward(context)
            * float(self.config.max_membrane_norm)
        )
        action_probabilities = torch.softmax(
            self.action_readout.forward(context), dim=0
        )
        outcome_probabilities = torch.softmax(
            self.outcome_readout.forward(context), dim=0
        )
        provenance_probabilities = torch.softmax(
            self.provenance_readout.forward(context), dim=0
        )

        clock_norm = self_clock.norm().clamp_min(1e-8)
        recalled_norm = time_code.norm().clamp_min(1e-8)
        recency = 0.5 * (
            1.0
            + float(
                (torch.dot(time_code, self_clock) / (recalled_norm * clock_norm)).item()
            )
        )
        selection = value_weight * value + (1.0 - value_weight) * novelty
        priority = familiarity_confidence * resonance * selection * recency
        accepted = priority >= float(self.config.replay_priority_threshold)

        trace = bound_norm(
            self.config.memory_trace_decay * previous.trace
            + (1.0 - self.config.memory_trace_decay) * activity,
            self.config.max_trace_norm,
        )
        next_state = MemoryState(
            activity=activity,
            trace=trace,
            cortical_feedback=cortical_projection.detach().clone(),
            threshold=previous.threshold.detach().clone(),
            inhibition=float(inhibition),
            last_confidence=float(confidence),
        )
        event = EpisodicReplay(
            pattern=activity.detach().clone(),
            cortical_projection=cortical_projection.detach().clone(),
            action_probabilities=action_probabilities.detach().clone(),
            outcome_probabilities=outcome_probabilities.detach().clone(),
            time_code=time_code.detach().clone(),
            episode_code=episode_code.detach().clone(),
            provenance_probabilities=provenance_probabilities.detach().clone(),
            novelty=float(novelty),
            value=float(value),
            familiarity=float(familiarity_confidence),
            resonance=float(resonance),
            priority=float(priority),
            expected_reward=float(expected_reward),
            accepted=bool(accepted),
        )
        return next_state, event

    def parameter_tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.association.edge_weight,
            self.action_readout.edge_weight,
            self.outcome_readout.edge_weight,
            self.reward_readout.edge_weight,
            self.familiarity_readout.edge_weight,
            self.cortical_readout.edge_weight,
            self.time_readout.edge_weight,
            self.episode_readout.edge_weight,
            self.provenance_readout.edge_weight,
        )

    def active_edge_count(self) -> int:
        return sum(
            projection.edge_count
            for projection in (
                self.association,
                self.action_readout,
                self.outcome_readout,
                self.reward_readout,
                self.familiarity_readout,
                self.cortical_readout,
                self.time_readout,
                self.episode_readout,
                self.provenance_readout,
            )
        )

    def dense_equivalent_edge_count(self) -> int:
        return sum(
            projection.dense_equivalent_count
            for projection in (
                self.association,
                self.action_readout,
                self.outcome_readout,
                self.reward_readout,
                self.familiarity_readout,
                self.cortical_readout,
                self.time_readout,
                self.episode_readout,
                self.provenance_readout,
            )
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cue_encoder": self.cue_encoder.to_payload(),
            "action_encoder": self.action_encoder.to_payload(),
            "outcome_encoder": self.outcome_encoder.to_payload(),
            "time_encoder": self.time_encoder.to_payload(),
            "episode_encoder": self.episode_encoder.to_payload(),
            "provenance_encoder": self.provenance_encoder.to_payload(),
            "reward_code": self.reward_code.detach().cpu().clone(),
            "association": self.association.to_payload(),
            "readout_receptors": self.readout_receptors.to_payload(),
            "action_readout": self.action_readout.to_payload(),
            "outcome_readout": self.outcome_readout.to_payload(),
            "reward_readout": self.reward_readout.to_payload(),
            "familiarity_readout": self.familiarity_readout.to_payload(),
            "cortical_readout": self.cortical_readout.to_payload(),
            "time_readout": self.time_readout.to_payload(),
            "episode_readout": self.episode_readout.to_payload(),
            "provenance_readout": self.provenance_readout.to_payload(),
            "write_count": self.write_count,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.cue_encoder.load_payload(payload["cue_encoder"])
        self.action_encoder.load_payload(payload["action_encoder"])
        self.outcome_encoder.load_payload(payload["outcome_encoder"])
        self.time_encoder.load_payload(payload["time_encoder"])
        self.episode_encoder.load_payload(payload["episode_encoder"])
        self.provenance_encoder.load_payload(payload["provenance_encoder"])
        reward_code = payload["reward_code"].detach().to(
            self.device, dtype=torch.float32
        )
        if reward_code.shape != self.reward_code.shape:
            raise ValueError("episodic reward code does not match architecture")
        self.reward_code = reward_code.clone()
        self.association.load_payload(payload["association"])
        self.readout_receptors.load_payload(payload["readout_receptors"])
        self.action_readout.load_payload(payload["action_readout"])
        self.outcome_readout.load_payload(payload["outcome_readout"])
        self.reward_readout.load_payload(payload["reward_readout"])
        self.familiarity_readout.load_payload(payload["familiarity_readout"])
        self.cortical_readout.load_payload(payload["cortical_readout"])
        self.time_readout.load_payload(payload["time_readout"])
        self.episode_readout.load_payload(payload["episode_readout"])
        self.provenance_readout.load_payload(payload["provenance_readout"])
        self.write_count = int(payload["write_count"])
        if self.write_count < 0:
            raise ValueError("episodic write count cannot be negative")
