"""Hierarchical predictive-error dynamics of the native Taiji fabric."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from .config import TaijiConfig
from .sparse import SparseSynapses, bound_norm
from .state import RegionState


class TaijiFabric:
    """A hierarchy of recurrent regions with reciprocal predictive synapses.

    A decoder predicts the region below from the previous local trace.  The
    same physical synapses carry bottom-up prediction error through their
    transpose.  Recurrent synapses predict the region's own next activity.
    No sequence matrix, content-addressed attention, or global router exists.
    """

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        lower_sizes = (config.alphabet_size, *config.region_sizes[:-1])
        self.decoders = tuple(
            SparseSynapses(
                lower_size,
                region_size,
                config.synapse_fan_in,
                generator=generator,
                init_scale=config.weight_init_scale,
                max_weight_norm=config.max_weight_norm,
                device=self.device,
            )
            for lower_size, region_size in zip(lower_sizes, config.region_sizes)
        )
        # A separate slow consolidation pathway keeps sleep learning from
        # overwriting the fast sparse predictor.  Every row shares the complete
        # signed cortical support, so learnability no longer depends on whether
        # a randomly sampled row happened to touch a residual coordinate.
        consolidation_rng = torch.Generator(device="cpu")
        consolidation_rng.manual_seed(
            config.seed + config.consolidation_seed_offset
        )
        self.consolidation_decoders = tuple(
            SparseSynapses(
                lower_size,
                region_size,
                region_size,
                generator=consolidation_rng,
                init_scale=config.weight_init_scale,
                max_weight_norm=config.max_weight_norm,
                device=self.device,
            )
            for lower_size, region_size in zip(lower_sizes, config.region_sizes)
        )
        for decoder in self.consolidation_decoders:
            decoder.edge_weight.zero_()
        # The running means supply the opponent origins.  They are homeostatic
        # statistics, not trainable parameters.
        self.trace_baselines = tuple(
            torch.zeros(region_size, device=self.device)
            for region_size in config.region_sizes
        )
        self.transitions = tuple(
            SparseSynapses(
                region_size,
                region_size,
                config.synapse_fan_in,
                generator=generator,
                init_scale=config.weight_init_scale,
                max_weight_norm=config.max_weight_norm,
                device=self.device,
                allow_self=False,
            )
            for region_size in config.region_sizes
        )
        # Lateral competition, one bank per region.  Its topology must not be
        # drawn from the shared generator: every downstream organ (motor, and
        # the forked memory stream) consumes the same sequence, so borrowing
        # draws here would silently shift their initial wiring.  A dedicated
        # offset stream keeps the lateral bank independent and reproducible.
        lateral_rng = torch.Generator(device="cpu")
        lateral_rng.manual_seed(config.seed + config.lateral_seed_offset)
        self.laterals = tuple(
            SparseSynapses(
                region_size,
                region_size,
                config.lateral_fan_in,
                generator=lateral_rng,
                init_scale=config.weight_init_scale,
                # A row of ones -- the uniform baseline below -- already has
                # norm sqrt(fan_in), so the bound is scaled with it.  The
                # headroom is what lets a single pair grow past the baseline.
                max_weight_norm=config.max_weight_norm
                * float(min(config.lateral_fan_in, region_size)) ** 0.5,
                device=self.device,
                allow_self=False,
            )
            for region_size in config.region_sizes
        )
        for lateral in self.laterals:
            # The old global law is not a second path that coexists with this
            # one; it is this bank's degenerate solution.  At W = 1 the mean
            # over a uniformly sampled neighbourhood is an unbiased estimate of
            # ``positive_drive.mean()``, so starting from ones reproduces the
            # tuned scalar dynamics and lets learning depart from it, rather
            # than throwing the working set point away for a random one.
            lateral.edge_weight.fill_(1.0)
        # Audit counter only: how many contacts the substrate has rewired.  It
        # describes the run, not the network, so it stays out of the payload.
        self.structural_events = 0

    def initial_state(self) -> Tuple[RegionState, ...]:
        lower_sizes = (self.config.alphabet_size, *self.config.region_sizes[:-1])
        states = []
        for lower_size, region_size in zip(lower_sizes, self.config.region_sizes):
            zero = torch.zeros(region_size, device=self.device)
            states.append(RegionState(
                membrane=zero.clone(),
                activity=zero.clone(),
                trace=zero.clone(),
                prediction=torch.zeros(lower_size, device=self.device),
                error=torch.zeros(lower_size, device=self.device),
                threshold=torch.full(
                    (region_size,), self.config.threshold_base, device=self.device
                ),
                inhibition=zero.clone(),
            ))
        return tuple(states)

    def opponent_trace(self, index: int, trace: torch.Tensor) -> torch.Tensor:
        if not 0 <= int(index) < len(self.trace_baselines):
            raise IndexError("region index outside the predictive fabric")
        expected = (self.config.region_sizes[int(index)],)
        if trace.shape != expected:
            raise ValueError(
                f"region trace must be {expected}, got {tuple(trace.shape)}"
            )
        return trace.to(self.device) - self.trace_baselines[int(index)]

    def consolidated_decode(self, index: int, trace: torch.Tensor) -> torch.Tensor:
        """Read only the slow shared-support consolidation pathway."""

        return self.consolidation_decoders[int(index)].forward(
            self.opponent_trace(index, trace)
        )

    def decode(
        self,
        index: int,
        trace: torch.Tensor,
        *,
        include_consolidated: bool = True,
    ) -> torch.Tensor:
        """Combine fast sparse prediction with slow opponent consolidation."""

        prediction = self.decoders[int(index)].forward(trace.to(self.device))
        if include_consolidated:
            prediction = prediction + self.consolidated_decode(index, trace)
        return prediction

    @torch.no_grad()
    def step(
        self,
        sensory_activity: torch.Tensor,
        previous: Sequence[RegionState],
        *,
        learn: bool,
        episodic_feedback: Optional[torch.Tensor] = None,
        learn_scale: float = 1.0,
        consolidation_learn_scale: float = 0.0,
        use_consolidated: bool = True,
        restructure: bool = False,
        adapt_homeostasis: bool = True,
    ) -> Tuple[Tuple[RegionState, ...], Tuple[float, ...], Tuple[float, ...]]:
        if sensory_activity.shape != (self.config.alphabet_size,):
            raise ValueError("sensory activity does not match the receptor population")
        if len(previous) != len(self.config.region_sizes):
            raise ValueError("region state count does not match the architecture")
        if not math.isfinite(learn_scale) or learn_scale < 0.0:
            raise ValueError("learn_scale must be a finite non-negative number")
        if (
            not math.isfinite(consolidation_learn_scale)
            or consolidation_learn_scale < 0.0
        ):
            raise ValueError(
                "consolidation_learn_scale must be a finite non-negative number"
            )
        if episodic_feedback is None:
            episodic_feedback = torch.zeros(
                self.config.cortical_context_dim, device=self.device
            )
        elif episodic_feedback.shape != (self.config.cortical_context_dim,):
            raise ValueError("episodic feedback does not match cortical context")
        else:
            episodic_feedback = episodic_feedback.to(self.device)

        lower_activity = sensory_activity.to(self.device)
        next_states = []
        activity_rates = []
        error_norms = []
        feedback_offset = 0
        feedback_trace_offset = sum(self.config.region_sizes)

        for index, (region_size, decoder, transition) in enumerate(zip(
            self.config.region_sizes, self.decoders, self.transitions
        )):
            old = previous[index]
            signed_trace = self.opponent_trace(index, old.trace)
            consolidated_prediction = self.consolidation_decoders[index].forward(
                signed_trace
            )
            lower_prediction = decoder.forward(old.trace)
            if use_consolidated:
                lower_prediction = lower_prediction + consolidated_prediction
            lower_error = lower_activity - lower_prediction
            recurrent_prediction = transition.forward(old.trace)
            bottom_up = decoder.backproject(lower_error)
            if use_consolidated:
                bottom_up = bottom_up + self.consolidation_decoders[
                    index
                ].backproject(lower_error)

            if index + 1 < len(previous):
                top_down = self.decode(
                    index + 1,
                    previous[index + 1].trace,
                    include_consolidated=use_consolidated,
                )
            else:
                top_down = torch.zeros(region_size, device=self.device)

            drive = (
                self.config.bottom_up_gain * bottom_up
                + self.config.recurrent_gain * recurrent_prediction
                + self.config.top_down_gain * top_down
                + self.config.memory_feedback_gain
                * (
                    episodic_feedback[
                        feedback_offset:feedback_offset + region_size
                    ]
                    + episodic_feedback[
                        feedback_trace_offset
                        + feedback_offset:feedback_trace_offset
                        + feedback_offset
                        + region_size
                    ]
                )
            )
            membrane = bound_norm(
                self.config.membrane_decay * old.membrane + drive,
                self.config.max_membrane_norm,
            )
            positive_drive = torch.relu(membrane - old.threshold)
            # Per-unit competition, not a master volume knob.  The old law
            # subtracted ``inhibition_gain * positive_drive.mean()`` from every
            # unit equally: it sharpens the threshold globally but by
            # construction cannot single out a unit for responding to
            # *everything*, and that promiscuity is precisely what carries the
            # rank-1 common mode -- measured at 28.5-35.0% of write energy
            # against the 1/k = 25% orthogonal floor for four actions.
            #
            # A learned lateral bank can.  ``lateral.forward`` divided by the
            # neighbourhood size is a per-unit weighted mean of the drive
            # reaching that unit's competitors, so with all weights at 1.0 it is
            # an unbiased estimate of the global mean and this expression
            # reduces to the tuned law above; anti-Hebbian learning then departs
            # from it, growing exactly the contacts between units that fire
            # together across actions.  Judging "co-active with the others"
            # requires *storing* that statistic, which is why no parameter-free
            # per-tick rule (kWTA, subtract-the-runner-up, local normalization)
            # can do it: while writing one action they cannot see the other
            # three, and a promiscuous unit ranks top-k under every one of them.
            lateral = self.laterals[index]
            competition = lateral.forward(positive_drive) / float(
                max(1, lateral.row_fan_in)
            )
            inhibition = (
                self.config.inhibition_decay * old.inhibition
                + (1.0 - self.config.inhibition_decay)
                * self.config.inhibition_gain
                * competition
            )
            activity = torch.tanh(torch.relu(
                membrane - old.threshold - inhibition
            ))
            active_indicator = (activity > 1e-6).to(activity.dtype)
            # Homeostasis is an integrator over the input a region actually
            # receives, and it can only find a useful set point if that input is
            # varied.  Replay is not: one symbol is driven for sixteen ticks with
            # no waking traffic to balance it, so a unit the engram drives gains
            # ``rate * (1 - target)`` every tick while a silent one sheds only
            # ``rate * target`` -- a 7:1 ratchet on exactly the units carrying
            # the memory.  Measured, that pushed the peak set point to 0.43,
            # twenty-one times ``threshold_base``, and since ``activity``
            # subtracts the threshold directly from the drive it collapsed the
            # write basis to 1/22 of the basis the probe reproduces.  ``local_update``
            # is linear in |trace|, so the write all but vanished, ``captured``
            # became arbitrary on a near-null trace, and one decoder row churned
            # through 118 rewires without ever terminating.
            #
            # Freezing the set point during replay -- rather than resetting it --
            # keeps whatever waking adaptation learned, and only denies a
            # degenerate burst the right to overwrite it.  Measured against the
            # reset alternative it wins on every behavioural column (true-cell
            # movement 3/4 either way, but mean |delta| 0.0088 vs 0.0073 and a
            # larger margin on three of four pairs) while leaving waking
            # homeostasis untouched.  Biological homeostatic plasticity is
            # hours-scale and population-driven for the same reason.
            if adapt_homeostasis:
                threshold = torch.clamp(
                    old.threshold
                    + self.config.homeostasis_rate
                    * (active_indicator - self.config.target_activity),
                    min=self.config.threshold_min,
                    max=self.config.threshold_max,
                )
            else:
                threshold = old.threshold
            trace = bound_norm(
                self.config.trace_decay * old.trace
                + (1.0 - self.config.trace_decay) * activity,
                self.config.max_trace_norm,
            )
            state_error = activity - recurrent_prediction

            if learn:
                # Structure moves before weights so a contact opened this tick
                # receives its first error × eligibility write immediately,
                # rather than idling at zero until the pattern recurs.  A swap
                # is discrete and cannot be scaled by learn_scale the way a
                # weight change can; the error gate is what limits it instead,
                # so only a row that is genuinely mispredicting rewires.
                #
                # Rewiring is confined to consolidation, and that boundary is
                # empirical, not conservative.  The silent-partner gate keeps a
                # row stable while one pattern is held, but waking input moves
                # on every tick: a partner recruited for this symbol falls
                # silent on the next and is retired again, so the support
                # churns.  Enabling it during a free run cost the learned byte
                # cycle its exactness after 66 steps.  The lottery it repairs is
                # a replay-time pathology to begin with -- waking learning has
                # thousands of writes and already reaches the cycle -- and
                # sleep-gated spine turnover is how brains do it.
                if restructure:
                    self.structural_events += decoder.structural_update(
                        lower_error,
                        old.trace,
                        turnover_ratio=self.config.structural_turnover_ratio,
                        capture_target=self.config.structural_capture_target,
                        error_threshold=self.config.structural_error_threshold,
                    )
                    self.structural_events += transition.structural_update(
                        state_error,
                        old.trace,
                        turnover_ratio=self.config.structural_turnover_ratio,
                        capture_target=self.config.structural_capture_target,
                        error_threshold=self.config.structural_error_threshold,
                    )
                # Decay belongs to the same plasticity event as potentiation, so it
                # must pass through the identical gate.  Ungated decay would make a
                # weakly gated replay a net forgetting operation.
                decay = self.config.synapse_decay * learn_scale
                decoder.local_update(
                    lower_error,
                    old.trace,
                    learning_rate=self.config.predictive_learning_rate
                    * learn_scale,
                    weight_decay=decay,
                )
                if consolidation_learn_scale > 0.0:
                    consolidation_error = (
                        lower_activity - consolidated_prediction
                    )
                    self.consolidation_decoders[index].local_update(
                        consolidation_error,
                        signed_trace,
                        learning_rate=self.config.predictive_learning_rate
                        * consolidation_learn_scale,
                        weight_decay=decay,
                    )
                transition.local_update(
                    state_error,
                    old.trace,
                    learning_rate=self.config.transition_learning_rate
                    * learn_scale,
                    weight_decay=decay,
                )
                # The competition is calibrated on the activity it produced, so
                # the baseline is this tick's own mean rate squared: that is the
                # co-activation two independent units of this rate would show,
                # and subtracting it is what makes the rule sensitive to
                # *correlation* rather than to overall firing.  Without it every
                # contact in an active region would grow and the bank would
                # collapse back into a global gain -- the very thing it
                # replaces.  The zero-mean form mirrors the replay fatigue term.
                mean_rate = float(activity.mean().item())
                lateral.anti_hebbian_update(
                    activity,
                    learning_rate=self.config.lateral_learning_rate * learn_scale,
                    baseline=mean_rate * mean_rate,
                )

            # The opponent origin is learned only from varied waking traffic.
            # Evaluation/generation and replay read the established statistic;
            # a repeated sleep burst cannot drag its own zero point toward the
            # pattern it is trying to consolidate.
            if learn and adapt_homeostasis:
                self.trace_baselines[index].lerp_(
                    trace, float(self.config.cortical_baseline_rate)
                )

            next_states.append(RegionState(
                membrane=membrane,
                activity=activity,
                trace=trace,
                prediction=lower_prediction,
                error=lower_error,
                threshold=threshold,
                inhibition=inhibition,
            ))
            activity_rates.append(float(active_indicator.mean().item()))
            error_norms.append(float(lower_error.norm().item()))
            lower_activity = activity
            feedback_offset += region_size

        return tuple(next_states), tuple(activity_rates), tuple(error_norms)

    def clear_dynamics(
        self, regions: Sequence[RegionState]
    ) -> Tuple[RegionState, ...]:
        """Silence activity while keeping every homeostatic set point.

        Unlike ``initial_state`` this preserves each region's adapted threshold and
        so can be used to segment one continuous stream into independent episodes
        without leaking a fresh set point back into waking behaviour.
        """

        if len(regions) != len(self.config.region_sizes):
            raise ValueError("region state count does not match the architecture")
        return tuple(
            RegionState(
                membrane=torch.zeros_like(region.membrane),
                activity=torch.zeros_like(region.activity),
                trace=torch.zeros_like(region.trace),
                prediction=torch.zeros_like(region.prediction),
                error=torch.zeros_like(region.error),
                threshold=region.threshold.detach().clone(),
                inhibition=torch.zeros_like(region.inhibition),
            )
            for region in regions
        )

    def cortical_context(self, regions: Sequence[RegionState]) -> torch.Tensor:
        """Expose time-separated fast activity and slow trace to an organ."""

        return torch.cat([
            *(region.activity for region in regions),
            *(region.trace for region in regions),
        ], dim=0)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "decoders": [decoder.to_payload() for decoder in self.decoders],
            "consolidation_decoders": [
                decoder.to_payload() for decoder in self.consolidation_decoders
            ],
            "trace_baselines": [
                baseline.detach().cpu().clone()
                for baseline in self.trace_baselines
            ],
            "transitions": [transition.to_payload() for transition in self.transitions],
            "laterals": [lateral.to_payload() for lateral in self.laterals],
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if len(payload["decoders"]) != len(self.decoders):
            raise ValueError("decoder count does not match architecture")
        if len(payload["consolidation_decoders"]) != len(
            self.consolidation_decoders
        ):
            raise ValueError(
                "consolidation decoder count does not match architecture"
            )
        if len(payload["trace_baselines"]) != len(self.trace_baselines):
            raise ValueError("trace baseline count does not match architecture")
        if len(payload["transitions"]) != len(self.transitions):
            raise ValueError("transition count does not match architecture")
        if len(payload["laterals"]) != len(self.laterals):
            raise ValueError("lateral count does not match architecture")
        for synapses, state in zip(self.decoders, payload["decoders"]):
            synapses.load_payload(state)
        for synapses, state in zip(
            self.consolidation_decoders, payload["consolidation_decoders"]
        ):
            synapses.load_payload(state)
        for target, stored in zip(
            self.trace_baselines, payload["trace_baselines"]
        ):
            baseline = stored.detach().to(self.device, dtype=torch.float32)
            if baseline.shape != target.shape:
                raise ValueError(
                    "trace baseline shape does not match architecture"
                )
            if not bool(torch.isfinite(baseline).all()):
                raise ValueError("trace baseline contains a non-finite value")
            target.copy_(baseline)
        for synapses, state in zip(self.transitions, payload["transitions"]):
            synapses.load_payload(state)
        for synapses, state in zip(self.laterals, payload["laterals"]):
            synapses.load_payload(state)

    def parameter_tensors(self) -> Tuple[torch.Tensor, ...]:
        return tuple(
            synapses.edge_weight
            for synapses in (
                *self.decoders,
                *self.consolidation_decoders,
                *self.transitions,
                *self.laterals,
            )
        )

    def active_edge_count(self) -> int:
        return sum(
            synapses.edge_count
            for synapses in (
                *self.decoders,
                *self.consolidation_decoders,
                *self.transitions,
                *self.laterals,
            )
        )

    def dense_equivalent_edge_count(self) -> int:
        return sum(
            synapses.dense_equivalent_count
            for synapses in (
                *self.decoders,
                *self.consolidation_decoders,
                *self.transitions,
                *self.laterals,
            )
        )
