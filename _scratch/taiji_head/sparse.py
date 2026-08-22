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
        self.excludes_self = bool(
            not allow_self
            and self.out_features == self.in_features
            and self.in_features > 1
        )

        selected_by_post = []
        for post in range(self.out_features):
            candidates = torch.arange(self.in_features)
            if self.excludes_self:
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
    def anti_hebbian_update(
        self,
        postsynaptic_activity: torch.Tensor,
        *,
        learning_rate: float,
        baseline: float,
    ) -> None:
        """Decorrelate a recurrent bank from its own co-activation statistics.

        Unlike ``local_update`` this carries no error signal: the target is not
        a value to predict but a statistic to remove.  ``a_i * a_j - baseline``
        is positive exactly for pairs that fire together more often than two
        independent units of the same mean rate would, so those contacts grow
        and mutually suppress; pairs below the baseline relax toward silence.
        Weights are clamped non-negative because an inhibitory contact that
        turned negative would become excitatory and invert the competition.
        """

        if postsynaptic_activity.shape != (self.out_features,):
            raise ValueError("postsynaptic activity dimension mismatch")
        if self.out_features != self.in_features:
            raise ValueError("anti-Hebbian competition requires a recurrent bank")
        activity = postsynaptic_activity.to(self.device)
        self.edge_weight.add_(
            float(learning_rate)
            * (
                activity.unsqueeze(1) * activity[self.pre_index]
                - float(baseline)
            )
        )
        self.edge_weight.clamp_(min=0.0)
        self._bound_rows()

    @torch.no_grad()
    def structural_update(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        turnover_ratio: float,
        capture_target: float,
        error_threshold: float,
    ) -> int:
        """Regrow weightless edges onto active partners; return contacts moved.

        A fixed fan-in drawn once at initialization is a lottery. Region codes
        are only nominally dense -- a settled trace lights 21-34 of 64 units
        but concentrates 60-98% of its energy in two or three of them -- so
        whether a row's 16 sampled contacts happen to touch the informative
        units decides, by geometry alone, whether that row can be taught at
        all. Measured across four contingencies the energy a row captured
        spanned three orders of magnitude, and the worst row needed millions
        of updates to move. That is not a dose problem, and no rescaling of
        the code fixes it: divisive normalization leaves the captured fraction
        bit-identical, because it changes the scale and not the shape.

        So the topology itself has to move. A row retires its presynaptically
        silent contacts, weakest first, and grows that many replacements onto
        the most active units it is not already touching. Silence is the whole
        admission test: a partner that is not firing on the pattern being
        written contributes exactly zero to this update, whatever its weight
        says. Weight enters only as the order of retirement, which is the right
        place for it -- a silent edge may still be carrying some *other*
        pattern, and going weakest-first spends the row's least committed
        contacts before its most committed ones. It cannot be an admission test
        as well, because a silent edge is often a strong one: measured on the
        four outcome rows, 5 to 11 of 16 contacts were silent, yet demanding
        they also fall under 5% of the row maximum admitted only 3, 1, 0 and 0
        of them, which stalled two rows outright.

        ``turnover_ratio`` caps how much of a row may move in any single update.
        Rewiring the 4 weakest of 16 contacts raised captured energy from 0.029
        to 0.971 on one row and from 0.0001 to 0.781 on the worst; pushing to 8
        sent a fourth row *backwards*, 0.962 down to 0.625, because by then
        retirement had reached contacts that were doing real work under other
        patterns. The ceiling is therefore a measured turnover rate rather than
        a safety margin -- but it is a rate, and a rate alone cannot terminate
        anything, which is what ``capture_target`` is for.

        The remaining gate is that the row must currently be mispredicting, and
        it is the gate that supplies selectivity. Rewiring on its own only
        supplies capacity -- pointing every row at the same active units raises
        what all of them capture, which discriminates nothing. But at a
        consolidation write the error is nearly one-hot: the row for the outcome
        actually being replayed carries an error near one while the rest of the
        alphabet sits two orders of magnitude below it. Thresholding on error
        therefore restructures exactly the row that needs the capacity and
        leaves the others alone, which is why error is used as a gate and never
        as a way to rank candidates -- within a row it is a single scalar and
        could not rank anything.

        Requiring the retired partner to be silent keeps a single held pattern
        from thrashing. A new contact opens at zero weight, so it is instantly
        among the weakest edges in its row, and a purely magnitude-based
        criterion would retire it again on the very next tick and never let it
        accumulate anything. A fresh donor is by construction one of the most
        active absent units, so silence excludes it from the retirement pool for
        exactly as long as it stays active. That protection lasts only while the
        pattern is held, though, which is why it does not by itself make the rule
        stable across a whole bout and why ``capture_target`` is the gate that
        does.

        Opening at zero weight also means the swap is behaviourally neutral at
        the instant it happens -- every edge it replaces contributed nothing to
        this pattern, being silent on it -- and the ordinary error x eligibility
        update grows it from there. Row fan-in never changes, so every stored
        shape is invariant and the payload format is untouched.
        """

        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError("postsynaptic error dimension mismatch")
        if presynaptic_trace.shape != (self.in_features,):
            raise ValueError("presynaptic trace dimension mismatch")
        if self.row_fan_in >= self.in_features:
            return 0

        postsynaptic_error = postsynaptic_error.to(self.device)
        presynaptic_trace = presynaptic_trace.to(self.device)
        activity = presynaptic_trace.abs()
        contacts = self.pre_index.long()

        # How much of the pattern's energy the row already reaches is its own
        # stopping signal, and without it the rule destroys what it repairs.
        # Silence cannot terminate anything: a settled trace occupies 21-34 of
        # 64 units, so every row keeps some silent contacts forever and keeps
        # trading them away forever.  Measured over a 96-cycle bout the swaps
        # never saturated -- 4, 16, 45, 83, 128 -- and rows ended with 3 of 16
        # original contacts and 2% of their accumulated weight, because each
        # donor opens at zero and the next bout retires the one before it had
        # time to grow.  The bout therefore had two phases: for the first ~16
        # cycles aiming dominated and all four rows improved together, then
        # capture saturated near 1.0 with nothing left to gain and continued
        # rewiring simply cannibalized the weight the writes had just built,
        # collapsing one row's logit from 6.9e-3 to 5.1e-4.  A row that already
        # sees the pattern needs writes, not new wiring, so aiming stops when
        # it is aimed.  This is measured from the trace and the topology in
        # hand, so it costs no counter, no synapse age, and no stored state.
        energy = presynaptic_trace * presynaptic_trace
        total_energy = float(energy.sum())
        if total_energy <= 0.0:
            return 0
        captured = energy[contacts].sum(dim=1) / total_energy
        unaimed = captured < float(capture_target)

        magnitude = self.edge_weight.abs()
        retirable = activity[contacts] == 0.0
        mispredicting = postsynaptic_error.abs() >= float(error_threshold)
        budget = int(float(turnover_ratio) * self.row_fan_in)
        if budget <= 0:
            return 0
        demand = retirable.sum(dim=1).clamp(max=budget) * mispredicting * unaimed
        if not bool((demand > 0).any()):
            return 0

        # Candidate donors are the units the row does not touch, ranked by
        # activity.  Marking taken columns below zero sorts them behind every
        # absent unit, so the leading entries of each row's ordering are its
        # absent partners from most to least active.  A recurrent projection
        # built to exclude self-contacts must keep excluding them, so such rows
        # count their own column as taken.
        taken = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.bool,
            device=self.device,
        )
        taken.scatter_(1, contacts, True)
        if self.excludes_self:
            diagonal = torch.arange(self.out_features, device=self.device)
            taken[diagonal, diagonal] = True
        available = (~taken) & (activity > 0.0).unsqueeze(0)
        ranking = activity.expand(self.out_features, -1).masked_fill(taken, -1.0)
        donor_order = ranking.argsort(dim=1, descending=True)

        # Each row moves as many contacts as it has dead ones, limited by how
        # many active partners are actually left to grow onto.
        moves = torch.minimum(demand, available.sum(dim=1))
        if not bool((moves > 0).any()):
            return 0

        # Retire the feeblest silent contact first, and pair the j-th
        # retirement with the j-th best donor.  Both orderings are per row and
        # the first ``moves`` entries of each are valid by construction, so one
        # mask selects the whole set of swaps at once.
        retire_order = magnitude.masked_fill(~retirable, float("inf")).argsort(dim=1)
        width = self.row_fan_in
        chosen = torch.arange(width, device=self.device).unsqueeze(0) < moves.unsqueeze(1)
        rows = torch.arange(self.out_features, device=self.device).unsqueeze(1).expand(-1, width)
        self.pre_index[rows[chosen], retire_order[chosen]] = (
            donor_order[:, :width][chosen].to(torch.int32)
        )
        self.edge_weight[rows[chosen], retire_order[chosen]] = 0.0
        return int(chosen.sum().item())

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
        """Restore weights and topology, validating the invariants that matter.

        Topology is learned, not derived, so a checkpoint's `pre_index` will not
        equal the one this instance drew from its seed.  Demanding equality
        would make every structurally plastic run unloadable.  What must still
        hold are the properties the kernel's correctness actually rests on: the
        stored shape, indices inside the presynaptic population, no duplicate
        contact within a row -- `backproject` scatter-adds over `pre_index`, so
        a repeated index would silently double-count that partner's error -- and
        no self-contact on a projection that was built to exclude it.
        """

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
        if pre_index.shape != self.pre_index.shape:
            raise ValueError("synapse presynaptic topology does not match architecture")
        if edge_weight.shape != self.edge_weight.shape:
            raise ValueError("synapse edge weights do not match architecture")
        if bool(((pre_index < 0) | (pre_index >= self.in_features)).any()):
            raise ValueError("synapse presynaptic index outside the population")
        rows = pre_index.long()
        if bool((rows.sort(dim=1).values.diff(dim=1) == 0).any()):
            raise ValueError("synapse presynaptic topology repeats a contact")
        if self.excludes_self:
            own = torch.arange(self.out_features, device=self.device).unsqueeze(1)
            if bool((rows == own).any()):
                raise ValueError("recurrent synapse topology contains a self-contact")
        self.pre_index = pre_index.clone()
        self.edge_weight = edge_weight.clone()
        self._bound_rows()
