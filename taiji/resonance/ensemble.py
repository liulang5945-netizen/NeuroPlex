"""Resonance ensemble — multi-round collaborative inference with gating.

The ensemble orchestrates multiple ResonanceNeurons through the
ResonanceField over 3-5 rounds of collaborative inference.

Each round:
1. All active neurons run forward (first round: independently;
   subsequent rounds: conditioned on field state)
2. Each neuron writes its L2-normalised field vector
3. Resonance scores are computed (cosine similarity with field state)
4. Low-resonance neurons are filtered out via dynamic thresholding

Experiment 12 revealed three critical mechanisms (now integrated):
- ConfidenceGate: skip resonance when prediction is already confident
- EarlyStopResonance: stop iterating when logits converge
- ResonanceTrigger: full trigger conditions for activation
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .field import ResonanceField
from .neuron import ResonanceNeuron
from .gating import ConfidenceGate, EarlyStopResonance, ResonanceTrigger
from .quality import QualityFilter
from .division import DivisionPath


class ResonanceEnsemble:
    """Orchestrates multi-round resonance inference across multiple neurons.

    v2: Integrated three gating mechanisms from Experiment 12:
    - ConfidenceGate: skip resonance for confident predictions
    - EarlyStopResonance: stop when logits converge
    - ResonanceTrigger: combined trigger conditions

    Usage:
        # Without gating (backward compatible):
        ensemble = ResonanceEnsemble(neurons, field)
        result = ensemble.forward(shared_embeddings)

        # With gating:
        ensemble = ResonanceEnsemble(
            neurons, field,
            confidence_gate=ConfidenceGate(threshold=0.9),
            early_stop=EarlyStopResonance(),
        )
        result = ensemble.forward(shared_embeddings, return_logits=True)
    """

    def __init__(
        self,
        neurons: Dict[str, ResonanceNeuron],
        field: ResonanceField,
        max_rounds: int = 3,
        diversity_lambda: float = 0.01,
        confidence_gate: Optional[ConfidenceGate] = None,
        early_stop: Optional[EarlyStopResonance] = None,
        resonance_trigger: Optional[ResonanceTrigger] = None,
        quality_filter: Optional[QualityFilter] = None,
        division_path: Optional[DivisionPath] = None,
    ):
        self.neurons = neurons
        self.field = field
        self.max_rounds = max_rounds
        self.diversity_lambda = diversity_lambda

        # ── Gating mechanisms (Experiment 12) ──
        self.confidence_gate = confidence_gate
        self.early_stop = early_stop
        self.resonance_trigger = resonance_trigger

        # ── Quality filtering (Experiment 9) ──
        self.quality_filter = quality_filter

        # ── Division-of-labor path ──
        self.division_path = division_path

        # Tracking
        self.round_scores: List[Dict[str, float]] = []
        self.n_active_history: List[int] = []

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        return_logits: bool = False,
        active_filter: bool = True,
        enable_gating: bool = True,
    ) -> Dict:
        """Run the full resonance loop with optional gating.

        Args:
            shared_embeddings: [B, L, base_embed_dim]
            return_logits: if True, each neuron also returns token logits
            active_filter: if True, filter out low-resonance neurons each round
            enable_gating: if False, skip all gating (backward compatible mode)

        Returns:
            dict with:
            - field_state: final field state vector
            - weighted_logits: resonance-weighted average logits (if return_logits)
            - final_scores: per-neuron resonance scores (final round)
            - n_rounds: actual number of rounds completed
            - skipped_resonance: True if gating skipped the resonance loop
            - skip_reason: explanation if resonance was skipped
        """
        self.field.reset()
        self.round_scores = []
        self.n_active_history = []

        neuron_ids = list(self.neurons.keys())
        active_ids = set(neuron_ids)

        # ── Quality filter: exclude weak neurons before resonance ──
        if self.quality_filter is not None:
            filtered_ids = self.quality_filter.filter(list(active_ids))
            if not filtered_ids:
                raise ValueError("All neurons filtered out by quality filter")
            active_ids = set(filtered_ids)

        vectors: Dict[str, torch.Tensor] = {}
        all_logits: Dict[str, torch.Tensor] = {}
        logits_history: List[torch.Tensor] = []

        # ── Round 1: all neurons run independently ──
        round_vecs: Dict[str, torch.Tensor] = {}
        round_logits: Dict[str, torch.Tensor] = {}

        for nid in active_ids:
            neuron = self.neurons[nid]
            result = neuron.forward(
                shared_embeddings,
                field_state=None,
                round_num=1,
                return_logits=return_logits,
            )
            round_vecs[nid] = result["field_vector"]
            if return_logits:
                round_logits[nid] = result["logits"]

        # Write round 1 to field
        for nid in active_ids:
            self.field.write(nid, round_vecs[nid])

        # Compute round 1 scores
        scores: Dict[str, float] = {}
        for nid in active_ids:
            scores[nid] = self.field.score(round_vecs[nid])
        self.round_scores.append(scores)

        # ── Gating check: should we resonate? ──
        if enable_gating and return_logits:
            if self.resonance_trigger is not None:
                best_nid = max(scores, key=scores.get)
                should_res, reason = self.resonance_trigger.should_resonate(
                    round_logits[best_nid], round_vecs
                )
                if not should_res:
                    self.n_active_history.append(len(active_ids))
                    return {
                        "field_state": self.field.get_state(),
                        "weighted_logits": round_logits[best_nid],
                        "final_scores": scores,
                        "n_rounds": 1,
                        "n_active_history": self.n_active_history,
                        "skipped_resonance": True,
                        "skip_reason": reason,
                    }
            elif self.confidence_gate is not None:
                best_nid = max(scores, key=scores.get)
                if not self.confidence_gate.should_resonate(round_logits[best_nid]):
                    confidence = self.confidence_gate.get_confidence(round_logits[best_nid])
                    self.n_active_history.append(len(active_ids))
                    return {
                        "field_state": self.field.get_state(),
                        "weighted_logits": round_logits[best_nid],
                        "final_scores": scores,
                        "n_rounds": 1,
                        "n_active_history": self.n_active_history,
                        "skipped_resonance": True,
                        "skip_reason": f"confident prediction (max_prob={confidence:.3f})",
                    }

        # Track round 1 for early stop
        if return_logits and round_logits:
            logits_history.append(self._average_logits(round_logits))
        vectors = round_vecs
        all_logits = round_logits

        # ── Rounds 2+: conditioned resonance ──
        for round_num in range(2, self.max_rounds + 1):
            round_vecs = {}
            round_logits = {}

            for nid in active_ids:
                neuron = self.neurons[nid]
                result = neuron.forward(
                    shared_embeddings,
                    field_state=self.field.get_normalised_state(),
                    round_num=round_num,
                    return_logits=return_logits,
                )
                round_vecs[nid] = result["field_vector"]
                if return_logits:
                    round_logits[nid] = result["logits"]

            for nid in active_ids:
                self.field.write(nid, round_vecs[nid])

            scores = {}
            for nid in active_ids:
                scores[nid] = self.field.score(round_vecs[nid])
            self.round_scores.append(scores)

            if active_filter and len(active_ids) > 1:
                active_vecs = [round_vecs[nid] for nid in active_ids]
                filtered = set()
                for nid in active_ids:
                    congestion = self.field.directional_congestion(round_vecs[nid], active_vecs)
                    threshold = self.field.compute_threshold(congestion)
                    if scores[nid] >= threshold:
                        filtered.add(nid)
                if not filtered:
                    best = max(active_ids, key=lambda nid: scores[nid])
                    filtered.add(best)
                self.field.scores = scores
                if len(filtered) <= 1 and round_num >= 2:
                    active_ids = filtered
                    vectors = round_vecs
                    all_logits = round_logits
                    break
                active_ids = filtered

            self.n_active_history.append(len(active_ids))
            vectors = round_vecs
            all_logits = round_logits

            # ── Early stop check ──
            if enable_gating and self.early_stop is not None and return_logits:
                logits_history.append(self._average_logits(all_logits))
                if self.early_stop.should_stop(logits_history):
                    break

        # ── Final output ──
        result = {
            "field_state": self.field.get_state(),
            "final_scores": self.round_scores[-1] if self.round_scores else {},
            "n_rounds": len(self.round_scores),
            "n_active_history": self.n_active_history,
            "skipped_resonance": False,
            "skip_reason": None,
        }

        if return_logits and all_logits:
            final_scores = self.round_scores[-1] if self.round_scores else scores

            if self.division_path is not None:
                # Use division-of-labor weighting (scale layering + cluster dominance)
                # Build clusters from neuron domains (simple: one cluster per neuron for now)
                clusters = {"default": {nid: vectors.get(nid, torch.zeros(1)) for nid in all_logits}}
                neuron_specs = {
                    nid: getattr(self.neurons[nid].config, "spec", "compact")
                    for nid in all_logits
                }
                # Use the first neuron's field vector as input_vector proxy
                input_vec = next(iter(vectors.values()), torch.zeros(1))
                final_weights_dict = self.division_path.compute_final_weights(
                    input_vector=input_vec.squeeze(),
                    clusters=clusters,
                    neuron_specs=neuron_specs,
                    resonance_scores=final_scores,
                )
                weight_list = [final_weights_dict.get(nid, 0.0) for nid in all_logits.keys()]
                weights = torch.tensor(weight_list, device=shared_embeddings.device)
            else:
                # Default: resonance-score softmax weighting
                score_tensor = torch.tensor(
                    [final_scores.get(nid, 0.0) for nid in all_logits.keys()],
                    device=shared_embeddings.device,
                )
                weights = F.softmax(score_tensor * 2.0, dim=0)

            weighted_logits = None
            for i, (nid, logits) in enumerate(all_logits.items()):
                w = weights[i]
                if weighted_logits is None:
                    weighted_logits = w * logits
                else:
                    weighted_logits = weighted_logits + w * logits
            result["weighted_logits"] = weighted_logits
            result["final_weights"] = {
                nid: float(weights[i]) for i, nid in enumerate(all_logits.keys())
            }

        return result

    def _average_logits(
        self, logits_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute simple average of logits across neurons for early stop."""
        if not logits_dict:
            return torch.zeros(1)
        stacked = torch.stack(list(logits_dict.values()))
        return stacked.mean(dim=0)

    def evaluate_ppl(
        self,
        dataloader,
        shared_embedding: nn.Embedding,
        tokenizer=None,
        max_batches: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate perplexity over a dataloader using the resonance ensemble.

        Uses teacher forcing: feeds the full sequence, gets predictions
        at all positions, computes cross-entropy loss.

        Args:
            dataloader: yields batches of token_ids [B, L]
            shared_embedding: the shared base embedding (Level 0)
            tokenizer: optional tokenizer for decoding (debug only)
            max_batches: maximum number of batches to evaluate
            verbose: print progress

        Returns:
            dict with 'ppl', 'loss', 'n_tokens'
        """
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids") or batch.get("tokens")
                target_ids = batch.get("labels") or batch.get("targets")
                if target_ids is None:
                    target_ids = input_ids
            elif isinstance(batch, torch.Tensor):
                input_ids = batch
                target_ids = batch
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
                target_ids = batch[1] if len(batch) > 1 else batch[0]
            else:
                continue

            if input_ids is None or input_ids.numel() == 0:
                continue

            # Get shared embeddings
            with torch.no_grad():
                shared_emb = shared_embedding(input_ids)  # [B, L, base_dim]

                # Run ensemble with logits
                result = self.forward(shared_emb, return_logits=True)

                if "weighted_logits" not in result:
                    continue

                logits = result["weighted_logits"]  # [B, L, vocab]

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = target_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

                total_loss += loss.item() * shift_targets.numel()
                total_tokens += shift_targets.numel()

            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_loss / max(total_tokens, 1))
                print(f"  Batch {batch_idx + 1}/{max_batches}, PPL: {current_ppl:.2f}")

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)

        return {"ppl": ppl, "loss": avg_loss, "n_tokens": total_tokens}

    @staticmethod
    def evaluate_single_neuron(
        neuron: ResonanceNeuron,
        dataloader,
        shared_embedding: nn.Embedding,
        max_batches: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate PPL for a single neuron (baseline comparison)."""
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids") or batch.get("tokens")
                target_ids = batch.get("labels") or batch.get("targets")
                if target_ids is None:
                    target_ids = input_ids
            elif isinstance(batch, torch.Tensor):
                input_ids = batch
                target_ids = batch
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
                target_ids = batch[1] if len(batch) > 1 else batch[0]
            else:
                continue

            if input_ids is None or input_ids.numel() == 0:
                continue

            with torch.no_grad():
                shared_emb = shared_embedding(input_ids)
                result = neuron.forward(shared_emb, return_logits=True)
                logits = result["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = target_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

                total_loss += loss.item() * shift_targets.numel()
                total_tokens += shift_targets.numel()

            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_loss / max(total_tokens, 1))
                print(f"  Batch {batch_idx + 1}/{max_batches}, PPL: {current_ppl:.2f}")

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)

        return {"ppl": ppl, "loss": avg_loss, "n_tokens": total_tokens}
