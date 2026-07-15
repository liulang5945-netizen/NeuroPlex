"""Joint training loop — multi-neuron synchronized training.

The core mechanism for training multiple neurons simultaneously,
with the resonance field acting as a shared memory for knowledge
transfer between neurons. Each neuron trains its own LM loss,
contrastive loss for field_write differentiation, and niche-seeking
loss for preventing homogenization.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointTrainingLoop:
    """Multi-neuron synchronized training with field as shared memory.

    Usage:
        loop = JointTrainingLoop(neurons, field, shared_embedding)
        loop.train_round(batch)
    """

    def __init__(
        self,
        neurons: Dict[str, nn.Module],
        field,
        shared_embedding: nn.Embedding,
        contrastive_weight: float = 0.1,
        niche_weight: float = 0.01,
        niche_start_progress: float = 0.3,
    ):
        self.neurons = neurons
        self.field = field
        self.shared_embedding = shared_embedding
        self.contrastive_weight = contrastive_weight
        self.niche_weight = niche_weight
        self.niche_start_progress = niche_start_progress
        self.progress = 0.0  # 0.0 to 1.0

    def train_round(self, batch, optimizer, domain_labels: Dict[str, int] = None):
        """One training round for all neurons.

        Args:
            batch: training batch (input_ids + target_ids).
            optimizer: shared or per-neuron optimizer.
            domain_labels: {neuron_id: domain_label} for contrastive loss.
        """
        # Handle batch format
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids") or batch.get("tokens")
            target_ids = batch.get("labels") or batch.get("targets") or input_ids
        elif isinstance(batch, (list, tuple)):
            input_ids, target_ids = batch[0], batch[1] if len(batch) > 1 else batch[0]
        else:
            input_ids = target_ids = batch

        # Shared embedding (all neurons reuse)
        shared_emb = self.shared_embedding(input_ids)

        # All neurons forward
        outputs = {}
        for nid, neuron in self.neurons.items():
            outputs[nid] = neuron.forward(shared_emb, return_logits=True)

        # Write to resonance field
        self.field.reset()
        for nid, out in outputs.items():
            self.field.write(nid, out["field_vector"])

        # Compute per-neuron loss
        total_loss = torch.tensor(0.0, device=input_ids.device)
        for nid, out in outputs.items():
            # Language modeling loss
            shift_logits = out["logits"][:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            neuron_loss = lm_loss

            # Contrastive loss (field_write differentiation, LUCKY v4 critical)
            if domain_labels and nid in domain_labels:
                field_vec = out["field_vector"]
                other_vecs = [
                    outputs[other]["field_vector"]
                    for other in outputs if other != nid
                ]
                if other_vecs:
                    contrastive_loss = self._contrastive_loss(
                        field_vec, domain_labels[nid], other_vecs
                    )
                    neuron_loss = neuron_loss + self.contrastive_weight * contrastive_loss

            # Niche-seeking loss (only after 30% progress, prevent early homogenization)
            if self.progress > self.niche_start_progress:
                niche_loss = self._niche_loss(
                    out["field_vector"],
                    [outputs[other]["field_vector"] for other in outputs if other != nid],
                )
                neuron_loss = neuron_loss + self.niche_weight * niche_loss

            total_loss = total_loss + neuron_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return float(total_loss.item())

    def _contrastive_loss(
        self,
        anchor: torch.Tensor,
        anchor_label: int,
        others: list[torch.Tensor],
    ) -> torch.Tensor:
        """Contrastive loss: pull same-domain, push different-domain.

        This is the critical mechanism for training field_write.
        Without it, field_write remains random and resonance routing fails.
        """
        if not others:
            return torch.tensor(0.0)

        # Placeholder: assumes all others are different domain
        anchor_norm = anchor / (anchor.norm(dim=-1, keepdim=True) + 1e-8)
        loss = torch.tensor(0.0)

        for other in others:
            other_norm = other / (other.norm(dim=-1, keepdim=True) + 1e-8)
            # Push away dissimilar neurons
            sim = (anchor_norm * other_norm).sum(dim=-1).mean()
            loss = loss + torch.clamp(sim + 1.0, min=0.0)  # hinge: sim should be -1

        return loss / len(others)

    def _niche_loss(
        self,
        current_vec: torch.Tensor,
        other_vecs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Niche-seeking loss: encourage neuron to occupy an empty direction.

        Penalizes high cosine similarity with other neurons' field vectors.
        """
        if not other_vecs:
            return torch.tensor(0.0)

        current_norm = current_vec / (current_vec.norm(dim=-1, keepdim=True) + 1e-8)
        similarities = []
        for other in other_vecs:
            other_norm = other / (other.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (current_norm * other_norm).sum(dim=-1).mean()
            similarities.append(sim)

        # Average similarity — we want this to be low (close to 0 or negative)
        avg_sim = torch.stack(similarities).mean()
        return avg_sim  # minimize similarity with other neurons
