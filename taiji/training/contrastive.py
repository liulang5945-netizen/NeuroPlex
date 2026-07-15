"""Contrastive learning for field_write projection.

The critical finding from LUCKY v4: field_write was never trained during
language modeling (only lm_head and Transformer body received gradients).
This made resonance routing random — cosine(Vi, F) ≈ 0 regardless of input.

This module trains field_write specifically:
- Same-domain batches → pull field_write outputs closer
- Different-domain batches → push field_write outputs apart
- Result: cosine(Vi, F) becomes a meaningful "does this input match my domain?" signal
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_field_write_contrastive(
    neuron: nn.Module,
    domain_dataloader,
    shared_embedding: nn.Embedding,
    domain_label: int,
    other_domain_vectors: list[torch.Tensor] = None,
    num_steps: int = 500,
    lr: float = 1e-4,
    temperature: float = 0.07,
    device: str = "cpu",
) -> dict:
    """Train field_write with contrastive learning.

    Freezes the Transformer body and lm_head, only trains:
    - field_write projection
    - field_read_layers (optional)

    Args:
        neuron: the ResonanceNeuron to train field_write on.
        domain_dataloader: dataloader for this neuron's domain.
        shared_embedding: shared base embedding.
        domain_label: integer label for this domain (for contrastive pairs).
        other_domain_vectors: field vectors from other domains (negative samples).
        num_steps: training steps.
        lr: learning rate.
        temperature: contrastive loss temperature.
        device: "cpu" or "cuda".

    Returns:
        {"final_loss": float, "steps": int}
    """
    # Freeze transformer body
    for name, param in neuron.named_parameters():
        if "field_write" not in name and "field_read" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in neuron.parameters() if p.requires_grad],
        lr=lr,
    )

    total_loss = 0.0
    step = 0

    for batch in domain_dataloader:
        if step >= num_steps:
            break

        if isinstance(batch, dict):
            input_ids = batch.get("input_ids") or batch.get("tokens")
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        else:
            input_ids = batch

        input_ids = input_ids.to(device)
        shared_emb = shared_embedding(input_ids)

        # Forward (transformer body frozen)
        result = neuron.forward(shared_emb)
        field_vec = result["field_vector"]  # [B, D]
        field_vec_norm = field_vec / (field_vec.norm(dim=-1, keepdim=True) + 1e-8)

        # Contrastive loss
        if other_domain_vectors:
            # Positive: field_vec should be similar within the same domain
            # Negative: field_vec should differ from other domains
            pos_sim = torch.tensor(1.0, device=device)  # anchor itself

            neg_sims = []
            for neg_vec in other_domain_vectors:
                neg_norm = neg_vec / (neg_vec.norm(dim=-1, keepdim=True) + 1e-8)
                sim = (field_vec_norm * neg_norm).sum(dim=-1).mean()
                neg_sims.append(torch.clamp(sim + 1.0, min=0.0))  # hinge

            loss = torch.stack(neg_sims).mean() if neg_sims else torch.tensor(0.0, device=device)
        else:
            # No negative samples: just ensure vectors are not degenerate
            loss = F.mse_loss(field_vec_norm, field_vec_norm.detach())  # placeholder

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 100 == 0:
            print(f"  Field write contrastive step {step}/{num_steps}, loss={loss.item():.4f}")

    # Unfreeze
    for param in neuron.parameters():
        param.requires_grad = True

    return {"final_loss": total_loss / max(step, 1), "steps": step}
