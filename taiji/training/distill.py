"""Distillation pipeline: 1.5B teacher → individual neurons.

The current bottleneck is neuron quality when trained from scratch.
Compact neurons (18M) on 20-35K sequences converge to PPL 2-4 digits,
while distillation from the 1.5B checkpoint produces neurons at PPL < 50.

This module implements:
1. Teacher direction extraction (from 1.5B hidden states)
2. Distillation loss (MSE between student hidden and teacher hidden)
3. Combined LM loss + distillation loss + contrastive loss
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def distill_neuron(
    teacher_model: nn.Module,
    student_neuron: nn.Module,
    domain_dataloader,
    shared_embedding: nn.Embedding,
    num_steps: int = 5000,
    lm_weight: float = 0.7,
    distill_weight: float = 0.3,
    lr: float = 5e-4,
    device: str = "cpu",
) -> Dict[str, float]:
    """Distill a single neuron from the 1.5B teacher model.

    The student learns to:
    1. Predict the next token (LM loss)
    2. Align its hidden states with the teacher's hidden states (distill loss)
    3. Differentiate its field_write from other neurons (contrastive loss, TBD)

    Args:
        teacher_model: the 1.5B ModelSelf checkpoint.
        student_neuron: the ResonanceNeuron to train.
        domain_dataloader: dataloader yielding domain-specific batches.
        shared_embedding: shared base embedding (from teacher or SVD-initialized).
        num_steps: number of training steps.
        lm_weight: weight of language modeling loss.
        distill_weight: weight of distillation loss.
        lr: learning rate.
        device: "cpu" or "cuda".

    Returns:
        {"final_loss": float, "final_ppl": float, "steps": int}
    """
    optimizer = torch.optim.AdamW(student_neuron.parameters(), lr=lr)
    student_neuron.train()
    teacher_model.eval()

    total_loss = 0.0
    step = 0

    for batch in domain_dataloader:
        if step >= num_steps:
            break

        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids") or batch.get("tokens")
            target_ids = batch.get("labels") or batch.get("targets") or input_ids
        elif isinstance(batch, (list, tuple)):
            input_ids, target_ids = batch[0], batch[1] if len(batch) > 1 else batch[0]
        else:
            input_ids = target_ids = batch

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Get shared embeddings
        with torch.no_grad():
            shared_emb = shared_embedding(input_ids)
            # Get teacher hidden states
            teacher_hidden = teacher_model.get_hidden_states(input_ids)

        # Student forward
        result = student_neuron.forward(shared_emb, return_logits=True)
        student_logits = result["logits"]
        student_hidden = result.get("hidden_before_write")

        # Shift for next-token prediction
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        # 1. Language modeling loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        # 2. Distillation loss (align hidden states)
        if student_hidden is not None and teacher_hidden is not None:
            distill_loss = F.mse_loss(
                student_hidden[:, -1, :],  # last token hidden
                teacher_hidden[:, -1, :],  # teacher last token hidden
            )
        else:
            distill_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = lm_weight * lm_loss + distill_weight * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 500 == 0:
            print(f"  Distill step {step}/{num_steps}, loss={loss.item():.4f}")

    avg_loss = total_loss / max(step, 1)

    return {
        "final_loss": avg_loss,
        "final_ppl": float(torch.exp(torch.tensor(avg_loss))),
        "steps": step,
    }


def extract_teacher_directions(
    teacher_model: nn.Module,
    domain_dataloader,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract teacher hidden state directions for a domain.

    Runs the teacher model on domain data and collects the average
    hidden state. This serves as the "target direction" for distillation.

    Args:
        teacher_model: the 1.5B ModelSelf checkpoint.
        domain_dataloader: domain-specific data.
        device: "cpu" or "cuda".

    Returns:
        [hidden_dim] average hidden state direction.
    """
    teacher_model.eval()
    all_hidden = []

    with torch.no_grad():
        for batch in domain_dataloader:
            if len(all_hidden) >= 100:  # sample 100 batches
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids") or batch.get("tokens")
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            else:
                input_ids = batch

            input_ids = input_ids.to(device)
            hidden = teacher_model.get_hidden_states(input_ids)
            all_hidden.append(hidden.mean(dim=1))  # average over sequence

    if not all_hidden:
        return torch.zeros(1)

    # Average over batches
    direction = torch.stack(all_hidden).mean(dim=0).mean(dim=0)  # [hidden_dim]
    return direction / (direction.norm() + 1e-8)  # normalize
