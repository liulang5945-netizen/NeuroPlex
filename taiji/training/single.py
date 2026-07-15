"""Single neuron training — train one neuron from scratch or continue.

For new domains where no existing neuron is close enough (TrainingScheduler
returns NewNeuronTask), this module trains a single ResonanceNeuron on
domain-specific data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_single_neuron(
    neuron: nn.Module,
    dataloader,
    shared_embedding: nn.Embedding,
    num_steps: int = 10000,
    lr: float = 5e-4,
    device: str = "cpu",
    log_every: int = 500,
) -> dict:
    """Train a single neuron on domain-specific data.

    Args:
        neuron: ResonanceNeuron to train.
        dataloader: domain-specific data.
        shared_embedding: shared base embedding.
        num_steps: training steps.
        lr: learning rate.
        device: "cpu" or "cuda".
        log_every: logging interval.

    Returns:
        {"final_loss": float, "final_ppl": float, "steps": int}
    """
    optimizer = torch.optim.AdamW(neuron.parameters(), lr=lr)
    neuron.train()

    total_loss = 0.0
    step = 0

    for batch in dataloader:
        if step >= num_steps:
            break

        if isinstance(batch, dict):
            input_ids = batch.get("input_ids") or batch.get("tokens")
            target_ids = batch.get("labels") or batch.get("targets") or input_ids
        elif isinstance(batch, (list, tuple)):
            input_ids, target_ids = batch[0], batch[1] if len(batch) > 1 else batch[0]
        else:
            input_ids = target_ids = batch

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        shared_emb = shared_embedding(input_ids)
        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % log_every == 0:
            current_ppl = float(torch.exp(torch.tensor(total_loss / step)))
            print(f"  Train step {step}/{num_steps}, loss={loss.item():.4f}, PPL={current_ppl:.2f}")

    avg_loss = total_loss / max(step, 1)
    ppl = float(torch.exp(torch.tensor(avg_loss)))

    return {"final_loss": avg_loss, "final_ppl": ppl, "steps": step}
