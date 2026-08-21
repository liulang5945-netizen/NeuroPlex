"""Local fast-memory rules for Taiji cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import bound_vector


@dataclass(frozen=True)
class AssociativeRecall:
    value: torch.Tensor
    confidence: float
    slot: int | None


@dataclass(frozen=True)
class AssociativeWrite:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    slot: int


def recall_association(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    usage: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
) -> AssociativeRecall:
    """Recall a local value without changing memory state."""

    if values.shape[0] != keys.shape[0] or usage.shape != (keys.shape[0],):
        raise ValueError("fast-memory slot dimensions do not match")
    zero = torch.zeros(values.shape[-1], device=values.device, dtype=values.dtype)
    active = usage > 0.0
    query_norm = query.norm()
    if not bool(active.any()) or float(query_norm.item()) < 1e-8:
        return AssociativeRecall(zero, 0.0, None)

    query_unit = query / query_norm.clamp_min(1e-8)
    key_units = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarities = key_units @ query_unit
    masked = similarities.masked_fill(~active, float("-inf"))
    best_slot = int(masked.argmax().item())
    best_similarity = float(masked[best_slot].item())
    if best_similarity < threshold:
        return AssociativeRecall(zero, 0.0, None)

    logits = masked * float(temperature)
    weights = torch.softmax(logits, dim=0)
    recalled = (weights.unsqueeze(-1) * values).sum(dim=0)
    similarity_confidence = (best_similarity - threshold) / max(1e-8, 1.0 - threshold)
    confidence = max(0.0, min(1.0, similarity_confidence))
    confidence *= max(0.0, min(1.0, float(usage[best_slot].item())))
    return AssociativeRecall(recalled, confidence, best_slot)


def store_association(
    cue: torch.Tensor,
    target: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    usage: torch.Tensor,
    *,
    strength: float,
    merge_threshold: float,
    max_value_norm: float,
) -> AssociativeWrite:
    """Write one association into one local slot.

    Existing matching keys are merged.  Otherwise the first empty slot is used;
    a full memory replaces only its least-used slot.  No other slot is modified.
    """

    if not 0.0 < strength <= 1.0:
        raise ValueError("association strength must be in (0, 1]")
    if cue.dim() != 1 or target.dim() != 1:
        raise ValueError("cue and target must be rank-1")
    if keys.shape != (usage.numel(), cue.numel()):
        raise ValueError("cue dimension does not match memory keys")
    if values.shape != (usage.numel(), target.numel()):
        raise ValueError("target dimension does not match memory values")
    cue_norm = cue.norm()
    if float(cue_norm.item()) < 1e-8:
        raise ValueError("cannot store a zero cue")

    cue_unit = cue / cue_norm.clamp_min(1e-8)
    target_value = bound_vector(target, max_value_norm)
    active = usage > 0.0
    slot: int
    merge = False
    if bool(active.any()):
        key_units = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarities = (key_units @ cue_unit).masked_fill(~active, float("-inf"))
        best_slot = int(similarities.argmax().item())
        if float(similarities[best_slot].item()) >= merge_threshold:
            slot = best_slot
            merge = True
        else:
            empty = (~active).nonzero(as_tuple=False).flatten()
            slot = int(empty[0].item()) if empty.numel() else int(usage.argmin().item())
    else:
        slot = 0

    new_keys = keys.detach().clone()
    new_values = values.detach().clone()
    new_usage = usage.detach().clone()
    if merge:
        merged_key = (1.0 - strength) * new_keys[slot] + strength * cue_unit
        new_keys[slot] = merged_key / merged_key.norm().clamp_min(1e-8)
        new_values[slot] = (
            (1.0 - strength) * new_values[slot] + strength * target_value
        )
    else:
        new_keys[slot] = cue_unit
        new_values[slot] = target_value
    new_usage[slot] = min(1.0, float(new_usage[slot].item()) + strength)
    return AssociativeWrite(new_keys, new_values, new_usage, slot)
