"""Short, in-memory dialogue pilot for the experimental ``micro`` neuron.

The pilot uses the production ResonanceNeuron path and the same dialogue
alignment/masking contract as the existing dialogue checkpoints.  It freezes
the population shared embedding, trains only the micro member, and never
writes a neuron checkpoint.  This isolates the question "can a smaller member
learn the dialogue target at lower local cost?" before testing population gain.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceNeuron, get_domain_neuron_config
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import batch_align_and_embed, build_position_alignment
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


BASE_CHECKPOINT = os.path.join("data", "neurons", "neuron_zh_aug0_dialogue.pt")
MAX_TEXTS = 100_000
TRAIN_CAP = 128
EVAL_CAP = 100
MAX_SEQ_LEN = 128
STEPS = 160
BATCH_SIZE = 4
LR = 5e-4


def _load_shared_embedding() -> torch.nn.Embedding:
    ckpt = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    del ckpt
    for parameter in shared.parameters():
        parameter.requires_grad = False
    shared.eval()
    return shared


def _encode(texts, domain_sp, general_sp, shared):
    """Precompute frozen input embeddings and masked domain targets."""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            embeddings, targets, mask, sft_mask = batch_align_and_embed(
                batch,
                domain_sp,
                general_sp,
                shared,
                max_seq_len=MAX_SEQ_LEN,
                answer_marker=SFT_ANSWER_MARKER,
            )
            chunks.append((embeddings, targets, mask, sft_mask))
    return tuple(torch.cat(items, dim=0) for items in zip(*chunks))


def _loss(neuron, batch):
    embeddings, targets, mask, sft_mask = batch
    logits = neuron(embeddings, return_logits=True)["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    valid = effective_sft_mask(
        shift_targets,
        mask[:, 1:].contiguous(),
        sft_mask[:, 1:].contiguous(),
    )
    masked_targets = shift_targets.clone()
    masked_targets[~valid] = -100
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        masked_targets.view(-1),
        ignore_index=-100,
    )
    return loss, logits, valid


def _first_token_metrics(first_logits, texts, domain_sp, general_sp) -> dict:
    ranks = []
    nlls = []
    top1 = 0
    for row_index, text in enumerate(texts):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker + len(SFT_ANSWER_MARKER)]
        _, targets = build_position_alignment(text, domain_sp, general_sp)
        answer_start = len(general_sp.encode(prompt))
        if answer_start <= 0 or answer_start > first_logits.shape[1]:
            continue
        target_id = int(targets[answer_start])
        if target_id < 0:
            continue
        next_logits = first_logits[row_index]
        target_logit = next_logits[target_id]
        ranks.append(int((next_logits > target_logit).sum()))
        nlls.append(float(-target_logit + torch.logsumexp(next_logits, dim=-1)))
        top1 += int(int(next_logits.argmax()) == target_id)
    ordered = sorted(ranks)
    return {
        "first_token_nll": round(sum(nlls) / max(len(nlls), 1), 6),
        "median_rank_zero_based": ordered[len(ordered) // 2] if ordered else None,
        "top1_rate": round(top1 / max(len(ranks), 1), 4),
    }


def _evaluate(neuron, encoded, texts, domain_sp, general_sp) -> dict:
    neuron.eval()
    total_ce = 0.0
    total_tokens = 0
    first_logits = []
    first_texts = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tuple(item[start:start + BATCH_SIZE] for item in encoded)
            loss, logits, valid = _loss(neuron, batch)
            total_ce += float(loss) * int(valid.sum())
            total_tokens += int(valid.sum())
            for row_index, text in enumerate(texts[start:start + BATCH_SIZE]):
                marker = text.find(SFT_ANSWER_MARKER)
                prompt = text[:marker + len(SFT_ANSWER_MARKER)]
                answer_start = len(general_sp.encode(prompt))
                if 0 < answer_start <= logits.shape[1]:
                    first_logits.append(logits[row_index, answer_start - 1].cpu())
                    first_texts.append(text)
    if first_logits:
        metrics = _first_token_metrics(
            torch.stack(first_logits), first_texts, domain_sp, general_sp
        )
    else:
        metrics = {
            "first_token_nll": None,
            "median_rank_zero_based": None,
            "top1_rate": None,
        }
    metrics.update({
        "answer_loss": round(total_ce / max(total_tokens, 1), 6),
        "corrected_ppl": round(math.exp(min(total_ce / max(total_tokens, 1), 20)), 4),
        "effective_answer_tokens": total_tokens,
    })
    return metrics


def run() -> dict:
    logging.disable(logging.CRITICAL)
    torch.manual_seed(20260819)
    random.seed(20260819)
    torch.set_num_threads(6)

    all_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    train_pool, eval_pool = split_train_eval(all_texts, eval_ratio=0.05, seed=42)
    train_texts = train_pool[:TRAIN_CAP]
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared = _load_shared_embedding()
    train_encoded = _encode(train_texts, domain_sp, general_sp, shared)
    eval_encoded = _encode(eval_texts, domain_sp, general_sp, shared)

    cfg = get_domain_neuron_config("zh", spec="micro")
    cfg.neuron_id = "zh_micro_dialogue_pilot"
    neuron = ResonanceNeuron(cfg)
    local_params = sum(parameter.numel() for parameter in neuron.parameters())
    report = {
        "contract": {
            "neuron_id": cfg.neuron_id,
            "spec": cfg.spec,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "train_samples": len(train_texts),
            "eval_samples": len(eval_texts),
            "max_seq_len": MAX_SEQ_LEN,
            "shared_embedding_frozen": True,
            "writes_checkpoint": False,
        },
        "architecture": {
            "local_params_m": round(local_params / 1_000_000, 6),
            "shared_embedding_params_m": 256_000 * 512 / 1_000_000,
            "hidden_size": cfg.hidden_size,
            "layers": cfg.num_hidden_layers,
            "field_dim": cfg.field_dim,
        },
    }

    report["before"] = _evaluate(neuron, eval_encoded, eval_texts, domain_sp, general_sp)
    optimizer = torch.optim.AdamW(neuron.parameters(), lr=LR, weight_decay=0.1)
    losses = []
    neuron.train()
    generator = torch.Generator().manual_seed(20260819)
    for step in range(1, STEPS + 1):
        indices = torch.randint(0, len(train_texts), (BATCH_SIZE,), generator=generator)
        batch = tuple(item[indices] for item in train_encoded)
        optimizer.zero_grad()
        loss, _, _ = _loss(neuron, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neuron.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 40 == 0:
            print(f"step {step}/{STEPS}: train_answer_loss={losses[-1]:.6f}", flush=True)

    report["after"] = _evaluate(neuron, eval_encoded, eval_texts, domain_sp, general_sp)
    report["loss_trace"] = {
        "first": round(losses[0], 6),
        "last": round(losses[-1], 6),
        "min": round(min(losses), 6),
    }
    del neuron, shared, optimizer
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
