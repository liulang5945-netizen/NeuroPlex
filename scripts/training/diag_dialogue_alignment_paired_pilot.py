#!/usr/bin/env python3
"""同初始化、同批次的 legacy/corrected alignment 对照 pilot。"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

import neuroplex.resonance.translator as translator_module
from neuroplex.resonance import ResonanceNeuron
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER, build_dialogue_prompt
from scripts.training.finetune_neuron_dialogue import effective_sft_mask
from scripts.training.utils import (
    SequentialSampler,
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    make_wsd_scheduler,
    split_train_eval,
)


NEURON_ID = "zh_aug0_dialogue"
MAX_TEXTS = 100_000
EVAL_CAP = 100
MAX_SEQ_LEN = 128
STEPS = 320
BATCH_SIZE = 8
LR = 1e-4
WARMUP_STEPS = 20
SEED = 20260820
GENERATION_QUESTIONS = ["你好，请介绍一下你自己", "什么是神经网络？", "什么是注意力机制？"]
PRODUCTION_ALIGNMENT = translator_module.build_position_alignment


def _legacy_position_alignment(text: str, domain_sp, general_sp):
    """保留旧行为：每个 general piece 都绑定其最大重叠 domain piece。"""
    domain_ids, domain_spans = translator_module._get_token_spans(domain_sp, text)
    general_ids, general_spans = translator_module._get_token_spans(general_sp, text)
    targets = torch.full((len(general_ids),), -100, dtype=torch.long)
    for j, (g_start, g_end) in enumerate(general_spans):
        best_i = -1
        best_overlap = 0
        for i, (d_start, d_end) in enumerate(domain_spans):
            overlap = max(0, min(g_end, d_end) - max(g_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i
        if best_i >= 0:
            targets[j] = domain_ids[best_i]
    return torch.tensor(general_ids, dtype=torch.long), targets


def _load_neuron() -> tuple[ResonanceNeuron, torch.nn.Embedding]:
    path = os.path.join("data", "neurons", f"neuron_{NEURON_ID}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"])
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    del ckpt
    return neuron, shared


def _batch_loss(neuron, shared, texts, domain_sp, general_sp) -> torch.Tensor:
    embeddings, targets, mask, sft_mask = translator_module.batch_align_and_embed(
        texts,
        domain_sp,
        general_sp,
        shared,
        max_seq_len=MAX_SEQ_LEN,
        answer_marker=SFT_ANSWER_MARKER,
    )
    logits = neuron(embeddings, return_logits=True)["logits"]
    shift_logits = logits[:, :-1].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    valid = effective_sft_mask(
        shift_targets,
        mask[:, 1:].contiguous(),
        sft_mask[:, 1:].contiguous(),
    )
    target_ids = shift_targets.clone()
    target_ids[~valid] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def _metrics(neuron, shared, texts, domain_sp, general_sp) -> dict:
    neuron.eval()
    shared.eval()
    total_ce = 0.0
    total_tokens = 0
    ranks = []
    top1 = 0
    skipped = 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        embeddings, targets, mask, sft_mask = translator_module.batch_align_and_embed(
            batch,
            domain_sp,
            general_sp,
            shared,
            max_seq_len=MAX_SEQ_LEN,
            answer_marker=SFT_ANSWER_MARKER,
        )
        logits = neuron(embeddings, return_logits=True)["logits"]
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        valid = effective_sft_mask(
            shift_targets,
            mask[:, 1:].contiguous(),
            sft_mask[:, 1:].contiguous(),
        )
        target_ids = shift_targets.clone()
        target_ids[~valid] = -100
        total_ce += F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        total_tokens += int(valid.sum())

        for row_index, text in enumerate(batch):
            marker = text.find(SFT_ANSWER_MARKER)
            prompt = text[:marker + len(SFT_ANSWER_MARKER)]
            _, aligned_targets = translator_module.build_position_alignment(
                text, domain_sp, general_sp
            )
            answer_start = len(general_sp.encode(prompt))
            if answer_start <= 0 or answer_start >= len(aligned_targets):
                skipped += 1
                continue
            target_id = int(aligned_targets[answer_start])
            if target_id < 0:
                skipped += 1
                continue
            next_logits = logits[row_index, answer_start - 1]
            target_logit = next_logits[target_id]
            ranks.append(int((next_logits > target_logit).sum()))
            top1 += int(int(next_logits.argmax()) == target_id)
    ordered = sorted(ranks)
    return {
        "ppl": round(math.exp(min(total_ce / max(total_tokens, 1), 20)), 4),
        "aligned_answer_tokens": total_tokens,
        "mean_first_target_rank_zero_based": round(sum(ranks) / max(len(ranks), 1), 2),
        "median_first_target_rank_zero_based": ordered[len(ordered) // 2] if ordered else None,
        "first_target_top1_rate": round(top1 / max(len(ranks), 1), 4),
        "rank_samples": len(ranks),
        "skipped_first_target_samples": skipped,
    }


@torch.no_grad()
def _generate(neuron, shared, domain_sp, general_sp, question: str) -> str:
    prompt = build_dialogue_prompt(question)
    general_ids = general_sp.encode(prompt) or [0]
    generated = []
    neuron.eval()
    shared.eval()
    torch.manual_seed(SEED)
    for _ in range(8):
        ids = torch.tensor([general_ids], dtype=torch.long)
        logits = neuron(shared(ids), return_logits=True)["logits"][:, -1, :]
        top_k = min(15, logits.shape[-1])
        top_values, top_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_values / 0.55, dim=-1)
        sampled = torch.multinomial(probs, 1)
        token_id = int(top_indices[0, sampled[0, 0]].item())
        if token_id == domain_sp.eos_id():
            break
        generated.append(token_id)
        general_ids.extend(general_sp.encode(domain_sp.decode([token_id])))
    return domain_sp.decode(generated)


def _run(alignment_name, alignment_fn, train_texts, eval_texts, domain_sp, general_sp):
    torch.manual_seed(SEED)
    translator_module.build_position_alignment = alignment_fn
    neuron, shared = _load_neuron()
    before = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    generation_before = {
        question: _generate(neuron, shared, domain_sp, general_sp, question)
        for question in GENERATION_QUESTIONS
    }

    neuron.train()
    shared.train()
    optimizer = torch.optim.AdamW(
        list(neuron.parameters()) + list(shared.parameters()),
        lr=LR,
        weight_decay=0.1,
    )
    scheduler = make_wsd_scheduler(
        optimizer,
        num_steps=STEPS,
        warmup_steps=WARMUP_STEPS,
        decay_ratio=0.85,
    )
    sampler = SequentialSampler(train_texts, BATCH_SIZE, seed=SEED)
    losses = []
    for step in range(STEPS):
        batch = sampler.sample_batch()
        optimizer.zero_grad()
        loss = _batch_loss(neuron, shared, batch, domain_sp, general_sp)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 40 == 0:
            print(f"  [{alignment_name}] step {step + 1}/{STEPS} loss={losses[-1]:.4f}", flush=True)

    after = _metrics(neuron, shared, eval_texts, domain_sp, general_sp)
    generation_after = {
        question: _generate(neuron, shared, domain_sp, general_sp, question)
        for question in GENERATION_QUESTIONS
    }
    result = {
        "alignment": alignment_name,
        "before": before,
        "after": after,
        "generation_before": generation_before,
        "generation_after": generation_after,
        "loss_trace": {
            "first": round(losses[0], 6),
            "last": round(losses[-1], 6),
            "min": round(min(losses), 6),
        },
    }
    del neuron, shared, optimizer, scheduler
    gc.collect()
    return result


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    full_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, eval_pool = split_train_eval(full_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    train_texts = full_texts
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    results = [
        _run("legacy", _legacy_position_alignment, train_texts, eval_texts, domain_sp, general_sp),
        _run("corrected", PRODUCTION_ALIGNMENT, train_texts, eval_texts, domain_sp, general_sp),
    ]
    translator_module.build_position_alignment = PRODUCTION_ALIGNMENT
    report = {
        "contract": {
            "neuron_id": NEURON_ID,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "train_samples": len(train_texts),
            "eval_samples": len(eval_texts),
            "same_initial_checkpoint": True,
            "same_sequential_batches": True,
            "writes_checkpoint": False,
        },
        "runs": results,
    }
    out_path = os.path.join("reports", "production_dialogue_alignment_paired_pilot_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
