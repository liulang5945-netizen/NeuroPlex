#!/usr/bin/env python3
"""单体 dialogue 生成上下文：逐 piece 回填 vs 完整文本重编码 A/B。"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceNeuron
from neuroplex.resonance.dialogue_format import build_dialogue_prompt
from scripts.training.utils import create_shared_embedding, load_domain_tokenizer, load_general_tokenizer


NEURON_ID = "zh_aug0_dialogue"
SEED = 20260820
TEMPERATURE = 0.55
TOP_K = 15
MAX_TOKENS = 8
QUESTIONS = ["你好，请介绍一下你自己", "什么是神经网络？", "什么是注意力机制？"]


def _load_neuron():
    path = os.path.join("data", "neurons", f"neuron_{NEURON_ID}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"])
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    del ckpt
    return neuron, shared


@torch.no_grad()
def _generate(neuron, shared, domain_sp, general_sp, question: str, mode: str) -> dict:
    prompt = build_dialogue_prompt(question)
    prefix_ids = general_sp.encode(prompt) or [0]
    general_ids = list(prefix_ids)
    incremental_context_ids = list(prefix_ids)
    prefix_text = general_sp.DecodeIds(general_ids)
    generated_ids = []
    context_equal_steps = []
    neuron.eval()
    shared.eval()
    torch.manual_seed(SEED)
    for _ in range(MAX_TOKENS):
        ids = torch.tensor([general_ids], dtype=torch.long)
        logits = neuron(shared(ids), return_logits=True)["logits"][:, -1, :]
        top_values, top_indices = torch.topk(logits, min(TOP_K, logits.shape[-1]))
        probs = F.softmax(top_values / TEMPERATURE, dim=-1)
        sampled = torch.multinomial(probs, 1)
        token_id = int(top_indices[0, sampled[0, 0]].item())
        if token_id == domain_sp.eos_id():
            break
        generated_ids.append(token_id)

        full_ids = general_sp.encode(prefix_text + domain_sp.DecodeIds(generated_ids)) or [0]
        incremental_ids = incremental_context_ids + general_sp.encode(domain_sp.decode([token_id]))
        context_equal_steps.append(incremental_ids == full_ids)
        incremental_context_ids = incremental_ids
        if mode == "incremental":
            general_ids = incremental_ids
        elif mode == "full_reencode":
            general_ids = full_ids
        else:
            raise ValueError(f"unknown mode: {mode}")

    return {
        "text": domain_sp.DecodeIds(generated_ids),
        "generated_ids": generated_ids,
        "context_equal_steps": context_equal_steps,
        "context_equal_rate": round(
            sum(context_equal_steps) / max(len(context_equal_steps), 1), 4
        ),
    }


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    neuron, shared = _load_neuron()
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    report = {
        "contract": {
            "neuron_id": NEURON_ID,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "max_tokens": MAX_TOKENS,
            "same_checkpoint": True,
            "production_checkpoint_written": False,
        },
        "prompts": {},
    }
    for question in QUESTIONS:
        report["prompts"][question] = {
            "incremental": _generate(neuron, shared, domain_sp, general_sp, question, "incremental"),
            "full_reencode": _generate(neuron, shared, domain_sp, general_sp, question, "full_reencode"),
        }
    out_path = os.path.join("reports", "production_dialogue_generation_context_ab_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
