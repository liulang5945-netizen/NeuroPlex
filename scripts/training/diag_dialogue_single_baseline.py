#!/usr/bin/env python3
"""P1.2：五个 dialogue neuron 的单体基线归因。

该脚本固定生产装配入口，只改变显式激活的 dialogue neuron：

    python -X utf8 -u scripts/training/diag_dialogue_single_baseline.py

输出每个 checkpoint 的来源字段、对齐后的 prompt NLL 和短生成，区分
“单体能力不足”和“群体路由/融合造成的退化”。不执行训练，不写 checkpoint。
"""
from __future__ import annotations

import json
import logging
import os
import time

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import build_dialogue_prompt


DIALOGUE_IDS = [
    "zh_aug0_dialogue",
    "zh_aug1_dialogue",
    "zh_aug2_dialogue",
    "zh_aug3_dialogue",
    "zh_std0_dialogue",
]
QUESTIONS = [
    "你好，请介绍一下你自己",
    "你是谁？",
    "什么是神经网络？",
    "什么是注意力机制？",
    "请解释梯度下降的原理",
    "你能帮我做什么？",
    "推荐一本好书",
    "今天天气怎么样？",
]
GENERATION_QUESTIONS = QUESTIONS[:4]


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def _checkpoint_metadata(nid: str) -> dict:
    path = os.path.join("data", "neurons", f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("neuron_config")
    result = ckpt.get("result") or {}
    return {
        "file_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
        "domain": ckpt.get("domain"),
        "data_source": ckpt.get("data_source"),
        "has_shared_embedding": "shared_embedding_state" in ckpt,
        "has_optimizer_state": "optimizer_state" in ckpt,
        "spec": getattr(cfg, "spec", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "field_dim": getattr(cfg, "field_dim", None),
        "steps": result.get("steps"),
        "best_step": result.get("best_step"),
        "best_val_ppl": result.get("best_val_ppl"),
        "base_id": result.get("base_id"),
        "finetune": result.get("finetune"),
    }


def _neuron_embeddings(cortex, nid: str, prompt: str) -> dict:
    general_ids = cortex._general_sp.encode(prompt)
    if not general_ids:
        general_ids = [0]
    ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
    embedding = cortex._neuron_shared_embeddings[nid](ids_tensor)
    return {nid: embedding}


def _prompt_nll(cortex, nid: str, prompt: str) -> float | None:
    _reset(cortex)
    result = cortex.think(
        active_nids=[nid],
        neuron_embeddings=_neuron_embeddings(cortex, nid, prompt),
        collab_mode="continuous",
    )
    quality = cortex._nll_quality_from_round1_logits(result, prompt, "zh")
    score = quality.get(nid)
    return -score if score is not None else None


def _generate(cortex, nid: str, prompt: str) -> str:
    _reset(cortex)
    torch.manual_seed(20260819)
    return cortex.generate(
        prompt=prompt,
        max_tokens=8,
        temperature=0.55,
        top_k=15,
        domain="zh",
        repetition_penalty=1.4,
        active_nids=[nid],
        collab_mode="continuous",
        fusion_mode="soft",
        auto_memory=False,
        instance_routing=False,
    )


def main() -> None:
    logging.disable(logging.CRITICAL)
    started = time.time()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    report = {
        "population": list(cortex.neurons.keys()),
        "quality_metric": "general_to_domain_aligned_prompt_nll",
        "max_generation_tokens": 8,
        "neurons": {},
    }
    for nid in DIALOGUE_IDS:
        nll_values = []
        for question in QUESTIONS:
            prompt = build_dialogue_prompt(question)
            value = _prompt_nll(cortex, nid, prompt)
            if value is not None:
                nll_values.append(value)
        generations = {}
        for question in GENERATION_QUESTIONS:
            prompt = build_dialogue_prompt(question)
            generations[question] = _generate(cortex, nid, prompt)
        report["neurons"][nid] = {
            "checkpoint": _checkpoint_metadata(nid),
            "nll": {
                "count": len(nll_values),
                "mean": round(sum(nll_values) / len(nll_values), 4) if nll_values else None,
                "min": round(min(nll_values), 4) if nll_values else None,
                "max": round(max(nll_values), 4) if nll_values else None,
            },
            "generations": generations,
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
