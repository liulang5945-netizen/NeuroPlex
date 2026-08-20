#!/usr/bin/env python3
"""追踪 dialogue continuous 路径的 leader 选择与首 token。

该脚本不改生成逻辑，只复现生成首轮 think 的输入，并记录：
  - round1 共振分 leader
  - prompt NLL 质量 leader
  - 二者融合后的 continuous leader（生成路径实际使用）
  - 各候选 zh neuron 的首 token

重点比较 5-dialogue 子群体和完整 9 成员群体，确认 full route 是否把
``zh`` 基础成员带入 dialogue 输出通路。
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
FULL_IDS = DIALOGUE_IDS + ["code", "en", "math", "zh"]
QUESTIONS = [
    "你好，请介绍一下你自己",
    "你是谁？",
    "什么是神经网络？",
    "什么是注意力机制？",
]
SEED = 20260820


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def _embeddings(cortex, active_nids: list[str], prompt: str) -> dict:
    ids = cortex._general_sp.encode(prompt)
    if not ids:
        ids = [0]
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=cortex.device)
    return {
        nid: cortex._neuron_shared_embeddings[nid](ids_tensor)
        for nid in active_nids
        if nid in cortex._neuron_shared_embeddings
    }


def _top_token(cortex, logits: torch.Tensor, tokenizer) -> dict:
    token_id = int(logits[0, -1, :].argmax().item())
    try:
        piece = tokenizer.id_to_piece(token_id)
    except Exception:
        piece = None
    try:
        decoded = tokenizer.decode([token_id])
    except Exception:
        decoded = None
    return {"token_id": token_id, "piece": piece, "decoded": decoded}


def _candidate_ids(active_nids: list[str]) -> list[str]:
    return [nid for nid in active_nids if nid == "zh" or nid.startswith("zh_")]


def _trace(cortex, active_nids: list[str], question: str) -> dict:
    _reset(cortex)
    torch.manual_seed(SEED)
    prompt = build_dialogue_prompt(question)
    result = cortex.think(
        active_nids=active_nids,
        neuron_embeddings=_embeddings(cortex, active_nids, prompt),
        collab_mode="continuous",
        fusion_mode="soft",
    )

    candidates = _candidate_ids(active_nids)
    round1_scores = result.get("round1_scores") or {}
    final_scores = result.get("final_scores") or {}
    domain_scores = {
        nid: value for nid, value in round1_scores.items() if nid in candidates
    }
    if not domain_scores:
        domain_scores = {nid: final_scores[nid] for nid in candidates if nid in final_scores}
    raw_leader = max(domain_scores, key=domain_scores.get) if domain_scores else None

    quality = cortex._nll_quality_from_round1_logits(result, prompt, "zh")
    domain_quality = {nid: value for nid, value in quality.items() if nid in candidates}
    quality_leader = max(domain_quality, key=domain_quality.get) if domain_quality else None
    fused = cortex._fuse_leader_quality(domain_scores, domain_quality)
    fused_leader = max(fused, key=fused.get) if fused else raw_leader

    zh_tokenizer = cortex._tokenizer_hub.get_tokenizer("zh")
    logits = result.get("round1_logits") or {}
    top_tokens = {
        nid: _top_token(cortex, logits[nid], zh_tokenizer)
        for nid in candidates
        if nid in logits and logits[nid].shape[-1] == zh_tokenizer.GetPieceSize()
    }
    return {
        "question": question,
        "active_nids": active_nids,
        "candidate_nids": candidates,
        "round1_scores": domain_scores,
        "final_scores": {nid: final_scores[nid] for nid in candidates if nid in final_scores},
        "nll_quality": domain_quality,
        "fused_leader_scores": fused,
        "raw_resonance_leader": raw_leader,
        "quality_leader": quality_leader,
        "predicted_continuous_leader": fused_leader,
        "top_tokens": top_tokens,
        "weighted_logits_vocab": (
            int(result["weighted_logits"].shape[-1])
            if result.get("weighted_logits") is not None else None
        ),
    }


def main() -> None:
    logging.disable(logging.CRITICAL)
    started = time.time()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=False,
        neuron_ids=DIALOGUE_IDS,
    )
    report = {
        "contract": {
            "seed": SEED,
            "questions": QUESTIONS,
            "population_shape": "5 dialogue + 4 general",
            "writes_checkpoint": False,
        },
        "modes": {},
    }
    for mode, active_nids in {
        "dialogue_5": DIALOGUE_IDS,
        "full_9": FULL_IDS,
    }.items():
        rows = []
        for question in QUESTIONS:
            try:
                rows.append(_trace(cortex, active_nids, question))
            except Exception as exc:
                rows.append({
                    "question": question,
                    "active_nids": active_nids,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["modes"][mode] = {
            "rows": rows,
            "predicted_continuous_leaders": [
                row.get("predicted_continuous_leader") for row in rows
            ],
            "leader_counts": {
                nid: sum(row.get("predicted_continuous_leader") == nid for row in rows)
                for nid in sorted({
                    row.get("predicted_continuous_leader")
                    for row in rows
                    if row.get("predicted_continuous_leader") is not None
                })
            },
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_leader_trace_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
