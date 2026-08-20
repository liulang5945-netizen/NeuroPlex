#!/usr/bin/env python3
"""逐 token 追踪 dialogue 生成：leader、候选一致性、采样与重复。"""
from __future__ import annotations

import json
import logging
import os
import time

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import build_dialogue_prompt
from diag_dialogue_capacity_ab import DIALOGUE_IDS, QUESTIONS


FULL_9 = DIALOGUE_IDS + ["code", "en", "math", "zh"]
FULL_8_NO_ZH = DIALOGUE_IDS + ["code", "en", "math"]
SEED = 20260820
MAX_TOKENS = 8
TEMPERATURE = 0.55
TOP_K = 15
REPETITION_PENALTY = 1.4


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def _candidate_ids(active_nids: list[str]) -> list[str]:
    return [nid for nid in active_nids if nid == "zh" or nid.startswith("zh_")]


def _decode_token(tokenizer, token_id: int) -> dict:
    try:
        piece = tokenizer.id_to_piece(token_id)
    except Exception:
        piece = None
    try:
        decoded = tokenizer.decode([token_id])
    except Exception:
        decoded = None
    return {"token_id": token_id, "piece": piece, "decoded": decoded}


def _leader_trace(
    cortex,
    result: dict,
    prompt: str,
    active_nids: list[str],
    topk_fn=None,
) -> dict:
    topk_fn = topk_fn or torch.topk
    candidates = _candidate_ids(active_nids)
    neuron_logits = result.get("neuron_logits") or {}
    round1_logits = result.get("round1_logits") or neuron_logits
    final_scores = result.get("final_scores") or {}
    resonance_scores = result.get("round1_scores") or final_scores
    domain_scores = {nid: resonance_scores[nid] for nid in candidates if nid in resonance_scores}
    if not domain_scores:
        domain_scores = {nid: final_scores[nid] for nid in candidates if nid in final_scores}
    quality = cortex._nll_quality_from_round1_logits(result, prompt, "zh")
    domain_quality = {nid: quality[nid] for nid in candidates if nid in quality}
    fused = cortex._fuse_leader_quality(domain_scores, domain_quality)
    if fused:
        leader = max(fused, key=fused.get)
    elif domain_scores:
        leader = max(domain_scores, key=domain_scores.get)
    else:
        leader = None

    tokenizer = cortex._tokenizer_hub.get_tokenizer("zh")
    top_tokens = {}
    for nid in candidates:
        logits = round1_logits.get(nid)
        if logits is None or logits.ndim != 3:
            continue
        values, indices = topk_fn(logits[0, -1, :], 5)
        top_tokens[nid] = {
            "top5": [
                {
                    **_decode_token(tokenizer, int(token_id)),
                    "logit": round(float(value), 6),
                }
                for value, token_id in zip(values.tolist(), indices.tolist())
            ]
        }
    top1 = {
        nid: values["top5"][0]["token_id"]
        for nid, values in top_tokens.items()
        if values["top5"]
    }
    return {
        "leader": leader,
        "resonance_scores": domain_scores,
        "nll_quality": domain_quality,
        "fused_scores": fused,
        "top_tokens": top_tokens,
        "top1_unique": len(set(top1.values())),
        "top1_agreement": len(set(top1.values())) <= 1 if top1 else None,
    }


def _trace_one(cortex, active_nids: list[str], question: str) -> dict:
    prompt = build_dialogue_prompt(question)
    think_records = []
    topk_events = []
    sample_events = []
    original_think = cortex.think
    original_topk = torch.topk
    original_multinomial = torch.multinomial

    def traced_think(*args, **kwargs):
        result = original_think(*args, **kwargs)
        think_records.append(
            _leader_trace(cortex, result, prompt, active_nids, topk_fn=original_topk)
        )
        return result

    def traced_topk(*args, **kwargs):
        result = original_topk(*args, **kwargs)
        source = args[0] if args else kwargs.get("input")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        if source is not None and k is not None and source.ndim == 2 and int(k) <= TOP_K:
            topk_events.append({
                "token_ids": result.indices[0].detach().cpu().tolist(),
                "values": result.values[0].detach().cpu().tolist(),
            })
        return result

    def traced_multinomial(*args, **kwargs):
        result = original_multinomial(*args, **kwargs)
        source = args[0] if args else kwargs.get("input")
        num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples")
        if source is not None and num_samples == 1 and source.ndim == 2:
            sample_events.append({
                "position": int(result.reshape(-1)[0].item()),
                "probabilities": source[0].detach().cpu().tolist(),
            })
        return result

    cortex.think = traced_think
    torch.topk = traced_topk
    torch.multinomial = traced_multinomial
    try:
        _reset(cortex)
        torch.manual_seed(SEED)
        generated = cortex._generate_p7(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            domain="zh",
            repetition_penalty=REPETITION_PENALTY,
            active_nids=active_nids,
            collab_mode="continuous",
            fusion_mode="soft",
            auto_memory=False,
            instance_routing=False,
        )
    finally:
        cortex.think = original_think
        torch.topk = original_topk
        torch.multinomial = original_multinomial

    tokenizer = cortex._tokenizer_hub.get_tokenizer("zh")
    steps = []
    generated_ids = []
    pair_history = []
    aligned_count = min(len(think_records), len(topk_events), len(sample_events))
    for index in range(aligned_count):
        leader_record = think_records[index]
        topk = topk_events[index]
        sample = sample_events[index]
        position = sample["position"]
        token_ids = topk["token_ids"]
        if position >= len(token_ids):
            continue
        token_id = int(token_ids[position])
        token = _decode_token(tokenizer, token_id)
        repeated_token = token_id in generated_ids
        pair = (generated_ids[-1], token_id) if generated_ids else None
        repeated_bigram = pair is not None and pair in pair_history
        if pair is not None:
            pair_history.append(pair)
        generated_ids.append(token_id)
        probs = torch.tensor(sample["probabilities"], dtype=torch.float32)
        entropy = float((-(probs * probs.clamp_min(1e-9).log()).sum()).item())
        steps.append({
            "step": index,
            "leader": leader_record["leader"],
            "leader_changed": (
                index > 0 and leader_record["leader"] != steps[-1]["leader"]
            ),
            "candidate_top1_unique": leader_record["top1_unique"],
            "candidate_top1_agreement": leader_record["top1_agreement"],
            "sampled_rank_in_topk": position + 1,
            "sample_entropy": round(entropy, 6),
            "repeated_token": repeated_token,
            "repeated_bigram": repeated_bigram,
            **token,
        })

    leaders = [step["leader"] for step in steps]
    return {
        "question": question,
        "generated": generated,
        "think_calls": len(think_records),
        "topk_calls": len(topk_events),
        "sample_calls": len(sample_events),
        "aligned_steps": len(steps),
        "steps": steps,
        "summary": {
            "leader_sequence": leaders,
            "leader_changes": sum(
                steps[i]["leader"] != steps[i - 1]["leader"]
                for i in range(1, len(steps))
            ),
            "candidate_disagreement_steps": sum(
                not step["candidate_top1_agreement"] for step in steps
            ),
            "repeated_token_steps": sum(step["repeated_token"] for step in steps),
            "repeated_bigram_steps": sum(step["repeated_bigram"] for step in steps),
            "mean_sampled_rank": round(
                sum(step["sampled_rank_in_topk"] for step in steps) / len(steps), 4
            ) if steps else None,
        },
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
            "max_generation_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "collab_mode": "continuous",
            "fusion_mode": "soft",
            "auto_memory": False,
            "instance_routing": False,
            "writes_checkpoint": False,
        },
        "modes": {},
    }
    for mode, active_nids in {"full_9": FULL_9, "full_8_no_zh": FULL_8_NO_ZH}.items():
        rows = []
        for question in QUESTIONS:
            try:
                rows.append(_trace_one(cortex, active_nids, question))
            except Exception as exc:
                rows.append({
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["modes"][mode] = {"active_nids": active_nids, "rows": rows}
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_decode_trace_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
