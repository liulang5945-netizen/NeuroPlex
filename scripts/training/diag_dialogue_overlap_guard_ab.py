#!/usr/bin/env python3
"""诊断-only：domain piece 字符重叠 guard 的最小 A/B。"""
from __future__ import annotations

import json
import logging
import os
import time

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.resonance.dialogue_format import build_dialogue_prompt
from diag_dialogue_capacity_ab import DIALOGUE_IDS, QUESTIONS, _text_metrics


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


def _overlap_chars(previous: str, current: str) -> int:
    previous = "".join(previous.split())
    current = "".join(current.split())
    overlap = 0
    for size in range(1, min(len(previous), len(current)) + 1):
        if previous[-size:] == current[:size]:
            overlap = size
    return overlap


def _generate(cortex, active_nids: list[str], question: str, guard: bool) -> dict:
    prompt = build_dialogue_prompt(question)
    tokenizer = cortex._tokenizer_hub.get_tokenizer("zh")
    topk_ids = None
    previous_pieces: list[str] = []
    guard_events = []
    original_topk = torch.topk
    original_multinomial = torch.multinomial

    def traced_topk(*args, **kwargs):
        nonlocal topk_ids
        result = original_topk(*args, **kwargs)
        source = args[0] if args else kwargs.get("input")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        if source is not None and k is not None and source.ndim == 2 and int(k) <= TOP_K:
            topk_ids = result.indices[0].detach().cpu().tolist()
        return result

    def traced_multinomial(*args, **kwargs):
        source = args[0] if args else kwargs.get("input")
        num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples")
        adjusted = source
        banned = []
        if (
            guard
            and source is not None
            and num_samples == 1
            and source.ndim == 2
            and topk_ids is not None
            and previous_pieces
        ):
            previous = previous_pieces[-1]
            for position, token_id in enumerate(topk_ids):
                current = tokenizer.decode([int(token_id)])
                if _overlap_chars(previous, current) >= 2:
                    banned.append({
                        "position": position,
                        "token_id": int(token_id),
                        "previous": previous,
                        "current": current,
                    })
            if banned and len(banned) < source.shape[-1]:
                adjusted = source.clone()
                adjusted[:, [event["position"] for event in banned]] = 0.0
                normalizer = adjusted.sum(dim=-1, keepdim=True)
                if torch.all(normalizer > 0):
                    adjusted = adjusted / normalizer
                else:
                    adjusted = source
                    banned = []
        result = original_multinomial(adjusted, *args[1:], **kwargs)
        if source is not None and num_samples == 1 and source.ndim == 2 and topk_ids is not None:
            position = int(result.reshape(-1)[0].item())
            token_id = int(topk_ids[position])
            current = tokenizer.decode([token_id])
            previous_pieces.append(current)
            if banned:
                guard_events.append({
                    "banned": banned,
                    "selected_position": position,
                    "selected_token_id": token_id,
                    "selected_piece": current,
                })
        return result

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
        torch.topk = original_topk
        torch.multinomial = original_multinomial
    return {
        "text": generated,
        **_text_metrics(generated),
        "guard": guard,
        "guard_events": guard_events,
        "guard_event_count": len(guard_events),
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
            "guard_rule": "ban a domain piece when its decoded text overlaps the previous piece by >=2 chars",
            "writes_checkpoint": False,
        },
        "modes": {},
    }
    for mode, active_nids in {"full_9": FULL_9, "full_8_no_zh": FULL_8_NO_ZH}.items():
        rows = []
        for question in QUESTIONS:
            try:
                current = _generate(cortex, active_nids, question, guard=False)
                guarded = _generate(cortex, active_nids, question, guard=True)
                rows.append({
                    "question": question,
                    "current": current,
                    "guarded": guarded,
                })
            except Exception as exc:
                rows.append({
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["modes"][mode] = {"active_nids": active_nids, "rows": rows}
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_overlap_guard_ab_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
