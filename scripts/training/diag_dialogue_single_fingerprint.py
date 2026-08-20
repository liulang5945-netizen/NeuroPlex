#!/usr/bin/env python3
"""记录 5 个 dialogue 单体的 token continuation fingerprint。"""
from __future__ import annotations

import json
import logging
import os
import time

from neuroplex.loader import assemble_cortex
from diag_dialogue_capacity_ab import DIALOGUE_IDS
from diag_dialogue_decode_trace import _trace_one


QUESTIONS = ["什么是神经网络？", "什么是注意力机制？", "你是谁？"]


def _overlap_events(steps: list[dict]) -> list[dict]:
    events = []
    for previous, current in zip(steps, steps[1:]):
        left = "".join((previous.get("decoded") or "").split())
        right = "".join((current.get("decoded") or "").split())
        overlap = 0
        for size in range(1, min(len(left), len(right)) + 1):
            if left[-size:] == right[:size]:
                overlap = size
        if overlap >= 2:
            events.append({
                "step": current["step"],
                "previous": left,
                "current": right,
                "overlap": overlap,
            })
    return events


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
            "seed": 20260820,
            "questions": QUESTIONS,
            "max_generation_tokens": 8,
            "temperature": 0.55,
            "top_k": 15,
            "repetition_penalty": 1.4,
            "collab_mode": "continuous",
            "fusion_mode": "soft",
            "auto_memory": False,
            "instance_routing": False,
            "writes_checkpoint": False,
        },
        "neurons": {},
    }
    for nid in DIALOGUE_IDS:
        rows = []
        for question in QUESTIONS:
            try:
                trace = _trace_one(cortex, [nid], question)
                rows.append({
                    "question": question,
                    "generated": trace["generated"],
                    "pieces": [step.get("decoded") for step in trace["steps"]],
                    "overlap_events": _overlap_events(trace["steps"]),
                    "trace": trace,
                })
            except Exception as exc:
                rows.append({
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["neurons"][nid] = {"rows": rows}

    for question in QUESTIONS:
        per_neuron = []
        for nid in DIALOGUE_IDS:
            row = next(row for row in report["neurons"][nid]["rows"] if row["question"] == question)
            per_neuron.append(row.get("pieces", []))
        max_steps = max((len(pieces) for pieces in per_neuron), default=0)
        common = []
        for step in range(max_steps):
            values = [pieces[step] for pieces in per_neuron if step < len(pieces)]
            common.append({
                "step": step,
                "values": values,
                "unique_values": sorted(set(values)),
                "all_same": len(set(values)) <= 1,
            })
        report.setdefault("cross_neuron", {})[question] = {
            "common_step_pieces": common,
            "neurons_with_overlap": [
                nid for nid in DIALOGUE_IDS
                if next(row for row in report["neurons"][nid]["rows"] if row["question"] == question)["overlap_events"]
            ],
        }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = os.path.join("reports", "production_dialogue_single_fingerprint_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
