"""Route-only canary for the 800-step 7.58M micro dialogue member.

This deliberately repeats only the bounded 90/10 mixed training needed to
recreate the in-memory micro member.  It does not run the expensive full
holdout evaluation and never writes a checkpoint.  The canary compares the
real nine-member population with the same population using either explicit
all-member fusion or automatic top-k activation under identical prompts and
seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from neuroplex.loader import assemble_cortex
from scripts.training.diag_micro_data_ab import (
    DEFAULT_STEPS,
    HF_DIR,
    SEED,
    _load_pools,
    _load_shared_embedding,
    _select_hf_for_ratio,
    _train_condition,
)
from scripts.training.diag_micro_dialogue_pilot import BATCH_SIZE, MAX_SEQ_LEN
from scripts.training.diag_micro_population_canary import (
    PROMPTS,
    _generate,
    _real_population_forward,
    _surface_metrics,
)
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer


DEFAULT_HF_RATIO = 0.10
ROUTE_MODES = (
    "base_9_all",
    "with_micro_10_all",
    "with_micro_10_auto_top1",
    "with_micro_10_auto_top2",
)


def _assemble_with_micro(micro, shared):
    with contextlib.redirect_stdout(io.StringIO()):
        cortex, _, _ = assemble_cortex(
            neurons_dir="data/neurons",
            extra_neurons_dir="data/foundation_v1_dual",
            collab_name="collab_v3_c24v2.ckpt.pt",
            device="cpu",
            max_rounds=3,
            wire_bio_modules=False,
            neuron_ids=list(DEFAULT_NEURON_IDS),
        )
    base_ids = list(cortex.neurons.keys())
    expected_general = {"code", "en", "math", "zh"}
    actual_general = set(base_ids) - set(DEFAULT_NEURON_IDS)
    if set(DEFAULT_NEURON_IDS) - set(base_ids) or not expected_general.issubset(actual_general):
        raise RuntimeError(f"real production population mismatch: {base_ids}")

    micro_id = "zh_micro_dialogue_ab"
    micro.config.neuron_id = micro_id
    cortex.ensemble.add_neuron(micro_id, micro)
    embeddings = dict(cortex._neuron_shared_embeddings or {})
    embeddings[micro_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    return cortex, base_ids, base_ids + [micro_id]


def run(steps: int = DEFAULT_STEPS, hf_ratio: float = DEFAULT_HF_RATIO) -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    pools = _load_pools()
    selected_hf = _select_hf_for_ratio(pools["current_train"], pools["hf_train"], hf_ratio)
    shared = _load_shared_embedding()
    for parameter in shared.parameters():
        parameter.requires_grad = False
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    train_texts = pools["current_train"] + selected_hf
    train_report, micro = _train_condition(
        "current_plus_hf_10",
        train_texts,
        {},
        shared,
        domain_sp,
        general_sp,
        steps,
        return_neuron=True,
    )

    cortex, base_ids, expanded_ids = _assemble_with_micro(micro, shared)
    active_sets = {
        "base_9_all": base_ids,
        "with_micro_10_all": expanded_ids,
        "with_micro_10_auto_top1": "auto_top1",
        "with_micro_10_auto_top2": "auto_top2",
    }
    report = {
        "contract": {
            "seed": SEED,
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "hf_ratio": hf_ratio,
            "hf_train_used": len(selected_hf),
            "shared_embedding_frozen": True,
            "writes_checkpoint": False,
            "base_population": "5 dialogue + 4 general",
            "base_ids": base_ids,
            "expanded_ids": expanded_ids,
            "route_modes": list(ROUTE_MODES),
        },
        "micro_train": train_report,
        "forward": {
            "base_9_all": _real_population_forward(cortex, base_ids),
            "with_micro_10_all": _real_population_forward(cortex, expanded_ids),
        },
        "generation": {mode: {} for mode in ROUTE_MODES},
    }
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        for mode, active_ids in active_sets.items():
            text = _generate(cortex, active_ids, prompt, seed)
            report["generation"][mode][prompt] = {
                "text": text,
                "surface": _surface_metrics(text),
            }

    del cortex, micro, shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--hf-ratio", type=float, default=DEFAULT_HF_RATIO)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(steps=args.steps, hf_ratio=args.hf_ratio)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
