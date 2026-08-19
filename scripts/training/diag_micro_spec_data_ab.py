"""Same-budget data training sweep for smaller production neuron candidates.

The sweep keeps one frozen population shared embedding in memory and trains the
three validated small configurations on the same deterministic 90/10 current
dialogue + HF pool.  Each candidate gets a full current/HF holdout evaluation
and a temporary 9+1 population canary.  No checkpoint or production loader
configuration is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from neuroplex.resonance import NeuronConfig
from scripts.training.diag_micro_data_ab import (
    DEFAULT_HF_RATIO,
    MAX_SEQ_LEN,
    SEED,
    _load_pools,
    _load_shared_embedding,
    _population_canary,
    _select_hf_for_ratio,
    _train_condition,
)
from scripts.training.diag_micro_spec_sweep import CANDIDATES
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer


BASE_EMBED_DIM = 512
ZH_VOCAB_SIZE = 50_000
SCREEN_SPECS = (
    "micro_2x128",
    "micro_3x128",
    "micro_4x128_field512",
)


def _make_config(spec_name: str) -> NeuronConfig:
    values = dict(CANDIDATES[spec_name])
    return NeuronConfig(
        **values,
        vocab_size=ZH_VOCAB_SIZE,
        base_embed_dim=BASE_EMBED_DIM,
        spec=spec_name,
        neuron_id=f"zh_{spec_name}",
    )


def run(
    steps: int = 800,
    hf_ratio: float = DEFAULT_HF_RATIO,
    eval_cap: int = 0,
    specs: tuple[str, ...] = SCREEN_SPECS,
) -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    pools = _load_pools(eval_cap=eval_cap)
    shared = _load_shared_embedding()
    for parameter in shared.parameters():
        parameter.requires_grad = False
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    eval_sets = {
        "current_eval": pools["current_eval"],
        "hf_eval": pools["hf_eval"],
    }
    selected_hf_train = _select_hf_for_ratio(
        pools["current_train"], pools["hf_train"], hf_ratio
    )
    train_texts = pools["current_train"] + selected_hf_train
    results = {}
    for spec_name in specs:
        print(f"[{spec_name}] starting {steps} steps", flush=True)
        report, neuron = _train_condition(
            spec_name,
            train_texts,
            eval_sets,
            shared,
            domain_sp,
            general_sp,
            steps,
            return_neuron=True,
            neuron_config=_make_config(spec_name),
        )
        report["spec_name"] = spec_name
        report["population_canary"] = _population_canary(
            neuron,
            shared,
            micro_id=f"zh_{spec_name}",
        )
        results[spec_name] = report
        del neuron
        gc.collect()
    report = {
        "contract": {
            "seed": SEED,
            "specs": list(specs),
            "steps_per_spec": steps,
            "hf_ratio_requested": hf_ratio,
            "max_seq_len": MAX_SEQ_LEN,
            "shared_embedding_frozen": True,
            "shared_embedding_loaded_once": True,
            "writes_checkpoint": False,
            "production_population_untouched": True,
        },
        "data": {
            "current_train": len(pools["current_train"]),
            "current_eval": len(pools["current_eval"]),
            "hf_train": len(pools["hf_train"]),
            "hf_train_used": len(selected_hf_train),
            "hf_eval": len(pools["hf_eval"]),
            "eval_cap": eval_cap,
            "train_samples": len(train_texts),
            "current_fraction": round(
                len(pools["current_train"]) / max(len(train_texts), 1), 6
            ),
            "hf_fraction": round(
                len(selected_hf_train) / max(len(train_texts), 1), 6
            ),
        },
        "candidates": results,
    }
    del shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hf-ratio", type=float, default=DEFAULT_HF_RATIO)
    parser.add_argument("--eval-cap", type=int, default=0)
    parser.add_argument(
        "--spec",
        dest="specs",
        action="append",
        choices=SCREEN_SPECS,
        help="run only the selected validated candidate; repeat for multiple specs",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    selected_specs = tuple(args.specs) if args.specs else SCREEN_SPECS
    report = run(
        steps=args.steps,
        hf_ratio=args.hf_ratio,
        eval_cap=args.eval_cap,
        specs=selected_specs,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
