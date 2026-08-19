"""Temporary three-member 6.97M specialist group experiment.

The experiment trains three fresh 6.974593M members against different data
roles (current-only, HF-only, and 90/10 mixed), while loading the frozen
population shared embedding once.  The members are evaluated individually and
then added together to the real 9-member population for a finite-forward and
generation canary.  No checkpoint or production loader configuration changes.
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
from neuroplex.resonance import NeuronConfig
from neuroplex.loader import assemble_cortex
from scripts.training.diag_micro_data_ab import (
    DEFAULT_HF_RATIO,
    MAX_SEQ_LEN,
    SEED,
    _load_pools,
    _load_shared_embedding,
    _real_population_forward,
    _select_hf_for_ratio,
    _surface_metrics,
    _train_condition,
)
from scripts.training.diag_micro_spec_sweep import CANDIDATES
from scripts.training.diag_micro_population_canary import PROMPTS, _generate
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer


SPECIALIST_ROLES = ("current_only", "hf_only", "current_plus_hf_10")
MICRO_SPEC = "micro_2x128"
BASE_EMBED_DIM = 512
ZH_VOCAB_SIZE = 50_000


def _make_config(neuron_id: str) -> NeuronConfig:
    return NeuronConfig(
        **dict(CANDIDATES[MICRO_SPEC]),
        vocab_size=ZH_VOCAB_SIZE,
        base_embed_dim=BASE_EMBED_DIM,
        spec=MICRO_SPEC,
        neuron_id=neuron_id,
    )


def _population_canary_multi(neurons: dict[str, torch.nn.Module], shared) -> dict:
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
    embeddings = dict(cortex._neuron_shared_embeddings or {})
    for neuron_id, neuron in neurons.items():
        neuron.eval()
        neuron.config.neuron_id = neuron_id
        cortex.ensemble.add_neuron(neuron_id, neuron)
        embeddings[neuron_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    expanded_ids = base_ids + list(neurons)
    report = {
        "base_ids": base_ids,
        "expanded_ids": expanded_ids,
        "specialist_ids": list(neurons),
        "base_population": "5 dialogue + 4 general",
        "micro_added_via": "ResonanceEnsemble.add_neuron",
        "writes_checkpoint": False,
        "base_forward": _real_population_forward(cortex, base_ids),
        "mixed_forward": _real_population_forward(cortex, expanded_ids),
        "generation": {"base_9": {}, "with_specialists_12": {}},
    }
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        base_text = _generate(cortex, base_ids, prompt, seed)
        mixed_text = _generate(cortex, expanded_ids, prompt, seed)
        report["generation"]["base_9"][prompt] = {
            "text": base_text,
            "surface": _surface_metrics(base_text),
        }
        report["generation"]["with_specialists_12"][prompt] = {
            "text": mixed_text,
            "surface": _surface_metrics(mixed_text),
        }
    del cortex
    gc.collect()
    return report


def run(steps: int = 800, hf_ratio: float = DEFAULT_HF_RATIO, eval_cap: int = 0) -> dict:
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
    train_sets = {
        "current_only": pools["current_train"],
        "hf_only": pools["hf_train"],
        "current_plus_hf_10": pools["current_train"] + selected_hf_train,
    }
    results = {}
    neurons = {}
    for role in SPECIALIST_ROLES:
        neuron_id = f"zh_micro_specialist_{role}"
        print(f"[{role}] starting {steps} steps", flush=True)
        report, neuron = _train_condition(
            role,
            train_sets[role],
            eval_sets,
            shared,
            domain_sp,
            general_sp,
            steps,
            return_neuron=True,
            neuron_config=_make_config(neuron_id),
        )
        results[role] = report
        neurons[neuron_id] = neuron
    group_canary = _population_canary_multi(neurons, shared)
    del neurons, shared
    gc.collect()
    return {
        "contract": {
            "seed": SEED,
            "micro_spec": MICRO_SPEC,
            "specialist_roles": list(SPECIALIST_ROLES),
            "steps_per_member": steps,
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
            "hf_train_used_for_mix": len(selected_hf_train),
            "hf_eval": len(pools["hf_eval"]),
            "eval_cap": eval_cap,
            "mix_train_samples": len(train_sets["current_plus_hf_10"]),
        },
        "members": results,
        "population_canary": group_canary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hf-ratio", type=float, default=DEFAULT_HF_RATIO)
    parser.add_argument("--eval-cap", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(steps=args.steps, hf_ratio=args.hf_ratio, eval_cap=args.eval_cap)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
