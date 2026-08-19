"""Read-only route/contribution audit for three trained 6.97M specialists.

The script recreates the temporary current-only, HF-only, and 90/10 members,
fits an independent external route projection for each, and measures route
rank, projected field magnitude, and generation under base/all/top-k active
sets.  The language body and original embed_adapter are frozen during route
fit.  Nothing is attached to a production checkpoint or loader.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from neuroplex.loader import assemble_cortex
from scripts.training.diag_micro_data_ab import (
    DEFAULT_HF_RATIO,
    MAX_SEQ_LEN,
    SEED,
    _load_pools,
    _load_shared_embedding,
    _select_hf_for_ratio,
    _train_condition,
)
from scripts.training.diag_micro_external_route import _fit_external_route_adapter
from scripts.training.diag_micro_population_canary import (
    PROMPTS,
    _generate,
    _surface_metrics,
)
from scripts.training.diag_micro_spec_sweep import CANDIDATES
from scripts.training.diag_micro_specialist_group import (
    MICRO_SPEC,
    SPECIALIST_ROLES,
    _make_config,
)
from scripts.training.diag_micro_dialogue_pilot import BATCH_SIZE
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer


DEFAULT_EVAL_CAP = 512
TOP_K = 3


def _assemble_population(neurons: dict[str, torch.nn.Module], shared):
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
    if set(DEFAULT_NEURON_IDS) - set(base_ids) or not expected_general.issubset(
        set(base_ids) - set(DEFAULT_NEURON_IDS)
    ):
        raise RuntimeError(f"real production population mismatch: {base_ids}")
    embeddings = dict(cortex._neuron_shared_embeddings or {})
    for neuron_id, neuron in neurons.items():
        neuron.eval()
        neuron.config.neuron_id = neuron_id
        cortex.ensemble.add_neuron(neuron_id, neuron)
        embeddings[neuron_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    return cortex, base_ids, base_ids + list(neurons)


def _route_scores(cortex, route_params: dict[str, tuple[torch.nn.Module, torch.Tensor]], prompt: str):
    ids = cortex._general_sp.encode(prompt)
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=cortex.device)
    pooled = cortex._shared_embedding(prompt_ids).mean(dim=1)
    scores = {}
    for neuron_id, neuron in cortex.neurons.items():
        if neuron_id in route_params:
            adapter, prototype = route_params[neuron_id]
            projected = adapter(pooled).squeeze(0)
        else:
            projected = neuron.embed_adapter(pooled).squeeze(0)
            prototype = neuron.domain_prototype
        if prototype is None or prototype.norm() < 1e-6:
            scores[neuron_id] = 0.0
        else:
            scores[neuron_id] = float(
                F.cosine_similarity(
                    projected.unsqueeze(0), prototype.unsqueeze(0), dim=-1
                ).item()
            )
    return scores


def _field_contributions(cortex, active_ids: list[str], prompt: str) -> dict:
    ids = cortex._general_sp.encode(prompt)
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=cortex.device)
    shared = cortex._shared_embedding(prompt_ids)
    result = {}
    with torch.no_grad():
        for neuron_id in active_ids:
            raw = cortex.neurons[neuron_id](shared, return_logits=False)["field_vector"]
            projected = cortex.ensemble._project_vec(neuron_id, raw)
            result[neuron_id] = {
                "raw_field_dim": int(raw.shape[-1]),
                "raw_field_norm": round(float(raw.norm().item()), 6),
                "projected_field_dim": int(projected.shape[-1]),
                "projected_field_norm": round(float(projected.norm().item()), 6),
                "projected_field_mean_abs": round(float(projected.abs().mean().item()), 6),
            }
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble_result = cortex.ensemble.forward(
                shared_embeddings=shared,
                active_nids=active_ids,
                active_filter=False,
                return_logits=False,
            )
        result["__ensemble__"] = {
            "field_state_norm": round(float(ensemble_result["field_state"].norm().item()), 6),
            "rounds": ensemble_result["n_rounds"],
        }
    return result


def _generation_and_routes(cortex, base_ids, expanded_ids, route_params):
    generation = {}
    routes = {}
    fields = {}
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        scores = _route_scores(cortex, route_params, prompt)
        ordered = sorted(scores, key=scores.get, reverse=True)
        active_sets = {
            "base_9": base_ids,
            "all_12": expanded_ids,
            "external_top1": ordered[:1],
            "external_top3": ordered[:TOP_K],
        }
        routes[prompt] = {
            "scores": scores,
            "ordered_ids": ordered,
            "specialist_ranks_zero_based": {
                neuron_id: ordered.index(neuron_id)
                for neuron_id in route_params
            },
            "top1": ordered[:1],
            "top3": ordered[:TOP_K],
        }
        fields[prompt] = _field_contributions(cortex, expanded_ids, prompt)
        generation[prompt] = {}
        for mode, active_ids in active_sets.items():
            text = _generate(cortex, active_ids, prompt, seed)
            generation[prompt][mode] = {
                "active_ids": list(active_ids),
                "text": text,
                "surface": _surface_metrics(text),
            }
    return routes, fields, generation


def run(steps: int = 800, hf_ratio: float = DEFAULT_HF_RATIO, eval_cap: int = DEFAULT_EVAL_CAP):
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    pools = _load_pools(eval_cap=eval_cap)
    shared = _load_shared_embedding()
    for parameter in shared.parameters():
        parameter.requires_grad = False
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    selected_hf = _select_hf_for_ratio(pools["current_train"], pools["hf_train"], hf_ratio)
    train_sets = {
        "current_only": pools["current_train"],
        "hf_only": pools["hf_train"],
        "current_plus_hf_10": pools["current_train"] + selected_hf,
    }
    eval_sets = {
        "current_eval": pools["current_eval"],
        "hf_eval": pools["hf_eval"],
    }
    neurons = {}
    members = {}
    for role in SPECIALIST_ROLES:
        neuron_id = f"zh_micro_specialist_{role}"
        print(f"[{role}] recreating {steps} steps for route audit", flush=True)
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
        members[role] = report
        neurons[neuron_id] = neuron

    cortex, base_ids, expanded_ids = _assemble_population(neurons, shared)
    route_params = {}
    calibrations = {}
    for role in SPECIALIST_ROLES:
        neuron_id = f"zh_micro_specialist_{role}"
        adapter, prototype, calibration = _fit_external_route_adapter(
            neurons[neuron_id],
            pools["current_train"],
            domain_sp,
            general_sp,
            shared,
        )
        route_params[neuron_id] = (adapter, prototype)
        calibrations[neuron_id] = calibration
    routes, fields, generation = _generation_and_routes(
        cortex, base_ids, expanded_ids, route_params
    )
    report = {
        "contract": {
            "seed": SEED,
            "micro_spec": MICRO_SPEC,
            "specialist_roles": list(SPECIALIST_ROLES),
            "steps_per_member": steps,
            "eval_cap": eval_cap,
            "shared_embedding_frozen": True,
            "shared_embedding_loaded_once": True,
            "language_body_frozen_during_route_fit": True,
            "original_embed_adapter_frozen_during_route_fit": True,
            "writes_checkpoint": False,
            "production_population_untouched": True,
            "base_ids": base_ids,
            "expanded_ids": expanded_ids,
        },
        "route_calibration": calibrations,
        "members": members,
        "routes": routes,
        "field_contributions": fields,
        "generation": generation,
    }
    del cortex, neurons, shared, route_params
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hf-ratio", type=float, default=DEFAULT_HF_RATIO)
    parser.add_argument("--eval-cap", type=int, default=DEFAULT_EVAL_CAP)
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
