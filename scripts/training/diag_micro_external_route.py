"""Experimental external route projection for the 7.58M micro member.

The language-facing ``embed_adapter`` is frozen.  A separate small projection
is fitted only to rank the micro member against the existing population.  The
projection is diagnostic-only and is never attached to production neurons or
written to a checkpoint.
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
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from scripts.training.diag_micro_data_ab import (
    DEFAULT_STEPS,
    SEED,
    _encode_batch,
    _evaluate,
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
from scripts.training.diag_micro_route_canary import _assemble_with_micro
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer
from neuroplex.loader import assemble_cortex


DEFAULT_HF_RATIO = 0.10
CALIBRATION_SAMPLES = 256
ALIGNMENT_EPOCHS = 2
DEFAULT_EVAL_CAP = 512
EXTERNAL_ROUTE_MODES = (
    "base_9_all",
    "with_micro_10_all",
    "with_micro_external_top1",
    "with_micro_external_top2",
)


def _fit_external_route_adapter(micro, texts, domain_sp, general_sp, shared):
    """Fit a route-only projection while leaving every micro parameter untouched."""

    torch.manual_seed(SEED + 9901)
    adapter = nn.Linear(shared.embedding_dim, micro.config.hidden_size, bias=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-4)
    selected = texts[:CALIBRATION_SAMPLES]
    losses = []
    for _ in range(ALIGNMENT_EPOCHS):
        for start in range(0, len(selected), BATCH_SIZE):
            batch = selected[start:start + BATCH_SIZE]
            embeddings = _encode_batch(batch, domain_sp, general_sp, shared)[0]
            with torch.no_grad():
                target = micro(embeddings, return_logits=False)["hidden_before_write"]
                target = F.normalize(target, dim=-1)
            projected = adapter(embeddings.mean(dim=1))
            loss = (1.0 - F.cosine_similarity(projected, target, dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

    targets = []
    with torch.no_grad():
        for start in range(0, len(selected), BATCH_SIZE):
            batch = selected[start:start + BATCH_SIZE]
            embeddings = _encode_batch(batch, domain_sp, general_sp, shared)[0]
            hidden = micro(embeddings, return_logits=False)["hidden_before_write"]
            targets.append(hidden)
    prototype = F.normalize(torch.cat(targets, dim=0).mean(dim=0), dim=0)
    return adapter, prototype, {
        "samples": len(selected),
        "epochs": ALIGNMENT_EPOCHS,
        "first_loss": round(losses[0], 6) if losses else None,
        "last_loss": round(losses[-1], 6) if losses else None,
        "projection_params": sum(p.numel() for p in adapter.parameters()),
        "prototype_norm": round(float(prototype.norm().item()), 6),
    }


def _external_route_ids(cortex, route_adapter, micro_prototype, prompt, top_k):
    general_ids = cortex._general_sp.encode(prompt)
    prompt_ids = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
    pooled = cortex._shared_embedding(prompt_ids).mean(dim=1)
    scores = {}
    for nid, neuron in cortex.neurons.items():
        if nid == "zh_micro_dialogue_ab":
            projected = route_adapter(pooled).squeeze(0)
            prototype = micro_prototype
        else:
            projected = neuron.embed_adapter(pooled).squeeze(0)
            prototype = neuron.domain_prototype
        if prototype.norm() < 1e-6:
            # Keep cold-start members in the ranking with neutral score, matching
            # the production auto-top-k behavior instead of silently dropping them.
            scores[nid] = 0.0
        else:
            scores[nid] = float(F.cosine_similarity(
                projected.unsqueeze(0), prototype.unsqueeze(0), dim=-1
            ).item())
    ordered = sorted(scores, key=scores.get, reverse=True)
    return ordered[:top_k], scores


def _generation_snapshot(cortex, base_ids, expanded_ids, route_adapter, prototype):
    generation = {mode: {} for mode in EXTERNAL_ROUTE_MODES}
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        route1, scores1 = _external_route_ids(cortex, route_adapter, prototype, prompt, 1)
        route2, scores2 = _external_route_ids(cortex, route_adapter, prototype, prompt, 2)
        active_sets = {
            "base_9_all": base_ids,
            "with_micro_10_all": expanded_ids,
            "with_micro_external_top1": route1,
            "with_micro_external_top2": route2,
        }
        route_scores = {
            "with_micro_external_top1": scores1,
            "with_micro_external_top2": scores2,
        }
        for mode, active_ids in active_sets.items():
            text = _generate(cortex, active_ids, prompt, seed)
            generation[mode][prompt] = {
                "active_ids": list(active_ids),
                "route_scores": route_scores.get(mode),
                "text": text,
                "surface": _surface_metrics(text),
            }
    return generation


def run(
    steps: int = DEFAULT_STEPS,
    hf_ratio: float = DEFAULT_HF_RATIO,
    eval_cap: int = DEFAULT_EVAL_CAP,
) -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    pools = _load_pools()
    selected_hf = _select_hf_for_ratio(pools["current_train"], pools["hf_train"], hf_ratio)
    shared = _load_shared_embedding()
    for parameter in shared.parameters():
        parameter.requires_grad = False
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    train_report, micro = _train_condition(
        "current_plus_hf_10",
        pools["current_train"] + selected_hf,
        {},
        shared,
        domain_sp,
        general_sp,
        steps,
        return_neuron=True,
    )
    micro.eval()

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
    micro_id = "zh_micro_dialogue_ab"
    cortex.ensemble.add_neuron(micro_id, micro)
    embeddings = dict(cortex._neuron_shared_embeddings or {})
    embeddings[micro_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    expanded_ids = base_ids + [micro_id]

    eval_sets = {
        "current_eval": pools["current_eval"][:eval_cap],
        "hf_eval": pools["hf_eval"][:eval_cap],
    }
    before_eval = {
        key: _evaluate(micro, texts, domain_sp, general_sp, shared)
        for key, texts in eval_sets.items()
    }
    route_adapter, route_prototype, calibration = _fit_external_route_adapter(
        micro, pools["current_train"], domain_sp, general_sp, shared
    )
    after_eval = {
        key: _evaluate(micro, texts, domain_sp, general_sp, shared)
        for key, texts in eval_sets.items()
    }
    report = {
        "contract": {
            "seed": SEED,
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "hf_ratio": hf_ratio,
            "hf_train_used": len(selected_hf),
            "eval_cap": eval_cap,
            "shared_embedding_frozen": True,
            "micro_language_body_frozen_during_route_fit": True,
            "micro_embed_adapter_frozen_during_route_fit": True,
            "writes_checkpoint": False,
            "base_population": "5 dialogue + 4 general",
            "base_ids": base_ids,
            "expanded_ids": expanded_ids,
            "route_modes": list(EXTERNAL_ROUTE_MODES),
        },
        "micro_train": train_report,
        "route_calibration": calibration,
        "micro_regression_screen": {
            "before_external_route_fit": before_eval,
            "after_external_route_fit": after_eval,
        },
        "forward": {
            "base_9_all": _real_population_forward(cortex, base_ids),
            "with_micro_10_all": _real_population_forward(cortex, expanded_ids),
        },
        "generation": _generation_snapshot(
            cortex, base_ids, expanded_ids, route_adapter, route_prototype
        ),
    }
    del cortex, micro, shared, route_adapter, route_prototype
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
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
