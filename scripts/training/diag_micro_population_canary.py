"""In-memory canary: append the trained micro member to the real 9-member population.

The script first runs the short micro dialogue pilot, then assembles the real
5 dialogue + 4 general production population, adds the pilot member through
``ResonanceEnsemble.add_neuron``, and compares fixed-prompt generation with
the same active-neuron set. No checkpoint or default loader configuration is
modified.
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

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from neuroplex.loader import assemble_cortex
from scripts.training.diag_micro_dialogue_pilot import run as run_micro_pilot


PROMPTS = [
    "问：你好，请介绍一下自己\n答：",
    "问：什么是人工智能？\n答：",
    "问：今天天气怎么样？\n答：",
    "问：请写一首关于春天的短诗。\n答：",
]
SEED = 20260819
MAX_TOKENS = 12
DEFAULT_PILOT_STEPS = 160


def _surface_metrics(text: str) -> dict:
    chars = list(text)
    bigrams = ["".join(chars[i:i + 2]) for i in range(max(0, len(chars) - 1))]
    repeated = len(bigrams) - len(set(bigrams))
    return {
        "chars": len(chars),
        "unique_char_ratio": round(len(set(chars)) / max(len(chars), 1), 4),
        "repeated_bigram_ratio": round(repeated / max(len(bigrams), 1), 4),
        "has_decode_artifact": any(token in text for token in ("<0x", "�", "▁")),
    }


def _generate(cortex, active_ids: list[str], prompt: str, seed: int) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    return cortex.generate(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.55,
        top_k=15,
        domain="zh",
        repetition_penalty=1.4,
        active_nids=active_ids,
        collab_mode="fusion",
        fusion_mode="soft",
        auto_memory=False,
        instance_routing=False,
    )


def _real_population_forward(cortex, active_ids: list[str]) -> dict:
    general_ids = cortex._general_sp.encode(PROMPTS[0])
    shared = cortex._shared_embedding(torch.tensor([general_ids], dtype=torch.long))
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        result = cortex.ensemble.forward(
            shared_embeddings=shared,
            active_nids=active_ids,
            active_filter=False,
            return_logits=False,
        )
    field_state = result["field_state"]
    return {
        "field_shape": list(field_state.shape),
        "finite": bool(torch.isfinite(field_state).all()),
        "rounds": result["n_rounds"],
    }


def run(pilot_steps: int = DEFAULT_PILOT_STEPS) -> dict:
    logging.disable(logging.CRITICAL)
    pilot_report, micro, micro_shared, _, _ = run_micro_pilot(
        return_state=True,
        steps=pilot_steps,
    )
    micro.eval()
    micro_id = "zh_micro_dialogue"
    micro.config.neuron_id = micro_id

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

    cortex.ensemble.add_neuron(micro_id, micro)
    embeddings = dict(cortex._neuron_shared_embeddings or {})
    embeddings[micro_id] = micro_shared
    cortex.set_neuron_shared_embeddings(embeddings)
    expanded_ids = base_ids + [micro_id]

    report = {
        "contract": {
            "base_population": "5 dialogue + 4 general",
            "base_ids": base_ids,
            "expanded_ids": expanded_ids,
            "micro_added_via": "ResonanceEnsemble.add_neuron",
            "writes_checkpoint": False,
            "changes_default_loader": False,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
        },
        "micro_pilot": pilot_report,
        "mixed_forward": _real_population_forward(cortex, expanded_ids),
        "generation": {"base_9": {}, "with_micro_10": {}},
    }
    for index, prompt in enumerate(PROMPTS):
        seed = SEED + index
        base_text = _generate(cortex, base_ids, prompt, seed)
        micro_text = _generate(cortex, expanded_ids, prompt, seed)
        report["generation"]["base_9"][prompt] = {
            "text": base_text,
            "surface": _surface_metrics(base_text),
        }
        report["generation"]["with_micro_10"][prompt] = {
            "text": micro_text,
            "surface": _surface_metrics(micro_text),
        }

    del cortex, micro_shared, micro
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--pilot-steps", type=int, default=DEFAULT_PILOT_STEPS)
    args = parser.parse_args()
    report = run(pilot_steps=args.pilot_steps)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
