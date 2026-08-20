"""Fresh-process reload canary for a three-member micro specialist group.

The loader reads only the temporary specialist checkpoints, reuses the canonical
frozen shared embedding once, and then evaluates the reloaded three-member
ensemble on holdout data plus fixed-prompt generation.  It never touches the
production nine-member loader configuration or writes production checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.loader import assemble_cortex
from scripts.training.diag_micro_data_ab import (
    MAX_SEQ_LEN,
    _evaluate,
    _load_pools,
    _load_shared_embedding,
    _real_population_forward,
)
from scripts.training.diag_micro_population_canary import PROMPTS, _generate, _surface_metrics
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer


SPECIALIST_ROLES = ("current_only", "hf_only", "current_plus_hf_10")
SPECIALIST_IDS = [f"zh_micro_specialist_{role}" for role in SPECIALIST_ROLES]


def _file_manifest(checkpoint_dir: Path) -> list[dict]:
    manifest = []
    for neuron_id in SPECIALIST_IDS:
        path = checkpoint_dir / f"neuron_{neuron_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing specialist checkpoint: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        })
    return manifest


def run(checkpoint_dir: Path, eval_cap: int = 0) -> dict:
    logging.disable(logging.CRITICAL)
    pools = _load_pools(eval_cap=eval_cap)
    shared = _load_shared_embedding()
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    manifest = _file_manifest(checkpoint_dir)

    with contextlib.redirect_stdout(io.StringIO()):
        cortex, _, _ = assemble_cortex(
            neurons_dir=str(checkpoint_dir),
            extra_neurons_dir=None,
            collab_name="",
            device="cpu",
            max_rounds=3,
            wire_bio_modules=False,
            neuron_ids=SPECIALIST_IDS,
        )

    loader_ids = list(cortex.neurons.keys())
    if set(loader_ids) != set(SPECIALIST_IDS):
        raise RuntimeError(f"reload population mismatch: {loader_ids}")
    # The filesystem loader is intentionally name-sorted; restore the
    # experiment's role order for stable metrics and report comparison.
    loaded_ids = [neuron_id for neuron_id in SPECIALIST_IDS if neuron_id in cortex.neurons]
    # The generic loader may prefer data/shared_embedding.pt.  The training
    # contract for this experiment uses the frozen table from the canonical
    # dialogue checkpoint, so override both shared-embedding paths explicitly.
    cortex.set_shared_embedding(shared)
    cortex.set_neuron_shared_embeddings({neuron_id: shared for neuron_id in SPECIALIST_IDS})
    shared_loaded = shared
    for neuron in cortex.neurons.values():
        neuron.eval()

    eval_sets = {
        "current_eval": pools["current_eval"],
        "hf_eval": pools["hf_eval"],
    }
    members = {}
    for neuron_id in SPECIALIST_IDS:
        neuron = cortex.neurons[neuron_id]
        members[neuron_id] = {
            "architecture": {
                "spec": neuron.config.spec,
                "local_params_m": round(
                    sum(parameter.numel() for parameter in neuron.parameters()) / 1_000_000,
                    6,
                ),
                "hidden_size": neuron.config.hidden_size,
                "layers": neuron.config.num_hidden_layers,
                "field_dim": neuron.config.field_dim,
            },
            "after_reload": {
                key: _evaluate(neuron, texts, domain_sp, general_sp, shared_loaded)
                for key, texts in eval_sets.items()
            },
        }

    forward = _real_population_forward(cortex, SPECIALIST_IDS)
    generation = {}
    for index, prompt in enumerate(PROMPTS):
        text = _generate(cortex, SPECIALIST_IDS, prompt, 20260819 + index)
        generation[prompt] = {
            "text": text,
            "surface": _surface_metrics(text),
        }

    report = {
        "contract": {
            "mode": "fresh_process_reload_only_three",
            "specialist_ids": SPECIALIST_IDS,
            "max_seq_len": MAX_SEQ_LEN,
            "shared_embedding_loaded_once": True,
            "production_population_untouched": True,
            "writes_checkpoint": False,
        },
        "checkpoint": {
            "directory_name": checkpoint_dir.name,
            "files": manifest,
        },
        "data": {
            "current_eval": len(pools["current_eval"]),
            "hf_eval": len(pools["hf_eval"]),
            "eval_cap": eval_cap,
        },
        "loaded_population": {
            "loaded_ids": loaded_ids,
            "member_count": len(loaded_ids),
            "field_dim": cortex.ensemble._field.dim,
            "shared_embedding_source": "canonical_base_checkpoint",
            "loader_default_embedding_overridden": True,
        },
        "members": members,
        "forward": forward,
        "generation_prompt_count": len(PROMPTS),
        "generation": generation,
    }
    del cortex, shared, shared_loaded
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-cap", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.checkpoint_dir, eval_cap=args.eval_cap)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
