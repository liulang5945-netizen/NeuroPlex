"""Scan small production ResonanceNeuron candidates before training.

This diagnostic deliberately does not add a fixed ``micro`` production spec.
The historical ~10M TinyStories model used tied embeddings and a different
token path, so the current population contract must be measured directly.

The scan reports local neuron parameters (including the domain lm_head) while
counting the 256K x 512 shared sensory embedding once at population level. It
also runs a real mixed-spec ResonanceEnsemble forward with a compact neuron,
which is the first compatibility gate for adding a small member beside the
existing population.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import torch

from neuroplex.resonance import (
    NeuronConfig,
    ResonanceEnsemble,
    ResonanceField,
    ResonanceNeuron,
    get_domain_neuron_config,
)


GENERAL_VOCAB = 256_000
BASE_EMBED_DIM = 512


# The candidates preserve the production Transformer/field path while varying
# the capacity. Field dimensions are intentionally smaller than the existing
# 2048/3072/4096 members; the ensemble must prove the cross-spec projection.
CANDIDATES = {
    "micro_2x128": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=384, field_dim=256),
    "micro_3x128": dict(hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=384, field_dim=256),
    "micro_4x128": dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=384, field_dim=256),
    "micro_4x128_field512": dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                                  num_key_value_heads=1, intermediate_size=384, field_dim=512),
    "micro_3x160": dict(hidden_size=160, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=480, field_dim=320),
    "micro_4x160": dict(hidden_size=160, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=480, field_dim=320),
    "micro_4x160_field640": dict(hidden_size=160, num_hidden_layers=4, num_attention_heads=4,
                                  num_key_value_heads=1, intermediate_size=480, field_dim=640),
    "micro_4x192": dict(hidden_size=192, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=1, intermediate_size=576, field_dim=384),
}


def _make_config(spec_name: str) -> NeuronConfig:
    cfg = NeuronConfig(
        **CANDIDATES[spec_name],
        vocab_size=50_000,
        base_embed_dim=BASE_EMBED_DIM,
        spec=spec_name,
        neuron_id=f"zh_{spec_name}",
    )
    return cfg


def _param_breakdown(neuron: ResonanceNeuron) -> dict[str, float]:
    groups = {"lm_head": 0, "body": 0, "field_and_interface": 0}
    for name, param in neuron.named_parameters():
        count = param.numel()
        if name.startswith("lm_head"):
            groups["lm_head"] += count
        elif name.startswith("layers.") or name.startswith("norm."):
            groups["body"] += count
        else:
            groups["field_and_interface"] += count
    return {key: value / 1_000_000 for key, value in groups.items()}


def _run_forward(neuron: ResonanceNeuron) -> dict:
    x = torch.randn(1, 8, BASE_EMBED_DIM)
    with torch.no_grad():
        out = neuron(x, return_logits=True)
    logits = out["logits"]
    field = out["field_vector"]
    return {
        "logits_shape": list(logits.shape),
        "field_shape": list(field.shape),
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(field).all()),
    }


def _run_mixed_population(candidate: ResonanceNeuron) -> dict:
    compact_cfg = get_domain_neuron_config("zh", spec="compact")
    compact_cfg.neuron_id = "zh_compact_probe"
    compact = ResonanceNeuron(compact_cfg)
    candidate.config.neuron_id = "zh_micro_probe"
    neurons = {"zh_compact_probe": compact, "zh_micro_probe": candidate}
    field = ResonanceField(dim=max(compact_cfg.field_dim, candidate.config.field_dim))
    ensemble = ResonanceEnsemble(neurons=neurons, field=field, max_rounds=2)
    embeddings = torch.randn(1, 8, BASE_EMBED_DIM)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        result = ensemble.forward(
            shared_embeddings=embeddings,
            return_logits=False,
            active_filter=False,
        )
    state = result["field_state"]
    return {
        "field_dim": field.dim,
        "forward_finite": bool(torch.isfinite(state).all()),
        "cross_spec_forward": "zh_micro_probe" in ensemble._cross_spec_projectors,
        "cross_spec_backward": "zh_micro_probe" in ensemble._cross_spec_back_projectors,
        "rounds": result["n_rounds"],
    }


def run() -> dict:
    rows = []
    for name in CANDIDATES:
        cfg = _make_config(name)
        neuron = ResonanceNeuron(cfg)
        local_params = sum(param.numel() for param in neuron.parameters())
        rows.append({
            "name": name,
            "config": {
                key: getattr(cfg, key)
                for key in (
                    "hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "intermediate_size", "field_dim",
                    "vocab_size", "base_embed_dim",
                )
            },
            "approx_local_params_m": cfg.approx_params_m,
            "actual_local_params_m": local_params / 1_000_000,
            "population_shared_embedding_m": GENERAL_VOCAB * BASE_EMBED_DIM / 1_000_000,
            "breakdown_m": _param_breakdown(neuron),
            "single_forward": _run_forward(neuron),
            "mixed_compact_forward": _run_mixed_population(neuron),
        })
        del neuron
    return {
        "contract": "production_resonance_neuron_micro_sweep",
        "shared_embedding_counted_once": True,
        "candidate_count": len(rows),
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
