"""Controlled data A/B pilot for the 7.58M micro dialogue neuron.

The experiment keeps the production five dialogue neurons and the shared
embedding frozen.  It trains two fresh in-memory micro members from the same
random initialization:

* ``current_only``: the canonical local dialogue pool;
* ``current_plus_hf``: the same pool plus the audited HF candidate pool.

Both members are evaluated on the local holdout and the independent HF
holdout.  The HF member is then added in memory to the real 5-dialogue +
4-general population for a finite-forward/generation canary.  No checkpoint
or default loader configuration is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from neuroplex.loader import assemble_cortex
from neuroplex.resonance import ResonanceNeuron, get_domain_neuron_config
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import (
    batch_align_and_embed,
    build_position_alignment,
)
from scripts.training.diag_micro_dialogue_pilot import (
    BATCH_SIZE,
    LR,
    MAX_SEQ_LEN,
    _load_shared_embedding,
    _loss,
)
from scripts.training.diag_micro_population_canary import (
    PROMPTS,
    _generate,
    _real_population_forward,
    _surface_metrics,
)
from scripts.training.utils import (
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


SEED = 20260819
CURRENT_MAX_TEXTS = 100_000
HF_DIR = Path("data/hf_candidates/moss_003_dialogue")
DEFAULT_STEPS = 160


def _valid_first_token_position(answer_start: int, seq_len: int) -> int | None:
    """Return the logits position for the first answer token, if visible."""

    if answer_start <= 0 or answer_start >= seq_len:
        return None
    return answer_start - 1


def _encode_batch(texts, domain_sp, general_sp, shared):
    return batch_align_and_embed(
        texts,
        domain_sp,
        general_sp,
        shared,
        max_seq_len=MAX_SEQ_LEN,
        answer_marker=SFT_ANSWER_MARKER,
    )


def _load_pools(eval_cap: int = 0) -> dict[str, list[str]]:
    current = load_dialogue_texts_multi("data/simple_zh", max_texts=CURRENT_MAX_TEXTS)
    current_train, current_eval = split_train_eval(current, eval_ratio=0.05, seed=42)
    hf_train = load_dialogue_texts_multi(
        str(HF_DIR), filenames=["train.jsonl"], max_texts=100_000
    )
    hf_eval = load_dialogue_texts_multi(
        str(HF_DIR), filenames=["eval.jsonl"], max_texts=100_000
    )
    if eval_cap > 0:
        current_eval = current_eval[:eval_cap]
        hf_eval = hf_eval[:eval_cap]
    return {
        "current_train": current_train,
        "current_eval": current_eval,
        "hf_train": hf_train,
        "hf_eval": hf_eval,
    }


def _evaluate(neuron, texts, domain_sp, general_sp, shared) -> dict:
    neuron.eval()
    total_ce = 0.0
    total_tokens = 0
    ranks = []
    nlls = []
    top1 = 0
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            encoded = _encode_batch(batch_texts, domain_sp, general_sp, shared)
            loss, logits, valid = _loss(neuron, encoded)
            valid_tokens = int(valid.sum())
            total_ce += float(loss) * valid_tokens
            total_tokens += valid_tokens
            for row_index, text in enumerate(batch_texts):
                marker = text.find(SFT_ANSWER_MARKER)
                prompt = text[:marker + len(SFT_ANSWER_MARKER)]
                _, targets = build_position_alignment(text, domain_sp, general_sp)
                answer_start = len(general_sp.encode(prompt))
                logit_position = _valid_first_token_position(answer_start, logits.shape[1])
                if logit_position is None or answer_start >= len(targets):
                    continue
                target_id = int(targets[answer_start])
                if target_id < 0:
                    continue
                next_logits = logits[row_index, logit_position]
                target_logit = next_logits[target_id]
                ranks.append(int((next_logits > target_logit).sum()))
                nlls.append(float(-target_logit + torch.logsumexp(next_logits, dim=-1)))
                top1 += int(int(next_logits.argmax()) == target_id)
    mean_loss = total_ce / max(total_tokens, 1)
    ordered = sorted(ranks)
    return {
        "samples": len(texts),
        "effective_answer_tokens": total_tokens,
        "answer_loss": round(mean_loss, 6),
        "corrected_ppl": round(math.exp(min(mean_loss, 20)), 4),
        "first_token_nll": round(sum(nlls) / max(len(nlls), 1), 6),
        "median_rank_zero_based": ordered[len(ordered) // 2] if ordered else None,
        "first_token_top1": round(top1 / max(len(ranks), 1), 4),
    }


def _train_condition(
    name: str,
    train_texts: list[str],
    eval_sets: dict[str, list[str]],
    shared,
    domain_sp,
    general_sp,
    steps: int,
    return_neuron: bool = False,
):
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg = get_domain_neuron_config("zh", spec="micro")
    cfg.neuron_id = f"zh_micro_dialogue_{name}"
    neuron = ResonanceNeuron(cfg)
    local_params = sum(parameter.numel() for parameter in neuron.parameters())
    before = {
        key: _evaluate(neuron, texts, domain_sp, general_sp, shared)
        for key, texts in eval_sets.items()
    }
    optimizer = torch.optim.AdamW(neuron.parameters(), lr=LR, weight_decay=0.1)
    generator = torch.Generator().manual_seed(SEED)
    losses = []
    neuron.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(train_texts), (BATCH_SIZE,), generator=generator)
        batch_texts = [train_texts[int(index)] for index in indices]
        encoded = _encode_batch(batch_texts, domain_sp, general_sp, shared)
        optimizer.zero_grad()
        loss, _, _ = _loss(neuron, encoded)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neuron.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 40 == 0 or step == steps:
            print(f"[{name}] step {step}/{steps}: train_answer_loss={losses[-1]:.6f}", flush=True)
    after = {
        key: _evaluate(neuron, texts, domain_sp, general_sp, shared)
        for key, texts in eval_sets.items()
    }
    report = {
        "condition": name,
        "train_samples": len(train_texts),
        "steps": steps,
        "architecture": {
            "spec": cfg.spec,
            "local_params_m": round(local_params / 1_000_000, 6),
            "hidden_size": cfg.hidden_size,
            "layers": cfg.num_hidden_layers,
            "field_dim": cfg.field_dim,
            "shared_embedding_frozen": True,
        },
        "before": before,
        "after": after,
        "loss_trace": {
            "first": round(losses[0], 6),
            "last": round(losses[-1], 6),
            "min": round(min(losses), 6),
        },
    }
    del optimizer
    gc.collect()
    if return_neuron:
        return report, neuron
    del neuron
    return report, None


def _population_canary(micro, shared) -> dict:
    micro.eval()
    micro_id = "zh_micro_dialogue_ab"
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
    embeddings[micro_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    expanded_ids = base_ids + [micro_id]
    report = {
        "base_ids": base_ids,
        "expanded_ids": expanded_ids,
        "base_population": "5 dialogue + 4 general",
        "micro_added_via": "ResonanceEnsemble.add_neuron",
        "writes_checkpoint": False,
        "mixed_forward": _real_population_forward(cortex, expanded_ids),
        "generation": {},
    }
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        text = _generate(cortex, expanded_ids, prompt, seed)
        report["generation"][prompt] = {
            "text": text,
            "surface": _surface_metrics(text),
        }
    del cortex
    gc.collect()
    return report


def run(
    steps: int = DEFAULT_STEPS,
    include_population_canary: bool = True,
    eval_cap: int = 0,
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
    current_report, _ = _train_condition(
        "current_only",
        pools["current_train"],
        eval_sets,
        shared,
        domain_sp,
        general_sp,
        steps,
    )
    plus_report, plus_neuron = _train_condition(
        "current_plus_hf",
        pools["current_train"] + pools["hf_train"],
        eval_sets,
        shared,
        domain_sp,
        general_sp,
        steps,
        return_neuron=include_population_canary,
    )
    report = {
        "contract": {
            "seed": SEED,
            "steps_per_condition": steps,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "shared_embedding_frozen": True,
            "writes_checkpoint": False,
            "five_dialogue_checkpoints_untouched": True,
        },
        "data": {
            "current_train": len(pools["current_train"]),
            "current_eval": len(pools["current_eval"]),
            "hf_train": len(pools["hf_train"]),
            "hf_eval": len(pools["hf_eval"]),
            "eval_cap": eval_cap,
            "plus_train": len(pools["current_train"]) + len(pools["hf_train"]),
            "current_eval_hash_overlap_hf_eval": len(
                set(pools["current_eval"]) & set(pools["hf_eval"])
            ),
        },
        "conditions": {
            "current_only": current_report,
            "current_plus_hf": plus_report,
        },
    }
    if include_population_canary and plus_neuron is not None:
        report["population_canary"] = _population_canary(plus_neuron, shared)
        del plus_neuron
    del shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-population-canary", action="store_true")
    parser.add_argument("--eval-cap", type=int, default=0, help="debug cap per eval set; 0=all")
    args = parser.parse_args()
    report = run(
        steps=args.steps,
        include_population_canary=not args.skip_population_canary,
        eval_cap=args.eval_cap,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
