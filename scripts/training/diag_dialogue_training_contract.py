#!/usr/bin/env python3
"""P1.2：核对 dialogue fine-tune 的数据与监督目标契约。

该脚本复现 ``finetune_neuron_dialogue.py`` 的默认数据口径，不执行训练：

* ``max_texts=100000``，多文件合并后去重；
* hash holdout 5%，实际验证集取前 100 条；
* ``max_seq_len=128``，首个 ``答：`` 之后才计算 SFT loss；
* 检查截断、answer mask、对齐目标和 checkpoint 元数据。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_training_contract.py
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import build_position_alignment
from scripts.training.utils import load_dialogue_texts_multi, split_train_eval


CHECKPOINT_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
DATA_DIR = "data/simple_zh"
MAX_TEXTS = 100000
MAX_SEQ_LEN = 128
EVAL_RATIO = 0.05
EVAL_CAP = 100


def _training_shape(text: str, domain_sp, general_sp) -> dict:
    """Replicate the token/mask portion of batch_align_and_embed."""
    general_ids, domain_targets = build_position_alignment(text, domain_sp, general_sp)
    try:
        general_eos = general_sp.eos_id()
    except Exception:
        general_eos = 1
    try:
        domain_eos = domain_sp.eos_id()
    except Exception:
        domain_eos = 1

    original_len = len(general_ids) + 1
    general_ids = torch.cat([general_ids, torch.tensor([general_eos])])
    domain_targets = torch.cat([domain_targets, torch.tensor([domain_eos])])
    truncated = original_len > MAX_SEQ_LEN
    if truncated:
        general_ids = torch.cat([general_ids[: MAX_SEQ_LEN - 1], torch.tensor([general_eos])])
        domain_targets = torch.cat([domain_targets[: MAX_SEQ_LEN - 1], torch.tensor([domain_eos])])

    marker_idx = text.find(SFT_ANSWER_MARKER)
    prefix = text[: marker_idx + len(SFT_ANSWER_MARKER)] if marker_idx >= 0 else ""
    prefix_len = len(general_sp.encode(prefix)) if prefix else 0
    answer_start = min(prefix_len, MAX_SEQ_LEN) if marker_idx >= 0 else 0
    sequence_len = len(general_ids)

    # The training loop shifts positions by one before applying the mask.
    shifted_positions = max(sequence_len - 1, 0)
    answer_positions = max(sequence_len - max(answer_start, 1), 0)
    aligned_answer_positions = int(
        ((domain_targets[1:sequence_len] >= 0)
         & (torch.arange(1, sequence_len) >= answer_start)).sum()
    )
    return {
        "sequence_len": sequence_len,
        "prefix_len": prefix_len,
        "answer_start": answer_start,
        "shifted_positions": shifted_positions,
        "answer_mask_positions": answer_positions,
        "aligned_answer_positions": aligned_answer_positions,
        "unaligned_answer_positions": answer_positions - aligned_answer_positions,
        "truncated": truncated,
        "marker_count": text.count(SFT_ANSWER_MARKER),
    }


def _aggregate(texts: list[str], domain_sp, general_sp) -> dict:
    rows = [_training_shape(text, domain_sp, general_sp) for text in texts]
    total_shifted = sum(row["shifted_positions"] for row in rows)
    total_answer = sum(row["answer_mask_positions"] for row in rows)
    total_aligned = sum(row["aligned_answer_positions"] for row in rows)
    return {
        "samples": len(rows),
        "truncated_samples": sum(row["truncated"] for row in rows),
        "multi_marker_samples": sum(row["marker_count"] > 1 for row in rows),
        "zero_answer_mask_samples": sum(row["answer_mask_positions"] == 0 for row in rows),
        "shifted_positions": total_shifted,
        "answer_mask_positions": total_answer,
        "aligned_answer_positions": total_aligned,
        "unaligned_answer_positions": sum(row["unaligned_answer_positions"] for row in rows),
        "answer_mask_fraction": round(total_answer / max(total_shifted, 1), 6),
        "alignment_coverage": round(total_aligned / max(total_answer, 1), 6),
        "mean_answer_mask_positions": round(total_answer / max(len(rows), 1), 2),
    }


def _checkpoint_metadata(neuron_id: str) -> dict:
    path = os.path.join("data", "neurons", f"neuron_{neuron_id}.pt")
    # Importing the loader registers the historical taiji pickle aliases. Meta
    # tensors keep this metadata-only audit from allocating multi-GB weights.
    from neuroplex.loader import assemble_cortex  # noqa: F401

    ckpt = torch.load(path, map_location="meta", weights_only=False)
    cfg = ckpt.get("neuron_config")
    result = ckpt.get("result") or {}
    metadata = {
        "file_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
        "domain": ckpt.get("domain"),
        "data_source": ckpt.get("data_source"),
        "has_shared_embedding": "shared_embedding_state" in ckpt,
        "has_optimizer_state": "optimizer_state" in ckpt,
        "has_scheduler_state": "scheduler_state" in ckpt,
        "spec": getattr(cfg, "spec", None),
        "steps": result.get("steps"),
        "best_step": result.get("best_step"),
        "best_val_ppl": result.get("best_val_ppl"),
        "base_id": result.get("base_id"),
        "finetune": result.get("finetune"),
    }
    del ckpt
    gc.collect()
    return metadata


def main() -> None:
    logging.disable(logging.CRITICAL)
    texts = load_dialogue_texts_multi(DATA_DIR, max_texts=MAX_TEXTS)
    train_texts, eval_pool = split_train_eval(texts, eval_ratio=EVAL_RATIO, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]

    from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer

    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    report = {
        "training_entry_defaults": {
            "max_texts": MAX_TEXTS,
            "deduplicated_loaded_texts": len(texts),
            "train_texts": len(train_texts),
            "eval_pool": len(eval_pool),
            "eval_used_by_training": len(eval_texts),
            "eval_ratio": EVAL_RATIO,
            "max_seq_len": MAX_SEQ_LEN,
            "answer_marker": SFT_ANSWER_MARKER,
            "answer_marker_mode": "first",
        },
        "data_contract": {
            "eval_pool": _aggregate(eval_pool, domain_sp, general_sp),
            "eval_used_by_training": _aggregate(eval_texts, domain_sp, general_sp),
        },
        "checkpoints": {
            neuron_id: _checkpoint_metadata(neuron_id)
            for neuron_id in CHECKPOINT_IDS
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
