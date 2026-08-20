#!/usr/bin/env python3
"""量化 general→domain 对齐中的 domain target 重复绑定。"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.resonance.translator import _get_token_spans, build_position_alignment
from scripts.training.utils import load_dialogue_texts_multi, load_domain_tokenizer, load_general_tokenizer


DATA_DIR = "data/simple_zh"
SAMPLE_CAP = 10_000
PHRASE = "是一种基于"


def _best_domain_indices(text: str, domain_sp, general_sp) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    domain_ids, domain_spans = _get_token_spans(domain_sp, text)
    general_ids, general_spans = _get_token_spans(general_sp, text)
    mapped = []
    for g_start, g_end in general_spans:
        best_i = -1
        best_overlap = 0
        for i, (d_start, d_end) in enumerate(domain_spans):
            overlap = max(0, min(g_end, d_end) - max(g_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i
        mapped.append(best_i)
    return domain_ids, general_ids, domain_spans, general_spans, mapped


def _causal_first_occurrence(mapped: list[int]) -> list[int]:
    seen = set()
    result = []
    for domain_index in mapped:
        if domain_index < 0 or domain_index in seen:
            result.append(-1)
        else:
            result.append(domain_index)
            seen.add(domain_index)
    return result


def _row(text: str, domain_sp, general_sp) -> dict:
    domain_ids, general_ids, domain_spans, general_spans, mapped = _best_domain_indices(
        text, domain_sp, general_sp
    )
    counts = Counter(index for index in mapped if index >= 0)
    repeated_positions = sum(count - 1 for count in counts.values() if count > 1)
    causal = _causal_first_occurrence(mapped)
    return {
        "text": text,
        "domain_pieces": [domain_sp.id_to_piece(token_id) for token_id in domain_ids],
        "domain_spans": domain_spans,
        "general_pieces": [general_sp.id_to_piece(token_id) for token_id in general_ids],
        "general_spans": general_spans,
        "legacy_domain_indices": mapped,
        "legacy_target_pieces": [
            domain_sp.id_to_piece(domain_ids[index]) if index >= 0 else None
            for index in mapped
        ],
        "production_first_occurrence_indices": causal,
        "production_target_pieces": [
            domain_sp.id_to_piece(domain_ids[index]) if index >= 0 else None
            for index in causal
        ],
        "legacy_repeated_target_positions": repeated_positions,
    }


def main() -> None:
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    texts = load_dialogue_texts_multi(DATA_DIR, max_texts=SAMPLE_CAP)
    total_general_positions = 0
    total_legacy_repeated_positions = 0
    total_domain_tokens = 0
    total_mapped_domain_tokens = 0
    phrase_rows = []
    for text in texts:
        domain_ids, general_ids, domain_spans, general_spans, mapped = _best_domain_indices(
            text, domain_sp, general_sp
        )
        counts = Counter(index for index in mapped if index >= 0)
        total_general_positions += len(general_ids)
        total_legacy_repeated_positions += sum(count - 1 for count in counts.values() if count > 1)
        total_domain_tokens += len(domain_ids)
        total_mapped_domain_tokens += len(counts)
        if PHRASE in text and len(phrase_rows) < 5:
            phrase_rows.append(_row(text, domain_sp, general_sp))

    example = _row(
        "神经网络是一种基于人工神经网络的机器学习方法",
        domain_sp,
        general_sp,
    )
    report = {
        "contract": {
            "data_dir": DATA_DIR,
            "sample_cap": SAMPLE_CAP,
            "samples_loaded": len(texts),
            "legacy_alignment_rule": "max-overlap mapping at every general position (pre-fix)",
            "production_alignment_rule": "max-overlap mapping with only the first general position per domain token index",
            "writes_checkpoint": False,
        },
        "aggregate": {
            "general_positions": total_general_positions,
            "domain_tokens": total_domain_tokens,
            "mapped_domain_tokens": total_mapped_domain_tokens,
            "legacy_repeated_target_positions": total_legacy_repeated_positions,
            "legacy_repeated_target_rate_over_general_positions": round(
                total_legacy_repeated_positions / max(total_general_positions, 1), 6
            ),
            "production_repeated_target_positions": 0,
            "domain_token_mapping_coverage": round(
                total_mapped_domain_tokens / max(total_domain_tokens, 1), 6
            ),
        },
        "canonical_example": example,
        "phrase_examples": phrase_rows,
    }
    out_path = os.path.join("reports", "production_dialogue_alignment_causality_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
