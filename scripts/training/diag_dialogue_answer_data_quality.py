#!/usr/bin/env python3
"""P1.2：审计 dialogue holdout 的答案首 token 数据质量。

固定复现 dialogue fine-tune 的 100 条验证样本，报告：

* 首答案 token 的来源类别与原始来源文件；
* 英文、代码信号、异常标记和 128 token 截断比例；
* 五个 dialogue checkpoint 在这些类别上的首 token rank / Top-1。

只做前向诊断，不修改权重、不执行训练。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from neuroplex.resonance import ResonanceNeuron
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from neuroplex.resonance.translator import build_position_alignment
from scripts.training.diag_dialogue_training_contract import _training_shape
from scripts.training.utils import (
    DIALOGUE_DATA_FILES,
    create_shared_embedding,
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
MAX_TEXTS = 100000
EVAL_CAP = 100
BATCH_SIZE = 4
MAX_SEQ_LEN = 128


def _source_map() -> dict[str, str]:
    """Reconstruct the first source file retained by the multi-file loader."""
    result: dict[str, str] = {}
    loaded = 0
    for filename in DIALOGUE_DATA_FILES:
        path = os.path.join("data/simple_zh", filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if loaded >= MAX_TEXTS:
                    return result
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if len(text) < 20 or SFT_ANSWER_MARKER not in text:
                    continue
                result.setdefault(text, filename)
                loaded += 1
    return result


def _first_token_category(piece: str, answer: str) -> str:
    if piece.startswith("<0x") or "<unk>" in piece or "<unk>" in answer[:16]:
        return "special_or_byte"
    if re.search(r"[A-Za-z]", piece):
        return "latin"
    if re.search(r"[0-9]", piece):
        return "numeric"
    if any(mark in piece or mark in answer[:32] for mark in ("```", "\\n", "\\t", "{", "}", ";")):
        return "code_signal"
    if re.search(r"[\u4e00-\u9fff]", piece):
        return "han"
    return "other"


def _data_rows(texts: list[str], domain_sp, general_sp, source_map: dict[str, str]) -> list[dict]:
    rows = []
    for index, text in enumerate(texts):
        marker = text.find(SFT_ANSWER_MARKER)
        prompt = text[:marker + len(SFT_ANSWER_MARKER)]
        answer = text[marker + len(SFT_ANSWER_MARKER):]
        full_general, targets = build_position_alignment(text, domain_sp, general_sp)
        prompt_general = general_sp.encode(prompt)
        answer_start = len(prompt_general)
        target_id = int(targets[answer_start]) if answer_start < len(targets) else -100
        piece = domain_sp.id_to_piece(target_id) if target_id >= 0 else ""
        shape = _training_shape(text, domain_sp, general_sp)
        rows.append({
            "sample_index": index,
            "source_file": source_map.get(text, "unknown"),
            "category": _first_token_category(piece, answer),
            "first_target_piece": piece,
            "first_answer_preview": answer[:80],
            "answer_chars": len(answer),
            "answer_marker_count": text.count(SFT_ANSWER_MARKER),
            "truncated": shape["truncated"],
            "first_target_id": target_id,
            "prompt_general_ids": prompt_general,
        })
    return rows


def _load_neuron(neuron_id: str):
    path = os.path.join("data", "neurons", f"neuron_{neuron_id}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neuron = ResonanceNeuron(ckpt["neuron_config"])
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    shared = create_shared_embedding("cpu")
    shared.load_state_dict(ckpt["shared_embedding_state"])
    neuron.eval()
    shared.eval()
    del ckpt
    return neuron, shared


def _rank_rows(rows: list[dict], neuron_id: str, neuron, shared, general_sp) -> None:
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            max_len = max(len(row["prompt_general_ids"]) for row in batch)
            ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            positions = []
            for row_index, row in enumerate(batch):
                token_ids = row["prompt_general_ids"]
                ids[row_index, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
                positions.append(len(token_ids) - 1)
            logits = neuron(shared(ids), return_logits=True)["logits"]
            for row_index, row in enumerate(batch):
                next_logits = logits[row_index, positions[row_index]]
                target_id = row["first_target_id"]
                if target_id < 0:
                    rank = None
                    top1 = False
                else:
                    target_logit = next_logits[target_id]
                    rank = int((next_logits > target_logit).sum())
                    top1 = int(next_logits.argmax()) == target_id
                row.setdefault("ranks_by_neuron", {})[neuron_id] = rank
                row.setdefault("top1_by_neuron", {})[neuron_id] = top1


def _summarize(rows: list[dict], neuron_id: str) -> dict:
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        ranks = [row["ranks_by_neuron"][neuron_id] for row in subset
                 if row["ranks_by_neuron"][neuron_id] is not None]
        top1 = [row["top1_by_neuron"][neuron_id] for row in subset]
        ranks.sort()
        by_category[category] = {
            "samples": len(subset),
            "truncated": sum(row["truncated"] for row in subset),
            "mean_rank_zero_based": round(sum(ranks) / max(len(ranks), 1), 2),
            "median_rank_zero_based": ranks[len(ranks) // 2] if ranks else None,
            "top1_rate": round(sum(top1) / max(len(top1), 1), 4),
        }
    return by_category


def main() -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    all_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=MAX_TEXTS)
    _, eval_pool = split_train_eval(all_texts, eval_ratio=0.05, seed=42)
    eval_texts = eval_pool[:EVAL_CAP]
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    rows = _data_rows(eval_texts, domain_sp, general_sp, _source_map())
    source_counts = {}
    category_counts = {}
    for row in rows:
        source_counts[row["source_file"]] = source_counts.get(row["source_file"], 0) + 1
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1

    report = {
        "contract": {
            "eval_samples": len(rows),
            "max_seq_len": MAX_SEQ_LEN,
            "data_files": DIALOGUE_DATA_FILES,
        },
        "data": {
            "source_file_counts": source_counts,
            "first_token_category_counts": category_counts,
            "truncated_samples": sum(row["truncated"] for row in rows),
            "multi_marker_samples": sum(row["answer_marker_count"] > 1 for row in rows),
            "answer_char_mean": round(sum(row["answer_chars"] for row in rows) / max(len(rows), 1), 2),
            "preview": [
                {key: row[key] for key in (
                    "sample_index", "source_file", "category", "first_target_piece",
                    "first_answer_preview", "truncated",
                )}
                for row in rows[:12]
            ],
        },
        "neurons": {},
    }
    for index, neuron_id in enumerate(DIALOGUE_IDS, start=1):
        print(f"[{index}/{len(DIALOGUE_IDS)}] ranking {neuron_id}...", flush=True)
        neuron, shared = _load_neuron(neuron_id)
        _rank_rows(rows, neuron_id, neuron, shared, general_sp)
        report["neurons"][neuron_id] = {
            "all": _summarize(rows, neuron_id),
        }
        del neuron, shared
        gc.collect()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
