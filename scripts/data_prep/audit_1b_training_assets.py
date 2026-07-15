#!/usr/bin/env python3
"""Audit local Taiji assets for the 1B training plan."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "taiji_data" / "training_data"
PRETRAIN_MIX_DIR = TRAINING_DIR / "pretrain_mix_v1"
MULTIMODAL_ANN_PATH = PROJECT_ROOT / "taiji_data" / "multimodal" / "annotations" / "train.jsonl"
OUTPUT_JSON = TRAINING_DIR / "reports" / "audit_1b_training_assets.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "1B_DATA_GAP_REPORT_CN.md"

TOKEN_GOALS = {
    "base_min": 5_000_000_000,
    "base_recommended": 10_000_000_000,
    "english_recommended": 2_500_000_000,
    "multimodal_pairs_min": 100_000,
}


@dataclass(frozen=True)
class MixBucket:
    name: str
    category: str
    path: Path


KNOWN_BUCKETS: dict[str, str] = {
    "fineweb_edu": "english_general",
    "fineweb2_en": "english_general",
    "falcon_refinedweb_en": "english_general",
    "fineweb2_zh": "chinese_general",
    "skypile_zh": "chinese_general",
    "openwebmath": "math",
    "codeparrot_code": "code",
}


def get_mix_buckets() -> list[MixBucket]:
    buckets: list[MixBucket] = []
    for name, category in KNOWN_BUCKETS.items():
        buckets.append(MixBucket(name, category, PRETRAIN_MIX_DIR / f"{name}.jsonl"))
    return buckets


def tokenizer_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "taiji" / "tokenizer_native_v2" / "sentencepiece.model",
        PROJECT_ROOT / "tokenizer_native_v2" / "sentencepiece.model",
    ]


def load_sentencepiece() -> spm.SentencePieceProcessor:
    for path in tokenizer_candidates():
        if path.exists():
            sp = spm.SentencePieceProcessor()
            sp.load(str(path))
            return sp
    raise FileNotFoundError("No sentencepiece.model found in known tokenizer paths")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def extract_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "code", "completion", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(record.get("messages"), list):
        parts = []
        for msg in record["messages"]:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        if parts:
            return "\n".join(parts)
    return json.dumps(record, ensure_ascii=False)


def sample_token_ratio(path: Path, sp: spm.SentencePieceProcessor, sample_size: int = 1000) -> dict[str, float]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return {"sampled": 0, "chars": 0.0, "tokens": 0.0, "token_per_char": 0.0}

    sampled_indices = random.sample(range(len(lines)), min(sample_size, len(lines)))
    total_chars = 0
    total_tokens = 0
    sampled = 0
    for idx in sampled_indices:
        line = lines[idx]
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            text = line
        else:
            text = extract_text(obj) if isinstance(obj, dict) else str(obj)
        if not text:
            continue
        total_chars += len(text)
        total_tokens += len(sp.encode(text))
        sampled += 1

    token_per_char = (total_tokens / total_chars) if total_chars else 0.0
    return {
        "sampled": sampled,
        "chars": float(total_chars),
        "tokens": float(total_tokens),
        "token_per_char": token_per_char,
    }


def read_manifest() -> dict[str, Any]:
    manifest_path = PRETRAIN_MIX_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def audit_mix(sp: spm.SentencePieceProcessor) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    buckets: list[dict[str, Any]] = []
    total_tokens = 0

    for bucket in get_mix_buckets():
        if not bucket.path.exists():
            continue
        line_count = count_lines(bucket.path)
        size_bytes = bucket.path.stat().st_size
        ratio = sample_token_ratio(bucket.path, sp)
        estimated_tokens = int(size_bytes * ratio["token_per_char"])
        total_tokens += estimated_tokens
        by_category[bucket.category] = by_category.get(bucket.category, 0) + estimated_tokens
        buckets.append(
            {
                "name": bucket.name,
                "category": bucket.category,
                "path": str(bucket.path),
                "line_count": line_count,
                "size_bytes": size_bytes,
                "sampled_records": int(ratio["sampled"]),
                "sample_token_per_char": ratio["token_per_char"],
                "estimated_tokens": estimated_tokens,
            }
        )

    return {
        "total_estimated_tokens": total_tokens,
        "by_category": by_category,
        "buckets": buckets,
        "manifest": read_manifest(),
    }


def audit_multimodal() -> dict[str, Any]:
    if not MULTIMODAL_ANN_PATH.exists():
        return {"annotation_pairs": 0, "size_bytes": 0}
    return {
        "annotation_pairs": count_lines(MULTIMODAL_ANN_PATH),
        "size_bytes": MULTIMODAL_ANN_PATH.stat().st_size,
        "path": str(MULTIMODAL_ANN_PATH),
    }


def build_gap_summary(mix: dict[str, Any], multimodal: dict[str, Any]) -> dict[str, Any]:
    total_tokens = int(mix["total_estimated_tokens"])
    english_tokens = int(mix["by_category"].get("english_general", 0))
    multimodal_pairs = int(multimodal.get("annotation_pairs", 0))
    return {
        "base_token_gap_min": max(0, TOKEN_GOALS["base_min"] - total_tokens),
        "base_token_gap_recommended": max(0, TOKEN_GOALS["base_recommended"] - total_tokens),
        "english_token_gap_recommended": max(0, TOKEN_GOALS["english_recommended"] - english_tokens),
        "multimodal_pair_gap_min": max(0, TOKEN_GOALS["multimodal_pairs_min"] - multimodal_pairs),
    }


def write_markdown(report: dict[str, Any]) -> None:
    mix = report["pretrain_mix_v1"]
    gaps = report["gaps"]
    multimodal = report["multimodal"]
    by_category = mix["by_category"]
    total_tokens = mix["total_estimated_tokens"]

    lines = [
        "# 态极 1B 数据缺口报告",
        "",
        "## 结论",
        "",
        f"- 当前 `pretrain_mix_v1` 估算总量约 `{total_tokens:,}` tokens，只够做词表训练、smoke run 和短程 stage1，不够支撑完整 1B 从零预训练。",
        f"- 距离最低可接受的 `5B tokens` 还差约 `{gaps['base_token_gap_min']:,}` tokens。",
        f"- 距离推荐目标的 `10B tokens` 还差约 `{gaps['base_token_gap_recommended']:,}` tokens。",
        f"- 英文主料当前约 `{by_category.get('english_general', 0):,}` tokens，距离建议的 `2.5B` 英文主料还差约 `{gaps['english_token_gap_recommended']:,}` tokens。",
        f"- 多模态标注当前仅 `{multimodal.get('annotation_pairs', 0):,}` 对，离最小可用的 `100,000` 对还差 `{gaps['multimodal_pair_gap_min']:,}` 对。",
        "",
        "## 当前主料",
        "",
        f"- 中文通用：`{by_category.get('chinese_general', 0):,}` tokens",
        f"- 英文通用：`{by_category.get('english_general', 0):,}` tokens",
        f"- 代码：`{by_category.get('code', 0):,}` tokens",
        f"- 数学：`{by_category.get('math', 0):,}` tokens",
        "",
        "## 训练建议",
        "",
        "- 先把这批 `pretrain_mix_v1` 用于 `native-v2` tokenizer 重练和 1B smoke run。",
        "- 文本基座正式开训前，优先补英文高质量通用文本和中文第二批 shard。",
        "- 多模态不要混入当前 base pretrain 主流程，单独准备图文/音文对齐阶段。",
        "- `pretrain_final.jsonl` 这类 SFT/合成混合数据不应作为 1B base pretrain 主料。",
        "",
        "## 明确缺口",
        "",
        "- 文本总量缺口：至少再补到 `5B-10B tokens`。",
        "- 英文缺口：优先补到 `2B+`，更稳妥是 `2.5B` 左右。",
        "- 多模态缺口：至少先补到 `100k` 图文/音文标注对，再谈 1B 多模态对齐。",
        "",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(42)
    sp = load_sentencepiece()
    mix = audit_mix(sp)
    multimodal = audit_multimodal()
    report = {
        "token_goals": TOKEN_GOALS,
        "pretrain_mix_v1": mix,
        "multimodal": multimodal,
    }
    report["gaps"] = build_gap_summary(mix, multimodal)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)

    print(
        json.dumps(
            {
                "json_report": str(OUTPUT_JSON),
                "markdown_report": str(OUTPUT_MD),
                "total_estimated_tokens": mix["total_estimated_tokens"],
                "gaps": report["gaps"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
