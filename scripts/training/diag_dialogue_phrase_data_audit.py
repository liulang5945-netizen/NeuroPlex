#!/usr/bin/env python3
"""审计 dialogue canonical answer 中的高频续写片段。"""
from __future__ import annotations

import json
import os
from collections import Counter

from experiment_config import DIALOGUE_DATA_FILES


DATA_DIR = os.path.join("data", "simple_zh")
ANSWER_MARKER = "答："
PHRASES = [
    "神经网络",
    "注意力机制",
    "是一种",
    "是一种基于",
    "它通过",
    "通过",
    "能够",
    "计算机",
]
CONTINUATION_WIDTH = 12
EXAMPLE_LIMIT = 5


def _phrase_bucket() -> dict:
    return {
        "occurrences": 0,
        "sample_count": 0,
        "continuations": Counter(),
        "examples": [],
    }


def main() -> None:
    file_stats = {}
    phrase_stats = {phrase: _phrase_bucket() for phrase in PHRASES}
    total_samples = 0
    total_answer_chars = 0

    for filename in DIALOGUE_DATA_FILES:
        path = os.path.join(DATA_DIR, filename)
        stats = {"samples": 0, "answer_chars": 0, "invalid_json": 0}
        if not os.path.exists(path):
            file_stats[filename] = {**stats, "missing": True}
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                if len(text) < 20 or ANSWER_MARKER not in text:
                    continue
                answer = text.split(ANSWER_MARKER, 1)[1]
                stats["samples"] += 1
                stats["answer_chars"] += len(answer)
                total_samples += 1
                total_answer_chars += len(answer)
                for phrase, bucket in phrase_stats.items():
                    positions = []
                    start = 0
                    while True:
                        index = answer.find(phrase, start)
                        if index < 0:
                            break
                        positions.append(index)
                        start = index + 1
                    if not positions:
                        continue
                    bucket["occurrences"] += len(positions)
                    bucket["sample_count"] += 1
                    for index in positions:
                        continuation = answer[
                            index + len(phrase): index + len(phrase) + CONTINUATION_WIDTH
                        ]
                        bucket["continuations"][continuation] += 1
                    if len(bucket["examples"]) < EXAMPLE_LIMIT:
                        index = positions[0]
                        left = max(0, index - 24)
                        right = min(len(answer), index + len(phrase) + 48)
                        bucket["examples"].append({
                            "source_file": filename,
                            "line": line_number,
                            "context": answer[left:right],
                        })
        file_stats[filename] = stats

    report = {
        "contract": {
            "data_files": DIALOGUE_DATA_FILES,
            "answer_region": "text after first 答：",
            "min_text_chars": 20,
            "continuation_width": CONTINUATION_WIDTH,
        },
        "data": {
            "total_samples": total_samples,
            "total_answer_chars": total_answer_chars,
            "files": file_stats,
        },
        "phrases": {},
    }
    for phrase, bucket in phrase_stats.items():
        report["phrases"][phrase] = {
            "occurrences": bucket["occurrences"],
            "sample_count": bucket["sample_count"],
            "sample_rate": round(bucket["sample_count"] / max(total_samples, 1), 6),
            "occurrences_per_1000_samples": round(
                bucket["occurrences"] / max(total_samples, 1) * 1000, 4
            ),
            "top_continuations": [
                {"text": text, "count": count}
                for text, count in bucket["continuations"].most_common(10)
            ],
            "examples": bucket["examples"],
        }

    out_path = os.path.join("reports", "production_dialogue_phrase_data_audit_20260820.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
