"""Download and audit a licensed Hugging Face dialogue candidate.

This is deliberately a candidate-data workflow.  It never writes into the
production ``data/simple_zh`` directory and it does not alter any neuron
checkpoint.  The output uses the same SFT contract as the active dialogue
loader::

    {"text": "问：...\n答：..."}

The selected source is the no-plugin portion of MOSS-003.  Its HF dataset
card declares CC BY 4.0.  The license is checked at runtime so a renamed or
replaced repository cannot silently enter the training path.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "fnlp/moss-003-sft-data"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
RAW_FILENAME = "moss-003-sft-no-tools.jsonl.zip"
EXPECTED_LICENSE = "cc-by-4.0"
MIN_CHINESE_RATIO = 0.20
DEFAULT_OUTPUT = Path("data/hf_candidates/moss_003_dialogue")
DEFAULT_EXISTING_DIR = Path("data/simple_zh")


def normalize_text(value: Any) -> str:
    """Normalize text without changing meaningful line structure."""

    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    return "\n".join(line for line in lines if line)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_hashes(existing_dir: Path) -> set[str]:
    """Load hashes from the current dialogue corpus for leakage prevention."""

    hashes: set[str] = set()
    if not existing_dir.exists():
        return hashes
    for path in sorted(existing_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = normalize_text(row.get("text", ""))
                if text:
                    hashes.add(text_hash(text))
    return hashes


def extract_pairs(item: dict[str, Any]) -> Iterable[tuple[str, str, str, int]]:
    """Yield (question, answer, category, turn_index) from one MOSS item."""

    category = normalize_text(item.get("category", "unknown")) or "unknown"
    conversation = item.get("conversation")
    if isinstance(conversation, list):
        for turn_index, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                continue
            question = normalize_text(turn.get("human", turn.get("user", "")))
            answer = normalize_text(turn.get("assistant", turn.get("answer", "")))
            if question and answer:
                yield question, answer, category, turn_index
        return

    # The published no-plugin file uses chat.turn_N.Human/MOSS fields.
    chat = item.get("chat")
    if not isinstance(chat, dict):
        return
    turn_keys = sorted(
        (key for key in chat if str(key).startswith("turn_")),
        key=lambda key: int(str(key).split("_", 1)[1]),
    )
    for turn_key in turn_keys:
        turn = chat.get(turn_key)
        if not isinstance(turn, dict):
            continue
        question = normalize_text(turn.get("Human", ""))
        answer = normalize_text(turn.get("MOSS", ""))
        question = re.sub(r"^<\|Human\|>:\s*", "", question)
        answer = re.sub(r"^<\|MOSS\|>:\s*", "", answer)
        question = re.sub(r"<eoh>\s*$", "", question)
        answer = re.sub(r"<eom>\s*$", "", answer)
        if question and answer:
            turn_index = int(str(turn_key).split("_", 1)[1])
            yield question, answer, category, turn_index


def valid_pair(question: str, answer: str) -> bool:
    """Apply conservative filters compatible with the current 128-token pilot."""

    if len(question) < 2 or len(answer) < 2:
        return False
    if len(question) > 2000 or len(answer) > 4000:
        return False
    if "\x00" in question or "\x00" in answer:
        return False
    # Drop obvious empty/template answers while retaining ordinary terse replies.
    if answer in {"无", "暂无", "不知道", "好的", "好的。", "...", "……"}:
        return False
    return True


def chinese_ratio(text: str) -> float:
    non_space = sum(not char.isspace() for char in text)
    if not non_space:
        return 0.0
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    return cjk / non_space


def get_source_metadata() -> dict[str, Any]:
    info = HfApi().dataset_info(DATASET_ID, files_metadata=True)
    card_license = (info.cardData or {}).get("license")
    if card_license != EXPECTED_LICENSE:
        raise RuntimeError(
            f"Refusing dataset: expected license {EXPECTED_LICENSE!r}, "
            f"got {card_license!r} for {DATASET_URL}"
        )
    sibling = next((item for item in info.siblings or [] if item.rfilename == RAW_FILENAME), None)
    if sibling is None:
        raise RuntimeError(f"Required source file is missing: {RAW_FILENAME}")
    lfs = getattr(sibling, "lfs", None)
    return {
        "dataset": DATASET_ID,
        "dataset_url": DATASET_URL,
        "revision": info.sha,
        "license": card_license,
        "raw_filename": RAW_FILENAME,
        "raw_size_bytes": getattr(sibling, "size", None),
        "raw_sha256": getattr(lfs, "sha256", None),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_candidate(
    raw_zip: Path,
    output_dir: Path,
    existing_dir: Path,
    max_per_category: int,
    seed: int,
) -> dict[str, Any]:
    """Read the compressed source and create deterministic train/eval JSONL."""

    existing_hashes = load_existing_hashes(existing_dir)
    selected_hashes: set[str] = set()
    reservoirs: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    category_seen: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    source_items = 0

    with zipfile.ZipFile(raw_zip) as archive:
        members = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if not members:
            raise RuntimeError(f"No JSONL member found in {raw_zip}")
        for member in members:
            with archive.open(member, "r") as binary_handle:
                for raw_line in binary_handle:
                    try:
                        item = json.loads(raw_line)
                    except json.JSONDecodeError:
                        skipped["invalid_json"] += 1
                        continue
                    source_items += 1
                    conversation_id = item.get("conversation_id")
                    for question, answer, category, turn_index in extract_pairs(item):
                        if chinese_ratio(question + answer) < MIN_CHINESE_RATIO:
                            skipped["non_chinese_dominant"] += 1
                            continue
                        if not valid_pair(question, answer):
                            skipped["quality_filter"] += 1
                            continue
                        text = f"问：{question}\n答：{answer}"
                        digest = text_hash(text)
                        if digest in existing_hashes:
                            skipped["overlap_existing"] += 1
                            continue
                        if digest in selected_hashes:
                            skipped["duplicate"] += 1
                            continue
                        category_seen[category] += 1
                        row = {
                            "text": text,
                            "source_dataset": DATASET_ID,
                            "source_category": category,
                            "source_conversation_id": conversation_id,
                            "source_turn": turn_index,
                            "text_sha256": digest,
                        }
                        bucket = reservoirs[category]
                        priority = int(digest, 16)
                        if len(bucket) < max_per_category:
                            heapq.heappush(bucket, (-priority, digest, row))
                            selected_hashes.add(digest)
                        else:
                            # The heap root is the largest digest priority, i.e.
                            # the worst retained row.  This keeps the operation
                            # logarithmic and bounds memory to the final set.
                            worst_priority, worst_digest, _ = bucket[0]
                            if priority < -worst_priority:
                                heapq.heapreplace(bucket, (-priority, digest, row))
                                selected_hashes.remove(worst_digest)
                                selected_hashes.add(digest)

    rows = [
        row
        for category in sorted(reservoirs)
        for _, _, row in reservoirs[category]
    ]
    random.Random(seed).shuffle(rows)
    eval_count = max(1, round(len(rows) * 0.05)) if rows else 0
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    manifest = {
        "source": get_source_metadata(),
        "seed": seed,
        "max_per_category": max_per_category,
        "min_chinese_ratio": MIN_CHINESE_RATIO,
        "existing_dialogue_dir": str(existing_dir),
        "source_items_read": source_items,
        "accepted_pairs_before_split": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "categories_seen": dict(sorted(category_seen.items())),
        "categories_sampled": dict(sorted(Counter(row["source_category"] for row in rows).items())),
        "skipped": dict(sorted(skipped.items())),
        "format": "jsonl rows with text=问：...\\n答：...",
        "production_status": "candidate_only_not_in_production_data",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--existing-dir", type=Path, default=DEFAULT_EXISTING_DIR)
    parser.add_argument("--max-per-category", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_category <= 0:
        raise SystemExit("--max-per-category must be positive")

    metadata = get_source_metadata()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=RAW_FILENAME,
            local_dir=raw_dir,
        )
    )
    if metadata.get("raw_size_bytes") and raw_path.stat().st_size != metadata["raw_size_bytes"]:
        raise RuntimeError(
            f"Downloaded size mismatch: {raw_path.stat().st_size} != {metadata['raw_size_bytes']}"
        )
    manifest = build_candidate(
        raw_zip=raw_path,
        output_dir=args.output_dir,
        existing_dir=args.existing_dir,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
