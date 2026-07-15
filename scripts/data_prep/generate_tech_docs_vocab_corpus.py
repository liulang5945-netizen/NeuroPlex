#!/usr/bin/env python3
"""Generate a technical-docs supplement corpus for tokenizer balancing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


TEXT_KEYS = (
    "text",
    "content",
    "code",
    "instruction",
    "input",
    "output",
    "question",
    "answer",
    "description",
)

TEXT_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".vue",
    ".css",
    ".html",
    ".yml",
    ".yaml",
    ".toml",
    ".sh",
    ".ps1",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "model_cache",
    ".cache",
    ".pytest_cache",
    "raw_pretrain_mix_v1",
    "pretrain_mix_v1",
    "test_artifacts",
}

SKIP_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".model",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".ico",
    ".zip",
    ".gz",
    ".parquet",
    ".pdf",
}

CURATED_TECH_SEEDS = [
    "技术文档需要清楚说明输入、输出、边界条件、失败模式和可观测性指标。",
    "Tokenizer 词表训练要覆盖中文、英文、代码、数学、技术文档和系统特殊格式，不能只靠通用网页文本。",
    "A model runbook should record tokenizer version, dataset manifest, holdout split, checkpoint cadence, and resume procedure.",
    "科学与工程文本通常混合自然语言、公式、配置项、路径、命令行参数和代码片段。",
    "Frontend technical notes should describe layout rules, responsive behavior, state transitions, and accessibility constraints.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate technical-docs supplement corpus for tokenizer balancing")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--input-dir", action="append", default=None, help="Directory to scan; can be repeated.")
    parser.add_argument(
        "--output",
        default="taiji_data/tokenizer/local_vocab_sources/tech/tech_docs_vocab.jsonl",
    )
    parser.add_argument("--max-records", type=int, default=50_000)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--max-record-chars", type=int, default=4_000)
    return parser.parse_args()


def default_input_dirs(project_dir: Path) -> list[Path]:
    return [
        project_dir / "docs",
        project_dir / "scripts",
        project_dir / "taiji",
        project_dir / "api",
        project_dir / "frontend" / "src",
        project_dir / "tests",
    ]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def should_skip_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & SKIP_DIRS:
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def iter_text_from_json(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if not isinstance(obj, dict):
        return
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str):
            yield value
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                yield message["content"]


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def chunk_text(text: str, max_chars: int) -> Iterator[str]:
    normalized = normalize_text(text)
    if not normalized:
        return

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    buffer: list[str] = []
    buffer_chars = 0
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if buffer:
                yield "\n\n".join(buffer).strip()
                buffer = []
                buffer_chars = 0
            yield paragraph[:max_chars].strip()
            paragraph = paragraph[max_chars:].strip()
        additional = len(paragraph) + (2 if buffer else 0)
        if buffer and buffer_chars + additional > max_chars:
            yield "\n\n".join(buffer).strip()
            buffer = [paragraph]
            buffer_chars = len(paragraph)
        else:
            buffer.append(paragraph)
            buffer_chars += additional if buffer_chars else len(paragraph)

    if buffer:
        yield "\n\n".join(buffer).strip()


def iter_jsonl_texts(path: Path, max_chars: int) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield from chunk_text(line, max_chars)
            else:
                for text in iter_text_from_json(obj):
                    yield from chunk_text(text, max_chars)


def iter_source_texts(path: Path, max_chars: int) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl_texts(path, max_chars)
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                for item in iter_text_from_json(obj):
                    yield from chunk_text(item, max_chars)
                return
    yield from chunk_text(text, max_chars)


def collect_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in TEXT_SUFFIXES and not should_skip_path(root):
                files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not should_skip_path(path):
                files.append(path)
    return sorted(files)


def format_record_text(project_dir: Path, path: Path, block: str) -> str:
    try:
        relative = path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        relative = path
    return f"Path: {relative.as_posix()}\n{block}".strip()


def write_records(
    output: Path,
    *,
    project_dir: Path | None = None,
    input_dirs: list[Path] | None = None,
    max_records: int = 50_000,
    min_chars: int = 64,
    max_record_chars: int = 4_000,
) -> dict[str, object]:
    project_root = (project_dir or Path.cwd()).resolve()
    scan_dirs = input_dirs or default_input_dirs(project_root)
    files = collect_input_files(scan_dirs)

    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records = 0
    chars = 0
    used_files = 0

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for seed in CURATED_TECH_SEEDS:
            if records >= max_records:
                break
            digest = stable_hash(seed)
            if digest in seen:
                continue
            seen.add(digest)
            handle.write(json.dumps({"text": seed, "source": "tech_docs_vocab", "path": "curated_seed"}, ensure_ascii=False) + "\n")
            records += 1
            chars += len(seed)

        for path in files:
            if records >= max_records:
                break
            wrote_file = False
            for block in iter_source_texts(path, max_record_chars):
                text = format_record_text(project_root, path, block)
                if len(text) < min_chars:
                    continue
                digest = stable_hash(text[:4096])
                if digest in seen:
                    continue
                seen.add(digest)
                handle.write(
                    json.dumps(
                        {
                            "text": text,
                            "source": "tech_docs_vocab",
                            "path": str(path),
                            "suffix": path.suffix.lower(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                records += 1
                chars += len(text)
                wrote_file = True
                if records >= max_records:
                    break
            if wrote_file:
                used_files += 1

    return {
        "output": str(output),
        "records": records,
        "chars": chars,
        "files": used_files,
        "scan_dirs": [str(path) for path in scan_dirs],
    }


def main() -> None:
    args = parse_args()
    input_dirs = [Path(path) for path in args.input_dir] if args.input_dir else None
    result = write_records(
        Path(args.output),
        project_dir=Path(args.project_dir),
        input_dirs=input_dirs,
        max_records=args.max_records,
        min_chars=args.min_chars,
        max_record_chars=args.max_record_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
