#!/usr/bin/env python3
"""Download and normalize a first-pass 1B-scale Taiji pretraining mix."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    file_prefix: str
    suffix: str
    text_field: str
    file_format: str
    category: str
    note: str


SOURCES: dict[str, SourceSpec] = {
    "fineweb_edu": SourceSpec(
        name="fineweb_edu",
        repo_id="HuggingFaceFW/fineweb-edu",
        file_prefix="data/",
        suffix=".parquet",
        text_field="text",
        file_format="parquet",
        category="general_web",
        note="High-quality educational web corpus.",
    ),
    "fineweb2_en": SourceSpec(
        name="fineweb2_en",
        repo_id="HuggingFaceFW/fineweb-2",
        file_prefix="data/",
        suffix=".parquet",
        text_field="text",
        file_format="parquet",
        category="english_web",
        note="Modern English slice from FineWeb2 for base-language coverage; filtered to eng_Latn/train at match time.",
    ),
    "falcon_refinedweb_en": SourceSpec(
        name="falcon_refinedweb_en",
        repo_id="tiiuae/falcon-refinedweb",
        file_prefix="data/",
        suffix=".parquet",
        text_field="content",
        file_format="parquet",
        category="english_web",
        note="Mirror-friendly English web corpus with direct parquet shards.",
    ),
    "fineweb2_zh": SourceSpec(
        name="fineweb2_zh",
        repo_id="HuggingFaceFW/fineweb-2",
        file_prefix="data/cmn_Hani/train/",
        suffix=".parquet",
        text_field="text",
        file_format="parquet",
        category="multilingual_web",
        note="Modern Chinese slice from FineWeb2; shards are large.",
    ),
    "skypile_zh": SourceSpec(
        name="skypile_zh",
        repo_id="Skywork/SkyPile-150B",
        file_prefix="data/",
        suffix=".jsonl",
        text_field="text",
        file_format="jsonl",
        category="chinese_web",
        note="Large open Chinese web corpus.",
    ),
    "openwebmath": SourceSpec(
        name="openwebmath",
        repo_id="open-web-math/open-web-math",
        file_prefix="data/",
        suffix=".parquet",
        text_field="text",
        file_format="parquet",
        category="math",
        note="Math-heavy pretraining corpus.",
    ),
    "codeparrot_code": SourceSpec(
        name="codeparrot_code",
        repo_id="codeparrot/codeparrot-clean",
        file_prefix="file-",
        suffix=".json.gz",
        text_field="content",
        file_format="json.gz",
        category="code",
        note="Accessible code pretraining corpus.",
    ),
}

DEFAULT_SOURCES = [
    "fineweb_edu",
    "fineweb2_en",
    "skypile_zh",
    "openwebmath",
    "codeparrot_code",
]

SOURCE_PRESETS: dict[str, list[str]] = {
    "stage0_smoke": [
        "fineweb_edu",
        "skypile_zh",
        "openwebmath",
        "codeparrot_code",
    ],
    "stage1_base_default": DEFAULT_SOURCES,
    "english_boost": [
        "fineweb_edu",
        "fineweb2_en",
    ],
    "english_boost_mirror": [
        "fineweb_edu",
        "fineweb2_en",
        "falcon_refinedweb_en",
    ],
    "chinese_boost": [
        "skypile_zh",
        "fineweb2_zh",
    ],
    "base_mirror_safe": [
        "fineweb_edu",
        "fineweb2_en",
        "falcon_refinedweb_en",
        "skypile_zh",
        "openwebmath",
        "codeparrot_code",
    ],
    "base_full_text": [
        "fineweb_edu",
        "fineweb2_en",
        "falcon_refinedweb_en",
        "fineweb2_zh",
        "skypile_zh",
        "openwebmath",
        "codeparrot_code",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Taiji pretraining mix v1")
    parser.add_argument(
        "--preset",
        choices=sorted(SOURCE_PRESETS),
        help="Named source bundle for staged 1B data expansion.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated source names. Known: {', '.join(sorted(SOURCES))}",
    )
    parser.add_argument("--shards-per-source", type=int, default=1)
    parser.add_argument("--max-records-per-source", type=int, default=250_000)
    parser.add_argument("--raw-dir", default="taiji_data/training_data/raw_pretrain_mix_v1")
    parser.add_argument("--output-dir", default="taiji_data/training_data/pretrain_mix_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_sources(raw_value: str) -> list[SourceSpec]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)}")
    return [SOURCES[name] for name in names]


def resolve_sources_from_args(*, preset: str | None, raw_sources: str) -> list[SourceSpec]:
    if preset:
        return [SOURCES[name] for name in SOURCE_PRESETS[preset]]
    return resolve_sources(raw_sources)


def list_matching_files(api: HfApi, source: SourceSpec, limit: int) -> list[str]:
    files = api.list_repo_files(source.repo_id, repo_type="dataset")
    matched = [
        file_name
        for file_name in files
        if file_name.startswith(source.file_prefix) and file_name.endswith(source.suffix)
    ]
    if source.name == "fineweb2_en":
        matched = [file_name for file_name in matched if "/eng_Latn/train/" in file_name]
    if source.name == "fineweb2_zh":
        matched = [file_name for file_name in matched if "/cmn_Hani/train/" in file_name]
    matched.sort()
    return matched[:limit]


def get_size_map(api: HfApi, source: SourceSpec, files: list[str]) -> dict[str, int]:
    info_map: dict[str, int] = {}
    if not files:
        return info_map
    infos = api.get_paths_info(source.repo_id, files, repo_type="dataset")
    for info in infos:
        info_map[info.path] = int(getattr(info, "size", 0) or 0)
    return info_map


def download_source_files(
    source: SourceSpec,
    files: list[str],
    raw_dir: Path,
    force: bool,
) -> list[Path]:
    downloaded: list[Path] = []
    target_dir = raw_dir / source.name
    target_dir.mkdir(parents=True, exist_ok=True)
    for remote_file in files:
        local_path = hf_hub_download(
            repo_id=source.repo_id,
            filename=remote_file,
            repo_type="dataset",
            local_dir=str(target_dir),
            force_download=force,
        )
        downloaded.append(Path(local_path))
    return downloaded


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield {"text": line}
            else:
                if isinstance(obj, dict):
                    yield obj


def iter_json_gz_lines(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield {"text": line}
            else:
                if isinstance(obj, dict):
                    yield obj


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return text.strip()


def extract_text(row: dict[str, Any], text_field: str) -> str:
    value = row.get(text_field)
    if isinstance(value, str):
        return normalize_text(value)
    return ""


def iter_source_rows(path: Path, source: SourceSpec) -> Iterable[dict[str, Any]]:
    if source.file_format == "parquet":
        return iter_parquet_rows(path)
    if source.file_format == "jsonl":
        return iter_json_lines(path)
    if source.file_format == "json.gz":
        return iter_json_gz_lines(path)
    raise ValueError(f"Unsupported format: {source.file_format}")


def normalize_downloads(
    source: SourceSpec,
    files: list[Path],
    output_dir: Path,
    max_records: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.name}.jsonl"
    written = 0
    chars = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            for row in iter_source_rows(path, source):
                text = extract_text(row, source.text_field)
                if len(text) < 32:
                    continue
                record = {
                    "text": text,
                    "source": source.name,
                    "category": source.category,
                    "repo_id": source.repo_id,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                chars += len(text)
                if written >= max_records:
                    return {"records": written, "chars": chars}
    return {"records": written, "chars": chars}


def write_manifest(
    manifest_path: Path,
    selected_sources: list[SourceSpec],
    file_map: dict[str, list[str]],
    size_map: dict[str, dict[str, int]],
    normalized_stats: dict[str, dict[str, int]],
) -> None:
    manifest = {
        "mix_name": "pretrain_mix_v1",
        "sources": [],
    }
    for source in selected_sources:
        manifest["sources"].append(
            {
                "name": source.name,
                "repo_id": source.repo_id,
                "category": source.category,
                "note": source.note,
                "files": file_map.get(source.name, []),
                "download_bytes": sum(size_map.get(source.name, {}).values()),
                "normalized": normalized_stats.get(source.name, {}),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(
    selected_sources: list[SourceSpec],
    file_map: dict[str, list[str]],
    size_map: dict[str, dict[str, int]],
    normalized_stats: dict[str, dict[str, int]],
) -> None:
    total_bytes = 0
    print("Selected sources:")
    for source in selected_sources:
        source_bytes = sum(size_map.get(source.name, {}).values())
        total_bytes += source_bytes
        print(
            f"- {source.name}: files={len(file_map.get(source.name, []))} "
            f"download_gb={source_bytes / (1024 ** 3):.2f} note={source.note}"
        )
        stats = normalized_stats.get(source.name)
        if stats:
            print(
                f"  normalized_records={stats['records']} "
                f"normalized_chars={stats['chars']}"
            )
    print(f"Estimated raw download size: {total_bytes / (1024 ** 3):.2f} GB")


def main() -> None:
    args = parse_args()
    selected_sources = resolve_sources_from_args(preset=args.preset, raw_sources=args.sources)
    api = HfApi()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    file_map: dict[str, list[str]] = {}
    size_map: dict[str, dict[str, int]] = {}
    normalized_stats: dict[str, dict[str, int]] = {}

    for source in selected_sources:
        files = list_matching_files(api, source, args.shards_per_source)
        if not files:
            raise RuntimeError(f"No files matched source={source.name}")
        file_map[source.name] = files
        size_map[source.name] = get_size_map(api, source, files)

    print_summary(selected_sources, file_map, size_map, normalized_stats)
    if args.dry_run:
        return

    downloaded_map: dict[str, list[Path]] = {}
    if not args.normalize_only:
        for source in selected_sources:
            print(f"Downloading {source.name}...")
            downloaded_map[source.name] = download_source_files(
                source=source,
                files=file_map[source.name],
                raw_dir=raw_dir,
                force=args.force,
            )

    if not args.download_only:
        for source in selected_sources:
            print(f"Normalizing {source.name}...")
            local_files = downloaded_map.get(source.name)
            if local_files is None:
                local_files = [raw_dir / source.name / file_name for file_name in file_map[source.name]]
            normalized_stats[source.name] = normalize_downloads(
                source=source,
                files=local_files,
                output_dir=output_dir,
                max_records=args.max_records_per_source,
            )

    print_summary(selected_sources, file_map, size_map, normalized_stats)
    write_manifest(
        manifest_path=output_dir / "manifest.json",
        selected_sources=selected_sources,
        file_map=file_map,
        size_map=size_map,
        normalized_stats=normalized_stats,
    )


if __name__ == "__main__":
    main()
