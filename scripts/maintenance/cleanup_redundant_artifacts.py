"""Remove explicitly superseded local artifacts from the research workspace.

The repository intentionally keeps runtime code and plans in Git while model
weights and datasets remain ignored local state.  This cleanup is therefore
an explicit allowlist, not a recursive "delete everything old" operation.

Default mode is a dry run.  Use ``--apply`` only after reviewing the listed
targets.  Production artifacts retained by this script are:

* the five dialogue neurons;
* the active C24v2 collaboration checkpoint;
* the hub checkpoint pair used by the local slow contract tests;
* ``foundation_v1_dual`` general neurons;
* canonical dialogue data and the audited HF candidate directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_PATH = PROJECT_ROOT / "reports" / "storage_cleanup_20260819.json"

KEEP_NEURON_FILES = {
    "neuron_zh_aug0_dialogue.pt",
    "neuron_zh_aug1_dialogue.pt",
    "neuron_zh_aug2_dialogue.pt",
    "neuron_zh_aug3_dialogue.pt",
    "neuron_zh_std0_dialogue.pt",
    "collab_v3_c24v2.ckpt.pt",
    # Required by tests/resonance/test_real_checkpoint_slow.py.
    "hub_collab_v2.ckpt.pt",
}

REMOVE_DIRECTORIES = [
    DATA_ROOT / "neurons" / "pre_t12_backup",
    DATA_ROOT / "foundation_v1_general",
    DATA_ROOT / "foundation_v1",
    DATA_ROOT / "neurons_joint",
    DATA_ROOT / "distill",
]

REMOVE_SIMPLE_ZH_FILES = {
    "alpaca_zh_sft.jsonl",
    "sft_shared_core.jsonl",
    "sft_unique_0.jsonl",
    "sft_unique_1.jsonl",
    "sft_unique_2.jsonl",
    "sft_unique_3.jsonl",
    "sft_unique_4.jsonl",
    "sft_shared_core_clean.jsonl",
    "sft_unique_0_clean.jsonl",
    "sft_unique_1_clean.jsonl",
    "sft_unique_2_clean.jsonl",
    "sft_unique_3_clean.jsonl",
    "sft_unique_4_clean.jsonl",
}


def _assert_in_data_root(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA_ROOT.resolve()):
        raise RuntimeError(f"Refusing target outside data root: {resolved}")


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    neurons_dir = DATA_ROOT / "neurons"
    if neurons_dir.exists():
        for path in neurons_dir.iterdir():
            if path.is_file() and path.name not in KEEP_NEURON_FILES:
                targets.append(path)
    targets.extend(path for path in REMOVE_DIRECTORIES if path.exists())
    simple_zh = DATA_ROOT / "simple_zh"
    targets.extend(
        simple_zh / name
        for name in sorted(REMOVE_SIMPLE_ZH_FILES)
        if (simple_zh / name).exists()
    )
    for path in targets:
        _assert_in_data_root(path)
    return sorted(targets, key=lambda path: str(path).lower())


def describe(targets: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in targets:
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": size,
            "size_gb": round(size / (1024 ** 3), 3),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually remove the listed targets")
    args = parser.parse_args()

    targets = collect_targets()
    rows = describe(targets)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "keep_neuron_files": sorted(KEEP_NEURON_FILES),
        "targets": rows,
        "target_count": len(rows),
        "target_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "target_size_gb": round(sum(int(row["size_bytes"]) for row in rows) / (1024 ** 3), 3),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.apply:
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"removed={len(targets)} targets")
    else:
        print("dry-run only; re-run with --apply to remove exactly these targets")


if __name__ == "__main__":
    main()
