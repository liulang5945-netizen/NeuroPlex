"""Inspect the phase-1 evaluation report."""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[2] / "reports" / "seed_corpus_eval_phase1.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    print("panel:")
    for group, entry in report["panel"].items():
        print(f"  {group}: mean_surprise={entry['mean']:.3f} std={entry['std']:.3f}")
    print()
    for sample in report["samples"]:
        print(f"[{sample['group']}] {sample['prompt']}")
        print(f"  -> {sample['continuation']!r}")


if __name__ == "__main__":
    main()
