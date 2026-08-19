#!/usr/bin/env python3
"""Run the one-command NeuroPlex population bootstrap demo.

This is an engineering demo, not a trained language model showcase. It uses
the same tiny deterministic population as ``verify_population_baseline.py``
and demonstrates the public runtime contract without downloading or loading
private checkpoints:

    experience batch -> resonance field -> sparse routing -> state round-trip

The report is intentionally marked ``synthetic_probe_only``. Use
``verify_population_baseline.py`` when the full JSON diagnostics are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_population_baseline import DEFAULT_SEED, run_baseline  # noqa: E402


def _summary(report: dict) -> dict:
    dense = report["metrics"]["dense"]
    sparse = report["metrics"]["sparse"]
    route = sparse["route"]
    return {
        "architecture": report["architecture"],
        "quality_scope": report["quality_scope"],
        "seed": report["seed"],
        "population": report["config"]["population"],
        "dense_ppl": dense["ppl"],
        "sparse_ppl": sparse["ppl"],
        "dense_active_round2": len(report["config"]["population"]),
        "sparse_active_round2": route["average_active_round2"],
        "selected_counts": route["selected_counts"],
        "cortex_roundtrip_ok": report["checks"]["cortex_roundtrip_ok"],
        "status": report["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic NeuroPlex population bootstrap demo",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for the compact demo summary",
    )
    args = parser.parse_args()

    report = run_baseline(seed=args.seed, include_api=False)
    summary = _summary(report)

    print("NeuroPlex population bootstrap demo")
    print("=" * 40)
    print("scope: synthetic_probe_only")
    print(f"population: {len(summary['population'])} neurons")
    print(
        "resonance: dense PPL={dense:.3f} | sparse PPL={sparse:.3f}".format(
            dense=summary["dense_ppl"], sparse=summary["sparse_ppl"],
        )
    )
    print(
        "routing: {dense:.0f} -> {sparse:.1f} active neurons in round 2".format(
            dense=summary["dense_active_round2"],
            sparse=summary["sparse_active_round2"],
        )
    )
    print(f"selected: {summary['selected_counts']}")
    print(f"state round-trip: {'PASS' if summary['cortex_roundtrip_ok'] else 'FAIL'}")
    print(f"status: {summary['status']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"summary: {args.json_out}")

    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
