"""Throwaway: read the M6 sweep reports and print one line per seed.

PowerShell mangles inline f-strings, so the summary lives in a file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = (
    "before_sleep",
    "full_replay",
    "no_replay_control",
    "engram_content_lesion",
    "recurrent_association_lesion",
)
SHORT = {
    "before_sleep": "pre",
    "full_replay": "full",
    "no_replay_control": "ctrl",
    "engram_content_lesion": "eLes",
    "recurrent_association_lesion": "rLes",
}


def main() -> int:
    seeds = [int(a) for a in sys.argv[1:]] or [11, 17, 29, 43, 61]
    passes = 0
    for seed in seeds:
        path = Path(f"reports/_sweep_{seed}.json")
        if not path.exists():
            print(f"seed={seed} MISSING {path}")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        cells = []
        for arm in ARMS:
            block = metrics[arm]
            events = block.get("structural_events", "-")
            cells.append(
                "{}={:.2f}/{}".format(
                    SHORT[arm], block["contingency_accuracy"], events
                )
            )
        passes += report["status"] == "pass"
        print(
            "seed={:<3d} {:5s} gain={:+.2f}  {}".format(
                seed,
                report["status"],
                metrics["accuracy_gain_over_control"],
                "  ".join(cells),
            )
        )
    print("\n{}/{} pass".format(passes, len(seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
