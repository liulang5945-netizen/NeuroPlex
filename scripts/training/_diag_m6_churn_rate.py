"""Throwaway: what does rewiring actually cost per accepted replay, per arm?

``test_consolidation_rewiring_terminates`` bounds events per accepted replay, but
the bound was picked from the 384-cycle bout on the M6 benchmark corpus, not from
the small config the unit test uses.  If the defect does not exceed the bound at
the test's cycle count, the test is decoration.  This measures both arms across
cycle counts so the threshold can be set from data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _diag_m6_write_basis import arm_patch  # noqa: E402

sys.path.insert(0, str(ROOT / "tests" / "taiji_native"))
import test_endogenous_replay as suite  # noqa: E402

from taiji import Taiji  # noqa: E402


def main() -> int:
    print("arm               cycles  accepted  events  per-accepted")
    for arm in ("shipped", "adapt-homeostasis"):
        for cycles in (24, 48, 96, 192):
            model = Taiji(suite._config())
            suite._store_episodes(model)
            with arm_patch(arm):
                summary = model.consolidate(cycles=cycles, learn=True)
            ratio = summary.structural_events / max(1, summary.accepted)
            print(
                "{:<17s} {:6d}  {:8d}  {:6d}  {:.3f}".format(
                    arm, cycles, summary.accepted, summary.structural_events, ratio
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
