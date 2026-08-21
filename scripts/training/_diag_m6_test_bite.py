"""Throwaway: confirm the M6 tests actually fail when the defect is reinstated.

A regression test that passes under both the fix and the bug is worse than no
test at all, because it converts a silent failure into a false assurance.  The
``adapt-homeostasis`` arm in ``_diag_m6_write_basis`` reinstates the pre-fix
behaviour, so running the same assertions inside it is a direct check that each
test is load-bearing.
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


CASES = (
    "test_replay_reads_the_homeostatic_set_point_without_writing_it",
    "test_consolidation_rewiring_terminates",
    "test_consolidation_leaves_the_field_untouched",
)


def main() -> int:
    print("reinstating the defect via the 'adapt-homeostasis' arm\n")
    load_bearing = 0
    for name in CASES:
        case = getattr(suite, name)
        with arm_patch("adapt-homeostasis"):
            try:
                case()
            except AssertionError as error:
                load_bearing += 1
                print(f"FAILS (good)  {name}\n              {error}")
                continue
        print(f"PASSES (bad)  {name}")
    print(f"\n{load_bearing}/{len(CASES)} load-bearing under the defect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
