#!/usr/bin/env python3
"""诊断廿三：确认残余损伤源 = cue-chain 慢通路写入。

arm_cue_off: experience(learn=True) + consolidate(replay_cue_chain=False)
若 Δ≈wake-only 的 +0.13，则成熟场 cue-chain 写入即唯一损伤源。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "training"))
sys.path.insert(0, str(REPO))

import _seed_verify_common as common  # noqa: E402
from seed import SeedSleepScheduler  # noqa: E402

CHECKPOINT = REPO / "checkpoints" / "seed_corpus_800k_backup.pt"


def main() -> None:
    model = common.load_model(str(CHECKPOINT))
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)["groups"]
    scheduler = SeedSleepScheduler(model, judge)
    panel = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(panel, k=6)
    for text in targets:
        scheduler.experience(text, learn=True)
        if model.substrate.memory.write_count > 0:
            model.substrate.consolidate(cycles=8, learn=True, replay_cue_chain=False)
    after = common.measure_panel(model, judge)["groups"]
    overall_b = sum(g["mean"] for g in base.values()) / len(base)
    overall_a = sum(g["mean"] for g in after.values()) / len(after)
    print("=== arm_cue_off（experience + consolidate 无 cue-chain）===", flush=True)
    for name in base:
        print(
            f"  {name}: Δ={after[name]['mean'] - base[name]['mean']:+.4f}",
            flush=True,
        )
    print(f"  overall Δ = {overall_a - overall_b:+.4f}", flush=True)


if __name__ == "__main__":
    main()
