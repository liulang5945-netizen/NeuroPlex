"""诊断十九：慢通路证据通道假说。检查点慢通路解码器全零（评估证据项
恒为 0）；一夜之后回放写入使其非零，评估在 consolidation_read_gain=1.0
下读入全部面板。若把读出增益压到近零后面板恢复基线，则损伤通道确认。"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "training"))

import torch  # noqa: E402

import _seed_verify_common as common  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def main() -> None:
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)

    sched = SeedSleepScheduler(model, judge)
    panel = [i[2] for i in common.panel_texts_by_quality(judge)]
    targets = sched.select_for_sleep(panel, k=6)
    sched.night(targets, cycles_per_text=8, learn=True)

    after = common.measure_panel(model, judge)
    model.substrate.config = dataclasses.replace(
        model.substrate.config, consolidation_read_gain=1e-3
    )
    muted = common.measure_panel(model, judge)

    for name in base["groups"]:
        d_after = after["groups"][name]["mean"] - base["groups"][name]["mean"]
        d_muted = muted["groups"][name]["mean"] - base["groups"][name]["mean"]
        print(
            f"{name}: after={d_after:+.4f} muted={d_muted:+.4f}", flush=True
        )
    print(
        f"overall: after={after['overall_mean']-base['overall_mean']:+.4f} "
        f"muted={muted['overall_mean']-base['overall_mean']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
