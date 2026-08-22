"""诊断十四：扫描 replay_learning_scale（fast_scale=0 下即巩固强度），
测面板三组 Δ 与 6 条目标文本自身的质量变化。"""

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


def run(scale: float):
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    model.substrate.config = dataclasses.replace(
        model.substrate.config, replay_learning_scale=scale
    )
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)
    sched = SeedSleepScheduler(model, judge)
    panel = [i[2] for i in common.panel_texts_by_quality(judge)]
    targets = sched.select_for_sleep(panel, k=6)
    tbase = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    night = sched.night(targets, cycles_per_text=8, learn=True)
    after = common.measure_panel(model, judge)
    tafter = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    groups = {
        g: round(after["groups"][g]["mean"] - base["groups"][g]["mean"], 4)
        for g in base["groups"]
    }
    print(
        f"scale={scale}: overall={round(after['overall_mean']-base['overall_mean'],4)} "
        f"groups={groups} targets={round(tafter-tbase,4)} "
        f"accepted={night['accepted']:.0f}",
        flush=True,
    )


def main() -> None:
    for scale in (0.45, 0.20, 0.10, 0.05):
        run(scale)


if __name__ == "__main__":
    main()
