"""诊断十七：结构重布线假说。回放期唯一的剂量无关离散操作是 outcome
首写的 ``restructure``（快通路 decoder/transition 换线，不受 learn_scale
缩放）。把 structural_error_threshold 推到极高使其失效，看损伤底盘是否消失。"""

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


def run(threshold: float):
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    replaced = dataclasses.replace(
        model.substrate.config, structural_error_threshold=threshold
    )
    model.substrate.config = replaced
    model.substrate.fabric.config = replaced
    model.substrate.memory.config = replaced
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)
    sched = SeedSleepScheduler(model, judge)
    panel = [i[2] for i in common.panel_texts_by_quality(judge)]
    targets = sched.select_for_sleep(panel, k=6)
    tbase = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    before_struct = model.substrate.fabric.structural_events
    night = sched.night(targets, cycles_per_text=8, learn=True)
    struct = model.substrate.fabric.structural_events - before_struct
    after = common.measure_panel(model, judge)
    tafter = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    groups = {
        g: round(after["groups"][g]["mean"] - base["groups"][g]["mean"], 4)
        for g in base["groups"]
    }
    print(
        f"threshold={threshold}: overall={round(after['overall_mean']-base['overall_mean'],4)} "
        f"groups={groups} targets={round(tafter-tbase,4)} "
        f"accepted={night['accepted']:.0f} structural={struct}",
        flush=True,
    )


def main() -> None:
    run(1e9)  # 换线全禁（慢通路已断开，看残余损伤是否归零）


if __name__ == "__main__":
    main()
