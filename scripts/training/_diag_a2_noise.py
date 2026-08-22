"""诊断十八：回放噪声假说。最差文本恰是场存储最差的 engram，
再生噪声 0.75 把补全拉向垃圾吸引子→回放教会模型自己的错误。
扫描 replay_noise_scale ∈ {0.75, 0.25, 0.0}（确定性再生）。"""

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


def run(noise: float):
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    replaced = dataclasses.replace(model.substrate.config, replay_noise_scale=noise)
    model.substrate.config = replaced
    model.substrate.fabric.config = replaced
    model.substrate.memory.config = replaced
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
        f"noise={noise}: overall={round(after['overall_mean']-base['overall_mean'],4)} "
        f"groups={groups} targets={round(tafter-tbase,4)} "
        f"accepted={night['accepted']:.0f}",
        flush=True,
    )


def main() -> None:
    for noise in (0.25, 0.0):
        run(noise)


if __name__ == "__main__":
    main()
