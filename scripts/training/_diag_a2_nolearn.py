"""诊断十五：分离回放损伤来源——learn 开关 × cycles 剂量 × 同配置复跑确定性。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "training"))

import torch  # noqa: E402

import _seed_verify_common as common  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def _night_delta(learn: bool, cycles: int = 8):
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)
    sched = SeedSleepScheduler(model, judge)
    panel = [i[2] for i in common.panel_texts_by_quality(judge)]
    targets = sched.select_for_sleep(panel, k=6)
    tbase = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    night = sched.night(targets, cycles_per_text=cycles, learn=learn)
    after = common.measure_panel(model, judge)
    tafter = sum(float(judge.score(t)["quality"]) for t in targets) / len(targets)
    return round(after["overall_mean"] - base["overall_mean"], 4), round(
        tafter - tbase, 4
    ), night


def _det_delta(cycles: int = 8):
    """同配置同种子跑两次，检验损伤是否确定性（噪声主导则两次差异大）。"""
    results = []
    for run in (1, 2):
        ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
        torch.manual_seed(7)
        model = common.load_model(ckpt)
        judge = common.calibrated_judge(model)
        base = common.measure_panel(model, judge)
        sched = SeedSleepScheduler(model, judge)
        panel = [i[2] for i in common.panel_texts_by_quality(judge)]
        targets = sched.select_for_sleep(panel, k=6)
        sched.night(targets, cycles_per_text=cycles, learn=True)
        after = common.measure_panel(model, judge)
        results.append(round(after["overall_mean"] - base["overall_mean"], 4))
        print(f"  run{run}: {results[-1]}", flush=True)
    return results


def main() -> None:
    d, t, night = _night_delta(learn=False)
    print(f"learn=False cycles=8: overall={d} targets={t} accepted={night['accepted']:.0f}", flush=True)
    d, t, night = _night_delta(learn=True, cycles=1)
    print(f"learn=True cycles=1: overall={d} targets={t} accepted={night['accepted']:.0f}", flush=True)
    print("determinism learn=True cycles=8:", flush=True)
    _det_delta()


if __name__ == "__main__":
    main()
