#!/usr/bin/env python3
"""诊断廿五：A3 漂移拆分——纯情节写入（fabric 不学习）是否漂移面板。

arm_obs_only: observe(learn=False) + settle(learn_memory=True) + consolidate
              —— 只写情节、只回放，清醒预测器理论零更新。
若面板仍漂移，漂移来自评分特征（记忆熟悉度）而非参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "training"))
sys.path.insert(0, str(REPO))

import _seed_verify_common as common  # noqa: E402
from seed import SeedSleepScheduler  # noqa: E402
from seed.sleep import top_candidates  # noqa: E402

CHECKPOINT = REPO / "checkpoints" / "seed_corpus_800k_backup.pt"


def observe_only_experience(scheduler, text):
    seed = scheduler.seed
    quality = float(scheduler.judge.score(text)["quality"])
    boundary = seed.substrate.config.boundary_symbol
    seed.reset_dynamics(episode_id="sleep-obs-only")
    seed.observe(boundary, learn=False)
    for symbol in text[:64]:
        seed.observe(int(symbol), learn=False)
        probabilities = seed.snapshot().motor_probabilities
        candidates = top_candidates(probabilities, 8)
        seed.act(candidates, sample=False)
        seed.settle_action(
            quality, learn=False, learn_memory=True, provenance="experienced"
        )
    seed.observe(boundary, learn=False)


def main() -> None:
    model = common.load_model(str(CHECKPOINT))
    judge = common.calibrated_judge(model)
    base = common.measure_panel(model, judge)
    scheduler = SeedSleepScheduler(model, judge)
    panel = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(panel, k=4)
    before = [v.detach().clone() for v in model.substrate.parameter_tensors()]
    for _round in range(8):
        for text in targets:
            observe_only_experience(scheduler, text)
            if model.substrate.memory.write_count > 0:
                model.substrate.consolidate(cycles=4, learn=True)
    after = common.measure_panel(model, judge)
    param_delta = sum(
        float((v - b).abs().sum())
        for v, b in zip(model.substrate.parameter_tensors(), before)
    )
    print(f"参数变化总量 = {param_delta:.6f}", flush=True)
    for name in base["groups"]:
        delta = (
            after["groups"][name]["mean"] - base["groups"][name]["mean"]
        )
        print(f"  {name}: Δ={delta:+.4f}", flush=True)
    print(f"  overall Δ = {after['overall_mean'] - base['overall_mean']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
