"""诊断二十一：trust 门控 vs 换线全禁（同 seed 对照）。
A2 verify 无 seed 时 Δ=-0.105，诊断二十（全禁，seed=7）为 -0.025。
固定 seed=7 对比当前门控与 1e9 全禁，并打印首场 trust 判断。"""

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


def run(struct_threshold: float | None, slow_scale: float = 1.0):
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    torch.manual_seed(7)
    model = common.load_model(ckpt)
    kwargs = {"replay_outcome_slow_scale": slow_scale}
    if struct_threshold is not None:
        kwargs["structural_error_threshold"] = struct_threshold
    replaced = dataclasses.replace(model.substrate.config, **kwargs)
    model.substrate.config = replaced
    model.substrate.fabric.config = replaced
    model.substrate.memory.config = replaced
    cfg = model.substrate.config
    print(
        f"struct={struct_threshold} slow={slow_scale}: tick={model.tick} "
        f"maturity_gate={model.tick < cfg.replay_maturity_ticks}",
        flush=True,
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
        f"  overall={round(after['overall_mean']-base['overall_mean'],4)} "
        f"groups={groups} targets={round(tafter-tbase,4)} "
        f"accepted={night['accepted']:.0f} "
        f"struct={night.get('structural_events', -1)}",
        flush=True,
    )


def main() -> None:
    run(None, 1.0)      # 慢写开（当前）
    run(None, 0.0)      # 慢写关（诊断二十配置）


if __name__ == "__main__":
    main()
