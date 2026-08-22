"""诊断十二：把 A2 的损伤拆成 experience 写入 vs replay 回放两段。

测量三段面板质量：
  base          —— 检查点原始
  after_exp     —— 只 experience（写入读出口，不回放）
  after_replay  —— experience + consolidate（完整 A2 路径）
同时逐 cycle 打印回放分量（value/novelty/familiarity/resonance/peakedness）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "training"))

import torch  # noqa: E402

import _seed_verify_common as common  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def panel_mean(model, judge) -> float:
    return float(common.measure_panel(model, judge)["overall_mean"])


def main() -> None:
    torch.manual_seed(7)
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    judge_docs = None

    model = common.load_model(ckpt)
    judge = common.calibrated_judge(model)
    scheduler = SeedSleepScheduler(model, judge)
    panel_texts = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(panel_texts, k=6)

    base = panel_mean(model, judge)
    print(f"base overall_mean = {base:.4f}", flush=True)

    for text in targets:
        scheduler.experience(text, learn=True)
    after_exp = panel_mean(model, judge)
    print(f"after experience-only overall_mean = {after_exp:.4f} "
          f"(Δ={after_exp - base:+.4f})", flush=True)

    # 单文本 8 cycles 的聚合分量，看 value/novelty 谁主导选择
    field = model.substrate.memory
    print(f"\nwrite_count={field.write_count} "
          f"replay_priority_threshold={field.config.replay_priority_threshold}",
          flush=True)
    for text in targets[:1]:
        scheduler.experience(text, learn=True)
        result = model.consolidate(cycles=8, learn=True)
        print(
            f"single-text consolidate: accepted={result.accepted}/8 "
            f"mean_priority={result.mean_priority:.3f} "
            f"mean_value={result.mean_value:.3f} "
            f"mean_novelty={result.mean_novelty:.3f} "
            f"mean_confidence={result.mean_confidence:.3f}",
            flush=True,
        )

    after_replay = panel_mean(model, judge)
    print(f"\nafter experience+replay overall_mean = {after_replay:.4f} "
          f"(Δ={after_replay - base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
