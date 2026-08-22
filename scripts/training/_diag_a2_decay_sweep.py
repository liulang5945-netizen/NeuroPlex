"""诊断十三：扫描 memory_confidence_decay，测 experience 写入对面板原始
surprise（预测损伤，不含 judge 置信特征污染）的影响。"""

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


def panel_texts() -> list:
    panel = common.load_eval_module().PROMPT_PANEL
    texts = []
    for prompts in panel.values():
        for text in prompts:
            texts.append(text.encode("utf-8"))
    return texts


def raw_panel_surprise(model, texts) -> float:
    total = 0.0
    for data in texts:
        total += float(model.score_bytes(data)["mean_surprise"])
    return total / len(texts)


def set_decay(model, decay: float) -> None:
    model.substrate.config = dataclasses.replace(
        model.substrate.config, memory_confidence_decay=decay
    )
    model.substrate.memory.config = dataclasses.replace(
        model.substrate.memory.config, memory_confidence_decay=decay
    )


def main() -> None:
    torch.manual_seed(7)
    ckpt = str(REPO / "checkpoints" / "seed_corpus.pt")
    texts = panel_texts()

    # 一次性在干净模型上校准 judge 并选出最差 6 条
    selector = common.load_model(ckpt)
    judge = common.calibrated_judge(selector)
    scheduler = SeedSleepScheduler(selector, judge)
    all_texts = [item[2] for item in common.panel_texts_by_quality(judge)]
    targets = scheduler.select_for_sleep(all_texts, k=6)

    base_model = common.load_model(ckpt)
    base = raw_panel_surprise(base_model, texts)
    print(f"baseline raw panel surprise = {base:.4f}", flush=True)

    from seed.judge import SeedJudge
    for decay in (5e-4, 2e-3, 5e-3, 1e-2):
        model = common.load_model(ckpt)
        set_decay(model, decay)
        sch = SeedSleepScheduler(model, SeedJudge(model))
        for text in targets:
            sch.experience(text, learn=True)
        after = raw_panel_surprise(model, texts)
        print(
            f"decay={decay:.0e}: after={after:.4f} \u0394={after - base:+.4f} "
            f"wc={model.substrate.memory.write_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
