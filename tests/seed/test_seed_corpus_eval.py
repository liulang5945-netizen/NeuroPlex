"""阶段 1 评估管线的失败测试。

计划要求：``scripts/training/eval_seed_corpus.py`` 以 ``score_bytes`` 的
next-byte accuracy/surprise 换算文本 PPL，固定 prompt 面板（BOOTSTRAP A1 的
dialogue/knowledge/unfamiliar 三组共 24 条真实面板）生成质量评分，并与冻结
``neuroplex`` 基线同口径对比，结果落盘 ``reports/``。
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from seed import Seed, SeedConfig
from taiji import TaijiConfig

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "eval_seed_corpus.py"


def _module():
    spec = importlib.util.spec_from_file_location("eval_seed_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(12, 8),
            synapse_fan_in=4,
            motor_fan_in=6,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=6,
            memory_meta_dim=6,
            memory_iterations=2,
            memory_time_dim=4,
            memory_episode_dim=4,
            lateral_fan_in=4,
            seed=41,
        )
    )


def test_panel_matches_the_frozen_a1_contract() -> None:
    evaluator = _module()

    assert set(evaluator.PROMPT_PANEL) == {
        "dialogue",
        "knowledge",
        "unfamiliar",
    }
    for prompts in evaluator.PROMPT_PANEL.values():
        assert len(prompts) == 8
    assert evaluator.PROMPT_PANEL["dialogue"][0] == "你好，请问今天感觉怎么样？"
    assert evaluator.PROMPT_PANEL["knowledge"][0].startswith("水的沸点是多少")
    assert evaluator.PROMPT_PANEL["unfamiliar"][0].startswith("请用古亚述语")


def test_evaluation_reports_ppl_panel_scores_and_generation(tmp_path) -> None:
    evaluator = _module()
    model = Seed(_small_config(), episode_id="eval-test")
    holdout = "水的沸点在标准大气压下是一百摄氏度。".encode("utf-8")
    report_path = tmp_path / "eval.json"

    report = evaluator.evaluate_seed(
        model,
        holdout_bytes=holdout,
        report_path=report_path,
        generation_length=12,
    )

    # score_bytes surprise is in nats per byte: PPL must be exp(mean_surprise)
    scored = model.score_bytes(holdout)
    assert report["holdout"]["byte_ppl"] == math.exp(scored["mean_surprise"])
    assert report["holdout"]["accuracy"] == scored["accuracy"]

    for group in ("dialogue", "knowledge", "unfamiliar"):
        entry = report["panel"][group]
        assert len(entry["surprises"]) == 8
        assert entry["mean"] > 0.0
        assert "std" in entry

    assert len(report["samples"]) == 24
    assert all("prompt" in sample and "continuation" in sample
               for sample in report["samples"])

    baseline = report["neuroplex_baseline_reference"]
    assert baseline["report"] == "a1_judge_nll_std_real_20260820.json"
    assert baseline["groups"]["dialogue"]["std"] > 0.05

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["holdout"] == report["holdout"]
