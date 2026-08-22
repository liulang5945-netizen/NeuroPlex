"""阶段 2 原生自我评估器官（seed/judge.py）的失败测试。

计划要求：taiji 的 region prediction error + episodic recall confidence
天然构成自指信号。``SeedJudge`` 对给定文本输出可排序的质量分，基于
surprise、错误累积、场熟悉度的组合，权重由局部学习获得，不引入外部
评分模型。判分只读不写：不得改变已学参数。
"""

from __future__ import annotations

import math

import torch

from seed import Seed, SeedConfig
from seed.judge import SeedJudge
from taiji import TaijiConfig


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


def _trained_seed() -> Seed:
    model = Seed(_small_config())
    data = (
        "问：你好。\n答：你好，很高兴见到你。"
        "水的沸点在标准大气压下是一百摄氏度。"
    ).encode("utf-8")
    model.learn_bytes(data, epochs=3)
    return model


def test_judge_report_contains_self_referential_signals() -> None:
    judge = SeedJudge(_trained_seed())
    report = judge.score("水的沸点是一百摄氏度。".encode("utf-8"))

    for key in (
        "quality",
        "mean_surprise",
        "mean_error_norm",
        "mean_confidence",
        "accuracy",
        "observations",
    ):
        assert key in report, f"judge report 缺少自指信号 {key}"
    assert report["observations"] > 0
    for key in ("quality", "mean_surprise", "mean_error_norm",
                "mean_confidence", "accuracy"):
        assert math.isfinite(report[key]), f"{key} 必须是有限实数"
    assert 0.0 <= report["accuracy"] <= 1.0


def test_judge_is_read_only_and_deterministic() -> None:
    model = _trained_seed()
    judge = SeedJudge(model)
    text = "问：推荐一本书。\n答：可以读一读《活着》。".encode("utf-8")

    before = [tensor.detach().clone() for tensor in model.substrate.parameter_tensors()]
    first = judge.score(text)
    second = judge.score(text)
    after = model.substrate.parameter_tensors()

    assert first["quality"] == second["quality"]
    assert first["mean_surprise"] == second["mean_surprise"]
    assert len(before) == len(after)
    for previous, current in zip(before, after):
        assert torch.equal(previous, current.detach()), "judge 判分不得改变已学参数"


def test_judge_ranks_learned_text_above_noise() -> None:
    model = _trained_seed()
    judge = SeedJudge(model)

    learned = judge.score(
        "问：你好。\n答：你好，很高兴见到你。".encode("utf-8")
    )
    # 同一批字节打乱顺序后不再具备学到的结构，质量必须更低。
    raw = "问：你好。\n答：你好，很高兴见到你。".encode("utf-8")
    permuted = torch.randperm(len(raw), generator=torch.Generator().manual_seed(7))
    shuffled = bytes(raw[index] for index in permuted.tolist())
    corrupted = judge.score(shuffled)

    assert learned["quality"] > corrupted["quality"]


def test_local_calibration_learns_weights_from_known_pairs() -> None:
    judge = SeedJudge(_trained_seed())
    assert judge.weights.shape == (4,)

    # 构造已知质量对的合成校准集：目标是 surprise 越低质量越高。
    pairs = []
    generator = torch.Generator().manual_seed(3)
    for _ in range(24):
        surprise = 1.0 + 4.0 * float(torch.rand(1, generator=generator).item())
        features = torch.tensor(
            [surprise, surprise * 0.4, 0.5 - surprise * 0.05, max(0.0, 0.9 - surprise * 0.15)]
        )
        target = -surprise  # 已知好/坏：低 surprise = 好
        pairs.append((features, target))

    accuracy = judge.calibrate(pairs)

    assert accuracy >= 0.7
    assert judge.weights.shape == (4,)
    # 学到的权重必须把 surprise 方向压成负贡献。
    assert float(judge.weights[0]) < 0.0


def test_judge_rejects_empty_text() -> None:
    judge = SeedJudge(_trained_seed())
    try:
        judge.score(b"")
    except ValueError:
        return
    raise AssertionError("空文本没有可判定的观测，必须拒绝")
