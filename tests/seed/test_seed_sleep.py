"""阶段 3 原生 sleep 调度的失败测试（judge 驱动巩固）。

计划要求：由阶段 2 的 judge 信号选择巩固对象（判定差的样本/模式优先）
——"眼睛驱动手"的显式接线；机制必须落在内生 replay（``consolidate``），
禁止 Python replay list。原生路径：纯观察训练不写情景（无 act/settle），
因此睡眠调度器必须先用 ``act`` + ``settle_action``（reward 由自我评估给出）
+ 下一个 ``observe`` 产生内生情景写入，再让场按自身优先级回放。
"""

from __future__ import annotations

import torch

from seed import Seed, SeedConfig
from seed.judge import SeedJudge
from seed.sleep import SeedSleepScheduler
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
            seed=45,
        )
    )


PATTERN = "ababcdcdabcd"


def _pattern_seed() -> Seed:
    """先让皮层熟悉该模式的字节结构，但不产生任何情景写入。"""

    model = Seed(_small_config(), episode_id="sleep-test")
    model.learn_bytes(PATTERN.encode("ascii"), epochs=8)
    return model


def test_scheduler_experience_creates_endogenous_episodic_write() -> None:
    model = _pattern_seed()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    report = scheduler.experience(PATTERN.encode("ascii"))

    # reward 必须来自自我评估（可有限、可排序），动作必须真实结算
    assert "reward" in report and "actions" in report
    assert report["actions"] > 0
    assert report["reward"] is not None and abs(report["reward"]) < 10.0
    # 结算链闭合：无悬空动作/经历，且产生了可供内生回放的情景写入
    # （consolidate 对 write_count<=0 会直接抛错，故成功即证明写入存在）
    result = model.consolidate(cycles=2, learn=False)
    assert result.cycles == 2


def test_scheduler_night_consolidates_and_reports_endogenous_replay() -> None:
    model = _pattern_seed()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    night = scheduler.night(
        [PATTERN.encode("ascii")], cycles_per_text=4, learn=True
    )

    assert night["texts"] == 1
    assert night["cycles"] == 4
    # 内生回放必须有被场自身优先级门接受的条目（无外部回放列表）
    assert night["accepted"] >= 1
    # 调度器记录的是基质的结算报告字段，不是自造统计
    assert "mean_priority" in night


def test_judge_driven_selection_prefers_worse_patterns() -> None:
    model = _pattern_seed()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    good = PATTERN.encode("ascii")
    # 打乱字节破坏结构 → 自我评估质量更差
    raw = bytearray(good)
    generator = torch.Generator().manual_seed(11)
    permuted = torch.randperm(len(raw), generator=generator).tolist()
    bad = bytes(raw[index] for index in permuted)

    selected = scheduler.select_for_sleep([good, bad], k=1)

    assert selected == [bad], "judge 判定更差的样本必须优先巩固"


def test_sleep_read_only_when_learn_disabled() -> None:
    model = _pattern_seed()
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)

    before = [
        tensor.detach().clone()
        for tensor in model.substrate.parameter_tensors()
    ]
    scheduler.night([PATTERN.encode("ascii")], cycles_per_text=2, learn=False)
    after = model.substrate.parameter_tensors()

    for previous, current in zip(before, after):
        assert torch.equal(previous, current.detach()), (
            "learn=False 的睡眠不得改变已学参数"
        )
