"""阶段 3 原生 play 环境的失败测试。

计划要求：实现 ``TaijiEnvironment`` 协议的文本交互环境——模型 ``act()``
选择行为（选择探索哪个主题/补全主题字节），环境返回 sensation+reward，
用 observe/act/settle_action 标准序列驱动。环境转移必须真实依赖动作
（不是脚本规定的固定流），且带防锁定的探索支架（B1-bis 同语义：
连续选择同一主题超限后该主题暂时退出候选）。
"""

from __future__ import annotations

from taiji import TaijiEnvironment

from seed import Seed, SeedConfig
from seed.environments import TopicWorld, play
from taiji import TaijiConfig


def _small_seed() -> Seed:
    config = SeedConfig(
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
            seed=99,
        )
    )
    return Seed(config, episode_id="play-test")


TOPICS = [
    b"the sun rises in the east",
    b"water freezes at zero degrees",
    b"birds fly south in winter",
    b"stones sink in deep water",
]


def test_topic_world_satisfies_native_protocol() -> None:
    world = TopicWorld(TOPICS)

    assert isinstance(world, TaijiEnvironment)
    sensation, actions = world.reset()
    assert isinstance(sensation, int)
    assert len(actions) > 0
    # 候选动作就是各主题可区分前缀的收口字节
    assert set(actions) == {topic[0] for topic in TOPICS}


def test_world_transition_depends_on_action() -> None:
    world = TopicWorld(TOPICS)

    world.reset()
    pick = TOPICS[2][0]
    outcome = world.step(pick)

    # 动作决定环境走向：选中主题的字节被逐个呈现为 sensation
    assert outcome.terminal is False
    assert outcome.sensation == pick
    rest = TOPICS[2][1:]
    for expected in rest:
        outcome = world.step(ord("x"))  # 补全动作无需"正确"才有下一个感觉
        assert outcome.sensation == expected
    assert outcome.terminal is True
    assert world.last_topic_index == 2
    assert world.topics_visited == [2]


def test_forced_switch_prevents_lock_in() -> None:
    world = TopicWorld(TOPICS, force_switch_streak=2)

    world.reset()
    first = TOPICS[0][0]
    for _ in range(2):
        while not world.choosing:
            world.step(first)
        world.step(first)

    assert world.streak == 2
    while not world.choosing:
        world.step(first)
    _, actions = world.reset_choice()
    assert first not in actions, "连续选中同一主题超限后必须暂时退出候选"
    _, actions = world.reset()
    assert first not in actions
    assert world.last_topic_index is None


def test_recency_window_keeps_exploration_moving() -> None:
    world = TopicWorld(TOPICS, recency_window=2)

    world.reset()
    world.step(TOPICS[0][0])
    while not world.choosing:
        world.step(0)
    _, actions = world.reset_choice()
    assert TOPICS[0][0] not in actions, "刚访问过的主题必须暂时退出候选"
    world.step(TOPICS[1][0])
    while not world.choosing:
        world.step(0)
    _, actions = world.reset_choice()
    # 窗口=2：最近两个主题（0、1）都被挡住，候选来自剩余主题且永不为空
    assert TOPICS[0][0] not in actions
    assert TOPICS[1][0] not in actions
    assert set(actions) == {TOPICS[2][0], TOPICS[3][0]}
    world.step(TOPICS[2][0])
    while not world.choosing:
        world.step(0)
    _, actions = world.reset_choice()
    # 滑窗：最近两个是 1、2，主题 0 重新回到候选（探索不会死锁）
    assert TOPICS[0][0] in actions
    assert TOPICS[2][0] not in actions


def test_play_session_runs_standard_sequence_and_reports_stats() -> None:
    model = _small_seed()
    world = TopicWorld(TOPICS, force_switch_streak=2)

    stats = play(model, world, episodes=6, sample=False, learn=True)

    assert stats["episodes"] == 6
    assert stats["crashes"] == 0
    assert stats["actions"] > 0
    # 结算链无悬空：play 之后可以立即巩固（情景写入真实发生）
    model.consolidate(cycles=2, learn=False)
    # 防锁定支架生效：6 集里必然被迫换过主题
    assert stats["distinct_topics"] >= 2
    assert len(stats["topic_sequence"]) == 6
