"""守护：真实经历的写入不得破坏清醒预测能力（sleep safety invariant）。

800K 成人模型崩塌复盘（阶段 3）：``experience`` 每 tick 一次情景写入，
``memory.write`` 以 ``episodic_readout_learning_rate × strength`` 训练
action/outcome/cortical readout。同一段文本反复经历时上下文高度重复，
且自我评估质量为负值（quality≈-3）作为 reward 注入——几百次重复写入把
读出磨向最后方向、符号反转，召回证据经 ``memory_read_gain`` 注入运动
预测，清醒惊讶度从 3.1 涨到 10+。机制缺陷在写入路径本身：读出只应在
事件确实新颖时强化（海马只把"新"事件固化为可读写出），熟悉事件的重复
不得冲刷已学读出。

测试用 toy 模型复现该路径：大幅负 reward 的重复真实经历后，清醒序列的
预测能力必须保住。
"""

from __future__ import annotations

import torch

from taiji import Taiji, TaijiConfig


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=8,
        motor_fan_in=12,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=12,
        memory_meta_dim=12,
        memory_iterations=2,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=6,
        seed=31,
    )


STREAM = bytes((ord("a") + i % 6) for i in range(48))


def _trained() -> Taiji:
    model = Taiji(_config(), episode_id="waking")
    for _ in range(6):
        model.reset_dynamics(episode_id="waking")
        for symbol in STREAM:
            model.observe(symbol, learn=True, learn_motor=False)
    return model


def _surprise(model: Taiji) -> float:
    return model.score_bytes(STREAM)["mean_surprise"]


def test_lived_experience_does_not_damage_waking_competence() -> None:
    model = _trained()
    baseline = _surprise(model)

    # 同一段内容反复"经历"：大幅负 reward（自我评估质量的真实形态），
    # 每 tick 一次真实结算与情景写入
    for trial in range(6):
        model.reset_dynamics(episode_id=f"lived-{trial}")
        model.observe(model.config.boundary_symbol, learn=False)
        for symbol in STREAM:
            model.observe(int(symbol), learn=False)
            probabilities = model.snapshot().motor_probabilities
            candidates = [
                int(index)
                for index in torch.argsort(probabilities, descending=True)[:8]
            ]
            model.act(candidates, sample=False)
            model.settle_action(-3.0, learn=False, learn_memory=True)
        model.observe(model.config.boundary_symbol, learn=False)

    after = _surprise(model)
    assert after <= baseline + 0.30, (
        f"重复真实经历把清醒惊讶度从 {baseline:.3f} 推到 {after:.3f}——"
        "情景写入路径正在破坏清醒能力"
    )
