"""长时间流式训练下突触权重不得蒸发的守护测试。

实测根因（800K 语料训练）：``synapse_decay=1e-5`` 以全局乘性方式挂在
每个学习 tick 的 ``local_update`` 上，(1-1e-5)^800000 ≈ e^-8 ≈ 3e-4。
模型越学越好时误差补写变小，而蒸发速率不变，皮层解码/转移权重从
0.052 跌到 1.7e-5（约 1/3000），holdout 惊讶度从 2.88 崩到 4.10。

衰减属于可塑性事件，就必须随该事件的资格迹（eligibility）门控：
突触前沉默的接触可以放松，被本次事件点亮的接触不得被抽走。
"""

from __future__ import annotations

import torch

from seed import Seed, SeedConfig
from taiji import TaijiConfig


def _tiny_config() -> SeedConfig:
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
            seed=43,
        )
    )


def _decoder_mass(model: Seed) -> float:
    fabric = model.substrate.fabric
    return sum(
        float(bank.edge_weight.abs().mean()) for bank in fabric.decoders
    ) / len(fabric.decoders)


def test_streaming_training_does_not_evaporate_cortical_synapses() -> None:
    model = Seed(_tiny_config())
    data = (
        "问：你好。\n答：你好，很高兴见到你。"
        "水的沸点在标准大气压下是一百摄氏度。"
    ).encode("utf-8")

    # 先建立记忆，再长时间持续暴露于同一内容（学会之后误差变小，
    # 正是旧全局衰减蒸发占主导的区间）。
    model.learn_bytes(data, epochs=6)
    learned_mass = _decoder_mass(model)
    learned_surprise = model.score_bytes(data)["mean_surprise"]

    for _ in range(150):
        model.reset_dynamics(episode_id="stream")
        for symbol in model.substrate.sensor.symbols(data):
            model.observe(symbol, learn=True)

    final_mass = _decoder_mass(model)
    assert final_mass >= 0.9 * learned_mass, (
        f"皮层解码权重在持续学习中蒸发：{learned_mass:.5f} → {final_mass:.5f}"
    )

    # 记忆必须仍然可读：蒸发的权重不会自我报告，行为才是证据。
    final_surprise = model.score_bytes(data)["mean_surprise"]
    assert final_surprise <= learned_surprise + 0.15, (
        f"长期暴露后模型丢掉了已学内容："
        f"surprise {learned_surprise:.3f} → {final_surprise:.3f}"
    )
