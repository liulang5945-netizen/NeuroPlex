"""探针：小模型学习后的 surprise 基线。"""
import sys

sys.path.insert(0, r"e:\Seed")

from seed import Seed, SeedConfig
from taiji import TaijiConfig

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
        seed=43,
    )
)
data = (
    "问：你好。\n答：你好，很高兴见到你。"
    "水的沸点在标准大气压下是一百摄氏度。"
).encode("utf-8")

for epochs in (6, 12, 30, 60):
    model = Seed(config)
    model.learn_bytes(data, epochs=epochs)
    print(epochs, round(model.score_bytes(data)["mean_surprise"], 4))
