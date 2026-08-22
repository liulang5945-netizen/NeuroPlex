"""在崩塌检查点上跑一段流，观察皮层活动率/误差/记忆置信度的动力学。"""
import sys

sys.path.insert(0, r"e:\Seed")

import torch

from seed import Seed

model = Seed.from_checkpoint(
    torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", weights_only=False)
)
text = (
    "问：你好。\n答：你好，很高兴见到你。"
    "水的沸点在标准大气压下是一百摄氏度。"
).encode("utf-8")

model.reset_dynamics(episode_id="diag")
step = model.observe(256, learn=False)
sums = [0.0, 0.0, 0.0, 0.0]
count = 0
for symbol in text:
    step = model.observe(int(symbol), learn=False)
    if step.prior_prediction is None:
        continue
    count += 1
    sums[0] += sum(step.activity_rates) / len(step.activity_rates)
    sums[1] += sum(step.local_error_norms) / len(step.local_error_norms)
    sums[2] += step.memory_recall.confidence
    sums[3] += float(step.surprise)

mean_activity, mean_error, mean_conf, mean_surprise = (
    value / count for value in sums
)
print(f"ticks={count}")
print(f"mean activity_rate = {mean_activity:.5f} (target=0.12)")
print(f"mean error_norm    = {mean_error:.5f}")
print(f"mean confidence    = {mean_conf:.5f}")
print(f"mean surprise      = {mean_surprise:.4f}")

# 阈值与抑制的病态检查
state = model.snapshot()
for index, region in enumerate(state.regions):
    print(
        f"region[{index}] threshold mean={float(region.threshold.mean()):.4f} "
        f"max={float(region.threshold.max()):.4f} "
        f"inhibition mean={float(region.inhibition.mean()):.4f} "
        f"activity frac={float((region.activity > region.threshold).float().mean()):.4f}"
    )
print(
    "memory threshold mean =",
    float(state.memory.threshold.mean()),
    "memory activity frac =",
    float((state.memory.activity > state.memory.threshold).float().mean()),
)
