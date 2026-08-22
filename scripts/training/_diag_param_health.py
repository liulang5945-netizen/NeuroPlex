"""检查崩塌检查点的参数健康度（NaN/Inf/范数溢出）。"""
import sys

sys.path.insert(0, r"e:\Seed")

import torch

from seed import Seed

model = Seed.from_checkpoint(
    torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", weights_only=False)
)
tensors = model.substrate.parameter_tensors()
print("n_tensors", len(tensors))
overall_max = 0.0
for index, tensor in enumerate(tensors):
    abs_max = float(tensor.abs().max())
    overall_max = max(overall_max, abs_max)
    nan = bool(torch.isnan(tensor).any())
    inf = bool(torch.isinf(tensor).any())
    if nan or inf or abs_max > 2.5:
        print(
            f"tensor[{index}] shape={tuple(tensor.shape)} "
            f"absmax={abs_max:.4f} nan={nan} inf={inf}"
        )
print("overall max abs =", overall_max)

# motor bias 分布：崩塌常表现为 bias 一边倒
bias = model.substrate.motor.bias
top = torch.topk(bias.detach(), 5)
print("motor bias top5:", [round(float(v), 3) for v in top.values])
print(
    "motor bias min/max:",
    float(bias.min()),
    float(bias.max()),
    "mean:",
    float(bias.mean()),
)
