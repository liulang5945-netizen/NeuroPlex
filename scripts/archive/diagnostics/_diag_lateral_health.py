"""崩塌检查点的侧向竞争权重体检。"""
import sys

sys.path.insert(0, r"e:\Seed")

import torch

from seed import Seed

model = Seed.from_checkpoint(
    torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", weights_only=False)
)
fabric = model.substrate.fabric
for index, lateral in enumerate(fabric.laterals):
    w = lateral.edge_weight
    print(
        f"lateral[{index}] shape={tuple(w.shape)} "
        f"mean={float(w.mean()):.6f} max={float(w.max()):.6f} "
        f"frac_nonzero={float((w > 0).float().mean()):.4f}"
    )

# 对照：刚初始化（未训练）模型的侧向权重
fresh = Seed(model.config)
for index, lateral in enumerate(fresh.substrate.fabric.laterals):
    w = lateral.edge_weight
    print(
        f"fresh lateral[{index}] mean={float(w.mean()):.6f} "
        f"max={float(w.max()):.6f} frac_nonzero={float((w > 0).float().mean()):.4f}"
    )
