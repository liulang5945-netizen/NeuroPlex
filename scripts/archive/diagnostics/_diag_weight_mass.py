"""崩塌检查点与新生模型的突触权重总量对比。

假设：``synapse_decay=1e-5`` 在每个学习 tick 的 ``local_update`` 里乘
``(1 - decay)``，800K tick 累计 (1-1e-5)^800000 ≈ e^-8 ≈ 3e-4；若误差
补写不足以补偿，已学突触会被蒸发 → 全局崩塌。
"""
import sys

sys.path.insert(0, r"e:\Seed")

import torch

from seed import Seed

model = Seed.from_checkpoint(
    torch.load(r"e:\Seed\checkpoints\seed_corpus.pt", weights_only=False)
)
fresh = Seed(model.config)


def fabric_stats(name, fabric):
    groups = (
        ("decoder", fabric.decoders),
        ("consolidation", fabric.consolidation_decoders),
        ("transition", fabric.transitions),
        ("lateral", fabric.laterals),
    )
    for label, banks in groups:
        for index, bank in enumerate(banks):
            w = bank.edge_weight
            print(
                f"{name} {label}[{index}]: "
                f"mean|w|={float(w.abs().mean()):.6f} "
                f"frac_nonzero={float((w.abs() > 1e-6).float().mean()):.4f}"
            )


fabric_stats("collapsed", model.substrate.fabric)
fabric_stats("fresh", fresh.substrate.fabric)

for name, substrate in (("collapsed", model.substrate), ("fresh", fresh.substrate)):
    w = substrate.motor.synapses.edge_weight
    b = substrate.motor.bias
    print(
        f"{name} motor: mean|w|={float(w.abs().mean()):.6f} "
        f"frac_nonzero={float((w.abs() > 1e-6).float().mean()):.4f} "
        f"bias_mean|.|={float(b.abs().mean()):.6f}"
    )
