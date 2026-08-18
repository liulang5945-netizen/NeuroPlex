"""ckpt 迁移脚本：v2 → v3 格式。

将旧格式 checkpoint 转换为新格式，使其原生支持：
- lm_head 低秩分解（可选，默认保留传统模式）
- side_channels → excite_channels + inhibit_channels
- 新增 neuron_type / refractory_counter 等字段

使用方法：
    python scripts/migrate_ckpt_v3.py --neurons-dir data/neurons
    python scripts/migrate_ckpt_v3.py --neurons-dir data/neurons --enable-low-rank --rank 64

策略：
- 默认保守：保留旧 lm_head（lm_head_rank=0），仅重命名 side_channels
- --enable-low-rank：把旧 lm_head 作为 W_base，新增零初始化的 delta_u/delta_v
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# 让脚本能直接运行：把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuroplex.resonance.config import NeuronConfig


def migrate_state_dict(
    sd: dict,
    cfg: NeuronConfig,
    enable_low_rank: bool = False,
    rank: int = 64,
) -> tuple[dict, NeuronConfig]:
    """迁移单个 state_dict。

    Args:
        sd: 旧 state_dict
        cfg: 旧 NeuronConfig
        enable_low_rank: 是否启用低秩分解
        rank: 低秩分解的 rank

    Returns:
        (new_sd, new_cfg)
    """
    new_sd = dict(sd)
    new_cfg = NeuronConfig(**cfg.__dict__)  # 拷贝

    # 补全新 cfg 字段（如果旧 cfg 没有这些字段）
    if not hasattr(new_cfg, "neuron_type"):
        new_cfg.neuron_type = "excitatory"
    if not hasattr(new_cfg, "refractory_cooldown"):
        new_cfg.refractory_cooldown = 2
    if not hasattr(new_cfg, "lm_head_rank"):
        new_cfg.lm_head_rank = 0  # 默认保留传统模式

    sd_keys = set(new_sd.keys())

    # 1. side_channels → excite_channels
    side_keys = [k for k in sd_keys if k.startswith("side_channels.")]
    for k in side_keys:
        new_k = k.replace("side_channels.", "excite_channels.", 1)
        new_sd[new_k] = new_sd.pop(k)

    # 2. lm_head 处理
    if enable_low_rank and "lm_head.weight" in sd_keys:
        # 启用低秩：把旧 lm_head 作为 W_base 的初始值
        # 但 W_base 应该是共享的，这里只保留 lm_head.weight 作为 per-neuron 的"伪 W_base"
        # 真正的共享 W_base 由 Cortex 在运行时注入
        # 迁移阶段：删除 lm_head.weight，新增零初始化的 delta_u/delta_v
        old_lm_head_weight = new_sd.pop("lm_head.weight")
        # 保留旧权重到一个特殊键，供 Cortex 加载时作为 W_base 候选
        new_sd["_migrated_lm_head_weight"] = old_lm_head_weight
        # 新增零初始化的 delta_u/delta_v
        hidden = new_cfg.hidden_size
        vocab = new_cfg.vocab_size
        new_sd["lm_head_delta_u.weight"] = torch.zeros(rank, hidden)
        new_sd["lm_head_delta_v.weight"] = torch.zeros(vocab, rank)
        new_cfg.lm_head_rank = rank
        print(f"    启用低秩分解 (rank={rank})，旧 lm_head 保留为 _migrated_lm_head_weight")
    else:
        # 保留传统模式
        new_cfg.lm_head_rank = 0
        if "lm_head.weight" not in sd_keys:
            print(f"    警告：lm_head.weight 不存在，可能需要手动检查")

    return new_sd, new_cfg


def migrate_ckpt(
    ckpt_path: str,
    output_path: str | None = None,
    enable_low_rank: bool = False,
    rank: int = 64,
    dry_run: bool = False,
) -> bool:
    """迁移单个 ckpt 文件。

    Args:
        ckpt_path: 输入 ckpt 路径
        output_path: 输出路径（None 则覆盖原文件）
        enable_low_rank: 是否启用低秩分解
        rank: 低秩 rank
        dry_run: 只打印不保存

    Returns:
        True 如果迁移成功
    """
    if not os.path.exists(ckpt_path):
        print(f"  跳过（不存在）: {ckpt_path}")
        return False

    print(f"  迁移: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "neuron_config" not in ckpt or "state_dict" not in ckpt:
        print(f"    跳过（格式不符）: 缺少 neuron_config 或 state_dict")
        return False

    cfg: NeuronConfig = ckpt["neuron_config"]
    sd = ckpt["state_dict"]

    new_sd, new_cfg = migrate_state_dict(sd, cfg, enable_low_rank, rank)

    if dry_run:
        print(f"    [dry-run] 不保存")
        return True

    out_path = output_path or ckpt_path
    # 备份原文件（仅当覆盖时）
    if out_path == ckpt_path:
        backup_path = ckpt_path + ".v2.bak"
        if not os.path.exists(backup_path):
            torch.save(ckpt, backup_path)
            print(f"    备份原文件: {backup_path}")

    new_ckpt = {
        "neuron_config": new_cfg,
        "state_dict": new_sd,
    }
    torch.save(new_ckpt, out_path)
    print(f"    保存: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="v2 → v3 ckpt 迁移")
    parser.add_argument(
        "--neurons-dir",
        default="data/neurons",
        help="神经元 ckpt 目录",
    )
    parser.add_argument(
        "--enable-low-rank",
        action="store_true",
        help="启用 lm_head 低秩分解（默认保留传统模式）",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="低秩分解的 rank（默认 64）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印不保存",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["zh", "en", "code", "math", "general"],
        help="要迁移的 domain 列表",
    )
    args = parser.parse_args()

    print(f"=== v2 → v3 ckpt 迁移 ===")
    print(f"neurons_dir: {args.neurons_dir}")
    print(f"enable_low_rank: {args.enable_low_rank}")
    print(f"rank: {args.rank}")
    print(f"dry_run: {args.dry_run}")
    print()

    # 查找所有 ckpt 文件
    migrated = 0
    failed = 0
    for domain in args.domains:
        # 同时处理 base 和 fieldcond
        for suffix in ["", "_fieldcond"]:
            ckpt_path = os.path.join(
                args.neurons_dir, f"neuron_{domain}{suffix}.pt"
            )
            if os.path.exists(ckpt_path):
                try:
                    if migrate_ckpt(
                        ckpt_path,
                        enable_low_rank=args.enable_low_rank,
                        rank=args.rank,
                        dry_run=args.dry_run,
                    ):
                        migrated += 1
                except Exception as e:
                    print(f"    失败: {e}")
                    failed += 1

    print()
    print(f"=== 迁移完成 ===")
    print(f"成功: {migrated}")
    print(f"失败: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
