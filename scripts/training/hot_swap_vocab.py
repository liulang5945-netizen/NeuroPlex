#!/usr/bin/env python3
"""T12: zh 词表库热插拔（第二步：token 映射 + lm_head 权重迁移）。

原理：
    neuron 不拥有独立文本 embedding（走 shared_embedding 256K），
    唯一与 zh vocab 相关的权重是 lm_head.weight [vocab, hidden]。
    因此"词表库热插拔"= 用新 tokenizer 的 token 映射迁移 lm_head 权重 +
    更新 cfg.vocab_size，无需重训神经元。

迁移策略（按优先级）：
    1. 精确 piece 匹配：旧 token 的 piece 字符串在新 tokenizer 中相同 → 行直接拷贝
    2. 特殊 token（<pad>/<unk>/<s>/</s>，id 0-3）：两代 tokenizer id 约定一致，直接对齐
    3. 子 piece 分解：新 token 不在旧词表 → 用旧 tokenizer 切分该 piece，
       若全部子 piece 可识别（无 unk），新行 = 子 piece 行均值
    4. 兜底：随机初始化 std=hidden**-0.5（与 neuron 创建一致），由增量微调学习

用法：
    python scripts/training/hot_swap_vocab.py [--neurons-dir data/neurons]
                                              [--domain zh]
                                              [--backup-dir data/neurons/pre_t12_backup]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ckpt 内 pickle 了 taiji.* 对象（如 NeuronConfig），必须能导入 taiji
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOMAIN_DIR = PROJECT_ROOT / "taiji" / "domains"
OLD_BACKUP = DOMAIN_DIR / "zh" / "sp_zh_v20k.model"


# ======================== 核心函数（供 smoke test 导入） ========================


def build_token_id_map(old_sp, new_sp) -> dict[int, int]:
    """构建 old_id → new_id 映射（piece 字符串精确匹配）。

    Args:
        old_sp: 旧 tokenizer（SentencePieceProcessor）
        new_sp: 新 tokenizer

    Returns:
        {old_id: new_id}，仅包含能匹配的 old token
    """
    new_piece_to_id: dict[str, int] = {}
    for i in range(new_sp.GetPieceSize()):
        new_piece_to_id[new_sp.IdToPiece(i)] = i

    id_map: dict[int, int] = {}
    for old_id in range(old_sp.GetPieceSize()):
        piece = old_sp.IdToPiece(old_id)
        new_id = new_piece_to_id.get(piece)
        if new_id is not None:
            id_map[old_id] = new_id
    return id_map


def compute_new_embeddings(
    old_w: torch.Tensor,
    id_map: dict[int, int],
    new_vocab: int,
    old_sp,
    new_sp,
) -> torch.Tensor:
    """从旧 lm_head 权重迁移出新 lm_head 权重。

    Args:
        old_w: [V_old, H]
        id_map: old_id → new_id
        new_vocab: 新 vocab 大小
        old_sp / new_sp: 新旧 tokenizer（用于子 piece 分解）

    Returns:
        new_w: [new_vocab, H]
    """
    old_vocab, hidden = old_w.shape
    new_w = torch.zeros(new_vocab, hidden)

    # 1. 精确匹配：直接拷贝
    for old_id, new_id in id_map.items():
        new_w[new_id] = old_w[old_id]

    # 2. 未匹配的新 token：子 piece 分解平均 或 随机初始化
    unk_id = old_sp.unk_id()
    std = hidden ** -0.5
    matched_new = set(id_map.values())
    n_avg = 0
    n_random = 0
    for new_id in range(new_vocab):
        if new_id in matched_new:
            continue
        piece = new_sp.IdToPiece(new_id)
        # 子 piece 分解：用旧 tokenizer 切分该 piece，无 unk 则取均值
        sub_ids = old_sp.encode(piece)
        if sub_ids and all(s != unk_id for s in sub_ids):
            new_w[new_id] = old_w[sub_ids].mean(dim=0)
            n_avg += 1
        else:
            new_w[new_id] = torch.randn(hidden, dtype=old_w.dtype) * std
            n_random += 1

    print(f"  [migrate] 精确匹配 {len(id_map)} / 子 piece 平均 {n_avg} / 随机初始化 {n_random} "
          f"（共 {new_vocab}）")
    return new_w


def migrate_lm_head_state(
    state_dict: dict,
    old_sp,
    new_sp,
    new_vocab: int,
) -> int:
    """迁移 state_dict 中的 lm_head.weight，返回迁移后的 vocab 大小。

    就地修改 state_dict。若不存在 lm_head.weight（低秩模式）则跳过。

    Returns:
        迁移后的 vocab 大小（未迁移时返回 0）
    """
    key = "lm_head.weight"
    if key not in state_dict:
        return 0
    old_w = state_dict[key]
    id_map = build_token_id_map(old_sp, new_sp)
    new_w = compute_new_embeddings(old_w, id_map, new_vocab, old_sp, new_sp)
    state_dict[key] = new_w
    return new_vocab


# ======================== ckpt 迁移 ========================


def migrate_neuron_ckpt(
    ckpt_path: Path,
    old_sp,
    new_sp,
    new_vocab: int,
    backup_dir: Path,
) -> dict:
    """迁移单个 neuron ckpt：lm_head 权重 + cfg.vocab_size。

    Args:
        ckpt_path: neuron_*.pt 路径
        old_sp / new_sp: 新旧 tokenizer
        new_vocab: 新 vocab 大小
        backup_dir: 原文件备份目录（不存在则跳过备份）

    Returns:
        迁移报告 dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    report = {"path": str(ckpt_path), "migrated": False, "old_vocab": None,
              "new_vocab": None, "lm_head_shape": None}

    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or {}
    if "lm_head.weight" not in sd:
        print(f"  [skip] {ckpt_path.name}: 无 lm_head.weight（非标准结构），跳过")
        return report

    old_shape = tuple(sd["lm_head.weight"].shape)
    report["old_vocab"] = old_shape[0]
    if old_shape[0] == new_vocab:
        print(f"  [skip] {ckpt_path.name}: lm_head 已是 {new_vocab} vocab，无需迁移")
        report["new_vocab"] = new_vocab
        report["lm_head_shape"] = old_shape
        return report

    # 备份原文件
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_path, backup_dir / ckpt_path.name)
        print(f"  [backup] {ckpt_path.name} → {backup_dir.name}/")

    # 迁移权重
    migrated = migrate_lm_head_state(sd, old_sp, new_sp, new_vocab)
    if not migrated:
        print(f"  [skip] {ckpt_path.name}: lm_head 迁移失败（无匹配 key）")
        return report

    # 更新 cfg.vocab_size（dataclass 直接改属性）
    cfg = ckpt.get("neuron_config")
    if cfg is not None:
        old_vocab = getattr(cfg, "vocab_size", None)
        cfg.vocab_size = new_vocab
        report["old_cfg_vocab"] = old_vocab
        print(f"  [cfg] {ckpt_path.name}: neuron_config.vocab_size "
              f"{old_vocab} → {new_vocab}")

    # 保存（写临时文件再原子替换，避免中断损坏）
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)

    report["migrated"] = True
    report["new_vocab"] = new_vocab
    report["lm_head_shape"] = tuple(sd["lm_head.weight"].shape)
    print(f"  [migrate] {ckpt_path.name}: lm_head {old_shape} → {tuple(sd['lm_head.weight'].shape)}")
    return report


def main():
    parser = argparse.ArgumentParser(description="T12: zh 词表库热插拔")
    parser.add_argument("--neurons-dir", type=str, default=str(PROJECT_ROOT / "data" / "neurons"))
    parser.add_argument("--domain", type=str, default="zh")
    parser.add_argument("--new-vocab", type=int, default=50000)
    parser.add_argument("--backup-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "neurons" / "pre_t12_backup"))
    parser.add_argument("--no-backup", action="store_true", help="不备份原 ckpt")
    args = parser.parse_args()

    import sentencepiece as spm

    new_model = DOMAIN_DIR / args.domain / f"sp_{args.domain}.model"
    if not new_model.exists():
        raise FileNotFoundError(f"新 tokenizer 不存在: {new_model}（先运行 upgrade_tokenizer.py）")
    if not OLD_BACKUP.exists():
        raise FileNotFoundError(f"旧 tokenizer 备份不存在: {OLD_BACKUP}（先运行 upgrade_tokenizer.py）")

    new_sp = spm.SentencePieceProcessor(str(new_model))
    old_sp = spm.SentencePieceProcessor(str(OLD_BACKUP))
    print(f"旧 tokenizer: {old_sp.GetPieceSize()} vocab | 新 tokenizer: {new_sp.GetPieceSize()} vocab")

    # 预检：新 vocab 是否与目标一致
    if new_sp.GetPieceSize() != args.new_vocab:
        print(f"[warn] 新 tokenizer 实际 vocab={new_sp.GetPieceSize()}，"
              f"目标 {args.new_vocab}（以实际为准）")
        args.new_vocab = new_sp.GetPieceSize()

    # 扫描目标域 ckpt（含 _aug/_dialogue/_fieldcond 变体）
    pattern = os.path.join(args.neurons_dir, f"neuron_{args.domain}*.pt")
    ckpt_paths = sorted(glob.glob(pattern))
    if not ckpt_paths:
        print(f"未找到 {args.domain} 域 ckpt: {pattern}")
        return

    print(f"\n迁移 {args.domain} 域 ckpt（共 {len(ckpt_paths)} 个）...")
    backup_dir = None if args.no_backup else Path(args.backup_dir)
    migrated = 0
    for path in ckpt_paths:
        report = migrate_neuron_ckpt(Path(path), old_sp, new_sp,
                                     args.new_vocab, backup_dir)
        if report["migrated"]:
            migrated += 1

    print(f"\n[T12-2 完成] 迁移 {migrated}/{len(ckpt_paths)} 个 ckpt")
    print(f"下一步：更新 DOMAIN_VOCAB_SIZES 后运行对话验证")
    print(f"  python scripts/training/verify_cortex_reload.py")


if __name__ == "__main__":
    main()
