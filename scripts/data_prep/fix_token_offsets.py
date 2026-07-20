"""P0-1c 修复:把 data/real/*.pt 中错位的 token ID 修正回 v2 contract。

问题:
  download_distill_data.py 用 sp.EncodeAsIds(text) 直接输出 sentencepiece ID,
  没有加 text_offset=13388,导致文本 token 落在 image/audio/control 区间,
  teacher 看到错位 token,PPL=24M (而训练时 loss=1.26, PPL≈3.5)。

修复规则:
  - ID ∈ {0,1,2,3} (control: pad/unk/bos/eos) → 保持不变
  - ID >= 4 → + text_offset (13388)
  这样 sentencepiece ID 4 变成 token ID 13392,落在 text range [13388, 256000)

验证:
  teacher PPL 应从 24M 降到 ~3.5 (与 training_state.json loss=1.26 一致)
"""
from __future__ import annotations

import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

TEXT_OFFSET = 13388
CONTROL_TOKENS = {0, 1, 2, 3}  # pad, unk, bos, eos


def fix_ids(ids: torch.Tensor) -> torch.Tensor:
    """修复 token ID: control 0-3 保留,其他 + TEXT_OFFSET。

    data/real/*.pt 是用 sp.EncodeAsIds 直接输出 raw sentencepiece ID,
    所有 ID 都是 sentencepiece ID (范围 [0, 242612))。
    v2 contract 要求 text token ID = sentencepiece ID + text_offset (13388)。
    所以修复规则:ID ∈ {0,1,2,3} (control) 保持不变,其他 + 13388。
    修复后范围:[13388, 256000),正好落在 text range。

    注:sentencepiece vocab=242612,最大 ID=242611,+13388=255999 < 256000 ✓
    """
    fixed = ids.clone()
    mask = fixed >= 4  # control 0-3 保持不变,其他全部 + TEXT_OFFSET
    fixed[mask] += TEXT_OFFSET
    return fixed


def verify_with_teacher(data_dir: str, sample_count: int = 2):
    """用 teacher 验证修复后的 PPL。"""
    print(f"\n[验证] 用 teacher 测试修复后的 PPL ...")
    from taiji.training.checkpoint_bridge import load_teacher_model

    teacher, _ = load_teacher_model("e:/taiji-neuron/checkpoint-481000", device="cpu")

    for domain in ["zh", "en", "code", "math", "general"]:
        path = os.path.join(data_dir, f"{domain}.pt")
        if not os.path.exists(path):
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        sample = data[:sample_count, :64] if data.dim() == 2 else data[:sample_count]

        # 修复后的样本
        fixed_sample = fix_ids(sample)

        # 原始 PPL
        with torch.no_grad():
            out_orig = teacher(sample)
            logits_orig = out_orig.logits if hasattr(out_orig, "logits") else out_orig
        shift_o = logits_orig[:, :-1, :].contiguous()
        targets_o = sample[:, 1:].contiguous()
        loss_o = F.cross_entropy(shift_o.view(-1, shift_o.size(-1)), targets_o.view(-1))
        ppl_o = math.exp(min(loss_o.item(), 20.0))

        # 修复后 PPL
        with torch.no_grad():
            out_fix = teacher(fixed_sample)
            logits_fix = out_fix.logits if hasattr(out_fix, "logits") else out_fix
        shift_f = logits_fix[:, :-1, :].contiguous()
        targets_f = fixed_sample[:, 1:].contiguous()
        loss_f = F.cross_entropy(shift_f.view(-1, shift_f.size(-1)), targets_f.view(-1))
        ppl_f = math.exp(min(loss_f.item(), 20.0))

        # top-1
        preds = shift_f.argmax(dim=-1)
        correct = (preds == targets_f).float().mean().item()

        print(f"  {domain:8s}: 原始 PPL={ppl_o:>12.2f}, 修复后 PPL={ppl_f:>10.2f}, top-1={correct*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/real")
    parser.add_argument("--dry_run", action="store_true", help="只验证不写文件")
    parser.add_argument("--no_verify", action="store_true", help="跳过 teacher 验证")
    args = parser.parse_args()

    print("=" * 70)
    print("P0-1c 修复:data/real/*.pt 的 token ID 偏移")
    print(f"  data_dir: {args.data_dir}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  TEXT_OFFSET: {TEXT_OFFSET}")
    print("=" * 70)

    if not os.path.isdir(args.data_dir):
        print(f"❌ 目录不存在: {args.data_dir}")
        return

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".pt"))
    print(f"\n找到 {len(files)} 个 .pt 文件: {files}")

    for fname in files:
        path = os.path.join(args.data_dir, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(data, dict):
            # domain_datasets.pt 是 dict
            print(f"\n[{fname}] dict with {len(data)} domains")
            fixed_dict = {}
            for k, v in data.items():
                fixed_dict[k] = fix_ids(v)
                # 统计
                orig_lt = ((v >= 4) & (v < TEXT_OFFSET)).sum().item()
                print(f"  {k}: shape={tuple(v.shape)}, 修复 {orig_lt} 个 token")
            if not args.dry_run:
                # 备份原文件
                backup = path + ".bak"
                if not os.path.exists(backup):
                    torch.save(data, backup)
                    print(f"  备份: {backup}")
                torch.save(fixed_dict, path)
                print(f"  写入: {path}")
        elif isinstance(data, torch.Tensor):
            print(f"\n[{fname}] tensor shape={tuple(data.shape)}")
            orig_lt = ((data >= 4) & (data < TEXT_OFFSET)).sum().item()
            total = data.numel()
            print(f"  需要修复的 token: {orig_lt}/{total} ({orig_lt/total*100:.2f}%)")
            fixed = fix_ids(data)
            if not args.dry_run:
                backup = path + ".bak"
                if not os.path.exists(backup):
                    torch.save(data, backup)
                    print(f"  备份: {backup}")
                torch.save(fixed, path)
                print(f"  写入: {path}")
            # 显示前 10 个 token 修复前后
            print(f"  前 10 token (原): {data.flatten()[:10].tolist()}")
            print(f"  前 10 token (修复): {fixed.flatten()[:10].tolist()}")
        else:
            print(f"\n[{fname}] 未知类型: {type(data)}, 跳过")

    # 验证
    if not args.no_verify:
        verify_with_teacher(args.data_dir)

    print("\n" + "=" * 70)
    print("[结论]")
    if args.dry_run:
        print("  dry_run 模式,未写入文件。去掉 --dry_run 实际修复。")
    else:
        print("  修复完成!原文件备份为 .bak。")
    print("  接下来需要:")
    print("  1. 重新蒸馏神经元 (用修复后的 data/real/*.pt)")
    print("  2. 修复 download_distill_data.py 防止以后再错")
    print("  3. 跑 1+1>2 验证")
    print("=" * 70)


if __name__ == "__main__":
    main()
