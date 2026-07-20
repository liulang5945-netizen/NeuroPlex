"""P0-1c: tokenizer offset 假设检验。

假设: data/real/*.pt 是用「无 text_offset」的旧 tokenizer 编码的,
而 teacher 训练时使用 v2 contract (text_offset=13388)。
导致 sentencepiece ID 直接被当作 token ID,大量文本 token 落在
image/audio/control 区间,teacher 看到错误的 token,PPL 极差。

验证方法:
1. 统计 data 中 token ID 分布,看是否有大量 ID < 13388 (非控制 token)
2. 对 ID < 13388 (排除控制 0-3) 的 token 做 +13388 偏移,看 PPL 是否改善
3. 对比几种偏移策略
"""
from __future__ import annotations

import os
import sys
import math
from collections import Counter

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from taiji.training.checkpoint_bridge import load_teacher_model

TEXT_OFFSET = 13388


def analyze_distribution(data: torch.Tensor, name: str):
    """统计 token ID 分布。"""
    flat = data.flatten()
    print(f"\n[{name}] shape={tuple(data.shape)}, total tokens={flat.numel()}")

    # 分桶
    ranges = [
        ("control [0,3]", 0, 4),
        ("taiji_special [4,999]", 4, 1000),
        ("image [1000,9191]", 1000, 9192),
        ("audio [9192,13287]", 9192, 13288),
        ("mm_control [13288,13387]", 13288, 13388),
        ("text [13388,255999]", 13388, 256000),
    ]
    for label, lo, hi in ranges:
        mask = (flat >= lo) & (flat < hi)
        cnt = int(mask.sum().item())
        pct = cnt / flat.numel() * 100
        print(f"  {label:30s}: {cnt:>10d}  ({pct:5.2f}%)")

    # top-10 最常见 ID
    cnt = Counter(flat.tolist())
    print(f"  top-10 最常见 ID:")
    for tid, c in cnt.most_common(10):
        print(f"    id={tid}: {c} 次")


def compute_ppl(teacher, ids: torch.Tensor, name: str, max_samples: int = 2):
    """计算 PPL。"""
    sample = ids[:max_samples, :64] if ids.dim() == 2 else ids[:max_samples]
    with torch.no_grad():
        out = teacher(sample)
        logits = out.logits if hasattr(out, "logits") else out
    shift = logits[:, :-1, :].contiguous()
    targets = sample[:, 1:].contiguous()
    loss = F.cross_entropy(shift.view(-1, shift.size(-1)), targets.view(-1))
    ppl = math.exp(min(loss.item(), 20.0))
    preds = shift.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    print(f"  [{name}] loss={loss.item():.4f}, PPL={ppl:.2f}, top-1={correct*100:.4f}%")
    return loss.item(), ppl


def shift_strategy(data: torch.Tensor, strategy: str) -> torch.Tensor:
    """应用不同的偏移策略。"""
    d = data.clone()
    if strategy == "original":
        return d
    if strategy == "shift_lt_text_offset":
        # 对所有 ID >= 4 且 < 13388 的 token 加 13388
        mask = (d >= 4) & (d < TEXT_OFFSET)
        d[mask] += TEXT_OFFSET
        return d
    if strategy == "shift_lt_1000":
        # 只对 [4, 1000) 范围加 13388(假设只有 taiji_special 被错位)
        mask = (d >= 4) & (d < 1000)
        d[mask] += TEXT_OFFSET
        return d
    if strategy == "shift_image_range":
        # 对 image/audio/mm_control 范围 [1000, 13388) 加 13388
        mask = (d >= 1000) & (d < TEXT_OFFSET)
        d[mask] += TEXT_OFFSET
        return d
    if strategy == "mod_text_vocab":
        # 对所有 ID 取模 text_vocab_size (242612) + text_offset
        text_vocab = 242612
        d = (d % text_vocab) + TEXT_OFFSET
        return d
    return d


def main():
    print("=" * 70)
    print("P0-1c: tokenizer text_offset 假设检验")
    print("=" * 70)

    # 加载 teacher
    print("\n[1] 加载 teacher ...")
    teacher, _ = load_teacher_model("e:/taiji-neuron/checkpoint-481000", device="cpu")

    # 加载数据
    print("\n[2] 加载 data/real/zh.pt ...")
    data_path = "data/real/zh.pt"
    if not os.path.exists(data_path):
        print(f"  ❌ {data_path} 不存在,退出")
        return
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    print(f"  shape: {tuple(data.shape)}, dtype: {data.dtype}")

    # 分析分布
    print("\n[3] 原始数据 token ID 分布")
    analyze_distribution(data, "original")

    # 尝试不同偏移策略
    print("\n[4] 不同偏移策略下的 PPL 对比")
    strategies = [
        "original",
        "shift_lt_text_offset",
        "shift_lt_1000",
        "shift_image_range",
        "mod_text_vocab",
    ]
    for s in strategies:
        shifted = shift_strategy(data, s)
        if s != "original":
            # 显示偏移后的分布前几行
            analyze_distribution(shifted, f"after {s}")
        compute_ppl(teacher, shifted, s)

    # 重点测试:如果 shift_lt_text_offset 显著改善 PPL,说明假设成立
    print("\n" + "=" * 70)
    print("[结论]")
    print("  如果 shift_lt_text_offset PPL 显著低于 original,")
    print("  则 data/real/zh.pt 是用无 text_offset 的旧 tokenizer 编码的,")
    print("  需要重新用 v2 contract tokenizer 编码数据。")
    print("=" * 70)


if __name__ == "__main__":
    main()
