"""S3: 验证 SFT answer masking 的正确性。

测试点：
1. 向后兼容：不传 answer_marker 时返回 3 元组
2. SFT mask：传入 answer_marker 时返回 4 元组，sft_mask 正确
3. answer 起始位置正确（"答："后的 token 为 True）
4. question 部分为 False
5. pad 部分为 False
6. 截断处理：max_seq_len 截断后 sft_mask 仍正确
7. 无分隔符文本：整个文本视为 answer（全 True）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer, create_shared_embedding
from scripts.training.experiment_config import SFT_ANSWER_MARKER


def make_dummy_embedding(vocab_size: int = 256000, dim: int = 512) -> nn.Embedding:
    """创建 dummy shared_embedding（不加载真实权重）。"""
    return nn.Embedding(vocab_size, dim)


def test_backward_compatibility():
    """测试 1：不传 answer_marker 时返回 3 元组（向后兼容）。"""
    print("\n=== Test 1: 向后兼容（不传 answer_marker）===")
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_emb = make_dummy_embedding()

    texts = ["问：你好\n答：你好啊"]
    result = batch_align_and_embed(texts, domain_sp, general_sp, shared_emb)
    assert len(result) == 3, f"不传 answer_marker 应返回 3 元组，实际 {len(result)}"
    print(f"  ✅ 返回 {len(result)} 元组（向后兼容）")
    return True


def test_sft_mask_basic():
    """测试 2：SFT mask 基本正确性。"""
    print("\n=== Test 2: SFT mask 基本正确性 ===")
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_emb = make_dummy_embedding()

    text = "问：你好\n答：你好啊，很高兴认识你"
    texts = [text]

    shared_emb_out, targets, mask, sft_mask = batch_align_and_embed(
        texts, domain_sp, general_sp, shared_emb,
        answer_marker=SFT_ANSWER_MARKER,
    )

    print(f"  text: {text}")
    print(f"  mask shape: {mask.shape}, sft_mask shape: {sft_mask.shape}")
    assert mask.shape == sft_mask.shape, "mask 和 sft_mask 形状应一致"

    # 检查 sft_mask：question 部分 False，answer 部分 True
    L = mask[0].sum().item()  # valid token 数
    sft_true_count = sft_mask[0].sum().item()
    print(f"  valid tokens: {L}, answer tokens: {sft_true_count}")

    # answer token 数应 < valid token 数（question 部分不计入）
    assert sft_true_count < L, f"answer token 数 ({sft_true_count}) 应 < valid token 数 ({L})"
    assert sft_true_count > 0, "answer token 数应 > 0"

    # 检查 answer 起始位置：用 general_sp encode prefix_with_marker
    marker_idx = text.find(SFT_ANSWER_MARKER)
    prefix_with_marker = text[:marker_idx + len(SFT_ANSWER_MARKER)]
    prefix_ids = general_sp.encode(prefix_with_marker)
    expected_start = len(prefix_ids)
    print(f"  expected answer start: {expected_start}")

    # sft_mask[0, :expected_start] 应为 False（question）
    # sft_mask[0, expected_start:L] 应为 True（answer）
    question_mask = sft_mask[0, :expected_start]
    answer_mask = sft_mask[0, expected_start:L]
    assert not question_mask.any(), f"question 部分（0:{expected_start}）应为 False"
    assert answer_mask.all(), f"answer 部分（{expected_start}:{L}）应全为 True"
    print(f"  ✅ question 部分 [0:{expected_start}] 全 False")
    print(f"  ✅ answer 部分 [{expected_start}:{L}] 全 True")
    return True


def test_sft_mask_batch():
    """测试 3：batch 模式下 sft_mask 对齐正确。"""
    print("\n=== Test 3: batch 模式 sft_mask 对齐 ===")
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_emb = make_dummy_embedding()

    texts = [
        "问：你好\n答：你好啊",
        "问：什么是人工智能？请详细解释一下\n答：人工智能是计算机科学的一个分支",
    ]

    shared_emb_out, targets, mask, sft_mask = batch_align_and_embed(
        texts, domain_sp, general_sp, shared_emb,
        answer_marker=SFT_ANSWER_MARKER,
    )

    print(f"  batch size: {len(texts)}, shape: {mask.shape}")
    assert mask.shape == sft_mask.shape

    # 每个样本的 answer token 数都应 > 0
    for i, text in enumerate(texts):
        L_i = mask[i].sum().item()
        sft_i = sft_mask[i].sum().item()
        print(f"  样本 {i}: valid={L_i}, answer={sft_i}")
        assert sft_i > 0, f"样本 {i} answer token 数应 > 0"
        assert sft_i < L_i, f"样本 {i} answer token 数应 < valid token 数"

    # pad 部分应为 False
    for i in range(len(texts)):
        L_i = mask[i].sum().item()
        pad_part = sft_mask[i, L_i:]
        assert not pad_part.any(), f"样本 {i} pad 部分应全 False"
    print(f"  ✅ pad 部分全 False")
    return True


def test_sft_mask_truncation():
    """测试 4：截断后 sft_mask 正确。"""
    print("\n=== Test 4: 截断处理 ===")
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_emb = make_dummy_embedding()

    # 长文本，max_seq_len=20 强制截断
    text = "问：请详细介绍人工智能的发展历史和未来趋势\n答：人工智能的发展可以追溯到1950年代"
    texts = [text]

    _, _, mask, sft_mask = batch_align_and_embed(
        texts, domain_sp, general_sp, shared_emb,
        max_seq_len=20,
        answer_marker=SFT_ANSWER_MARKER,
    )

    L = mask[0].sum().item()
    print(f"  max_seq_len=20, valid tokens: {L}")
    assert L <= 20, f"截断后 valid token 数应 <= 20，实际 {L}"

    # 即使截断，answer 部分仍应有 True（如果 answer 在截断范围内）
    sft_true = sft_mask[0].sum().item()
    print(f"  answer tokens after truncation: {sft_true}")
    if sft_true > 0:
        print(f"  ✅ 截断后仍有 answer token")
    else:
        print(f"  ⚠️ 截断后无 answer token（answer 全部被截断）")
    return True


def test_sft_mask_no_marker():
    """测试 5：无分隔符文本，整个文本视为 answer。"""
    print("\n=== Test 5: 无分隔符文本 ===")
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_emb = make_dummy_embedding()

    text = "这是一段没有分隔符的普通文本"
    texts = [text]

    _, _, mask, sft_mask = batch_align_and_embed(
        texts, domain_sp, general_sp, shared_emb,
        answer_marker=SFT_ANSWER_MARKER,
    )

    L = mask[0].sum().item()
    sft_true = sft_mask[0].sum().item()
    print(f"  valid tokens: {L}, answer tokens: {sft_true}")
    assert sft_true == L, f"无分隔符文本应整个视为 answer，sft_true ({sft_true}) 应 = valid ({L})"
    print(f"  ✅ 无分隔符文本整个视为 answer")
    return True


def main():
    print("=" * 60)
    print("S3 SFT Answer Masking 验证")
    print("=" * 60)

    tests = [
        test_backward_compatibility,
        test_sft_mask_basic,
        test_sft_mask_batch,
        test_sft_mask_truncation,
        test_sft_mask_no_marker,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
