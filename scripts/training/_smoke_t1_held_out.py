"""T1 held-out 评估集分桶 smoke test.

验证 split_train_eval 的正确性：
1. 评估集占比约 5%
2. 训练集和评估集无交集
3. 确定性：同一 seed + 同一文本 → 同一分桶（跨运行一致）
4. 不同 seed → 不同分桶
5. 空列表安全处理
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.training.utils import split_train_eval


def test_eval_ratio():
    """[1] 评估集占比约 5%。"""
    print("\n[1] 评估集占比约 5%")
    texts = [f"这是第{i}条测试文本，内容各不相同。" for i in range(10000)]
    train, eval_ = split_train_eval(texts, eval_ratio=0.05)

    ratio = len(eval_) / len(texts)
    assert 0.03 < ratio < 0.07, f"评估集占比应约 5%, 实际 {ratio:.3f}"
    print(f"  PASS: 评估集占比 {ratio:.3f} (train={len(train)}, eval={len(eval_)})")


def test_no_intersection():
    """[2] 训练集和评估集无交集。"""
    print("\n[2] 训练集和评估集无交集")
    texts = [f"文本_{i}" for i in range(1000)]
    train, eval_ = split_train_eval(texts, eval_ratio=0.05)

    train_set = set(train)
    eval_set = set(eval_)
    intersection = train_set & eval_set

    assert len(intersection) == 0, f"训练集和评估集有 {len(intersection)} 条交集"
    assert len(train) + len(eval_) == len(texts), "train + eval 应等于总量"
    print(f"  PASS: 无交集 (train={len(train)}, eval={len(eval_)}, total={len(texts)})")


def test_deterministic():
    """[3] 确定性：同一 seed + 同一文本 → 同一分桶。"""
    print("\n[3] 确定性（跨运行一致）")
    texts = [f"确定性测试文本_{i}" for i in range(500)]

    train1, eval1 = split_train_eval(texts, eval_ratio=0.05, seed=42)
    train2, eval2 = split_train_eval(texts, eval_ratio=0.05, seed=42)

    assert train1 == train2, "同一 seed 的训练集应完全一致"
    assert eval1 == eval2, "同一 seed 的评估集应完全一致"
    print(f"  PASS: 同一 seed 跨运行一致 (eval={len(eval1)})")


def test_different_seed():
    """[4] 不同 seed → 不同分桶。"""
    print("\n[4] 不同 seed 不同分桶")
    texts = [f"多 seed 测试_{i}" for i in range(1000)]

    _, eval1 = split_train_eval(texts, eval_ratio=0.05, seed=42)
    _, eval2 = split_train_eval(texts, eval_ratio=0.05, seed=123)

    # 不同 seed 应产生不同的评估集
    set1 = set(eval1)
    set2 = set(eval2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    assert len(only_in_1) > 0, "不同 seed 应产生不同分桶"
    assert len(only_in_2) > 0, "不同 seed 应产生不同分桶"
    print(f"  PASS: 不同 seed 产生不同分桶 (seed=42 eval={len(eval1)}, seed=123 eval={len(eval2)})")


def test_empty_list():
    """[5] 空列表安全处理。"""
    print("\n[5] 空列表安全处理")
    train, eval_ = split_train_eval([], eval_ratio=0.05)
    assert len(train) == 0
    assert len(eval_) == 0
    print(f"  PASS: 空列表安全 (train={len(train)}, eval={len(eval_)})")


def test_small_list():
    """[6] 小列表也能工作。"""
    print("\n[6] 小列表也能工作")
    texts = ["短文本1", "短文本2", "短文本3"]
    train, eval_ = split_train_eval(texts, eval_ratio=0.05)
    assert len(train) + len(eval_) == 3, "总量应保持"
    print(f"  PASS: 小列表正常 (train={len(train)}, eval={len(eval_)})")


def main():
    print("=" * 70)
    print("T1 held-out 评估集分桶 smoke test")
    print("=" * 70)

    test_eval_ratio()
    test_no_intersection()
    test_deterministic()
    test_different_seed()
    test_empty_list()
    test_small_list()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
