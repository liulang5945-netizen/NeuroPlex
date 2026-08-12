"""T4 数据增强 smoke test.

验证数据增强模块的正确性：
1. parse_dialogue 正确解析对话
2. rewrite_question 同义替换 + 前缀/后缀包装
3. augment_dialogue_text 保留 answer 不变
4. multi_turn_concatenate 生成多轮格式
5. DialogueAugmenter：epoch 种子变化 → 不同变体；epoch 内可复现
6. answer_marker_mode="last"：多轮只对最后 answer 计 loss
7. 向后兼容：answer_marker_mode 默认 "first" 行为不变
8. 神经元改写（TINY_TEST neuron 少量样本）
"""
from __future__ import annotations

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from scripts.training.data_augmentation import (
    parse_dialogue, rewrite_question, augment_dialogue_text,
    multi_turn_concatenate, DialogueAugmenter, generate_neuron_augmented_data,
)
from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron


SAMPLE = "问：你好，请告诉我什么是人工智能？\n答：人工智能是让机器模拟人类智能的技术。"
SAMPLE2 = "问：如何学习编程？\n答：先学基础语法，再动手写项目。"


def test_parse_dialogue():
    """[1] parse_dialogue 正确解析。"""
    print("\n[1] parse_dialogue")
    q, a = parse_dialogue(SAMPLE)
    assert q == "你好，请告诉我什么是人工智能？", f"question 解析错误: {q}"
    assert a == "人工智能是让机器模拟人类智能的技术。", f"answer 解析错误: {a}"
    assert parse_dialogue("没有标记的文本") is None
    assert parse_dialogue("") is None
    print(f"  PASS: question={q[:12]}..., answer={a[:12]}...")


def test_rewrite_question():
    """[2] rewrite_question 同义替换 + 前缀包装。"""
    print("\n[2] rewrite_question")
    rng = random.Random(42)
    q = "你好，请告诉我什么是人工智能？"
    # 全概率改写：必然触发替换和前缀
    rw = rewrite_question(q, rng, prefix_prob=1.0, suffix_prob=1.0, synonym_prob=1.0)
    assert "你好" not in rw, f"应替换'你好', got: {rw}"
    assert "请" not in rw, f"应替换'请', got: {rw}"
    assert len(rw) > len(q) // 2, f"改写后不应过短: {rw}"
    # 零概率：原样返回
    rw0 = rewrite_question(q, random.Random(0), prefix_prob=0.0, suffix_prob=0.0, synonym_prob=0.0)
    assert rw0 == q, f"零概率应原样返回, got: {rw0}"
    print(f"  PASS: 改写示例: '{q}' → '{rw}'")


def test_augment_keep_answer():
    """[3] augment_dialogue_text 保留 answer 不变。"""
    print("\n[3] augment_dialogue_text 保留 answer")
    rng = random.Random(42)
    aug = augment_dialogue_text(SAMPLE, rng, rewrite_prob=1.0)
    _, a_orig = parse_dialogue(SAMPLE)
    _, a_aug = parse_dialogue(aug)
    assert a_orig == a_aug, f"answer 应保持不变, orig={a_orig[:10]}..., aug={a_aug[:10]}..."
    # rewrite_prob=0：原样返回
    aug0 = augment_dialogue_text(SAMPLE, random.Random(0), rewrite_prob=0.0)
    assert aug0 == SAMPLE
    # 无效格式：原样返回
    assert augment_dialogue_text("无格式", random.Random(0), rewrite_prob=1.0) == "无格式"
    print(f"  PASS: answer 保持不变，question 改写: '{SAMPLE.splitlines()[0][:20]}...'")


def test_multi_turn():
    """[4] multi_turn_concatenate 生成多轮格式。"""
    print("\n[4] 多轮拼接")
    rng = random.Random(42)
    pool = [SAMPLE2, "问：什么是机器学习？\n答：机器学习是数据驱动的学习范式。"]
    mt = multi_turn_concatenate(SAMPLE, pool, rng, extra_turns=(1, 1), prob=1.0)
    n_q = mt.count("问：")
    n_a = mt.count("答：")
    assert n_q == 2 and n_a == 2, f"应拼成 2 轮, got 问:{n_q} 答:{n_a}"
    assert mt.endswith(SAMPLE.splitlines()[-1]), "最终轮应为当前对话"
    # prob=0：原样返回
    mt0 = multi_turn_concatenate(SAMPLE, pool, random.Random(0), prob=0.0)
    assert mt0 == SAMPLE
    print(f"  PASS: 2 轮拼接示例: {mt[:60]}...")


def test_augmenter_epoch():
    """[5] DialogueAugmenter：epoch 间变体不同，epoch 内可复现。"""
    print("\n[5] DialogueAugmenter epoch 种子")
    texts = [SAMPLE, SAMPLE2, "问：什么是机器学习？\n答：机器学习是数据驱动的学习范式。"]
    aug_e0 = DialogueAugmenter(rewrite_prob=0.9, multi_turn_prob=0.9)
    aug_e0.set_context_pool(texts)
    aug_e0.set_epoch(0)
    v0 = [aug_e0.augment(t) for t in texts]

    # 同 epoch 可复现
    aug_e0b = DialogueAugmenter(rewrite_prob=0.9, multi_turn_prob=0.9)
    aug_e0b.set_context_pool(texts)
    aug_e0b.set_epoch(0)
    v0b = [aug_e0b.augment(t) for t in texts]
    assert v0 == v0b, "同 epoch 应可复现"

    # 不同 epoch 应有差异（高概率下至少 1 条不同）
    aug_e1 = DialogueAugmenter(rewrite_prob=0.9, multi_turn_prob=0.9)
    aug_e1.set_context_pool(texts)
    aug_e1.set_epoch(1)
    v1 = [aug_e1.augment(t) for t in texts]
    diffs = sum(1 for a, b in zip(v0, v1) if a != b)
    assert diffs >= 1, f"epoch 间应产生差异, diffs={diffs}"
    print(f"  PASS: epoch0 可复现，epoch0/1 差异 {diffs}/3 条")


def test_last_marker_mode():
    """[6] answer_marker_mode="last"：多轮只对最后 answer 计 loss。"""
    print("\n[6] answer_marker_mode=last")
    from taiji.resonance.translator import batch_align_and_embed
    from taiji.resonance.translator import TokenizerHub
    # 用真实 tokenizer（zh 域）验证 mask 位置
    hub = TokenizerHub()
    try:
        domain_sp = hub.get_domain_sp("zh")
        general_sp = hub.get_general_sp()
    except Exception:
        from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer
        domain_sp = load_domain_tokenizer("zh")
        general_sp = load_general_tokenizer()

    shared_emb = torch.nn.Embedding(general_sp.vocab_size(), 512)
    mt_text = "问：你好\n答：你好呀！\n问：什么是AI？\n答：人工智能。"
    _, _, mask, sft_first = batch_align_and_embed(
        [mt_text], domain_sp, general_sp, shared_emb,
        answer_marker="答：", answer_marker_mode="first",
    )
    _, _, mask, sft_last = batch_align_and_embed(
        [mt_text], domain_sp, general_sp, shared_emb,
        answer_marker="答：", answer_marker_mode="last",
    )

    # first 模式：第一个 "答：" 之后全 True（含中间轮 question）
    n_first = sft_first[0].sum().item()
    # last 模式：只有最后一个 "答：" 之后 True（更少）
    n_last = sft_last[0].sum().item()
    assert n_last < n_first, f"last 模式 mask 应更少, first={n_first}, last={n_last}"
    # last 模式的 answer 起始位置应更靠后
    first_true_first = int(sft_first[0].nonzero()[0].item())
    first_true_last = int(sft_last[0].nonzero()[0].item())
    assert first_true_last > first_true_first, "last 模式 answer 起点应更靠后"
    print(f"  PASS: first 模式 answer 起点={first_true_first} (n={n_first}), "
          f"last 模式起点={first_true_last} (n={n_last})")


def test_backward_compat():
    """[7] 向后兼容：answer_marker_mode 默认 "first"。"""
    print("\n[7] 向后兼容")
    from taiji.resonance.translator import batch_align_and_embed
    from taiji.resonance.translator import TokenizerHub
    hub = TokenizerHub()
    try:
        domain_sp = hub.get_domain_sp("zh")
        general_sp = hub.get_general_sp()
    except Exception:
        from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer
        domain_sp = load_domain_tokenizer("zh")
        general_sp = load_general_tokenizer()

    shared_emb = torch.nn.Embedding(general_sp.vocab_size(), 512)
    # 不传 answer_marker_mode：与显式 "first" 一致
    _, _, mask, sft_default = batch_align_and_embed(
        [SAMPLE], domain_sp, general_sp, shared_emb, answer_marker="答：",
    )
    _, _, mask, sft_first = batch_align_and_embed(
        [SAMPLE], domain_sp, general_sp, shared_emb,
        answer_marker="答：", answer_marker_mode="first",
    )
    assert torch.equal(sft_default, sft_first), "默认模式应与 first 一致"
    # 不传 answer_marker：返回 3 元组（兼容旧调用）
    out = batch_align_and_embed([SAMPLE], domain_sp, general_sp, shared_emb)
    assert len(out) == 3, f"不传 answer_marker 应返回 3 元组, got {len(out)}"
    print(f"  PASS: 默认行为完全向后兼容")


def test_neuron_aug():
    """[8] 神经元改写（TINY_TEST neuron 少量样本）。"""
    print("\n[8] 神经元改写")
    torch.manual_seed(42)
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 1000
    cfg.neuron_id = "n_aug"
    neuron = ResonanceNeuron(cfg)
    neuron.eval()

    from taiji.resonance.translator import TokenizerHub
    hub = TokenizerHub()
    try:
        domain_sp = hub.get_domain_sp("zh")
        general_sp = hub.get_general_sp()
    except Exception:
        from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer
        domain_sp = load_domain_tokenizer("zh")
        general_sp = load_general_tokenizer()

    shared_emb = torch.nn.Embedding(general_sp.vocab_size(), 512)
    rng = random.Random(42)
    texts = [SAMPLE, SAMPLE2]
    augmented = generate_neuron_augmented_data(
        texts, neuron, shared_emb, domain_sp, general_sp,
        rng, max_new_tokens=16, max_samples=2,
    )
    # 随机初始化的 neuron 生成可能失败，但函数不应抛异常
    assert isinstance(augmented, list), "应返回列表"
    print(f"  PASS: 神经元改写返回 {len(augmented)} 条（随机初始化 neuron 生成质量有限，接口正确）")


def main():
    print("=" * 70)
    print("T4 数据增强 smoke test")
    print("=" * 70)

    test_parse_dialogue()
    test_rewrite_question()
    test_augment_keep_answer()
    test_multi_turn()
    test_augmenter_epoch()
    test_last_marker_mode()
    test_backward_compat()
    test_neuron_aug()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
