#!/usr/bin/env python3
"""T12 smoke test: 验证词表库热插拔（token 映射 + lm_head 迁移 + ckpt 加载）。

测试方法：
    用内存文本训练两个 mini SentencePiece tokenizer（旧 vocab=100 / 新 vocab=160，
    新语料包含旧语料 + 更多文本，保证 piece 重叠），构造假 lm_head 权重，
    调用 hot_swap_vocab 的核心函数验证：
      [1] 特殊 token（0-3）映射正确
      [2] 精确 piece 匹配映射覆盖率
      [3] lm_head 迁移后形状正确 [new_vocab, hidden]
      [4] 匹配行权重完全一致（拷贝正确）
      [5] 子 piece 分解初始化（可分解新 token = 旧子 piece 均值）
      [6] 兜底随机初始化 std 合理（非全零）
      [7] ckpt 迁移：cfg.vocab_size 更新 + 保存后可 torch.load
      [8] 全部断言通过

用法：
    python scripts/training/_smoke_t12_hotswap.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
from scripts.training.hot_swap_vocab import (
    build_token_id_map,
    compute_new_embeddings,
    migrate_lm_head_state,
    migrate_neuron_ckpt,
)

# 旧语料：基础中文（含常用词，保证与新语料重叠）
OLD_TEXTS = [
    "人工智能是研究如何让计算机模拟人类智能行为的学科",
    "深度学习通过多层神经网络进行表征学习",
    "机器学习是人工智能的一个重要分支",
    "神经网络由大量神经元组成",
    "语言模型能够生成自然流畅的文本",
    "自然语言处理是计算机科学的研究方向",
    "今天天气很好，我们一起去公园散步",
    "中华人民共和国成立于一九四九年",
    "数据科学家需要掌握统计学和编程技能",
    "量子计算利用量子叠加原理进行计算",
]
# 新语料：旧语料 + 更多词汇（覆盖旧词表没有的词）
NEW_TEXTS = OLD_TEXTS + [
    "变压器架构推动了自然语言处理领域的革命性进展",
    "注意力机制让模型能够关注输入序列中的关键信息",
    "强化学习通过奖励信号来指导智能体学习最优策略",
    "卷积神经网络在图像识别任务中表现出色",
    "生成对抗网络由生成器和判别器两部分组成",
    "推荐系统根据用户的历史行为进行个性化推荐",
    "自动驾驶汽车依赖传感器融合和决策算法",
    "区块链技术实现了去中心化的可信数据存储",
    "元宇宙概念催生了虚拟现实与增强现实的融合发展",
    "开源社区协作模式推动了软件技术的快速进步",
]


def train_mini_tokenizer(texts: list[str], vocab_size: int) -> spm.SentencePieceProcessor:
    """训练 mini tokenizer（参数与生产一致）。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for _ in range(200):  # 重复扩展训练数据
            for t in texts:
                f.write(t + "\n")
        corpus_path = f.name

    prefix = corpus_path[:-len(".txt")]
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        split_digits=True,
        max_sentence_length=16384,
        num_threads=4,
        hard_vocab_limit=False,
    )
    os.unlink(corpus_path)
    return spm.SentencePieceProcessor(f"{prefix}.model")


def run():
    passed = []
    fail = lambda msg: (_ for _ in ()).throw(AssertionError(msg))

    # ---- 准备 mini tokenizer ----
    # byte_fallback 需要 256 字节 piece + 4 特殊 token + 字符，vocab 需 ≥ 300
    print("训练 mini tokenizer（旧 vocab=400 / 新 vocab=520）...")
    old_sp = train_mini_tokenizer(OLD_TEXTS, 400)
    new_sp = train_mini_tokenizer(NEW_TEXTS, 520)
    print(f"  旧 vocab={old_sp.GetPieceSize()} 新 vocab={new_sp.GetPieceSize()}")

    # ---- [1] 特殊 token 映射 ----
    id_map = build_token_id_map(old_sp, new_sp)
    for old_id, name in [(0, "pad"), (1, "unk"), (2, "bos"), (3, "eos")]:
        piece = old_sp.IdToPiece(old_id)
        assert id_map.get(old_id) == new_sp.PieceToId(piece), \
            f"[1] 特殊 token {name}(id={old_id}) 映射错误: {id_map.get(old_id)}"
    passed.append("[1] 特殊 token（0-3）映射正确")

    # ---- [2] 精确匹配覆盖率 ----
    coverage = len(id_map) / old_sp.GetPieceSize()
    print(f"  精确匹配覆盖率: {coverage:.1%} ({len(id_map)}/{old_sp.GetPieceSize()})")
    assert coverage > 0.3, f"[2] 匹配覆盖率过低: {coverage:.1%}"
    passed.append("[2] 精确 piece 匹配映射覆盖正常")

    # ---- 构造假 lm_head 权重 ----
    hidden = 64
    old_vocab = old_sp.GetPieceSize()
    old_w = torch.randn(old_vocab, hidden)
    torch.manual_seed(42)

    # ---- [3] 迁移后形状 ----
    new_vocab = new_sp.GetPieceSize()
    new_w = compute_new_embeddings(old_w, id_map, new_vocab, old_sp, new_sp)
    assert new_w.shape == (new_vocab, hidden), \
        f"[3] 迁移形状错误: {tuple(new_w.shape)} vs {(new_vocab, hidden)}"
    passed.append(f"[3] lm_head 迁移形状正确 {tuple(new_w.shape)}")

    # ---- [4] 匹配行权重一致 ----
    for old_id, new_id in list(id_map.items())[:5]:
        assert torch.equal(new_w[new_id], old_w[old_id]), \
            f"[4] 匹配行权重不一致: old {old_id} → new {new_id}"
    passed.append("[4] 精确匹配行权重完全一致")

    # ---- [5] 子 piece 分解初始化 ----
    n_avg = 0
    matched_new = set(id_map.values())
    for new_id in range(new_vocab):
        if new_id in matched_new:
            continue
        piece = new_sp.IdToPiece(new_id)
        sub_ids = old_sp.encode(piece)
        if sub_ids and all(s != old_sp.unk_id() for s in sub_ids):
            expect = old_w[sub_ids].mean(dim=0)
            assert torch.allclose(new_w[new_id], expect, atol=1e-6), \
                f"[5] 子 piece 平均错误: piece={piece}, sub={sub_ids}"
            n_avg += 1
            break  # 验证一个即可
    assert n_avg >= 1, "[5] 未找到可子 piece 分解的新 token（语料问题？）"
    passed.append("[5] 子 piece 分解初始化正确（均值）")

    # ---- [6] 随机初始化兜底 ----
    std = hidden ** -0.5
    random_ids = []
    for i in range(new_vocab):
        if i in matched_new:
            continue
        sub = old_sp.encode(new_sp.IdToPiece(i))
        if sub and all(s != old_sp.unk_id() for s in sub):
            continue  # 走子 piece 平均路径
        random_ids.append(i)
    if random_ids:
        random_rows = new_w[random_ids]
        assert random_rows.abs().sum() > 0, "[6] 随机行全零"
        assert abs(random_rows.std().item() - std) < 0.1, \
            f"[6] 随机行 std 偏差: {random_rows.std().item():.3f} vs {std:.3f}"
    passed.append("[6] 随机初始化 std 合理（非全零）")

    # ---- [7] ckpt 迁移：cfg.vocab_size 更新 + 保存后加载 ----
    from taiji.resonance.config import get_domain_neuron_config
    cfg = get_domain_neuron_config("zh")
    cfg.vocab_size = old_vocab  # 模拟旧 ckpt 的 cfg
    fake_ckpt = {
        "neuron_config": cfg,
        "state_dict": {"lm_head.weight": old_w.clone()},
        "shared_embedding_state": torch.randn(100, 8),
        "domain": "zh",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "neuron_zh_smoke.pt"
        torch.save(fake_ckpt, ckpt_path)
        backup_dir = Path(tmpdir) / "backup"
        report = migrate_neuron_ckpt(ckpt_path, old_sp, new_sp, new_vocab, backup_dir)
        assert report["migrated"], f"[7] ckpt 未迁移: {report}"
        assert report["new_vocab"] == new_vocab, f"[7] 迁移 vocab 错误: {report}"
        assert report["lm_head_shape"] == (new_vocab, hidden), \
            f"[7] 迁移后形状错误: {report}"
        # 保存后可加载验证
        reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert reloaded["neuron_config"].vocab_size == new_vocab, \
            f"[7] 重载 cfg.vocab_size 未更新: {reloaded['neuron_config'].vocab_size}"
        assert reloaded["state_dict"]["lm_head.weight"].shape == (new_vocab, hidden), \
            "[7] 重载 lm_head 形状错误"
        # 备份存在
        assert (backup_dir / "neuron_zh_smoke.pt").exists(), "[7] 备份文件缺失"
        # 匹配行仍一致
        for old_id, new_id in list(id_map.items())[:3]:
            assert torch.equal(reloaded["state_dict"]["lm_head.weight"][new_id],
                               old_w[old_id]), "[7] 迁移后匹配行不一致"
    passed.append("[7] ckpt 迁移后 cfg.vocab_size 更新 + 可加载 + 备份完整")

    # ---- [8] migrate_lm_head_state 顶层入口 ----
    sd = {"lm_head.weight": old_w.clone()}
    vocab = migrate_lm_head_state(sd, old_sp, new_sp, new_vocab)
    assert vocab == new_vocab, f"[8] 顶层入口返回错误: {vocab}"
    assert sd["lm_head.weight"].shape == (new_vocab, hidden), "[8] 顶层入口迁移形状错误"
    passed.append("[8] migrate_lm_head_state 顶层入口正常")

    print(f"\n{'='*50}")
    print(f"T12 smoke test: {len(passed)}/8 通过")
    for p in passed:
        print(f"  ✓ {p}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
