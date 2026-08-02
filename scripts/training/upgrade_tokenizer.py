#!/usr/bin/env python3
"""T12: 升级 zh 域 tokenizer（词表库热插拔第一步）。

背景：
    现有 zh tokenizer 用 30K 行语料训练（vocab=20K），覆盖率 ~70%，
    导致长词/常见词被切碎（如"表""征"分离），对话质量受限。
    本脚本用 1314 万行纯文本语料（data/distill/zh_texts.jsonl）采样训练
    新的 zh tokenizer（vocab=50K），实现词表库热插拔，不重训神经元。

流程：
    1. 备份旧模型 taiji/domains/zh/sp_zh.model → sp_zh_v20k.model
       （hot_swap_vocab.py 依赖旧模型构建 token id 映射）
    2. 从大语料均匀采样 ~200 万行（跳过空行/过短行）
    3. 训练新 BPE tokenizer（vocab=50K，参数与 build_domain_tokenizers 一致）
    4. 覆盖写入 taiji/domains/zh/sp_zh.model
    5. 诊断：新旧 tokenizer 在测试文本上的 token 数/覆盖率对比

输出：
    taiji/domains/zh/sp_zh.model        （新 50K tokenizer）
    taiji/domains/zh/sp_zh_v20k.model   （旧 20K tokenizer 备份，供映射）

用法：
    python scripts/training/upgrade_tokenizer.py [--vocab-size 50000]
                                                [--max-lines 2000000]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "distill" / "zh_texts.jsonl"
# T12-B: 混合对话语料，让 BPE 合并对话高频词（三原色/保持健康等）
DIALOGUE_PATH = PROJECT_ROOT / "data" / "simple_zh" / "alpaca_zh_sft.jsonl"
DOMAIN_DIR = PROJECT_ROOT / "taiji" / "domains" / "zh"
MODEL_PATH = DOMAIN_DIR / "sp_zh.model"
OLD_BACKUP = DOMAIN_DIR / "sp_zh_v20k.model"

# 与 build_domain_tokenizers.py 保持一致的训练参数
TRAIN_KWARGS = dict(
    model_type="bpe",
    character_coverage=0.9999,
    byte_fallback=True,
    normalization_rule_name="identity",
    add_dummy_prefix=True,
    remove_extra_whitespaces=False,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    split_digits=True,
    split_by_whitespace=True,
    split_by_unicode_script=True,
    split_by_number=True,
    max_sentence_length=16384,
    num_threads=8,
    input_sentence_size=0,
    shuffle_input_sentence=True,
    hard_vocab_limit=False,
)


def sample_corpus(max_lines: int, min_len: int = 10) -> list[str]:
    """从百科语料均匀采样，叠加对话语料（T12-B）。

    百科语料每行一条纯文本（无 JSON 包装），均匀采样保证覆盖全局分布；
    对话语料（alpaca_zh_sft.jsonl）全量读入并重复 3 次，
    使 BPE 合并对话高频词（"三原色""保持健康"等），服务对话场景。
    """
    texts: list[str] = []
    # step 6：1314 万行 → ~219 万行采样（略超目标 200 万）
    step = 6
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if len(texts) >= max_lines:
                break
            if i % step != 0:
                continue
            text = raw.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
            if len(text) < min_len:
                continue
            texts.append(text)
    if len(texts) < max_lines:
        # 若百科采样未达上限，继续取更多行（step 降为 1）
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if len(texts) >= max_lines:
                    break
                text = raw.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
                if len(text) < min_len:
                    continue
                texts.append(text)

    # 对话语料：全量 + 重复 3 次（占比 ~15%）
    if DIALOGUE_PATH.exists():
        import json
        n_dialog = 0
        for _rep in range(3):
            with open(DIALOGUE_PATH, encoding="utf-8") as f:
                for raw in f:
                    if len(texts) >= max_lines + 300_000:
                        break
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = (obj.get("text") or "").strip()
                    if len(text) < min_len:
                        continue
                    texts.append(text)
                    n_dialog += 1
        print(f"  对话语料: {n_dialog} 条（混合 {n_dialog/len(texts)*100:.1f}%）")
    return texts


def backup_old_model() -> None:
    """备份旧 tokenizer 为 sp_zh_v20k.model（hot_swap 依赖旧模型）。"""
    if OLD_BACKUP.exists():
        print(f"[backup] 旧模型备份已存在: {OLD_BACKUP.name}")
        return
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, OLD_BACKUP)
        print(f"[backup] 旧模型已备份: {OLD_BACKUP.name}")
    else:
        print(f"[backup] 未找到旧模型 {MODEL_PATH}（首次训练？）")


def train_tokenizer(texts: list[str], vocab_size: int) -> spm.SentencePieceProcessor:
    """用采样语料训练新 tokenizer。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
        corpus_path = f.name

    model_prefix = str(MODEL_PATH)[:-len(".model")]
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        **TRAIN_KWARGS,
    )

    Path(corpus_path).unlink()
    return spm.SentencePieceProcessor(str(MODEL_PATH))


def diagnose(old_sp, new_sp) -> None:
    """对比新旧 tokenizer 的分词质量。"""
    test_cases = [
        "人工智能是研究如何让计算机模拟人类智能行为的学科",
        "深度学习通过多层神经网络进行表征学习，是机器学习的一个重要分支",
        "量子计算利用量子叠加和量子纠缠原理，有望突破经典计算极限",
        "请用通俗易懂的语言解释一下什么是黑洞",
    ]
    print(f"\n诊断（旧 vocab={old_sp.GetPieceSize()} vs 新 vocab={new_sp.GetPieceSize()}）:")
    total_old = total_new = 0
    for text in test_cases:
        old_ids = old_sp.encode(text)
        new_ids = new_sp.encode(text)
        total_old += len(old_ids)
        total_new += len(new_ids)
        print(f"  \"{text[:30]}...\"")
        print(f"    旧: {len(old_ids)} tokens | 新: {len(new_ids)} tokens")
        print(f"    新 pieces: {' | '.join(new_sp.encode(text, out_type=str)[:15])}")
    print(f"\n  平均 token 数: 旧 {total_old/len(test_cases):.1f} → 新 {total_new/len(test_cases):.1f}"
          f"（压缩率 {total_new/total_old:.2f}×）")

    # 覆盖率粗查：用新语料 10K 行统计 unk 频率（旧 tokenizer 无 byte_fallback 统计）
    unk_id = old_sp.unk_id()
    new_unk = 0
    n_new_tokens = 0
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i >= 10000:
                break
            text = raw.strip()
            if not text:
                continue
            ids = new_sp.encode(text)
            new_unk += sum(1 for x in ids if x == unk_id)
            n_new_tokens += len(ids)
    print(f"  新 tokenizer unk 率（10K 行采样）: {new_unk}/{n_new_tokens} = {new_unk/max(n_new_tokens,1)*100:.3f}%")


def main():
    parser = argparse.ArgumentParser(description="T12: 升级 zh tokenizer")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--max-lines", type=int, default=2_000_000)
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"语料不存在: {CORPUS_PATH}")

    print(f"{'='*50}")
    print(f"T12: 升级 zh tokenizer（vocab 20K → {args.vocab_size}）")
    print(f"语料: {CORPUS_PATH}")
    print(f"{'='*50}")

    # 1. 备份旧模型
    backup_old_model()

    # 2. 采样语料
    print(f"\n采样语料（max_lines={args.max_lines}）...")
    texts = sample_corpus(args.max_lines)
    print(f"采样完成: {len(texts)} 行")
    if len(texts) < 100_000:
        raise RuntimeError(f"语料不足（{len(texts)} 行），中止升级")

    # 3. 训练
    print(f"\n训练新 tokenizer（vocab={args.vocab_size}，此步骤耗时数分钟）...")
    new_sp = train_tokenizer(texts, args.vocab_size)
    print(f"新 tokenizer 已保存: {MODEL_PATH}（实际词表 {new_sp.GetPieceSize()}）")

    # 4. 诊断
    old_sp = spm.SentencePieceProcessor(str(OLD_BACKUP))
    diagnose(old_sp, new_sp)

    print(f"\n[T12-1 完成] 词表库已升级，下一步运行:")
    print(f"  python scripts/training/hot_swap_vocab.py")


if __name__ == "__main__":
    main()
