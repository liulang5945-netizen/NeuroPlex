#!/usr/bin/env python3
"""为每个领域构建专用 SentencePiece tokenizer。

输出路径统一为 taiji/domains/<domain>/sp_<domain>.model，
与 load_domain_tokenizer / load_general_tokenizer 的加载路径一致（修复 T13 路径不一致）。

general 域使用混合语料（zh+en+code+math），vocab=256K，覆盖全词（修复 S2 隐性天花板）。
"""

from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/training/ → 项目根
DATA_DIR = PROJECT_ROOT / "taiji_data/training_data/pretrain_mix_v1"
# 修复 T13: 输出路径与 load_domain_tokenizer/load_general_tokenizer 的加载路径一致
OUTPUT_DIR = PROJECT_ROOT / "taiji" / "domains"

# general 域使用混合语料（zh+en+code+math），vocab=256K
# 其他域用专用语料，vocab=10K-20K
DOMAINS = {
    "zh":      ("skypile_zh.jsonl",      20000, 30000, "中文", None),
    "en":      ("falcon_refinedweb_en.jsonl", 16000, 20000, "英文", None),
    "code":    ("codeparrot_code.jsonl",  12000, 15000, "代码", None),
    "math":    ("openwebmath.jsonl",      10000, 10000, "数学", None),
    # general: 混合语料，256K vocab，max_lines 表示每个域取多少行合并
    "general": (None,                    256000, 200000, "通用(混合语料)", "mixed"),
}


def extract_text(path: Path, max_lines: int) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if len(lines) >= max_lines:
                break
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "") or obj.get("content", "") or obj.get("output", "") or ""
            text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\x00", "").strip()
            if len(text) > 60:
                lines.append(text)
    return lines


def train_tokenizer(domain: str, texts: list[str], vocab_size: int) -> spm.SentencePieceProcessor:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
        corpus_path = f.name

    model_prefix = str(OUTPUT_DIR / f"sp_{domain}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
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

    Path(corpus_path).unlink()
    sp = spm.SentencePieceProcessor(f"{model_prefix}.model")
    return sp


def diagnose(sp, domain: str):
    """验证 tokenizer 质量。"""
    test_cases = {
        "zh":   ["深度学习是人工智能的一个分支", "中华人民共和国宪法"],
        "en":   ["Inspector General Report on Tax-Exempt Scrutiny",
                 "Understanding the fundamentals of machine learning"],
        "code": ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                 "class NeuralNetwork(nn.Module):"],
        "math": ["f(x) = \\int_{0}^{\\infty} e^{-x^2} dx",
                 "Let G be a finite group of order n"],
    }

    print(f"\n  {domain} tokenizer (vocab={sp.GetPieceSize()}):")
    for text in test_cases.get(domain, [])[:2]:
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        print(f"    \"{text[:60]}...\"")
        print(f"    → {len(ids)} tokens: {' | '.join(pieces[:20])}")


def load_mixed_corpus(max_lines_per_domain: int) -> list[str]:
    """为 general 域加载混合语料（zh+en+code+math）。

    每个域取 max_lines_per_domain 行，合并后作为 general tokenizer 训练语料。
    """
    mixed_texts = []
    sub_domains = [
        ("skypile_zh.jsonl",         max_lines_per_domain, "中文"),
        ("falcon_refinedweb_en.jsonl", max_lines_per_domain, "英文"),
        ("codeparrot_code.jsonl",    max_lines_per_domain // 2, "代码"),  # 代码量减半
        ("openwebmath.jsonl",        max_lines_per_domain // 4, "数学"),  # 数学量减半
    ]
    for fname, n, desc in sub_domains:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  {desc}: 文件不存在，跳过")
            continue
        texts = extract_text(path, n)
        mixed_texts.extend(texts)
        print(f"  {desc}: {len(texts)} 行")
    return mixed_texts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, (fname, vocab_size, max_lines, desc, corpus_type) in DOMAINS.items():
        print(f"\n{'='*50}")
        print(f"训练 {domain} tokenizer ({desc}, vocab={vocab_size})")
        print(f"{'='*50}")

        if corpus_type == "mixed":
            # general 域：混合语料
            # max_lines 是每个子域的行数上限，总语料 = 4 个子域之和
            per_domain = max_lines // 4
            texts = load_mixed_corpus(per_domain)
        else:
            # 专用域：单一语料
            path = DATA_DIR / fname
            if not path.exists():
                print(f"  {domain}: 文件不存在，跳过")
                continue
            texts = extract_text(path, max_lines)

        if len(texts) < 1000:
            print(f"  {domain}: 语料不足（{len(texts)} 行），跳过")
            continue

        print(f"  语料: {len(texts)} 行")
        sp = train_tokenizer(domain, texts, vocab_size)
        print(f"  实际词表: {sp.GetPieceSize()}")

        diagnose(sp, domain)

    print(f"\n输出: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
