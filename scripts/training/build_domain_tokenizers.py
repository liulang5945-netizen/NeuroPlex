#!/usr/bin/env python3
"""为每个领域构建专用 SentencePiece tokenizer。"""

from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "taiji_data/training_data/pretrain_mix_v1"
OUTPUT_DIR = PROJECT_ROOT / "domain_tokenizers"

DOMAINS = {
    "zh":      ("skypile_zh.jsonl",      20000, 30000, "中文"),
    "en":      ("falcon_refinedweb_en.jsonl", 16000, 20000, "英文"),
    "code":    ("codeparrot_code.jsonl",  12000, 15000, "代码"),
    "math":    ("openwebmath.jsonl",      10000, 10000, "数学"),
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, (fname, vocab_size, max_lines, desc) in DOMAINS.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  {domain}: 文件不存在，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"训练 {domain} tokenizer ({desc}, vocab={vocab_size})")
        print(f"{'='*50}")

        texts = extract_text(path, max_lines)
        print(f"  语料: {len(texts)} 行")

        sp = train_tokenizer(domain, texts, vocab_size)
        print(f"  实际词表: {sp.GetPieceSize()}")

        diagnose(sp, domain)

    print(f"\n输出: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
