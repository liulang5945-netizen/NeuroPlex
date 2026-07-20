"""从 HuggingFace datasets-server API 下载真实数据用于神经元蒸馏。

纯标准库实现（urllib + json），不依赖 datasets/huggingface_hub/pandas。

数据源（每域 10K 样本）：
- zh:      BelleGroup/train_1.5M_CN（中文指令）
- en:      teknium/OpenHermes-2.5（英文对话）
- code:    sahil2801/CodeAlpaca-20k（代码指令）
- math:    openai/gsm8k（数学推理，main + socratic）
- general: databricks/databricks-dolly-15k（通用问答）

输出格式（与 download_distill_data.py 兼容）：
  data/distill/{domain}.pt              — torch.long [N, 256]
  data/distill/domain_datasets.pt       — dict[str, Tensor]

v2 tokenizer contract: text token ID = sentencepiece ID + TEXT_OFFSET (13388)
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

# sentencepiece 装在 _libs/ 下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS):
    sys.path.insert(0, _LIBS)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import sentencepiece as spm

# v2 contract: text token range [13388, 256000)
TEXT_OFFSET = 13388
SEQ_LEN = 256
DEFAULT_SAMPLES = 10000
DEFAULT_TEACHER = "E:/taiji-neuron/checkpoint-481000"
DEFAULT_OUTPUT = "data/distill"

# HF datasets-server API
API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # 每次请求 100 条


# ── 数据源配置 ──
# 每个域配置一个 list，按顺序取样直到达到目标样本数
# config: HF dataset 的 config name（通常 "default"）
DOMAIN_SOURCES = {
    "zh": [
        {
            "dataset": "shibing624/alpaca-zh",
            "config": "default",
            "split": "train",
            "text_field": "instruction",
            "input_field": "input",      # instruction + input + output 拼接
            "answer_field": "output",
            "max_samples": 12000,
        },
    ],
    "en": [
        {
            "dataset": "tatsu-lab/alpaca",
            "config": "default",
            "split": "train",
            "text_field": "instruction",
            "input_field": "input",
            "answer_field": "output",
            "max_samples": 12000,
        },
    ],
    "code": [
        {
            "dataset": "sahil2801/CodeAlpaca-20k",
            "config": "default",
            "split": "train",
            "text_field": "instruction",
            "input_field": "input",
            "answer_field": "output",    # output 是代码
            "max_samples": 12000,
        },
    ],
    "math": [
        {
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": "train",
            "text_field": "question",
            "answer_field": "answer",
            "max_samples": 8000,
        },
        {
            "dataset": "openai/gsm8k",
            "config": "socratic",
            "split": "train",
            "text_field": "question",
            "answer_field": "answer",
            "max_samples": 4000,
        },
    ],
    "general": [
        {
            "dataset": "databricks/databricks-dolly-15k",
            "config": "default",
            "split": "train",
            "text_field": "instruction",
            "input_field": "context",
            "answer_field": "response",
            "max_samples": 12000,
        },
    ],
}


def extract_text(sample: dict, cfg: dict) -> str:
    """从 HF 样本中提取纯文本。"""
    if cfg.get("is_conversation"):
        # OpenHermes: conversations = [{from: human, value: ...}, {from: gpt, value: ...}]
        convs = sample.get("conversations", [])
        if isinstance(convs, list):
            parts = []
            for c in convs:
                if isinstance(c, dict) and "value" in c:
                    parts.append(str(c["value"]))
            return "\n".join(parts)
        return ""

    text = str(sample.get(cfg["text_field"], "")).strip()
    if not text:
        return ""

    # 拼接 input/context（如果有）
    if cfg.get("input_field"):
        inp = str(sample.get(cfg["input_field"], "")).strip()
        if inp:
            text = f"{text}\n{inp}"

    # 拼接 answer（如果有）
    if cfg.get("answer_field"):
        ans = str(sample.get(cfg["answer_field"], "")).strip()
        if ans:
            text = f"{text}\n{ans}"

    return text


def fetch_page(dataset: str, config: str, split: str, offset: int,
               length: int = PAGE_SIZE, retries: int = 3) -> list[dict]:
    """从 HF datasets-server 获取一页数据。返回 rows 列表（每项是 {row: {...}, row_idx: n}）。"""
    url = (f"{API_BASE}?dataset={dataset}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "taiji-neuron/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("rows", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"      404: {url}")
                return []
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"      HTTP {e.code}: {e.reason}")
            return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"      请求失败: {e}")
            return []
    return []


def download_domain(domain: str, target_n: int, sp: spm.SentencePieceProcessor) -> list[list[int]]:
    """下载一个域的数据并 tokenize。"""
    sources = DOMAIN_SOURCES[domain]
    tokens_list: list[list[int]] = []

    for src_idx, cfg in enumerate(sources):
        if len(tokens_list) >= target_n:
            break

        dataset = cfg["dataset"]
        config = cfg["config"]
        split = cfg["split"]
        max_samples = cfg.get("max_samples", target_n)

        print(f"  [{domain}] 源 {src_idx+1}/{len(sources)}: "
              f"{dataset} (config={config}, split={split})")

        fetched = 0
        offset = 0
        empty_pages = 0

        while fetched < max_samples and len(tokens_list) < target_n:
            rows = fetch_page(dataset, config, split, offset)
            if not rows:
                empty_pages += 1
                if empty_pages >= 2:
                    print(f"    连续 {empty_pages} 次空页，停止此源")
                    break
                offset += PAGE_SIZE
                continue
            empty_pages = 0

            for item in rows:
                sample = item.get("row", {})
                text = extract_text(sample, cfg)
                text = text.strip().replace("\n", " ")
                if len(text) < 50:
                    continue

                # tokenize + text_offset
                encoded = [tid + TEXT_OFFSET for tid in sp.EncodeAsIds(text)]

                # 长度过滤
                if len(encoded) >= SEQ_LEN:
                    tokens_list.append(encoded[:SEQ_LEN])
                elif len(encoded) >= 20:
                    padded = encoded.copy()
                    while len(padded) < SEQ_LEN:
                        padded = padded + encoded
                    tokens_list.append(padded[:SEQ_LEN])
                else:
                    continue

                fetched += 1
                if fetched >= max_samples or len(tokens_list) >= target_n:
                    break

            offset += PAGE_SIZE
            if fetched % 1000 < PAGE_SIZE:
                print(f"    fetched={fetched}, total={len(tokens_list)}, offset={offset}")

        print(f"    完成: fetched={fetched}, total={len(tokens_list)}")

    return tokens_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="每域目标样本数（默认 10K）")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--tokenizer", default=None,
                        help="sentencepiece.model 路径（默认自动查找）")
    args = parser.parse_args()

    # 自动查找 tokenizer
    if args.tokenizer is None:
        candidates = [
            os.path.join(DEFAULT_TEACHER, "sentencepiece.model"),
            os.path.join(PROJECT_ROOT, "checkpoint-481000", "sentencepiece.model"),
            os.path.join(PROJECT_ROOT, "taiji", "tokenizer_native_v2", "sentencepiece.model"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.tokenizer = c
                break
        if args.tokenizer is None:
            print(f"❌ 找不到 sentencepiece.model，尝试过: {candidates}")
            sys.exit(1)

    print("=" * 70)
    print(f"HF 真实数据下载（每域 {args.samples} 样本，纯标准库）")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  output:    {args.output_dir}")
    print(f"  seq_len:   {args.seq_len}")
    print(f"  TEXT_OFFSET: {TEXT_OFFSET}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer loaded: {sp.GetPieceSize()} tokens\n")

    all_data = {}
    for domain in ["zh", "en", "code", "math", "general"]:
        print(f"\n── {domain} ──────────────────────────────────")
        t0 = time.time()
        tokens_list = download_domain(domain, args.samples, sp)
        elapsed = time.time() - t0

        if len(tokens_list) < args.samples // 2:
            print(f"  ⚠️  {domain} 只获取到 {len(tokens_list)} 条（目标 {args.samples}）")

        if not tokens_list:
            print(f"  ❌ {domain} 无数据，跳过")
            continue

        t = torch.tensor(tokens_list, dtype=torch.long)
        all_data[domain] = t
        torch.save(t, os.path.join(args.output_dir, f"{domain}.pt"))
        print(f"  ✓ {domain}: {t.shape}, "
              f"range=[{t.min().item()}, {t.max().item()}], "
              f"耗时 {elapsed:.0f}s")

    # 合并保存
    domain_path = os.path.join(args.output_dir, "domain_datasets.pt")
    torch.save(all_data, domain_path)
    print(f"\n{'=' * 70}")
    print(f"✅ 全部完成: {domain_path}")
    print(f"{'=' * 70}")
    total_tokens = 0
    for d, t in all_data.items():
        n_tokens = t.numel()
        total_tokens += n_tokens
        print(f"  {d:8s}: {t.shape}, {n_tokens:,} tokens")
    print(f"  {'total':8s}: {total_tokens:,} tokens "
          f"({total_tokens / 1e6:.1f}M tokens)")

    print(f"\n下一步:")
    print(f"  1. python scripts/training/precompute_teacher_cache.py")
    print(f"  2. python scripts/training/build_shared_projections.py")
    print(f"  3. python scripts/training/distill_neurons.py "
          f"--checkpoint {DEFAULT_TEACHER} --data_dir {args.output_dir} "
          f"--output_dir data/neurons --steps 2000")


if __name__ == "__main__":
    main()
