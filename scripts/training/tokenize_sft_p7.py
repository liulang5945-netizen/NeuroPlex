"""P8-2: 用域专用 tokenizer 重新 tokenize SFT 数据。

P7 架构每 neuron 有独立 embedding + 独立 lm_head + 域专用 vocab，
需要将 SFT 数据从旧共享 tokenizer 转换为域专用 tokenizer。

输入: data/sft/sft_datasets.pt（raw text: prompt + response）
输出: data/sft/p7_{domain}_tokenized.pt（domain-tokenized tensors）

用法:
    python scripts/training/tokenize_sft_p7.py
    python scripts/training/tokenize_sft_p7.py --seq-len 512 --samples 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.translator import TokenizerHub

# ── 配置 ──
SEQ_LEN = 256
PAD_ID = 0
IGNORE_LABEL = -100
# general 域复用 en tokenizer
GENERAL_TOKENIZER_DOMAIN = "en"
OUTPUT_DIR = "data/sft"

DOMAINS = ["zh", "en", "code", "math", "general"]


def tokenize_domain(
    hub: TokenizerHub,
    domain: str,
    samples: List[dict],
    seq_len: int = 256,
) -> Dict[str, torch.Tensor]:
    """用域 tokenizer 将 raw text 转换为 input_ids + labels 张量。

    Args:
        hub: TokenizerHub 实例
        domain: 域名 ("zh"/"en"/"code"/"math"/"general")
        samples: list of dict with "prompt" and "response" keys
        seq_len: 最大序列长度

    Returns:
        {"input_ids": [N, seq_len], "labels": [N, seq_len]}
    """
    # general 域复用 en tokenizer
    tokenizer_domain = "en" if domain == "general" else domain

    all_input_ids = []
    all_labels = []
    skipped = 0

    for item in samples:
        prompt = item["prompt"]
        response = item["response"]

        # 域 tokenizer encode
        prompt_ids = hub.encode(prompt, domain=tokenizer_domain)
        response_ids = hub.encode(response, domain=tokenizer_domain)

        # 跳过空 response（罕见）
        if not response_ids:
            skipped += 1
            continue

        # EOS
        eos_id = hub.eos_token_id(tokenizer_domain)

        # 拼接: prompt_ids + response_ids + eos
        full_ids = prompt_ids + response_ids + [eos_id]

        # 截断或填充
        if len(full_ids) > seq_len:
            full_ids = full_ids[:seq_len]
        else:
            full_ids = full_ids + [PAD_ID] * (seq_len - len(full_ids))

        # labels: prompt 部分为 -100，response+eos 部分保留
        prompt_len = min(len(prompt_ids), seq_len)
        response_len = min(len(response_ids) + 1, seq_len - prompt_len)

        labels = [IGNORE_LABEL] * prompt_len
        # response_ids + eos 对应的 label
        for i in range(response_len):
            if prompt_len + i < seq_len:
                labels.append(full_ids[prompt_len + i])
        # 剩余 padding 填充 -100
        while len(labels) < seq_len:
            labels.append(IGNORE_LABEL)

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    if skipped:
        print(f"  [{domain}] skipped {skipped} samples with empty response")

    if not all_input_ids:
        raise ValueError(f"[{domain}] no valid samples remain after filtering")

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)

    # 验证
    assert input_ids.shape == (len(all_input_ids), seq_len), \
        f"input_ids shape mismatch: {input_ids.shape}"
    assert labels.shape == (len(all_labels), seq_len), \
        f"labels shape mismatch: {labels.shape}"
    # prompt 位置 label 应为 -100
    assert (labels[:, 0] == IGNORE_LABEL).all(), \
        "first position should be prompt (label=-100)"

    print(f"  [{domain}] {len(all_input_ids)} samples, "
          f"input_ids={input_ids.shape}, labels={labels.shape}")

    return {"input_ids": input_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="P8-2: tokenize SFT data with domain tokenizers")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="max sequence length")
    parser.add_argument("--samples", type=int, default=None,
                        help="max samples per domain (None=all)")
    parser.add_argument("--input", type=str, default="data/sft/sft_datasets.pt",
                        help="input raw SFT data path")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()

    # 1. 加载 TokenizerHub
    print("=== P8-2: Tokenize SFT data with domain tokenizers ===")
    hub = TokenizerHub.load_default_domains()
    domains_info = {d: hub.vocab_size(d) for d in hub.list_domains()}
    print(f"Loaded domains: {domains_info}")
    print(f"GENERAL domain uses 'en' tokenizer (vocab={domains_info.get('en', 'N/A')})")

    # 2. 加载原始 SFT 数据
    input_path = args.input
    if not os.path.exists(input_path):
        # 尝试 per-domain 文件
        print(f"Warning: {input_path} not found, trying per-domain files...")
        domain_data = {}
        for domain in DOMAINS:
            path = f"data/sft/{domain}_sft.pt"
            if os.path.exists(path):
                d = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(d, list):
                    domain_data[domain] = d
                    print(f"  Loaded {domain}: {len(d)} samples from {path}")
        if not domain_data:
            raise FileNotFoundError(f"No SFT data found at {input_path} or per-domain files")
    else:
        raw = torch.load(input_path, map_location="cpu", weights_only=False)
        domain_data = {}
        for domain in DOMAINS:
            if domain in raw:
                items = raw[domain]
                if isinstance(items, list):
                    domain_data[domain] = items
                    print(f"  Loaded {domain}: {len(items)} samples")
    print(f"Total domains: {list(domain_data.keys())}")

    # 3. Tokenize 每个域
    os.makedirs(args.output_dir, exist_ok=True)
    output_sizes = {}

    for domain in DOMAINS:
        if domain not in domain_data:
            print(f"  [{domain}] SKIP: no data")
            continue

        samples = domain_data[domain]
        if args.samples and len(samples) > args.samples:
            samples = samples[:args.samples]

        result = tokenize_domain(hub, domain, samples, seq_len=args.seq_len)

        # 保存
        out_path = os.path.join(args.output_dir, f"p7_{domain}_tokenized.pt")
        torch.save(result, out_path)
        output_sizes[domain] = result["input_ids"].shape[0]
        print(f"  Saved {out_path}")

    # 4. 汇总
    print(f"\n=== Done ===")
    for domain, n in output_sizes.items():
        print(f"  {domain}: {n} samples")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    print(f"Next: python scripts/training/train_neurons_from_scratch.py")


if __name__ == "__main__":
    main()
