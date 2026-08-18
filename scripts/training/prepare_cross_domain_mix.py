"""P8-2x: 生成跨域混合 SFT 数据（工作3：跨域神经元 Step 2 数据准备）。

策略（2026-08-05 用户指正 + 实测验证）：
- code/math neuron **保留各自域 tokenizer**（code 12K / math 10K），通过词库转译
  （general 256K 统一输入空间 + S6 alignment_table）实现跨域语义转换。
- 混合数据 = 域 SFT 数据（CodeAlpaca/GSM8K 英文）+ 英文对话数据（en_sft）。
  实测域 tokenizer 编码英文目标 byte_ratio 2-7% 高效。
- **不混中文对话**（code tokenizer 编中文 byte_fallback 57% 低效）；
  中文语义通过 general 256K 输入空间在协作层处理。

输入: data/sft/code_sft.pt, math_sft.pt, en_sft.pt
输出: data/sft/p7_{domain}_mixed_tokenized.pt
  {"input_ids": [N, 256], "labels": [N, 256]}  （与 p7_{domain}_tokenized.pt 同格式）

用法:
    python scripts/training/prepare_cross_domain_mix.py
    python scripts/training/prepare_cross_domain_mix.py --domains code math
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.resonance.translator import TokenizerHub

# ── 配置 ──
SEQ_LEN = 256
PAD_ID = 0
IGNORE_LABEL = -100
SFT_DIR = "data/sft"

# 每域混合：域数据（主）+ 英文对话（辅，提供对话能力）
MIX_RATIO_DOMAIN = 1.0   # 域 SFT 全部保留
MIX_RATIO_DIALOGUE = 1.0  # 英文对话等量混入（让 neuron 具备对话响应能力）


def load_sft(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"  ⚠️ 不存在: {path}", flush=True)
        return []
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, list):
        return d
    raise TypeError(f"预期 list[dict]，得到 {type(d)}")


def tokenize_domain_mix(
    hub: TokenizerHub,
    domain: str,
    domain_samples: List[dict],
    dialogue_samples: List[dict],
    seq_len: int = SEQ_LEN,
) -> Dict[str, torch.Tensor]:
    """用域 tokenizer 将混合 SFT 数据转为 input_ids + labels 张量。

    与 tokenize_sft_p7.tokenize_domain 相同格式，但数据源为混合集。
    目标（labels）用域 tokenizer 编码（域数据 + 英文对话均高效）。
    """
    tokenizer_domain = "en" if domain == "general" else domain
    samples = domain_samples + dialogue_samples
    random.Random(42).shuffle(samples)

    all_input_ids = []
    all_labels = []
    skipped = 0

    for item in samples:
        prompt = item["prompt"]
        response = item["response"]
        prompt_ids = hub.encode(prompt, domain=tokenizer_domain)
        response_ids = hub.encode(response, domain=tokenizer_domain)
        if not response_ids:
            skipped += 1
            continue

        eos_id = hub.eos_token_id(tokenizer_domain)
        full_ids = prompt_ids + response_ids + [eos_id]
        if len(full_ids) > seq_len:
            full_ids = full_ids[:seq_len]
        else:
            full_ids = full_ids + [PAD_ID] * (seq_len - len(full_ids))

        prompt_len = min(len(prompt_ids), seq_len)
        response_len = min(len(response_ids) + 1, seq_len - prompt_len)
        labels = [IGNORE_LABEL] * prompt_len
        for i in range(response_len):
            if prompt_len + i < seq_len:
                labels.append(full_ids[prompt_len + i])
        while len(labels) < seq_len:
            labels.append(IGNORE_LABEL)

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    if skipped:
        print(f"  [{domain}] skipped {skipped} samples with empty response", flush=True)
    if not all_input_ids:
        raise ValueError(f"[{domain}] no valid samples after filtering")

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)
    assert input_ids.shape == (len(all_input_ids), seq_len)
    assert labels.shape == (len(all_labels), seq_len)

    print(f"  [{domain}] mixed: domain={len(domain_samples)} + dialogue={len(dialogue_samples)} "
          f"→ {len(all_input_ids)} samples, input_ids={tuple(input_ids.shape)}", flush=True)
    return {"input_ids": input_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="P8-2x: 跨域混合 SFT 数据准备")
    parser.add_argument("--domains", nargs="+", default=["code", "math"],
                        help="要混合的域（默认 code math）")
    parser.add_argument("--dialogue", type=str, default="en",
                        help="对话数据域（默认 en_sft 英文对话，域 tokenizer 编码高效）")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--output-dir", type=str, default=SFT_DIR)
    args = parser.parse_args()

    print("=== P8-2x: 跨域混合 SFT 数据准备 ===")
    hub = TokenizerHub.load_default_domains()
    domains_info = {d: hub.vocab_size(d) for d in hub.list_domains()}
    print(f"Loaded domains: {domains_info}")

    dialogue_samples = load_sft(os.path.join(args.output_dir, f"{args.dialogue}_sft.pt"))
    print(f"对话数据({args.dialogue}_sft.pt): {len(dialogue_samples)} 条", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for domain in args.domains:
        domain_samples = load_sft(os.path.join(args.output_dir, f"{domain}_sft.pt"))
        print(f"域数据({domain}_sft.pt): {len(domain_samples)} 条", flush=True)
        if not domain_samples:
            print(f"  [{domain}] SKIP: 无域数据", flush=True)
            continue

        result = tokenize_domain_mix(
            hub, domain, domain_samples, dialogue_samples, seq_len=args.seq_len,
        )
        out_path = os.path.join(args.output_dir, f"p7_{domain}_mixed_tokenized.pt")
        torch.save(result, out_path)
        print(f"  Saved {out_path}", flush=True)

    print("\n=== Done ===")
    print("下一步: 用混合数据训练 code/math neuron 本体")
    print("  python scripts/training/train_neurons_from_scratch.py --domain code --data-dir data/sft")


if __name__ == "__main__":
    main()
