"""下载 SFT (Supervised Fine-Tuning) instruction-response 数据.

与 download_hf_distill_data.py 区别：
  - 蒸馏数据：纯文本拼接（instruction + input + output 一段），用于 forward 续写学习
  - SFT 数据：保留 instruction/response 结构，让模型学"回答"而非"续写"

数据源（每域 3K 样本）：
- zh:      shibing624/alpaca-zh（中文指令）
- en:      tatsu-lab/alpaca（英文指令）
- code:    sahil2801/CodeAlpaca-20k（代码指令）
- math:    openai/gsm8k（数学推理）
- general: databricks/databricks-dolly-15k（通用问答）

输出格式：
  data/sft/{domain}_sft.pt
    list[dict] 每项 {"instruction": str, "input": str, "response": str, "prompt": str, "full": str}
      - prompt:   用于输入 neuron 的文本（instruction + input）
      - response: 期望 neuron 生成的回答
      - full:     prompt + response（teacher-forcing 全序列）

  data/sft/sft_datasets.pt
    dict[domain] -> list[dict]   （与上同结构）

  data/sft/sft_tokenized.pt
    dict[domain] -> {"input_ids": LongTensor[N, L], "labels": LongTensor[N, L], "response_mask": LongTensor[N, L]}
    （已 tokenize，供训练脚本直接使用）
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
DEFAULT_SAMPLES = 3000

# HF datasets-server API
API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

SPM_PATH = "checkpoint-481000/sentencepiece.model"


# ── 数据源配置 ──
DOMAIN_SOURCES = {
    "zh": [
        {
            "dataset": "shibing624/alpaca-zh",
            "config": "default",
            "split": "train",
            "instruction_field": "instruction",
            "input_field": "input",
            "response_field": "output",
            "max_samples": 5000,
        },
    ],
    "en": [
        {
            "dataset": "tatsu-lab/alpaca",
            "config": "default",
            "split": "train",
            "instruction_field": "instruction",
            "input_field": "input",
            "response_field": "output",
            "max_samples": 5000,
        },
    ],
    "code": [
        {
            "dataset": "sahil2801/CodeAlpaca-20k",
            "config": "default",
            "split": "train",
            "instruction_field": "instruction",
            "input_field": "input",
            "response_field": "output",
            "max_samples": 5000,
        },
    ],
    "math": [
        {
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": "train",
            "instruction_field": "question",
            "input_field": None,
            "response_field": "answer",
            "max_samples": 4000,
        },
    ],
    "general": [
        {
            "dataset": "databricks/databricks-dolly-15k",
            "config": "default",
            "split": "train",
            "instruction_field": "instruction",
            "input_field": "context",
            "response_field": "response",
            "max_samples": 5000,
        },
    ],
}


def extract_sft_fields(sample: dict, cfg: dict) -> dict:
    """从 HF 样本中提取 instruction/input/response 三元组."""
    instruction = str(sample.get(cfg["instruction_field"], "")).strip()
    response = str(sample.get(cfg["response_field"], "")).strip()

    inp = ""
    if cfg.get("input_field"):
        inp = str(sample.get(cfg["input_field"], "")).strip()

    if not instruction or not response:
        return {}

    # 构造 prompt（与 neuron 推理时输入一致）
    prompt_parts = [instruction]
    if inp:
        prompt_parts.append(inp)
    prompt = "\n".join(prompt_parts)

    # full sequence: prompt + "\n" + response
    full = f"{prompt}\n{response}"

    return {
        "instruction": instruction,
        "input": inp,
        "response": response,
        "prompt": prompt,
        "full": full,
    }


def fetch_page(dataset: str, config: str, split: str, offset: int,
               length: int = PAGE_SIZE, retries: int = 5) -> list[dict]:
    """从 HF datasets-server 获取一页数据（带 429 限流处理）.

    429 限流策略：
      - 指数退避：5s, 15s, 45s, 135s, 405s
      - 最多 5 次重试
      - 每次成功后短暂 sleep 0.5s 避免 trigger 限流
    """
    url = (f"{API_BASE}?dataset={dataset}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    backoff_delays = [5, 15, 45, 135, 405]  # 指数退避
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "taiji-neuron/1.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                time.sleep(0.5)  # 避免 trigger 限流
                return data.get("rows", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"      404: {url}")
                return []
            if e.code == 429:
                # 限流，指数退避
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                print(f"      429 rate-limited, backoff {delay}s (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                print(f"      429 给 up after {retries} retries")
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


def download_domain_sft(domain: str, target_n: int) -> list[dict]:
    """下载一个域的 SFT 数据（原始 JSON 结构）."""
    sources = DOMAIN_SOURCES[domain]
    samples: list[dict] = []

    for src_idx, cfg in enumerate(sources):
        if len(samples) >= target_n:
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

        while fetched < max_samples and len(samples) < target_n:
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
                sft = extract_sft_fields(sample, cfg)
                if not sft:
                    continue
                # 过滤太短的 response
                if len(sft["response"]) < 5:
                    continue
                samples.append(sft)
                fetched += 1
                if len(samples) >= target_n:
                    break

            offset += PAGE_SIZE
            if fetched % 500 < PAGE_SIZE:
                print(f"    fetched={fetched}, total={len(samples)}")

        print(f"  [{domain}] 源 {src_idx+1} 完成: fetched={fetched}, total={len(samples)}")

    return samples[:target_n]


def tokenize_sft(sp: spm.SentencePieceProcessor, samples: list[dict],
                 seq_len: int = SEQ_LEN) -> dict:
    """把 SFT 样本 tokenize 成训练用的 input_ids + labels + response_mask.

    关键设计：
    - input_ids = tokenize(prompt) + tokenize(response) + [EOS]
    - labels = [-100] * len(prompt_ids) + response_ids + [EOS]
        （prompt 部分不计 loss，只对 response 计算 LM loss = teacher forcing SFT）
    - response_mask = [0] * len(prompt_ids) + [1] * (len(response_ids) + 1)
        （标识哪些位置是 response，用于统计训练效果）

    所有序列 pad/truncate 到 seq_len。
    """
    input_ids_list = []
    labels_list = []
    response_mask_list = []

    PAD_ID = 0
    EOS_ID = 3  # v2 tokenizer contract: </s>=3 (EOS)
    IGNORE_LABEL = -100

    for sft in samples:
        prompt_ids = [t + TEXT_OFFSET for t in sp.encode(sft["prompt"])]
        response_ids = [t + TEXT_OFFSET for t in sp.encode(sft["response"])]

        # 留 1 个位置给 EOS
        max_prompt = seq_len - 1
        if len(prompt_ids) > max_prompt:
            # prompt 太长，跳过
            continue
        max_response = seq_len - len(prompt_ids) - 1
        if len(response_ids) > max_response:
            response_ids = response_ids[:max_response]

        # 构造 input_ids: prompt + response + EOS
        full_ids = prompt_ids + response_ids + [EOS_ID]
        # pad 到 seq_len
        pad_len = seq_len - len(full_ids)
        full_ids = full_ids + [PAD_ID] * pad_len

        # labels: prompt 部分用 IGNORE_LABEL，response + EOS 用真实 token
        labels = ([IGNORE_LABEL] * len(prompt_ids)
                  + response_ids + [EOS_ID]
                  + [IGNORE_LABEL] * pad_len)

        # response_mask: prompt=0, response+EOS=1, pad=0
        response_mask = ([0] * len(prompt_ids)
                         + [1] * (len(response_ids) + 1)
                         + [0] * pad_len)

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        response_mask_list.append(response_mask)

    if not input_ids_list:
        return {
            "input_ids": torch.zeros((0, seq_len), dtype=torch.long),
            "labels": torch.zeros((0, seq_len), dtype=torch.long),
            "response_mask": torch.zeros((0, seq_len), dtype=torch.long),
        }

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_list, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"每域目标样本数（默认 {DEFAULT_SAMPLES}）")
    parser.add_argument("--output", type=str, default="data/sft",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载 tokenizer
    if not os.path.exists(SPM_PATH):
        print(f"ERROR: tokenizer not found at {SPM_PATH}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_PATH)
    print(f"Loaded tokenizer: vocab_size={sp.GetPieceSize()}, SPM_PATH={SPM_PATH}")

    all_datasets = {}
    all_tokenized = {}

    for domain in DOMAIN_SOURCES.keys():
        print(f"\n{'=' * 60}\n[Download] domain={domain}, target={args.samples}\n{'=' * 60}")
        samples = download_domain_sft(domain, args.samples)
        print(f"  Total samples: {len(samples)}")

        # 保存原始 SFT 数据
        domain_path = os.path.join(args.output, f"{domain}_sft.pt")
        torch.save(samples, domain_path)
        print(f"  Saved raw SFT: {domain_path} ({len(samples)} samples)")

        # tokenize
        print(f"  Tokenizing...")
        tokenized = tokenize_sft(sp, samples, SEQ_LEN)
        print(f"  Tokenized: input_ids={tokenized['input_ids'].shape}, "
              f"labels={tokenized['labels'].shape}")
        # 统计 response token 数
        if len(tokenized["response_mask"]) > 0:
            avg_resp_len = tokenized["response_mask"].sum(dim=1).float().mean().item()
            print(f"  Avg response length: {avg_resp_len:.1f} tokens")

        all_datasets[domain] = samples
        all_tokenized[domain] = tokenized

    # 保存合并文件
    all_path = os.path.join(args.output, "sft_datasets.pt")
    torch.save(all_datasets, all_path)
    print(f"\nSaved combined raw: {all_path}")

    tok_path = os.path.join(args.output, "sft_tokenized.pt")
    torch.save(all_tokenized, tok_path)
    print(f"Saved combined tokenized: {tok_path}")

    # 摘要
    print(f"\n{'=' * 60}\n[Summary]\n{'=' * 60}")
    for domain, samples in all_datasets.items():
        tok = all_tokenized[domain]
        print(f"  {domain}: {len(samples)} samples, "
              f"input_ids={tok['input_ids'].shape}")


if __name__ == "__main__":
    main()
