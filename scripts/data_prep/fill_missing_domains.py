"""补全缺失的域数据（en, code），加入请求间隔避免 429 限流。

将新数据合并到现有的 data/distill/domain_datasets.pt。
"""
from __future__ import annotations

import os
import sys
import json
import time
import urllib.request
import urllib.error

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

TEXT_OFFSET = 13388
SEQ_LEN = 256
API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
REQUEST_INTERVAL = 0.3  # 每次请求间隔 0.3 秒，避免 429

DOMAIN_DATA_PATH = "data/distill/domain_datasets.pt"
# R18（REMEDIATION_PLAN 2026-08-14）：绝对路径改相对（仓库根目录为 cwd 约定），
# 可用环境变量 TAICHI_TEACHER_PATH 覆盖。
TEACHER_PATH = os.environ.get("TAICHI_TEACHER_PATH", "checkpoint-481000")

# 需要补全的域
MISSING_DOMAINS = {
    "en": {
        "dataset": "tatsu-lab/alpaca",
        "config": "default",
        "split": "train",
        "text_field": "instruction",
        "input_field": "input",
        "answer_field": "output",
        "target_samples": 8000,  # 降低到 8K 避免限流
    },
    "code": {
        "dataset": "sahil2801/CodeAlpaca-20k",
        "config": "default",
        "split": "train",
        "text_field": "instruction",
        "input_field": "input",
        "answer_field": "output",
        "target_samples": 8000,
    },
}


def extract_text(sample: dict, cfg: dict) -> str:
    text = str(sample.get(cfg["text_field"], "")).strip()
    if not text:
        return ""
    if cfg.get("input_field"):
        inp = str(sample.get(cfg["input_field"], "")).strip()
        if inp:
            text = f"{text}\n{inp}"
    if cfg.get("answer_field"):
        ans = str(sample.get(cfg["answer_field"], "")).strip()
        if ans:
            text = f"{text}\n{ans}"
    return text


def fetch_page(dataset, config, split, offset, length=PAGE_SIZE, retries=5):
    url = (f"{API_BASE}?dataset={dataset}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "taiji-neuron/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("rows", [])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # 429 Too Many Requests: 等待更长时间
                wait = 5 * (attempt + 1)
                print(f"      429, 等待 {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 404:
                return []
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"      请求失败: {e}")
            return []
    return []


def download_domain(domain, cfg, sp):
    dataset = cfg["dataset"]
    config = cfg["config"]
    split = cfg["split"]
    target = cfg["target_samples"]

    print(f"\n  [{domain}] {dataset} (config={config}, split={split}), 目标 {target} 条")

    tokens_list = []
    fetched = 0
    offset = 0
    empty_pages = 0

    while fetched < target:
        rows = fetch_page(dataset, config, split, offset)
        time.sleep(REQUEST_INTERVAL)  # 关键：请求间隔

        if not rows:
            empty_pages += 1
            if empty_pages >= 3:
                print(f"    连续 {empty_pages} 次空页，停止")
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

            encoded = [tid + TEXT_OFFSET for tid in sp.EncodeAsIds(text)]
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
            if fetched >= target:
                break

        offset += PAGE_SIZE
        if fetched % 1000 < PAGE_SIZE:
            print(f"    fetched={fetched}, total={len(tokens_list)}, offset={offset}")

    print(f"    完成: fetched={fetched}, total={len(tokens_list)}")
    return tokens_list


def main():
    print("=" * 70)
    print("补全缺失域数据（en, code）")
    print("=" * 70)

    # 加载 tokenizer
    sp_path = os.path.join(TEACHER_PATH, "sentencepiece.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    print(f"Tokenizer: {sp.GetPieceSize()} tokens")

    # 加载现有数据
    if os.path.exists(DOMAIN_DATA_PATH):
        all_data = torch.load(DOMAIN_DATA_PATH, map_location="cpu", weights_only=False)
        print(f"现有域: {list(all_data.keys())}")
        for k, v in all_data.items():
            print(f"  {k}: {v.shape}")
    else:
        all_data = {}
        print("⚠️ 现有 domain_datasets.pt 不存在，将创建新的")

    # 补全缺失域
    for domain, cfg in MISSING_DOMAINS.items():
        if domain in all_data and all_data[domain].shape[0] >= 5000:
            print(f"\n  [{domain}] 已有 {all_data[domain].shape[0]} 条，跳过")
            continue

        t0 = time.time()
        tokens_list = download_domain(domain, cfg, sp)
        elapsed = time.time() - t0

        if not tokens_list:
            print(f"  ❌ {domain} 无数据")
            continue

        t = torch.tensor(tokens_list, dtype=torch.long)
        all_data[domain] = t
        torch.save(t, f"data/distill/{domain}.pt")
        print(f"  ✓ {domain}: {t.shape}, "
              f"range=[{t.min().item()}, {t.max().item()}], "
              f"耗时 {elapsed:.0f}s")

    # 保存合并数据
    torch.save(all_data, DOMAIN_DATA_PATH)
    print(f"\n{'=' * 70}")
    print(f"✅ 全部完成: {DOMAIN_DATA_PATH}")
    print(f"{'=' * 70}")
    total_tokens = 0
    for d, t in all_data.items():
        n_tokens = t.numel()
        total_tokens += n_tokens
        print(f"  {d:8s}: {t.shape}, {n_tokens:,} tokens")
    print(f"  {'total':8s}: {total_tokens:,} tokens ({total_tokens / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
