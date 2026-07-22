"""补全 general 域数据：从多个通用问答数据集采样，避免单一数据集限流。

数据源（按顺序尝试，直到达到目标）：
1. databricks/databricks-dolly-15k（已有 1945 条，继续从此取）
2. tatsu-lab/alpaca（英文通用指令，作为 general 补充）
3. shibing624/alpaca-zh（中文通用指令，作为 general 补充）

输出：合并到 data/distill/domain_datasets.pt，更新 data/distill/general.pt
"""
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
REQUEST_INTERVAL = 0.5  # 慢一点避免 429
TARGET_SAMPLES = 8000  # general 域目标 8K

DOMAIN_DATA_PATH = "data/distill/domain_datasets.pt"
TEACHER_PATH = "E:/taiji-neuron/checkpoint-481000"

# 多数据源（通用问答/指令）
SOURCES = [
    {
        "dataset": "databricks/databricks-dolly-15k",
        "config": "default",
        "split": "train",
        "text_field": "instruction",
        "input_field": "context",
        "answer_field": "response",
        "max_samples": 6000,
    },
    {
        "dataset": "tatsu-lab/alpaca",
        "config": "default",
        "split": "train",
        "text_field": "instruction",
        "input_field": "input",
        "answer_field": "output",
        "max_samples": 3000,  # 补充英文通用
    },
]


def extract_text(sample, cfg):
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
                wait = 8 * (attempt + 1)
                print(f"      429, 等待 {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 404:
                return []
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"      请求失败: {e}")
            return []
    return []


def download_domain(sp):
    """下载 general 域数据（多源合并）。"""
    tokens_list = []

    for src_idx, cfg in enumerate(SOURCES):
        if len(tokens_list) >= TARGET_SAMPLES:
            break

        dataset = cfg["dataset"]
        config = cfg["config"]
        split = cfg["split"]
        max_samples = cfg.get("max_samples", TARGET_SAMPLES)

        print(f"\n  源 {src_idx+1}/{len(SOURCES)}: {dataset} (split={split})")

        fetched = 0
        offset = 0
        empty_pages = 0

        while fetched < max_samples and len(tokens_list) < TARGET_SAMPLES:
            rows = fetch_page(dataset, config, split, offset)
            time.sleep(REQUEST_INTERVAL)

            if not rows:
                empty_pages += 1
                if empty_pages >= 3:
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
                if fetched >= max_samples or len(tokens_list) >= TARGET_SAMPLES:
                    break

            offset += PAGE_SIZE
            if fetched % 1000 < PAGE_SIZE:
                print(f"    fetched={fetched}, total={len(tokens_list)}, offset={offset}")

        print(f"    完成: fetched={fetched}, total={len(tokens_list)}")

    return tokens_list


def main():
    print("=" * 70)
    print(f"补全 general 域数据（目标 {TARGET_SAMPLES} 条）")
    print("=" * 70)

    # 加载 tokenizer
    sp_path = os.path.join(TEACHER_PATH, "sentencepiece.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    print(f"Tokenizer: {sp.GetPieceSize()} tokens")

    # 加载现有数据
    all_data = torch.load(DOMAIN_DATA_PATH, map_location="cpu", weights_only=False)
    print(f"\n现有 general: {all_data['general'].shape}")

    # 下载新数据
    t0 = time.time()
    tokens_list = download_domain(sp)
    elapsed = time.time() - t0

    if not tokens_list:
        print(f"❌ 无数据下载")
        return

    t = torch.tensor(tokens_list, dtype=torch.long)
    all_data["general"] = t
    torch.save(t, "data/distill/general.pt")
    torch.save(all_data, DOMAIN_DATA_PATH)

    print(f"\n{'=' * 70}")
    print(f"✅ 完成: general {t.shape}, 耗时 {elapsed:.0f}s")
    print(f"{'=' * 70}")
    print(f"\n更新后各域:")
    total_tokens = 0
    for d, v in all_data.items():
        n_tokens = v.numel()
        total_tokens += n_tokens
        print(f"  {d:8s}: {v.shape}, {n_tokens:,} tokens")
    print(f"  {'total':8s}: {total_tokens:,} tokens ({total_tokens / 1e6:.1f}M tokens)")

    print(f"\n下一步: 重新蒸馏 general 神经元")
    print(f"  python scripts/training/distill_neurons.py --checkpoint {TEACHER_PATH} "
          f"--data_dir data/distill --output_dir data/neurons "
          f"--steps 1000 --skip_field_cond --device cpu "
          f"--domains general")


if __name__ == "__main__":
    main()
