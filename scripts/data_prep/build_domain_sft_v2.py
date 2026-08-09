"""C24 域 SFT 数据扩充 v2（2026-08-10）。

背景：C24 域生成碎片根因已闭环——每域仅 3000 条短 QA，数据不足（diag_c24_domain_generate
单独域头验证也碎片）。本脚本利用本地 HF cache + 本地语料重建每域 2-3 万条 SFT 数据：

    code: code_alpaca-20k 全量 20022 条（原 3000 条正是其子集）
    math: gsm8k main train+test 8792 条 QA + math_texts 22904 行续写样本 → 30000
    zh:   alpaca-zh 采样 30000
    en:   alpaca 采样 30000

输出格式与 train_domain_target_sft.py 完全兼容：
    data/sft/{domain}_sft.pt = List[dict{instruction, input, response, prompt, full}]
    prompt = instruction + ("\n" + input if input else "")
    full   = prompt + "\n" + response

Usage:
    python -u scripts/data_prep/build_domain_sft_v2.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import pyarrow.ipc as ipc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFT_DIR = os.path.join(PROJECT_ROOT, "data", "sft")
os.makedirs(SFT_DIR, exist_ok=True)

CACHE_ROOT = os.path.join(PROJECT_ROOT, "data", "cache")
CODE_ALPACA_ARROW = os.path.join(
    CACHE_ROOT, "sahil2801___code_alpaca-20k", "default", "0.0.0",
    "152bb5e9a29651266b018106053980070a0521a1", "code_alpaca-20k-train.arrow")
MATH_TEXTS = os.path.join(PROJECT_ROOT, "data", "distill", "math_texts.jsonl")

MAX_FULL_CHARS = 512  # 超长样本跳过（SEQ_LEN=192 token 装不下，截断会丢 answer 尾部）
SEED = 42


def to_sample(instruction: str, input_: str, response: str):
    """统一转为 {instruction, input, response, prompt, full} 格式。"""
    instruction = (instruction or "").strip()
    input_ = (input_ or "").strip()
    response = (response or "").strip()
    if not instruction or len(response) < 2:
        return None
    prompt = instruction if not input_ else instruction + "\n" + input_
    full = prompt + "\n" + response
    if len(full) > MAX_FULL_CHARS:
        return None
    return {"instruction": instruction, "input": input_, "response": response,
            "prompt": prompt, "full": full}


def load_code_alpaca() -> list:
    """code_alpaca-20k 本地 arrow 直读（load_dataset 名称不命中缓存）。"""
    with open(CODE_ALPACA_ARROW, "rb") as f:
        table = ipc.open_stream(f).read_all()
    out = []
    for row in table.to_pylist():
        s = to_sample(row.get("instruction", ""), row.get("input", ""), row.get("output", ""))
        if s:
            out.append(s)
    print(f"  [code] code_alpaca-20k {len(out)} 条", flush=True)
    return out


def load_math() -> list:
    """gsm8k main train+test 完整 QA + math_texts 行级续写样本 → 30000。"""
    import os as _os
    _os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets
    qa = []
    for split in ["train", "test"]:
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            s = to_sample(row["question"], "", row["answer"])
            if s:
                qa.append(s)
    print(f"  [math] gsm8k train+test {len(qa)} 条 QA", flush=True)

    # math_texts.jsonl = gsm8k 逐行摊开的文本行（问题/答案/####）。每行做续写样本：
    # prompt = 行前 40% 字符，response = 后 60%（教数学文本续写流利度，补齐 QA 量）。
    cont = []
    with open(MATH_TEXTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < 16:
                continue
            cut = max(4, int(len(line) * 0.4))
            s = to_sample(line[:cut].rstrip(), "", line[cut:].lstrip())
            if s:
                cont.append(s)
    print(f"  [math] math_texts 续写样本 {len(cont)} 条", flush=True)

    need = 30000 - len(qa)
    if need > 0:
        rng = random.Random(SEED)
        picked = rng.sample(cont, min(need, len(cont)))
        out = qa + picked
    else:
        out = qa[:30000]
    return out


def load_zh() -> list:
    import os as _os
    _os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets
    ds = datasets.load_dataset("shibing624/alpaca-zh", split="train")
    out = []
    for row in ds:
        s = to_sample(row["instruction"], row.get("input", ""), row.get("output", ""))
        if s:
            out.append(s)
    print(f"  [zh] alpaca-zh {len(out)} 条", flush=True)
    return out


def load_en() -> list:
    import os as _os
    _os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets
    ds = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for row in ds:
        s = to_sample(row["instruction"], row.get("input", ""), row.get("output", ""))
        if s:
            out.append(s)
    print(f"  [en] alpaca {len(out)} 条", flush=True)
    return out


def sample_to(samples: list, target: int, label: str) -> list:
    rng = random.Random(SEED)
    if len(samples) > target:
        samples = rng.sample(samples, target)
    print(f"  {label} 采样后 {len(samples)} 条（目标 {target}）", flush=True)
    return samples


def main():
    random.seed(SEED)
    print("=" * 60)
    print("C24 域 SFT 数据扩充 v2（本地缓存组合）")
    print("=" * 60, flush=True)

    builders = {
        "code": (load_code_alpaca, 20022),
        "math": (load_math, 30000),
        "zh": (load_zh, 30000),
        "en": (load_en, 30000),
    }
    for domain, (loader, target) in builders.items():
        print(f"\n[{domain}] 加载数据源...", flush=True)
        samples = loader()
        samples = sample_to(samples, target, domain)
        lens = [len(s["full"]) for s in samples]
        print(f"  [{domain}] avg_len={sum(lens)//len(lens)} max_len={max(lens)}", flush=True)
        path = os.path.join(SFT_DIR, f"{domain}_sft.pt")
        import torch
        torch.save(samples, path)
        print(f"  [{domain}] 已保存 {path}（{len(samples)} 条）", flush=True)

    # 验证回读
    print("\n" + "=" * 60)
    print("回读验证：")
    import torch
    for domain in builders:
        data = torch.load(os.path.join(SFT_DIR, f"{domain}_sft.pt"),
                          map_location="cpu", weights_only=False)
        keys = set(data[0].keys())
        assert keys == {"instruction", "input", "response", "prompt", "full"}, keys
        ok = all(s["full"].startswith(s["prompt"] + "\n") for s in data[:50])
        print(f"  {domain}: {len(data)} 条, 字段 OK, 前50条 prompt 前缀匹配 {ok}", flush=True)
    print("\n✅ 数据构建完成", flush=True)


if __name__ == "__main__":
    main()
