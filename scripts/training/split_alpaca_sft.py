"""将 alpaca-zh SFT 数据分割成 5 份，供 5 个神经元差异化训练。

每个神经元：30% 共享核心 + 70% 独有数据
输出：
  - data/simple_zh/sft_shared_core.jsonl  (30% 共享，~14600 条)
  - data/simple_zh/sft_unique_0.jsonl ~ sft_unique_4.jsonl  (70% 独有，每份 ~6800 条)
"""
import json
import os
import random

INPUT_PATH = "data/simple_zh/alpaca_zh_sft.jsonl"
OUTPUT_DIR = "data/simple_zh"
NUM_SPLITS = 5
SHARED_RATIO = 0.3
SEED = 42


def main():
    random.seed(SEED)

    # 加载全部数据
    all_items = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if len(text) >= 20:
                all_items.append(item)
    print(f"总数据: {len(all_items)} 条", flush=True)

    # 打乱
    random.shuffle(all_items)

    # 分割：30% 共享 + 70% 独有
    n_shared = int(len(all_items) * SHARED_RATIO)
    shared_core = all_items[:n_shared]
    unique_pool = all_items[n_shared:]

    # 独有数据分成 5 份
    n_unique = len(unique_pool)
    split_size = n_unique // NUM_SPLITS
    splits = []
    for i in range(NUM_SPLITS):
        start = i * split_size
        end = (i + 1) * split_size if i < NUM_SPLITS - 1 else n_unique
        splits.append(unique_pool[start:end])

    # 保存共享核心
    shared_path = os.path.join(OUTPUT_DIR, "sft_shared_core.jsonl")
    with open(shared_path, "w", encoding="utf-8") as f:
        for item in shared_core:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"共享核心: {len(shared_core)} 条 -> {shared_path}", flush=True)

    # 保存独有数据
    for i, split in enumerate(splits):
        path = os.path.join(OUTPUT_DIR, f"sft_unique_{i}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"独有数据 {i}: {len(split)} 条 -> {path}", flush=True)

    # 验证：每个神经元的总数据量
    print("\n每个神经元训练数据量:", flush=True)
    for i in range(NUM_SPLITS):
        total = len(shared_core) + len(splits[i])
        print(f"  神经元 {i}: {len(shared_core)} (shared) + {len(splits[i])} (unique) = {total} 条", flush=True)


if __name__ == "__main__":
    main()
