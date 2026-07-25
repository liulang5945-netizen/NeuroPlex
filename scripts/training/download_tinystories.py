"""下载 TinyStories 数据集并用 GPT-2 BPE 编码。

TinyStories (Microsoft 2023): 简单英文故事，3-4岁儿童词汇。
证明 <10M 参数可生成连贯多段落故事。

输出：
  data/tinystories/train.bin  — uint16 token ids (GPT-2 BPE)
  data/tinystories/val.bin    — uint16 token ids
"""
import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = "data/tinystories"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("下载 TinyStories 数据集")
print("=" * 60)

# GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
print(f"Tokenizer: GPT-2 BPE, vocab={enc.n_vocab}")

# 下载 TinyStories（streaming 模式避免全量下载）
print("\n[1] 下载 TinyStories (streaming 模式)...")
try:
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    print("  数据集连接成功")
except Exception as e:
    print(f"  streaming 失败: {e}")
    print("  尝试非 streaming 模式...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print("  数据集加载成功")

# 收集文本并编码
# CPU 训练验证：100K 条故事足够（约 50M tokens）
N_TRAIN = 80000   # 80K 条训练
N_VAL = 2000      # 2K 条验证
TOTAL = N_TRAIN + N_VAL

print(f"\n[2] 编码 {TOTAL} 条故事 (train={N_TRAIN}, val={N_VAL})...")

train_tokens = []
val_tokens = []
count = 0

for item in ds:
    if count >= TOTAL:
        break
    text = item.get("text", "")
    if len(text) < 50:
        continue
    tokens = enc.encode(text)
    if count < N_TRAIN:
        train_tokens.extend(tokens)
    else:
        val_tokens.extend(tokens)
    count += 1
    if count % 10000 == 0:
        print(f"  已处理 {count}/{TOTAL} 条, train_tokens={len(train_tokens)}, val_tokens={len(val_tokens)}")

print(f"\n[3] 编码完成:")
print(f"  Train: {len(train_tokens)} tokens ({len(train_tokens)/1e6:.1f}M)")
print(f"  Val:   {len(val_tokens)} tokens ({len(val_tokens)/1e6:.1f}M)")

# 保存为二进制文件（uint16，GPT-2 vocab=50257 < 65535）
train_arr = np.array(train_tokens, dtype=np.uint16)
val_arr = np.array(val_tokens, dtype=np.uint16)

train_path = os.path.join(OUT_DIR, "train.bin")
val_path = os.path.join(OUT_DIR, "val.bin")

train_arr.tofile(train_path)
val_arr.tofile(val_path)

print(f"\n[4] 保存完成:")
print(f"  {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")
print(f"  {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

# 统计
print(f"\n[5] 统计:")
print(f"  唯一 token 数: {len(set(train_tokens))}")
print(f"  平均故事长度: {len(train_tokens)/N_TRAIN:.0f} tokens")
print(f"  数据/参数比 (10M模型): {len(train_tokens)/10e6:.1f}")
print(f"  数据/参数比 (36M模型): {len(train_tokens)/36e6:.1f}")

print(f"\n{'='*60}")
print("下载完成！")
print(f"{'='*60}")
