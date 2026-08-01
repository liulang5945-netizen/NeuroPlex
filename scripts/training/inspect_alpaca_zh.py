"""检查 alpaca-zh SFT 数据格式。"""
from datasets import load_dataset
import os

cache_path = "data/cache/shibing624___alpaca-zh"
ds = load_dataset("shibing624/alpaca-zh", cache_dir=cache_path)
print(f"splits: {list(ds.keys())}")
train = ds['train']
print(f"数据集大小: {len(train)}")
print(f"字段: {train.column_names}")
print("\n样例（前 3 条）：")
for i in range(3):
    item = train[i]
    print(f"\n--- 样例 {i+1} ---")
    print(f"instruction: {item['instruction']}")
    print(f"input: {item['input']}")
    print(f"output: {item['output']}")

# 统计长度分布
lengths = [len(item['instruction']) + len(item['input']) + len(item['output']) for item in train]
print(f"\n长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
