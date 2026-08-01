"""将 alpaca-zh SFT 数据转换为训练用 jsonl 格式。

格式: "问：{instruction}\n答：{output}"
- 评估集保留完整对话（用于 PPL 评估）
- 训练集同格式

Output: data/simple_zh/alpaca_zh_sft.jsonl
"""
from datasets import load_dataset
import json
import os

OUTPUT_PATH = os.path.join("data", "simple_zh", "alpaca_zh_sft.jsonl")


def format_dialogue(instruction: str, output: str) -> str:
    """格式化为对话文本。"""
    return f"问：{instruction}\n答：{output}"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("加载 alpaca-zh...", flush=True)
    ds = load_dataset("shibing624/alpaca-zh", cache_dir="data/cache/shibing624___alpaca-zh")
    train = ds['train']
    print(f"原始数据: {len(train)} 条", flush=True)

    # 转换格式
    samples = []
    for item in train:
        instruction = item['instruction'].strip()
        output = item['output'].strip()
        if not instruction or not output:
            continue
        text = format_dialogue(instruction, output)
        samples.append({"text": text})

    print(f"有效样本: {len(samples)} 条", flush=True)

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已保存: {OUTPUT_PATH}", flush=True)

    # 长度统计
    lengths = [len(s['text']) for s in samples]
    print(f"长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}", flush=True)

    # 样例
    print("\n前 3 条样例：", flush=True)
    for i in range(3):
        print(f"\n--- 样例 {i+1} ---", flush=True)
        print(samples[i]['text'][:300], flush=True)


if __name__ == "__main__":
    main()
