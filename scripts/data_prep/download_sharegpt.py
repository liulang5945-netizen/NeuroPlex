"""
下载 ShareGPT 数据集
====================
从 HuggingFace 下载 ShareGPT 数据集
"""
import os
import json
import requests
from pathlib import Path

def download_from_hf_api(dataset_name, output_file, max_samples=50000):
    """使用 HuggingFace API 下载数据集"""
    try:
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset

        print(f"下载 {dataset_name}...")

        # 使用 datasets 库下载
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        print(f"保存到: {output_file}")

        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                # 根据不同数据集格式处理
                if 'conversations' in item:
                    # ShareGPT 格式
                    conversations = item['conversations']
                    messages = [{"role": "system", "content": "你是态极，一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。"}]

                    for conv in conversations:
                        role = 'user' if conv.get('from') in ['human', 'user'] else 'assistant'
                        content = conv.get('value', '')
                        if content:
                            messages.append({"role": role, "content": content})

                    if len(messages) >= 3:  # 至少有一轮对话
                        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
                        count += 1

                elif 'instruction' in item:
                    # Alpaca 格式
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output_text = item.get('output', '')

                    if instruction and output_text:
                        user_msg = instruction
                        if input_text:
                            user_msg = f"{instruction}\n\n{input_text}"

                        messages = [
                            {"role": "system", "content": "你是态极，一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。"},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": output_text}
                        ]
                        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
                        count += 1

                if count % 10000 == 0:
                    print(f"  已处理 {count} 条...")

                if count >= max_samples:
                    break

        print(f"完成! 共 {count} 条")
        return count

    except Exception as e:
        print(f"下载失败: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    print("=" * 60)
    print("下载 ShareGPT 数据集")
    print("=" * 60)

    sft_dir = Path("taiji_data/training_data/sft")
    sft_dir.mkdir(parents=True, exist_ok=True)

    # 尝试下载 ShareGPT 中文版
    datasets_to_try = [
        ("ShareGPT-Chinese", "sharegpt_chinese.jsonl", 50000),
        ("shibing624/sharegpt_zh", "sharegpt_zh.jsonl", 50000),
    ]

    for dataset_name, output_name, max_samples in datasets_to_try:
        output_file = sft_dir / output_name

        if output_file.exists():
            print(f"文件已存在: {output_file}")
            continue

        print(f"\n尝试下载: {dataset_name}")
        count = download_from_hf_api(dataset_name, str(output_file), max_samples)

        if count > 0:
            print(f"成功下载: {count} 条")
            break

    # 统计当前所有SFT数据
    print("\n" + "=" * 60)
    print("当前 SFT 数据统计")
    print("=" * 60)

    total = 0
    for f in sorted(sft_dir.glob("*.jsonl")):
        if f.name.endswith("_clean.jsonl"):
            continue
        with open(f, 'r', encoding='utf-8') as fh:
            count = sum(1 for line in fh if line.strip())
        print(f"  {f.name}: {count} 条")
        total += count

    print(f"\n总计: {total} 条")

if __name__ == '__main__':
    main()
