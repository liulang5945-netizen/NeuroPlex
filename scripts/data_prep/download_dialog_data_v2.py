"""
下载高质量对话数据集 (使用 datasets 库)
========================================
使用 HuggingFace datasets 库下载数据集
"""
import os
import json
from pathlib import Path

def download_belle_data():
    """下载 BELLE 数据集"""
    try:
        from datasets import load_dataset

        print("=" * 60)
        print("下载 BELLE-0.5M 中文指令数据")
        print("=" * 60)

        # 下载数据集
        print("\n正在下载...")
        dataset = load_dataset("BelleGroup/train_0.5M_CN", split="train")

        print(f"下载完成! 共 {len(dataset)} 条数据")

        # 转换为 messages 格式
        output_dir = Path("taiji_data/training_data/sft")
        output_file = output_dir / "belle_0.5m_cn.jsonl"

        print(f"\n转换格式并保存到: {output_file}")

        converted = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')

                if not instruction or not output_text:
                    continue

                # 构建用户消息
                user_msg = instruction
                if input_text:
                    user_msg = f"{instruction}\n\n{input_text}"

                # 构建messages格式
                messages = [
                    {"role": "system", "content": "你是态极，一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。"},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output_text}
                ]

                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
                converted += 1

        print(f"转换完成: {converted} 条")
        return str(output_file)

    except Exception as e:
        print(f"下载失败: {e}")
        return None

def download_firefly_data():
    """下载 Firefly 数据集"""
    try:
        from datasets import load_dataset

        print("\n" + "=" * 60)
        print("下载 Firefly 中文对话数据")
        print("=" * 60)

        # 下载数据集
        print("\n正在下载...")
        dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train")

        print(f"下载完成! 共 {len(dataset)} 条数据")

        # 转换为 messages 格式
        output_dir = Path("taiji_data/training_data/sft")
        output_file = output_dir / "firefly_1.1m.jsonl"

        print(f"\n转换格式并保存到: {output_file}")

        converted = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Firefly 格式: kind, input, target
                kind = item.get('kind', '')
                input_text = item.get('input', '')
                target_text = item.get('target', '')

                if not target_text:
                    continue

                # 构建用户消息
                user_msg = input_text if input_text else kind

                # 构建messages格式
                messages = [
                    {"role": "system", "content": "你是态极，一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。"},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": target_text}
                ]

                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
                converted += 1

        print(f"转换完成: {converted} 条")
        return str(output_file)

    except Exception as e:
        print(f"下载失败: {e}")
        return None

def sample_and_check(file_path, num_samples=5):
    """抽样检查数据质量"""
    print(f"\n抽样检查: {file_path}")
    print("-" * 40)

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            msgs = item['messages']
            print(f"\n样本 {i+1}:")
            print(f"  Q: {msgs[1]['content'][:80]}...")
            print(f"  A: {msgs[2]['content'][:80]}...")

def main():
    print("=" * 60)
    print("下载高质量对话数据集")
    print("=" * 60)

    downloaded_files = []

    # 下载 BELLE 数据
    belle_file = download_belle_data()
    if belle_file:
        downloaded_files.append(belle_file)
        sample_and_check(belle_file)

    # 下载 Firefly 数据
    firefly_file = download_firefly_data()
    if firefly_file:
        downloaded_files.append(firefly_file)
        sample_and_check(firefly_file)

    # 总结
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)

    if downloaded_files:
        print("\n下载的文件:")
        for f in downloaded_files:
            with open(f, 'r', encoding='utf-8') as fh:
                line_count = sum(1 for _ in fh)
            print(f"  {os.path.basename(f)}: {line_count} 条")

        print("\n下一步:")
        print("1. 检查数据质量")
        print("2. 合并所有数据进行训练")

if __name__ == '__main__':
    main()
