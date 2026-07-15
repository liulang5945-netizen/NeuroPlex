"""
合并清洗后的 SFT 数据，为微调准备最终数据集
==========================================
将清洗后的对话数据合并为单一文件，供后续 SFT 微调使用。

用法:
  python scripts/data_prep/merge_cleaned_sft.py
"""
import json
import os
from pathlib import Path
from collections import Counter


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def merge_cleaned_sft():
    """合并所有清洗后的SFT数据"""
    sft_dir = Path('taiji_data/training_data/sft')

    # 查找所有清洗后的文件
    clean_files = list(sft_dir.glob('*_clean.jsonl'))

    if not clean_files:
        print("未找到清洗后的文件！请先运行 clean_sft_for_pretrain.py")
        return

    print("=" * 60)
    print("合并清洗后的SFT数据")
    print("=" * 60)

    all_data = []
    source_stats = Counter()

    for file_path in sorted(clean_files):
        print(f"\n[加载] {file_path.name}")
        data = load_jsonl(str(file_path))
        print(f"  -> {len(data)} 条")
        all_data.extend(data)
        source_stats[file_path.stem] = len(data)

    # 去重 (基于问题内容)
    seen_questions = set()
    unique_data = []
    duplicates = 0

    for item in all_data:
        messages = item.get('messages', [])
        # 提取问题作为去重键
        question = ''
        for msg in messages:
            if msg.get('role') == 'user':
                question = msg['content'][:100]  # 取前100字符作为键
                break

        if question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)
        else:
            duplicates += 1

    print(f"\n去重: 移除 {duplicates} 条重复数据")

    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(unique_data)

    # 写入合并文件
    output_file = sft_dir.parent / 'sft_merged_clean.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n合并完成!")
    print(f"输出文件: {output_file}")
    print(f"总计: {len(unique_data)} 条高质量对话数据")

    print("\n数据来源统计:")
    for source, count in sorted(source_stats.items()):
        print(f"  {source}: {count} 条")

    print("\n" + "=" * 60)
    print("下一步: 使用清洗后的 SFT 数据做微调")
    print("=" * 60)
    print("""
# 当前仓库中，清洗后的 SFT 数据用于微调而不是预训练
python taiji/train/finetune_taiji.py
""")


if __name__ == '__main__':
    merge_cleaned_sft()
