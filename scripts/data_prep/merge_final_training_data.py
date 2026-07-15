"""
合并最终训练数据
================
将所有清洗和扩充后的数据合并为最终训练集
"""
import json
import os
from pathlib import Path
from collections import Counter

def load_jsonl(file_path):
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

def validate_messages(item):
    """验证 messages 格式数据"""
    messages = item.get('messages', [])
    if len(messages) < 2:
        return False

    # 检查是否有 user 和 assistant
    roles = [m.get('role') for m in messages]
    if 'user' not in roles or 'assistant' not in roles:
        return False

    # 检查内容非空
    for msg in messages:
        if not msg.get('content', '').strip():
            return False

    return True

def main():
    print("=" * 60)
    print("合并最终训练数据")
    print("=" * 60)

    sft_dir = Path("taiji_data/training_data/sft")
    lifeform_dir = Path("taiji_data/training_data/lifeform")
    supp_dir = Path("taiji_data/training_data/supplementary")

    all_data = []
    source_stats = Counter()

    # 1. SFT 数据
    print("\n[1] 加载 SFT 数据...")
    sft_files = [
        "taiji_conversation_data.jsonl",
        "taiji_graduation_v3_conversation.jsonl",
        "taiji_graduation_v3_react.jsonl",
        "taiji_complex_react.jsonl",
        "taiji_react_data.jsonl",
        "taiji_developer_story.jsonl",
        "gap_complex_tasks.jsonl",
        "gap_error_recovery.jsonl",
        "gap_memory_usage.jsonl",
        "gap_multimodal.jsonl",
        "gap_safety_refusal.jsonl",
        "final_error_recovery.jsonl",
        "final_language.jsonl",
        "final_memory.jsonl",
        "final_perception.jsonl",
        "final_planning.jsonl",
        "final_safety.jsonl",
        "final_tools.jsonl",
        "openhermes_2_5.jsonl",
        "alpaca_zh.jsonl",
        "alpaca_en.jsonl",
        "dolly_15k.jsonl",
        "code_alpaca_20k.jsonl",
        "coig_kaoyan.jsonl",
        "coig_logi_qa.jsonl",
        "coig_ruozhiba.jsonl",
        "identity_data.jsonl",
        "identity_taiji.jsonl",
        "memory_training_data.jsonl",
        "tool_call_training.jsonl",
    ]

    for filename in sft_files:
        filepath = sft_dir / filename
        if not filepath.exists():
            continue

        data = load_jsonl(str(filepath))
        valid_data = [item for item in data if validate_messages(item)]
        all_data.extend(valid_data)
        source_stats[f"sft/{filename}"] = len(valid_data)
        print(f"  {filename}: {len(valid_data)} 条")

    # 2. 生命体数据
    print("\n[2] 加载生命体数据...")
    lifeform_files = [
        "state_awareness.jsonl",
        "temporal_memory.jsonl",
        "proactive_interaction.jsonl",
        "boundary_refusal.jsonl",
        "ambiguity_clarification.jsonl",
        "emotional_empathy.jsonl",
        "memory_forgetting.jsonl",
        "tool_failure_recovery.jsonl",
        "lifeform_expanded.jsonl",
    ]

    for filename in lifeform_files:
        filepath = lifeform_dir / filename
        if not filepath.exists():
            continue

        data = load_jsonl(str(filepath))
        valid_data = [item for item in data if validate_messages(item)]
        all_data.extend(valid_data)
        source_stats[f"lifeform/{filename}"] = len(valid_data)
        print(f"  {filename}: {len(valid_data)} 条")

    # 3. 补充数据
    print("\n[3] 加载补充数据...")
    supp_files = [
        "safety_alignment.jsonl",
        "math_reasoning.jsonl",
        "long_context.jsonl",
        "science_knowledge.jsonl",
    ]

    for filename in supp_files:
        filepath = supp_dir / filename
        if not filepath.exists():
            continue

        data = load_jsonl(str(filepath))
        valid_data = [item for item in data if validate_messages(item)]
        all_data.extend(valid_data)
        source_stats[f"supplementary/{filename}"] = len(valid_data)
        print(f"  {filename}: {len(valid_data)} 条")

    # 去重
    print("\n[4] 去重...")
    seen = set()
    unique_data = []
    duplicates = 0

    for item in all_data:
        # 使用问题的前100字符作为去重键
        messages = item.get('messages', [])
        key = ''
        for msg in messages:
            if msg.get('role') == 'user':
                key = msg['content'][:100]
                break

        if key not in seen:
            seen.add(key)
            unique_data.append(item)
        else:
            duplicates += 1

    print(f"  去重前: {len(all_data)} 条")
    print(f"  去重后: {len(unique_data)} 条")
    print(f"  移除重复: {duplicates} 条")

    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(unique_data)

    # 保存
    output_file = Path("taiji_data/training_data/pretrain_final.jsonl")
    print(f"\n[5] 保存到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  保存完成: {len(unique_data)} 条")

    # 统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)

    print("\n数据来源:")
    for source, count in sorted(source_stats.items()):
        if count > 0:
            print(f"  {source}: {count} 条")

    print(f"\n总计: {len(unique_data)} 条")

    # 抽样检查
    print("\n" + "=" * 60)
    print("抽样检查")
    print("=" * 60)

    samples = random.sample(unique_data, min(5, len(unique_data)))
    for i, sample in enumerate(samples):
        msgs = sample.get('messages', [])
        print(f"\n样本 {i+1}:")
        print(f"  Q: {msgs[1]['content'][:100]}")
        print(f"  A: {msgs[2]['content'][:100]}")

    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)
    print(f"\n最终训练集: {output_file}")
    print(f"数据量: {len(unique_data)} 条")
    print("\n下一步:")
    print("1. 上传到 Autodl 平台")
    print("2. 运行预训练:")
    print("   python taiji/train/finetune_taiji.py")

if __name__ == '__main__':
    main()
