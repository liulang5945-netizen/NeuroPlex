"""
最终数据混合
============
将所有数据按推荐比例混合，构建面向未来的完美地基
"""
import os
import json
import random
from collections import Counter

# 数据源配置
DATA_SOURCES = [
    # === 基础预训练数据 (70%) ===
    {"file": "taiji_data/training_data/pretrain_final.jsonl", "weight": 1.0, "category": "基础预训练"},

    # === 工具调用数据 (10%) ===
    {"file": "taiji_data/training_data/react_converted.jsonl", "weight": 1.0, "category": "工具调用"},

    # === 生命体数据 (5%，高权重) ===
    {"file": "taiji_data/training_data/lifeform/state_awareness.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/temporal_memory.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/proactive_interaction.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/boundary_refusal.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/ambiguity_clarification.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/emotional_empathy.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/memory_forgetting.jsonl", "weight": 10.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/tool_failure_recovery.jsonl", "weight": 10.0, "category": "生命体"},

    # === 毕业级数据 (额外补充) ===
    {"file": "taiji_data/training_data/taiji_graduation_conversation.jsonl", "weight": 0.3, "category": "毕业对话"},
    {"file": "taiji_data/training_data/taiji_graduation_v2_conversation.jsonl", "weight": 0.2, "category": "毕业对话"},
    {"file": "taiji_data/training_data/taiji_graduation_v3_conversation.jsonl", "weight": 0.2, "category": "毕业对话"},
]

OUTPUT_FILE = "taiji_data/training_data/pretrain_final.jsonl"

def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if "messages" in item and len(item["messages"]) >= 2:
                        data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def validate_item(item):
    """验证数据质量"""
    messages = item.get("messages", [])
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False
    for msg in messages:
        if not msg.get("content", "").strip():
            return False
    return True

def main():
    print("=" * 60)
    print("构建面向未来的完美地基")
    print("=" * 60)

    all_data = []
    category_counts = Counter()

    for source in DATA_SOURCES:
        file_path = source["file"]

        if not os.path.exists(file_path):
            print(f"  [跳过] {os.path.basename(file_path)}: 文件不存在")
            continue

        print(f"  [加载] {os.path.basename(file_path)}...", end=" ")
        data = load_jsonl(file_path)
        valid_data = [item for item in data if validate_item(item)]

        # 按权重复制
        weight = source["weight"]
        if weight > 1:
            valid_data = valid_data * int(weight)
        elif weight < 1:
            # 按比例采样
            sample_size = int(len(valid_data) * weight)
            valid_data = random.sample(valid_data, min(sample_size, len(valid_data)))

        category = source["category"]
        category_counts[category] += len(valid_data)
        all_data.extend(valid_data)

        print(f"{len(valid_data):,} 条 (权重: {weight})")

    # 打乱顺序
    print(f"\n  打乱数据顺序...")
    random.shuffle(all_data)

    # 保存
    print(f"  保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    total = len(all_data)
    print()
    print("=" * 60)
    print("最终数据集统计")
    print("=" * 60)
    print(f"  总数据量: {total:,} 条")
    print()
    print("  类别分布:")
    for category, count in category_counts.most_common():
        print(f"    {category}: {count:,} 条 ({count/total*100:.1f}%)")

    # 检查比例
    tool_ratio = category_counts.get("工具调用", 0) / total * 100
    life_ratio = category_counts.get("生命体", 0) / total * 100
    print()
    print("  比例检查:")
    print(f"    工具调用: {tool_ratio:.1f}% (目标: 10%)")
    print(f"    生命体: {life_ratio:.1f}% (目标: 5%)")

    # 语言检测
    chinese_count = 0
    english_count = 0
    for item in all_data[:10000]:
        user_content = next((m["content"] for m in item["messages"] if m.get("role") == "user"), "")
        if any('一' <= c <= '鿿' for c in user_content):
            chinese_count += 1
        else:
            english_count += 1

    total_sampled = chinese_count + english_count
    print()
    print("  语言分布 (抽样):")
    print(f"    中文: {chinese_count/total_sampled*100:.1f}%")
    print(f"    英文: {english_count/total_sampled*100:.1f}%")

    print()
    print("=" * 60)
    print("[DONE] 面向未来的完美地基构建完成！")
    print("=" * 60)
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  总数据量: {total:,} 条")

if __name__ == "__main__":
    main()
