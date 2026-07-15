"""
合并最终预训练数据集
====================
将所有高质量数据合并为面向未来的地基
"""
import os
import json
import random
from collections import Counter

# 数据源配置：文件路径 + 权重 + 类别
DATA_SOURCES = [
    # === 原始清洗数据 ===
    {"file": "taiji_data/training_data/pretrain_cleaned.jsonl", "weight": 1.0, "category": "基础"},

    # === 中文数据 ===
    {"file": "taiji_data/training_data/alpaca_zh_51k.jsonl", "weight": 2.0, "category": "中文"},
    {"file": "taiji_data/training_data/alpaca_zh.jsonl", "weight": 2.0, "category": "中文"},
    {"file": "taiji_data/training_data/alpaca中文版.jsonl", "weight": 2.0, "category": "中文"},

    # === 英文指令数据 ===
    {"file": "taiji_data/training_data/alpaca_en.jsonl", "weight": 1.5, "category": "英文"},
    {"file": "taiji_data/training_data/dolly_15k.jsonl", "weight": 1.5, "category": "知识"},

    # === 代码数据 ===
    {"file": "taiji_data/training_data/code_alpaca.jsonl", "weight": 2.0, "category": "代码"},
    {"file": "taiji_data/training_data/code_alpaca_20k.jsonl", "weight": 2.0, "category": "代码"},

    # === 工具调用数据 ===
    {"file": "taiji_data/training_data/react_data.jsonl", "weight": 2.0, "category": "工具"},
    {"file": "taiji_data/training_data/taiji_react_data.jsonl", "weight": 2.0, "category": "工具"},

    # === 对话数据 ===
    {"file": "taiji_data/training_data/taiji_conversation_data.jsonl", "weight": 1.5, "category": "对话"},
    {"file": "taiji_data/training_data/conversation_data.jsonl", "weight": 1.0, "category": "对话"},

    # === 生命体数据 ===
    {"file": "taiji_data/training_data/lifeform/state_awareness.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/temporal_memory.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/proactive_interaction.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/boundary_refusal.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/ambiguity_clarification.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/emotional_empathy.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/memory_forgetting.jsonl", "weight": 3.0, "category": "生命体"},
    {"file": "taiji_data/training_data/lifeform/tool_failure_recovery.jsonl", "weight": 3.0, "category": "生命体"},

    # === 专项能力数据 ===
    {"file": "taiji_data/training_data/final_error_recovery.jsonl", "weight": 2.0, "category": "专项"},
    {"file": "taiji_data/training_data/final_language.jsonl", "weight": 1.5, "category": "专项"},
    {"file": "taiji_data/training_data/final_memory.jsonl", "weight": 2.0, "category": "专项"},
    {"file": "taiji_data/training_data/final_perception.jsonl", "weight": 1.5, "category": "专项"},
    {"file": "taiji_data/training_data/final_planning.jsonl", "weight": 2.0, "category": "专项"},
    {"file": "taiji_data/training_data/final_safety.jsonl", "weight": 2.0, "category": "专项"},
    {"file": "taiji_data/training_data/final_tools.jsonl", "weight": 2.0, "category": "专项"},

    # === 高质量数据 ===
    {"file": "taiji_data/training_data/hq_conversation.jsonl", "weight": 3.0, "category": "高质量"},
    {"file": "taiji_data/training_data/hq_error_recovery.jsonl", "weight": 3.0, "category": "高质量"},
    {"file": "taiji_data/training_data/hq_memory.jsonl", "weight": 3.0, "category": "高质量"},
    {"file": "taiji_data/training_data/hq_react.jsonl", "weight": 3.0, "category": "高质量"},
    {"file": "taiji_data/training_data/hq_safety.jsonl", "weight": 3.0, "category": "高质量"},

    # === 补强数据 ===
    {"file": "taiji_data/training_data/gap_complex_tasks.jsonl", "weight": 2.0, "category": "补强"},
    {"file": "taiji_data/training_data/gap_error_recovery.jsonl", "weight": 2.0, "category": "补强"},
    {"file": "taiji_data/training_data/gap_memory_usage.jsonl", "weight": 2.0, "category": "补强"},
    {"file": "taiji_data/training_data/gap_multimodal.jsonl", "weight": 1.5, "category": "补强"},
    {"file": "taiji_data/training_data/gap_safety_refusal.jsonl", "weight": 2.0, "category": "补强"},
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
                    # 验证格式
                    if "messages" in item and len(item["messages"]) >= 2:
                        data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def validate_item(item):
    """验证单条数据质量"""
    messages = item.get("messages", [])

    # 检查是否有用户和助手消息
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False

    # 检查内容是否为空
    for msg in messages:
        if not msg.get("content", "").strip():
            return False

    # 检查是否是回声模式
    user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
    assistant_content = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    if user_content == assistant_content:
        return False

    return True

def main():
    print("=" * 60)
    print("合并最终预训练数据集")
    print("=" * 60)

    all_data = []
    category_counts = Counter()
    file_stats = []

    for source in DATA_SOURCES:
        file_path = source["file"]

        if not os.path.exists(file_path):
            print(f"  [跳过] {os.path.basename(file_path)}: 文件不存在")
            continue

        print(f"  [加载] {os.path.basename(file_path)}...", end=" ")
        data = load_jsonl(file_path)

        # 验证质量
        valid_data = [item for item in data if validate_item(item)]

        # 按权重复制
        weight = source["weight"]
        if weight > 1:
            repeat_times = int(weight)
            valid_data = valid_data * repeat_times

        category = source["category"]
        category_counts[category] += len(valid_data)
        all_data.extend(valid_data)

        print(f"{len(valid_data):,} 条 (权重: {weight})")

        file_stats.append({
            "file": os.path.basename(file_path),
            "original": len(data),
            "valid": len(valid_data),
            "weight": weight,
            "category": category,
        })

    # 打乱顺序
    print(f"\n  打乱数据顺序...")
    random.shuffle(all_data)

    # 保存
    print(f"  保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    print()
    print("=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"  总数据量: {len(all_data):,} 条")
    print()
    print("  类别分布:")
    for category, count in category_counts.most_common():
        print(f"    {category}: {count:,} 条 ({count/len(all_data)*100:.1f}%)")

    # 语言检测
    chinese_count = 0
    english_count = 0
    for item in all_data[:10000]:  # 抽样检测
        user_content = next((m["content"] for m in item["messages"] if m.get("role") == "user"), "")
        if any('一' <= c <= '鿿' for c in user_content):
            chinese_count += 1
        else:
            english_count += 1

    total_sampled = chinese_count + english_count
    print()
    print("  语言分布 (抽样):")
    print(f"    中文: {chinese_count:,} ({chinese_count/total_sampled*100:.1f}%)")
    print(f"    英文: {english_count:,} ({english_count/total_sampled*100:.1f}%)")

    # 保存统计报告
    report = {
        "total_samples": len(all_data),
        "category_distribution": dict(category_counts),
        "language_distribution": {
            "chinese_ratio": chinese_count / total_sampled,
            "english_ratio": english_count / total_sampled,
        },
        "file_stats": file_stats,
    }

    report_file = "taiji_data/training_data/final_dataset_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"  统计报告: {report_file}")
    print()
    print("=" * 60)
    print("[DONE] 面向未来的地基构建完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
