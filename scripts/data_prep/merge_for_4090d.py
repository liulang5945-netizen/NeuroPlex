"""
合并补充数据到预训练数据集
==========================
将下载的补充数据合并为统一的预训练文件,供4090D继续预训练使用。
"""
import os
import json
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "taiji_data" / "training_data"
SUPPLEMENT_DIR = DATA_DIR / "supplementary"
OUTPUT_FILE = DATA_DIR / "pretrain_for_4090d.jsonl"


def load_jsonl(filepath):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "messages" in item and len(item["messages"]) >= 2:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def main():
    all_data = []
    category_counts = Counter()

    # 1. 加载原始预训练数据
    original_file = DATA_DIR / "pretrain_final.jsonl"
    if original_file.exists():
        print(f"[加载] pretrain_final.jsonl ...", end=" ")
        data = load_jsonl(original_file)
        for item in data:
            item["_category"] = "original"
        all_data.extend(data)
        category_counts["原始预训练"] = len(data)
        print(f"{len(data)} 条")

    # 2. 加载补充数据
    supplement_files = {
        "math_reasoning.jsonl": "数学推理",
        "science_knowledge.jsonl": "科学知识",
        "safety_alignment.jsonl": "安全对齐",
        "long_context.jsonl": "长文档理解",
        "chinese_pretrain.jsonl": "中文预训练",
        "lifeform_expanded.jsonl": "生命体",
    }

    for filename, category in supplement_files.items():
        filepath = SUPPLEMENT_DIR / filename
        if filepath.exists():
            print(f"[加载] {filename} ...", end=" ")
            data = load_jsonl(filepath)
            for item in data:
                item["_category"] = category
            all_data.extend(data)
            category_counts[category] = len(data)
            print(f"{len(data)} 条")

    # 3. 加载原有专项数据 (生命体、工具调用等)
    extra_files = [
        ("lifeform/state_awareness.jsonl", "生命体"),
        ("lifeform/temporal_memory.jsonl", "生命体"),
        ("lifeform/proactive_interaction.jsonl", "生命体"),
        ("lifeform/boundary_refusal.jsonl", "生命体"),
        ("lifeform/ambiguity_clarification.jsonl", "生命体"),
        ("lifeform/emotional_empathy.jsonl", "生命体"),
        ("lifeform/memory_forgetting.jsonl", "生命体"),
        ("lifeform/tool_failure_recovery.jsonl", "生命体"),
        ("taiji_react_data.jsonl", "工具调用"),
        ("taiji_complex_react.jsonl", "工具调用"),
        ("gap_error_recovery.jsonl", "错误恢复"),
        ("gap_memory_usage.jsonl", "记忆使用"),
        ("gap_complex_tasks.jsonl", "复杂任务"),
        ("gap_safety_refusal.jsonl", "安全拒绝"),
    ]

    for rel_path, category in extra_files:
        filepath = DATA_DIR / rel_path
        if filepath.exists():
            print(f"[加载] {rel_path} ...", end=" ")
            data = load_jsonl(filepath)
            for item in data:
                item["_category"] = category
            all_data.extend(data)
            category_counts[category] += len(data)
            print(f"{len(data)} 条")

    # 4. 打乱顺序
    random.seed(42)
    random.shuffle(all_data)

    # 5. 保存 (移除 _category 标记)
    print(f"\n[保存] 合并数据 -> {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_data:
            item.pop("_category", None)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. 统计
    print("\n" + "=" * 60)
    print("数据集统计:")
    print("=" * 60)
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count:,} 条")
    print(f"  {'总计':}: {len(all_data):,} 条")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
