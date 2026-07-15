"""
构建完美预训练数据集
====================
1. 审计现有数据
2. 识别数据缺口
3. 下载补充数据
4. 验证质量
5. 输出最终报告
"""
import os
import json
import requests
import time
from collections import Counter

# ============ 配置 ============

EXISTING_DATA = "taiji_data/training_data/pretrain_all_v2.jsonl"
OUTPUT_DATA = "taiji_data/training_data/pretrain_final.jsonl"
QUALITY_REPORT = "taiji_data/training_data/data_quality_report.json"

# 目标：每个类别的最低样本数
TARGETS = {
    "中文对话": 200000,
    "英文对话": 100000,
    "代码编程": 100000,
    "推理数学": 50000,
    "工具调用": 50000,
    "知识问答": 100000,
}

# ============ 数据源 ============

DATASETS = [
    # 中文数据
    {
        "name": "Alpaca中文版",
        "url": "https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/data/alpaca_data_zh_51k.json",
        "format": "alpaca",
        "category": "中文对话",
        "priority": 1,
    },
    {
        "name": "BELLE 1.5M",
        "url": "https://huggingface.co/datasets/BelleGroup/train_1.5M_CN/resolve/main/Belle_open_source_1.5M.json",
        "format": "belle",
        "category": "中文对话",
        "priority": 1,
        "max_samples": 300000,
    },
    # 代码数据
    {
        "name": "Code Alpaca 20K",
        "url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
        "format": "alpaca",
        "category": "代码编程",
        "priority": 1,
    },
    {
        "name": "Python Instructions 18K",
        "url": "https://huggingface.co/datasets/iamtarun/python_code_instructions_18k/resolve/main/python_code_instructions_18k.json",
        "format": "alpaca",
        "category": "代码编程",
        "priority": 2,
    },
    {
        "name": "Code Instruct 75K",
        "url": "https://huggingface.co/datasets/nickrosh/Code-Instruct/resolve/main/code_instruct_75k.json",
        "format": "alpaca",
        "category": "代码编程",
        "priority": 2,
    },
    # 英文对话
    {
        "name": "Alpaca 英文",
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "format": "alpaca",
        "category": "英文对话",
        "priority": 1,
    },
    {
        "name": "OpenAssistant",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/refs%2Fpr%2F9/oasst_ready.jsonl",
        "format": "oasst",
        "category": "英文对话",
        "priority": 2,
        "max_samples": 50000,
    },
    # 知识问答
    {
        "name": "Dolly 15K",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        "format": "dolly",
        "category": "知识问答",
        "priority": 1,
    },
]


def analyze_existing_data(file_path):
    """分析现有数据"""
    print("=" * 60)
    print("第一步：分析现有数据")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"  文件不存在: {file_path}")
        return {}

    stats = {
        "total": 0,
        "valid": 0,
        "categories": Counter(),
        "languages": Counter(),
        "avg_turns": 0,
        "total_turns": 0,
    }

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1

            try:
                item = json.loads(line)
                messages = item.get("messages", [])

                if not messages:
                    continue

                stats["valid"] += 1
                stats["total_turns"] += len(messages)

                # 检测语言
                user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
                if any('一' <= c <= '鿿' for c in user_msg):
                    stats["languages"]["中文"] += 1
                else:
                    stats["languages"]["英文"] += 1

                # 检测类别
                content = " ".join(m.get("content", "") for m in messages)
                if any(kw in content.lower() for kw in ["代码", "code", "python", "javascript", "编程"]):
                    stats["categories"]["代码编程"] += 1
                elif any(kw in content.lower() for kw in ["计算", "数学", "math", "推理"]):
                    stats["categories"]["推理数学"] += 1
                elif any(kw in content.lower() for kw in ["工具", "tool", "搜索", "search"]):
                    stats["categories"]["工具调用"] += 1
                else:
                    stats["categories"]["对话问答"] += 1

            except json.JSONDecodeError:
                continue

    if stats["valid"] > 0:
        stats["avg_turns"] = stats["total_turns"] / stats["valid"]

    print(f"  总数据: {stats['total']:,} 条")
    print(f"  有效数据: {stats['valid']:,} 条")
    print(f"  平均轮次: {stats['avg_turns']:.1f}")
    print()
    print("  语言分布:")
    for lang, count in stats["languages"].most_common():
        print(f"    {lang}: {count:,} ({count/stats['valid']*100:.1f}%)")
    print()
    print("  类别分布:")
    for cat, count in stats["categories"].most_common():
        print(f"    {cat}: {count:,} ({count/stats['valid']*100:.1f}%)")

    return stats


def identify_gaps(stats):
    """识别数据缺口"""
    print()
    print("=" * 60)
    print("第二步：识别数据缺口")
    print("=" * 60)

    gaps = {}

    for category, target in TARGETS.items():
        current = stats.get("categories", {}).get(category, 0)
        # 也检查语言相关
        if category == "中文对话":
            current = stats.get("languages", {}).get("中文", 0)
        elif category == "英文对话":
            current = stats.get("languages", {}).get("英文", 0)

        gap = max(0, target - current)
        if gap > 0:
            gaps[category] = gap
            print(f"  {category}: 需要补充 {gap:,} 条 (当前 {current:,} / 目标 {target:,})")
        else:
            print(f"  {category}: [OK] 已满足 ({current:,} / {target:,})")

    return gaps


def download_dataset(ds_info, output_file):
    """下载单个数据集"""
    print(f"  下载: {ds_info['name']}...")

    try:
        response = requests.get(ds_info["url"], timeout=300)
        response.raise_for_status()

        # 保存原始文件
        temp_file = output_file + ".tmp"

        if ds_info["url"].endswith(".jsonl"):
            # JSONL 格式
            with open(temp_file, "wb") as f:
                f.write(response.content)
        else:
            # JSON 格式
            with open(temp_file, "wb") as f:
                f.write(response.content)

        return temp_file

    except Exception as e:
        print(f"    下载失败: {e}")
        return None


def convert_dataset(temp_file, output_file, format_type, max_samples=None):
    """转换数据集格式"""
    count = 0

    try:
        if format_type in ["alpaca", "belle"]:
            with open(temp_file, encoding="utf-8") as f:
                data = json.load(f)

            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    if max_samples and count >= max_samples:
                        break

                    instruction = item.get("instruction", "")
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")

                    if not instruction or not output_text:
                        continue

                    user_content = instruction
                    if input_text:
                        user_content += "\n" + input_text

                    # 判断是否中文
                    is_chinese = any('一' <= c <= '鿿' for c in instruction)
                    system_prompt = "你是态极，一个有帮助的AI助手。" if is_chinese else "You are Taiji, a helpful AI assistant."

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output_text}
                    ]

                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    count += 1

        elif format_type == "dolly":
            with open(temp_file, encoding="utf-8") as f:
                for line in f:
                    if max_samples and count >= max_samples:
                        break

                    try:
                        item = json.loads(line)
                        instruction = item.get("instruction", "")
                        context = item.get("context", "")
                        response = item.get("response", "")

                        if not instruction or not response:
                            continue

                        user_content = instruction
                        if context:
                            user_content = f"背景：{context}\n\n问题：{instruction}"

                        messages = [
                            {"role": "system", "content": "你是态极，一个有帮助的AI助手。"},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": response}
                        ]

                        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                        count += 1

                    except json.JSONDecodeError:
                        continue

        elif format_type == "oasst":
            with open(temp_file, encoding="utf-8") as f:
                for line in f:
                    if max_samples and count >= max_samples:
                        break

                    try:
                        item = json.loads(line)
                        if item.get("rank") != 0:
                            continue

                        text = item.get("text", "")
                        role = item.get("role", "")

                        if role != "assistant" or not text:
                            continue

                        messages = [
                            {"role": "system", "content": "You are Taiji, a helpful AI assistant."},
                            {"role": "user", "content": "Tell me something interesting."},
                            {"role": "assistant", "content": text}
                        ]

                        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                        count += 1

                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        print(f"    转换失败: {e}")

    return count


def validate_quality(file_path):
    """验证数据质量"""
    print()
    print("=" * 60)
    print("第三步：验证数据质量")
    print("=" * 60)

    issues = {
        "empty_content": 0,
        "echo_pattern": 0,
        "too_short": 0,
        "too_long": 0,
        "format_error": 0,
    }

    total = 0
    valid = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            total += 1

            try:
                item = json.loads(line)
                messages = item.get("messages", [])

                if not messages:
                    issues["format_error"] += 1
                    continue

                valid += 1

                for msg in messages:
                    content = msg.get("content", "")

                    # 检查空内容
                    if not content:
                        issues["empty_content"] += 1
                        break

                    # 检查长度
                    if len(content) < 10:
                        issues["too_short"] += 1
                        break

                    if len(content) > 10000:
                        issues["too_long"] += 1

                # 检查回声模式
                user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
                assistant_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
                if user_msg and assistant_msg and user_msg == assistant_msg:
                    issues["echo_pattern"] += 1

            except json.JSONDecodeError:
                issues["format_error"] += 1

    print(f"  总数据: {total:,} 条")
    print(f"  有效数据: {valid:,} 条")
    print()
    print("  质量检查:")
    for issue, count in issues.items():
        status = "[OK]" if count < 100 else "[WARN]"
        print(f"    {status} {issue}: {count:,}")

    return issues


def main():
    print("=" * 60)
    print("构建完美预训练数据集")
    print("=" * 60)

    # 第一步：分析现有数据
    stats = analyze_existing_data(EXISTING_DATA)

    # 第二步：识别缺口
    gaps = identify_gaps(stats)

    # 第三步：下载补充数据
    print()
    print("=" * 60)
    print("第三步：下载补充数据")
    print("=" * 60)

    os.makedirs("taiji_data/training_data", exist_ok=True)

    downloaded_files = []

    for ds in DATASETS:
        category = ds["category"]
        if category not in gaps or gaps[category] <= 0:
            print(f"  跳过 {ds['name']}: {category} 已满足")
            continue

        output_file = f"taiji_data/training_data/{ds['name'].replace(' ', '_').lower()}.jsonl"

        if os.path.exists(output_file):
            with open(output_file, encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            if existing > 100:
                print(f"  已存在 {ds['name']}: {existing:,} 条")
                downloaded_files.append(output_file)
                gaps[category] = max(0, gaps[category] - existing)
                continue

        temp_file = download_dataset(ds, output_file)
        if temp_file:
            count = convert_dataset(temp_file, output_file, ds["format"], ds.get("max_samples"))
            print(f"    完成: {count:,} 条")
            downloaded_files.append(output_file)
            gaps[category] = max(0, gaps[category] - count)

            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # 第四步：合并所有数据
    print()
    print("=" * 60)
    print("第四步：合并所有数据")
    print("=" * 60)

    all_files = [EXISTING_DATA] + downloaded_files
    total_lines = 0

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f_out:
        for file_path in all_files:
            if not os.path.exists(file_path):
                continue

            print(f"  合并: {os.path.basename(file_path)}")
            with open(file_path, encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_lines += 1

    print(f"\n  总计: {total_lines:,} 条 -> {OUTPUT_DATA}")

    # 第五步：最终质量验证
    validate_quality(OUTPUT_DATA)

    # 保存报告
    report = {
        "total_samples": total_lines,
        "source_files": all_files,
        "gaps_remaining": gaps,
    }

    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("[DONE] 完成！")
    print("=" * 60)
    print(f"  最终数据: {OUTPUT_DATA}")
    print(f"  质量报告: {QUALITY_REPORT}")


if __name__ == "__main__":
    main()
