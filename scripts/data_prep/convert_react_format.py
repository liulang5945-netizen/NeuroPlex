"""
转换 ReAct 格式为 Messages 格式
================================
将 task/steps 格式转换为 messages 格式，用于预训练
"""
import os
import json
import random

# ReAct 格式文件
REACT_FILES = [
    "taiji_data/training_data/react_data.jsonl",
    "taiji_data/training_data/taiji_react_data.jsonl",
    "taiji_data/training_data/taiji_graduation_react.jsonl",
    "taiji_data/training_data/taiji_graduation_v2_react.jsonl",
    "taiji_data/training_data/taiji_graduation_v3_react.jsonl",
    "taiji_data/training_data/taiji_ultimate_react.jsonl",
]

OUTPUT_FILE = "taiji_data/training_data/react_converted.jsonl"

def convert_react_to_messages(item):
    """将 ReAct 格式转换为 Messages 格式"""
    task = item.get("task", "")
    steps = item.get("steps", [])

    if not task or not steps:
        return None

    # 构建助手回答
    assistant_parts = []

    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action")
        action_args = step.get("action_args", {})
        final_answer = step.get("final_answer", "")

        if thought:
            assistant_parts.append(f"思考：{thought}")

        if action:
            args_str = json.dumps(action_args, ensure_ascii=False) if action_args else ""
            assistant_parts.append(f"行动：{action}({args_str})")

        if final_answer:
            # 替换 {observation} 占位符
            answer = final_answer.replace("{observation}", "[执行结果]")
            assistant_parts.append(answer)

    assistant_content = "\n\n".join(assistant_parts)

    if not assistant_content:
        return None

    messages = [
        {"role": "system", "content": "你是态极，一个会使用工具的AI助手。"},
        {"role": "user", "content": task},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"messages": messages}

def main():
    print("=" * 60)
    print("转换 ReAct 格式为 Messages 格式")
    print("=" * 60)

    all_converted = []

    for file_path in REACT_FILES:
        if not os.path.exists(file_path):
            print(f"  [跳过] {os.path.basename(file_path)}: 文件不存在")
            continue

        print(f"  [转换] {os.path.basename(file_path)}...", end=" ")

        count = 0
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    converted = convert_react_to_messages(item)
                    if converted:
                        all_converted.append(converted)
                        count += 1
                except json.JSONDecodeError:
                    continue

        print(f"{count:,} 条")

    # 打乱顺序
    random.shuffle(all_converted)

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"转换完成！总计: {len(all_converted):,} 条")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
