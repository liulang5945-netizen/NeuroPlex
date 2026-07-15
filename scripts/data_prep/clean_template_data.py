"""
清理包含模板变量的训练数据
==========================
删除包含 {expr}, {file}, {topic} 等未填充模板变量的样本
"""
import json
import re
import os

def has_template_vars(text):
    """检查文本是否包含模板变量"""
    template_patterns = [
        r'\{[a-zA-Z_]+\}',  # {var}, {file_name}, {topic}
        r'\{expr\}',
        r'\{file\}',
        r'\{topic\}',
        r'\{input\}',
        r'\{output\}',
        r'\{path\}',
        r'\{content\}',
    ]
    for pattern in template_patterns:
        if re.search(pattern, text):
            return True
    return False

def clean_react_file(input_path, output_path):
    """清理ReAct格式文件中的模板数据"""
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return 0, 0

    total = 0
    kept = 0
    cleaned = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 检查task字段
            task = item.get('task', '')
            if has_template_vars(task):
                continue

            # 检查steps中的action_args
            steps = item.get('steps', [])
            has_template = False
            for step in steps:
                action_args = step.get('action_args', {})
                args_str = json.dumps(action_args, ensure_ascii=False)
                if has_template_vars(args_str):
                    has_template = True
                    break
                thought = step.get('thought', '')
                if has_template_vars(thought):
                    has_template = True
                    break
                final_answer = step.get('final_answer', '')
                if has_template_vars(final_answer):
                    has_template = True
                    break

            if not has_template:
                cleaned.append(item)
                kept += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return total, kept

def clean_messages_file(input_path, output_path):
    """清理messages格式文件中的模板数据"""
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return 0, 0

    total = 0
    kept = 0
    cleaned = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = item.get('messages', [])
            has_template = False

            for msg in messages:
                content = msg.get('content', '')
                if has_template_vars(content):
                    has_template = True
                    break

            if not has_template:
                cleaned.append(item)
                kept += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return total, kept

def main():
    print("=" * 60)
    print("清理包含模板变量的训练数据")
    print("=" * 60)

    sft_dir = "taiji_data/training_data/sft"

    # 清理 synthetic_react_data.jsonl
    print("\n[处理] synthetic_react_data.jsonl")
    total, kept = clean_react_file(
        f"{sft_dir}/synthetic_react_data.jsonl",
        f"{sft_dir}/synthetic_react_data_clean.jsonl"
    )
    print(f"  总计: {total}, 保留: {kept}, 移除: {total - kept}")

    # 清理 react_data.jsonl
    print("\n[处理] react_data.jsonl")
    total, kept = clean_react_file(
        f"{sft_dir}/react_data.jsonl",
        f"{sft_dir}/react_data_clean.jsonl"
    )
    print(f"  总计: {total}, 保留: {kept}, 移除: {total - kept}")

    # 清理 taiji_long_context.jsonl (代码中包含TODO)
    print("\n[处理] taiji_long_context.jsonl (移除含TODO的代码)")
    if os.path.exists(f"{sft_dir}/taiji_long_context.jsonl"):
        total = 0
        kept = 0
        cleaned = []
        with open(f"{sft_dir}/taiji_long_context.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    item = json.loads(line)
                    messages = item.get('messages', [])
                    has_todo = False
                    for msg in messages:
                        if 'TODO' in msg.get('content', ''):
                            has_todo = True
                            break
                    if not has_todo:
                        cleaned.append(item)
                        kept += 1
                except:
                    continue

        with open(f"{sft_dir}/taiji_long_context_clean.jsonl", 'w', encoding='utf-8') as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  总计: {total}, 保留: {kept}, 移除: {total - kept}")

    # 清理 code_alpaca.jsonl 和 code_alpaca_20k.jsonl (检查重叠)
    print("\n[检查] code_alpaca.jsonl 和 code_alpaca_20k.jsonl 重叠")
    if os.path.exists(f"{sft_dir}/code_alpaca.jsonl") and os.path.exists(f"{sft_dir}/code_alpaca_20k.jsonl"):
        # 读取两个文件的instruction
        instructions1 = set()
        instructions2 = set()

        with open(f"{sft_dir}/code_alpaca.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    msgs = item.get('messages', [])
                    if len(msgs) >= 2:
                        instructions1.add(msgs[1]['content'][:100])
                except:
                    pass

        with open(f"{sft_dir}/code_alpaca_20k.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    msgs = item.get('messages', [])
                    if len(msgs) >= 2:
                        instructions2.add(msgs[1]['content'][:100])
                except:
                    pass

        overlap = instructions1 & instructions2
        print(f"  code_alpaca.jsonl: {len(instructions1)} 条")
        print(f"  code_alpaca_20k.jsonl: {len(instructions2)} 条")
        print(f"  重叠: {len(overlap)} 条")

        if len(overlap) > 100:
            print("  建议: 删除 code_alpaca.jsonl (较小的文件)")

    print("\n" + "=" * 60)
    print("清理完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
