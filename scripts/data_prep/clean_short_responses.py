"""
清洗短回复数据
==============
删除assistant回复过短（<20字符）的样本
"""
import json
import os

def clean_short_responses(input_path, output_path, min_length=20):
    """删除回复过短的样本"""
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return 0, 0, 0

    total = 0
    kept = 0
    removed_short = 0
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

            # 找到最后一个assistant回复
            last_response = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    last_response = msg.get('content', '')
                    break

            # 检查回复长度
            if len(last_response) < min_length:
                removed_short += 1
                continue

            cleaned.append(item)
            kept += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return total, kept, removed_short

def main():
    print("=" * 60)
    print("清洗短回复数据")
    print("=" * 60)

    supp_dir = "taiji_data/training_data/supplementary"

    # 清洗 long_context.jsonl
    print("\n[处理] long_context.jsonl")
    total, kept, removed = clean_short_responses(
        f"{supp_dir}/long_context.jsonl",
        f"{supp_dir}/long_context_clean.jsonl",
        min_length=20
    )
    print(f"  总计: {total}")
    print(f"  保留: {kept} ({kept/max(total,1)*100:.1f}%)")
    print(f"  移除短回复: {removed}")

    # 清洗 science_knowledge.jsonl
    print("\n[处理] science_knowledge.jsonl")
    total, kept, removed = clean_short_responses(
        f"{supp_dir}/science_knowledge.jsonl",
        f"{supp_dir}/science_knowledge_clean.jsonl",
        min_length=20
    )
    print(f"  总计: {total}")
    print(f"  保留: {kept} ({kept/max(total,1)*100:.1f}%)")
    print(f"  移除短回复: {removed}")

    # 检查 chinese_pretrain.jsonl (缺少系统提示)
    print("\n[检查] chinese_pretrain.jsonl (缺少系统提示)")
    if os.path.exists(f"{supp_dir}/chinese_pretrain.jsonl"):
        with open(f"{supp_dir}/chinese_pretrain.jsonl", 'r', encoding='utf-8') as f:
            first_line = f.readline()
            item = json.loads(first_line)
            msgs = item.get('messages', [])
            has_system = any(m.get('role') == 'system' for m in msgs)
            print(f"  第一条样本有system提示: {has_system}")
            print(f"  格式: {'messages' if 'messages' in item else 'text'}")

    print("\n" + "=" * 60)
    print("清洗完成!")
    print("=" * 60)

    print("\n下一步:")
    print("1. 删除原始文件，保留清洗后的版本")
    print("2. 下载更多高质量对话数据补充")

if __name__ == '__main__':
    main()
