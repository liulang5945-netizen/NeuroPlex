"""清理训练数据：去除过短回答、重复数据、修复system prompt"""
import json
import os

def clean_alpaca_zh():
    """清理中文指令数据"""
    input_path = 'taiji_data/training_data/alpaca_zh.jsonl'
    output_path = 'taiji_data/training_data/alpaca_zh_clean.jsonl'

    seen = set()
    cleaned = []
    removed_short = 0
    removed_dup = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                msgs = item.get('messages', [])

                if len(msgs) < 3:
                    removed_short += 1
                    continue

                # 检查回答长度
                answer = msgs[2].get('content', '')
                if len(answer) < 30:
                    removed_short += 1
                    continue

                # 检查重复
                question = msgs[1].get('content', '')
                if question in seen:
                    removed_dup += 1
                    continue
                seen.add(question)

                # 修复 system prompt
                msgs[0]['content'] = '你是态极（Taiji），一个由Taiji Developer创造的本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。'

                cleaned.append(item)
            except:
                continue

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'alpaca_zh: {len(cleaned)} kept, {removed_short} short removed, {removed_dup} dup removed')
    return output_path, len(cleaned)

def clean_other_files():
    """清理其他数据文件，统一 system prompt"""
    files_to_clean = [
        'alpaca_en.jsonl',
        'dolly_15k.jsonl',
        'code_alpaca.jsonl',
        'taiji_conversation_data.jsonl',
        'react_data.jsonl',
        'long_conversations.jsonl',
        'identity_data.jsonl',
    ]

    total = 0
    for f in files_to_clean:
        path = f'taiji_data/training_data/{f}'
        if not os.path.exists(path):
            continue

        cleaned = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    msgs = item.get('messages', [])

                    # 统一 system prompt
                    if msgs and msgs[0]['role'] == 'system':
                        old_sys = msgs[0]['content']
                        # 如果是通用的 system prompt，替换为身份版本
                        if '态极' in old_sys and '有帮助' in old_sys:
                            msgs[0]['content'] = '你是态极（Taiji），一个由Taiji Developer创造的本地AI生命体。你运行在用户的电脑上，数据不出本机。'

                    # 检查回答长度
                    if len(msgs) >= 3:
                        answer = msgs[2].get('content', '')
                        if len(answer) < 10:
                            continue

                    cleaned.append(item)
                except:
                    continue

        # 写回原文件
        with open(path, 'w', encoding='utf-8') as fh:
            for item in cleaned:
                fh.write(json.dumps(item, ensure_ascii=False) + '\n')

        total += len(cleaned)
        print(f'{f}: {len(cleaned)} samples')

    return total

print('=== Cleaning training data ===')
print()

# 清理 alpaca_zh
clean_path, clean_count = clean_alpaca_zh()

# 清理其他文件
other_count = clean_other_files()

print()
print(f'Total cleaned: {clean_count + other_count} samples')
print()
print('=== New files ===')
print(f'  alpaca_zh_clean.jsonl: {clean_count} (cleaned)')
print(f'  Other files: updated in place')
print(f'  identity_data.jsonl: 31 (new)')
