"""
SFT 数据清洗脚本 - 为预训练准备高质量对话数据
==============================================
过滤低质量、模板化、占位符回答，保留有实质内容的对话。

用法:
  python scripts/data_prep/clean_sft_for_pretrain.py
  python scripts/data_prep/clean_sft_for_pretrain.py --input taiji_data/training_data/sft/conversation_data.jsonl
"""
import json
import os
import re
import argparse
from collections import Counter
from pathlib import Path

# ============== 质量过滤规则 ==============

# 模板化占位符 - 包含这些内容的回答直接丢弃
PLACEHOLDER_PATTERNS = [
    r'\d+\.\s*\.\.\.',           # 1. ... 2. ...
    r'-\s*\.\.\.',               # - ...
    r'如果需要更详细',
    r'如有.*问题.*请.*告诉我',
    r'我可以帮你搜索',
    r'有什么.*想了解的吗',
    r'还有什么.*问题吗',
    r'希望.*对你.*有帮助',
    r'以上.*仅供参考',
    r'请注意.*以上.*仅为',
]

# 低质量回答特征
LOW_QUALITY_INDICATORS = [
    '基本介绍：',
    '主要特点包括：',
    '核心思想是...',
    '您可以进一步',
    '建议您.*搜索',
    '更多.*信息.*请',
]

# 高质量回答特征 - 包含这些说明回答有实质内容
HIGH_QUALITY_INDICATORS = [
    '例如',
    '具体来说',
    '举个例子',
    '首先',
    '其次',
    '因为',
    '所以',
    '原理是',
    '步骤',
    '代码',
    'import ',
    'def ',
    'class ',
    '```',
]


def is_placeholder_response(text: str) -> bool:
    """检查是否是占位符回答"""
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def is_low_quality(text: str) -> bool:
    """检查是否是低质量回答"""
    for indicator in LOW_QUALITY_INDICATORS:
        if indicator in text:
            return True
    return False


def calculate_quality_score(text: str) -> float:
    """计算回答质量分数 (0-1)"""
    score = 0.5  # 基础分

    # 长度加分 (太短扣分，适中长度加分)
    length = len(text)
    if length < 30:
        score -= 0.3
    elif length < 50:
        score -= 0.1
    elif 100 <= length <= 500:
        score += 0.1
    elif length > 500:
        score += 0.05

    # 结构化内容加分
    if re.search(r'\d+\.', text):  # 有序列表
        score += 0.05
    if re.search(r'[-*]', text):  # 无序列表
        score += 0.05
    if re.search(r'\*\*.*\*\*', text):  # Markdown 加粗
        score += 0.05

    # 高质量特征加分
    for indicator in HIGH_QUALITY_INDICATORS:
        if indicator in text:
            score += 0.05
            break

    # 包含代码加分
    if '```' in text or 'import ' in text or 'def ' in text:
        score += 0.1

    return min(1.0, max(0.0, score))


def validate_message_structure(messages: list) -> bool:
    """验证消息结构是否有效"""
    if not messages or len(messages) < 2:
        return False

    # 检查角色顺序
    roles = [m.get('role') for m in messages]
    if 'user' not in roles or 'assistant' not in roles:
        return False

    # 检查内容非空
    for msg in messages:
        if not msg.get('content', '').strip():
            return False

    return True


def clean_sft_file(input_path: str, output_path: str, min_quality: float = 0.3,
                   min_response_length: int = 50, verbose: bool = True) -> dict:
    """
    清洗单个SFT文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        min_quality: 最低质量分数阈值
        min_response_length: 最低回答长度
        verbose: 是否打印详细信息

    Returns:
        清洗统计信息
    """
    stats = {
        'total': 0,
        'kept': 0,
        'removed_placeholder': 0,
        'removed_short': 0,
        'removed_low_quality': 0,
        'removed_invalid_structure': 0,
        'quality_scores': [],
    }

    cleaned_items = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats['total'] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  [跳过] 第{line_num}行: JSON解析失败")
                continue

            messages = item.get('messages', [])

            # 验证结构
            if not validate_message_structure(messages):
                stats['removed_invalid_structure'] += 1
                continue

            # 获取最后一个assistant回答
            assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
            if not assistant_msgs:
                stats['removed_invalid_structure'] += 1
                continue

            last_response = assistant_msgs[-1].get('content', '')

            # 检查占位符
            if is_placeholder_response(last_response):
                stats['removed_placeholder'] += 1
                if verbose and stats['removed_placeholder'] <= 3:
                    print(f"  [占位符] 问题: {messages[1]['content'][:50]}...")
                    print(f"          回答: {last_response[:80]}...")
                continue

            # 检查长度
            if len(last_response) < min_response_length:
                stats['removed_short'] += 1
                continue

            # 检查低质量
            if is_low_quality(last_response):
                stats['removed_low_quality'] += 1
                continue

            # 计算质量分数
            quality = calculate_quality_score(last_response)
            stats['quality_scores'].append(quality)

            if quality < min_quality:
                stats['removed_low_quality'] += 1
                continue

            # 标准化 system prompt
            if messages and messages[0].get('role') == 'system':
                messages[0]['content'] = '你是态极（Taiji），一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。'

            item['messages'] = messages
            item['_quality_score'] = round(quality, 2)
            cleaned_items.append(item)
            stats['kept'] += 1

    # 按质量分数排序 (高质量在前)
    cleaned_items.sort(key=lambda x: x.get('_quality_score', 0), reverse=True)

    # 移除临时字段并写入
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_items:
            if '_quality_score' in item:
                del item['_quality_score']
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description='清洗SFT数据用于预训练')
    parser.add_argument('--input', type=str, help='输入文件路径 (默认处理所有SFT对话文件)')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--min-quality', type=float, default=0.3, help='最低质量分数 (0-1)')
    parser.add_argument('--min-length', type=int, default=50, help='最低回答长度')
    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')
    args = parser.parse_args()

    print("=" * 60)
    print("SFT 数据清洗 - 为预训练准备高质量对话数据")
    print("=" * 60)

    # 定义要清洗的文件列表
    sft_dir = Path('taiji_data/training_data/sft')

    if args.input:
        files_to_clean = [Path(args.input)]
    else:
        # 默认清洗所有对话相关文件
        files_to_clean = [
            sft_dir / 'conversation_data.jsonl',
            sft_dir / 'taiji_conversation_data.jsonl',
            sft_dir / 'hq_conversation.jsonl',
            sft_dir / 'long_conversations.jsonl',
            sft_dir / 'identity_data.jsonl',
            sft_dir / 'identity_taiji.jsonl',
        ]

    total_stats = Counter()
    output_files = []

    for input_file in files_to_clean:
        if not input_file.exists():
            print(f"\n[跳过] 文件不存在: {input_file}")
            continue

        # 生成输出文件名
        if args.output and len(files_to_clean) == 1:
            output_file = Path(args.output)
        else:
            output_file = input_file.parent / f"{input_file.stem}_clean.jsonl"

        print(f"\n[处理] {input_file.name}")
        print(f"  -> {output_file.name}")

        stats = clean_sft_file(
            str(input_file),
            str(output_file),
            min_quality=args.min_quality,
            min_response_length=args.min_length,
            verbose=args.verbose,
        )

        # 打印统计
        print(f"  总计: {stats['total']} 条")
        print(f"  保留: {stats['kept']} 条 ({stats['kept']/max(stats['total'],1)*100:.1f}%)")
        print(f"  移除:")
        print(f"    - 占位符回答: {stats['removed_placeholder']}")
        print(f"    - 过短回答: {stats['removed_short']}")
        print(f"    - 低质量回答: {stats['removed_low_quality']}")
        print(f"    - 结构无效: {stats['removed_invalid_structure']}")

        if stats['quality_scores']:
            avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores'])
            print(f"  平均质量分: {avg_quality:.2f}")

        # 累计统计
        total_stats['total'] += stats['total']
        total_stats['kept'] += stats['kept']
        total_stats['removed_placeholder'] += stats['removed_placeholder']
        total_stats['removed_short'] += stats['removed_short']
        total_stats['removed_low_quality'] += stats['removed_low_quality']
        total_stats['removed_invalid_structure'] += stats['removed_invalid_structure']

        output_files.append(output_file)

    # 打印总计
    print("\n" + "=" * 60)
    print("清洗完成 - 总计")
    print("=" * 60)
    print(f"总处理: {total_stats['total']} 条")
    print(f"总保留: {total_stats['kept']} 条 ({total_stats['kept']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"总移除: {total_stats['total'] - total_stats['kept']} 条")

    print("\n输出文件:")
    for f in output_files:
        if f.exists():
            # 统计输出文件行数
            with open(f, 'r', encoding='utf-8') as fh:
                line_count = sum(1 for _ in fh)
            print(f"  {f.name}: {line_count} 条")

    # 合并建议
    print("\n" + "=" * 60)
    print("下一步建议:")
    print("=" * 60)
    print("1. 检查清洗后的数据质量:")
    print("   python -c \"import json; [print(json.loads(l)['messages'][1]['content'][:50]) for l in open('taiji_data/training_data/sft/taiji_conversation_data_clean.jsonl')]\"")
    print("\n2. 合并清洗后的数据:")
    print("   python scripts/data_prep/merge_cleaned_sft.py")
    print("\n3. 在 Autodl 上开始训练:")
    print("   python taiji/train/finetune_taiji.py")


if __name__ == '__main__':
    main()
