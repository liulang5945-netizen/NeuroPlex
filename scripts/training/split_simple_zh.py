"""按类别分割简单中文数据，为多神经元差异化训练准备。

将 TinyStoriesAdv-zh 按内容类别分成 3 组：
  A. 语文文学（chinese_class + tinybooks）→ 语言文学专精
  B. 百科数学（encyclopedias + math）→ 知识推理专精
  C. 故事对话（tinystories_adv + tinygames）→ 叙事对话专精

每组：30% 抽取为共享核心 + 70% 独有数据
每个神经元 = 共享核心 + 一类独有数据（保证差异性）

输出：
  data/simple_zh/shared_core.jsonl      （30% 共享核心，所有神经元都用）
  data/simple_zh/class_a_chinese.jsonl  （语文文学 70%）
  data/simple_zh/class_b_encyclopedia.jsonl（百科数学 70%）
  data/simple_zh/class_c_story.jsonl    （故事对话 70%）
"""
from __future__ import annotations

import os
import sys
import json
import random
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "fzmnm/TinyStoriesAdv-zh"
OUTPUT_DIR = "data/simple_zh"

# 类别映射：文件路径前缀 → 类别
CATEGORY_MAP = {
    "train_data/chinese_class/": "A_chinese",      # 语文（古诗文、字词、阅读）
    "train_data/tinybooks/": "A_chinese",           # 文学（经典简化版）
    "train_data/encyclopedias/": "B_encyclopedia",  # 百科
    "train_data/math/": "B_encyclopedia",            # 数学
    "train_data/tinystories_adv/": "C_story",       # 故事生成
    "train_data/tinygames/": "C_story",              # 游戏/对话
    "train_data/quizs/": "B_encyclopedia",           # 测验归到百科
}


def categorize_file(fname: str) -> str | None:
    """根据文件路径判断类别。"""
    for prefix, cat in CATEGORY_MAP.items():
        if fname.startswith(prefix):
            return cat
    return None


def extract_text(d: dict) -> str:
    """提取 text 字段，清理 meta_tag。"""
    text = d.get('text') or d.get('content') or d.get('story') or ''
    if not text or len(text) < 20:
        return ''
    if text.startswith('meta_tag:'):
        lines = text.split('\n', 2)
        if len(lines) >= 2:
            text = lines[1] if len(lines) == 2 else lines[2]
    text = text.strip()
    return text if len(text) >= 20 else ''


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = HfApi()

    print("=" * 60)
    print("按类别分割 TinyStoriesAdv-zh（为多神经元差异化训练）")
    print("=" * 60)

    files = api.list_repo_files(REPO_ID, repo_type='dataset')
    jsonl_files = [f for f in files if f.startswith('train_data/') and f.endswith('.jsonl')]

    # 按类别收集
    categories = {"A_chinese": [], "B_encyclopedia": [], "C_story": []}
    uncategorized = []

    for fname in jsonl_files:
        cat = categorize_file(fname)
        if cat:
            categories[cat].append(fname)
        else:
            uncategorized.append(fname)

    print("\n类别分布:")
    for cat, flist in categories.items():
        print(f"  {cat}: {len(flist)} 个文件")
    if uncategorized:
        print(f"  未分类: {len(uncategorized)} 个文件")

    # 下载并按类别保存
    raw_dir = os.path.join(OUTPUT_DIR, '_raw_split')
    os.makedirs(raw_dir, exist_ok=True)

    cat_texts = {"A_chinese": [], "B_encyclopedia": [], "C_story": []}
    total_stats = {"A_chinese": 0, "B_encyclopedia": 0, "C_story": 0}

    for cat, flist in categories.items():
        print(f"\n[{cat}] 下载 {len(flist)} 个文件...")
        for i, fname in enumerate(flist):
            print(f"  ({i+1}/{len(flist)}) {fname}...", end=' ', flush=True)
            try:
                local = hf_hub_download(REPO_ID, fname, repo_type='dataset', local_dir=raw_dir)
                count = 0
                with open(local, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = extract_text(d)
                        if text:
                            cat_texts[cat].append(text)
                            count += 1
                total_stats[cat] += count
                print(f"{count} 条")
            except Exception as e:
                print(f"失败: {e}")

    # 统计
    print(f"\n{'='*60}")
    print("各类别数据量:")
    for cat, texts in cat_texts.items():
        chars = sum(len(t) for t in texts)
        print(f"  {cat}: {len(texts)} 条, {chars/10000:.0f} 万字, ~{int(chars*0.6/1e6)}M tokens")

    # 分割：30% 共享核心 + 70% 独有
    random.seed(42)
    shared_core = []
    unique_data = {"A_chinese": [], "B_encyclopedia": [], "C_story": []}

    for cat, texts in cat_texts.items():
        indices = list(range(len(texts)))
        random.shuffle(indices)
        split_point = int(len(indices) * 0.3)
        shared_idx = indices[:split_point]
        unique_idx = indices[split_point:]
        shared_core.extend([texts[i] for i in shared_idx])
        unique_data[cat] = [texts[i] for i in unique_idx]

    random.shuffle(shared_core)

    # 保存
    output_files = {
        "shared_core": (os.path.join(OUTPUT_DIR, "shared_core.jsonl"), shared_core),
        "class_a_chinese": (os.path.join(OUTPUT_DIR, "class_a_chinese.jsonl"), unique_data["A_chinese"]),
        "class_b_encyclopedia": (os.path.join(OUTPUT_DIR, "class_b_encyclopedia.jsonl"), unique_data["B_encyclopedia"]),
        "class_c_story": (os.path.join(OUTPUT_DIR, "class_c_story.jsonl"), unique_data["C_story"]),
    }

    print(f"\n保存分割结果:")
    for name, (path, texts) in output_files.items():
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        chars = sum(len(t) for t in texts)
        print(f"  {name}: {len(texts)} 条, {chars/10000:.0f} 万字, ~{int(chars*0.6/1e6)}M tokens")

    # 计算每个神经元的训练数据量
    print(f"\n{'='*60}")
    print("每个神经元训练数据（共享核心 + 一类独有）:")
    shared_count = len(shared_core)
    for cat, label in [("A_chinese", "语文文学"), ("B_encyclopedia", "百科数学"), ("C_story", "故事对话")]:
        unique_count = len(unique_data[cat])
        total = shared_count + unique_count
        chars = sum(len(t) for t in shared_core) + sum(len(t) for t in unique_data[cat])
        print(f"  {label}: {shared_count}(共享) + {unique_count}(独有) = {total} 条, ~{int(chars*0.6/1e6)}M tokens")
        print(f"    数据/参数比(36M): {chars*0.6/1e6/36:.1f}:1")

    # 清理
    import shutil
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        print(f"\n已清理临时目录")

    print(f"\n{'='*60}")
    print("✅ 数据分割完成！")
    print(f"  共享核心: data/simple_zh/shared_core.jsonl")
    print(f"  语文文学: data/simple_zh/class_a_chinese.jsonl")
    print(f"  百科数学: data/simple_zh/class_b_encyclopedia.jsonl")
    print(f"  故事对话: data/simple_zh/class_c_story.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
