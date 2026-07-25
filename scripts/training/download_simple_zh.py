"""下载并预处理中文简单数据集 TinyStoriesAdv-zh（小学/幼儿园知识水平）。

数据来源：https://huggingface.co/datasets/fzmnm/TinyStoriesAdv-zh
约 1B tokens，专为小模型训练设计。

用途：为态极 compact(36M) 神经元准备"不复杂"的中文训练数据。
解决之前维基百科数据太复杂导致 36M 模型生成乱码的问题。

输出：data/simple_zh/simple_zh_texts.jsonl（统一格式，每行一个 text）
"""
from __future__ import annotations

import os
import sys
import json
import time
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO_ID = "fzmnm/TinyStoriesAdv-zh"
OUTPUT_DIR = "data/simple_zh"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "simple_zh_texts.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = HfApi()

    print("=" * 60)
    print("下载 TinyStoriesAdv-zh（小学/幼儿园知识水平中文数据）")
    print("=" * 60)

    # 列出所有文件
    files = api.list_repo_files(REPO_ID, repo_type='dataset')
    jsonl_files = [f for f in files if f.startswith('train_data/') and f.endswith('.jsonl')]
    val_files = [f for f in files if f.startswith('val_data/') and f.endswith('.jsonl')]
    print(f"训练文件: {len(jsonl_files)} 个 jsonl")
    print(f"验证文件: {len(val_files)} 个 jsonl")

    # 下载并合并
    total_texts = 0
    total_chars = 0
    t0 = time.time()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # 训练数据
        for i, fname in enumerate(jsonl_files):
            print(f"  [{i+1}/{len(jsonl_files)}] 下载 {fname}...", end=' ', flush=True)
            try:
                local = hf_hub_download(REPO_ID, fname, repo_type='dataset',
                                        local_dir=os.path.join(OUTPUT_DIR, '_raw'))
                count = 0
                chars = 0
                with open(local, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # 提取 text 字段（不同文件字段名可能不同）
                        text = d.get('text') or d.get('content') or d.get('story') or ''
                        if not text or len(text) < 20:
                            continue
                        # 清理：去掉 meta_tag 前缀行（如果有）
                        if text.startswith('meta_tag:'):
                            lines = text.split('\n', 2)
                            if len(lines) >= 2:
                                text = lines[1] if len(lines) == 2 else lines[2]
                        text = text.strip()
                        if len(text) < 20:
                            continue
                        out_f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                        chars += len(text)
                total_texts += count
                total_chars += chars
                print(f"{count} 条, {chars/10000:.1f} 万字")
            except Exception as e:
                print(f"失败: {e}")

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"下载完成！")
    print(f"  总条目: {total_texts:,} 条")
    print(f"  总字符: {total_chars:,} ({total_chars/10000:.0f} 万字)")
    print(f"  估算 tokens: ~{int(total_chars * 0.6):,} ({total_chars * 0.6 / 1e6:.0f}M tokens)")
    print(f"  文件大小: {file_size_mb:.1f} MB")
    print(f"  输出: {OUTPUT_FILE}")
    print(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  数据/参数比估算:")
    print(f"    compact(36M): {total_chars * 0.6 / 1e6 / 36:.1f}:1 (Chinchilla 最优 20:1)")
    print(f"    standard(131M): {total_chars * 0.6 / 1e6 / 131:.1f}:1")
    print(f"{'='*60}")

    # 清理临时目录
    import shutil
    raw_dir = os.path.join(OUTPUT_DIR, '_raw')
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        print(f"已清理临时目录: {raw_dir}")


if __name__ == "__main__":
    main()
