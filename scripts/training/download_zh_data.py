"""下载大量中文训练数据 —— 解决数据/参数比严重不足问题。

当前：3M tokens / 491M 参数 = 0.006（差 Chinchilla 定律 3000 倍）
目标：100M+ tokens / 491M 参数 = 0.2+（改善 30 倍）

数据源：
1. wikimedia/wikipedia (20231101.zh) — 中文维基百科，高质量百科文本
2. shibing624/alpaca-zh — 指令对话数据（多样性补充）
3. wangrui6/Zhihu-KOL — 知乎高质量问答（如果可用）

用法：
    python -u scripts/training/download_zh_data.py --target_articles 500000
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CACHE_PATH = "data/distill/zh_texts.jsonl"
TARGET_CHARS_PER_ARTICLE = 200  # 平均每条文本约 200 字


def download_wikipedia(max_articles, cache_path, existing_count):
    """下载中文维基百科文本。"""
    from datasets import load_dataset

    print(f"  下载中文维基百科 (wikimedia/wikipedia 20231101.zh)...")
    print(f"  目标: {max_articles} 篇文章", flush=True)

    count = 0
    try:
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.zh",
            split="train", streaming=True,
        )
        with open(cache_path, "a", encoding="utf-8") as f:
            for example in ds:
                if count >= max_articles:
                    break
                text = example.get("text", "").strip()
                if len(text) < 50:  # 跳过太短的文章
                    continue
                # 维基文章可能很长，分段写入（每段 ~500 字）
                while len(text) > 500:
                    f.write(text[:500] + "\n")
                    text = text[500:]
                    count += 1
                    if count >= max_articles:
                        break
                    if count % 50000 == 0:
                        print(f"    已下载 {count + existing_count} 条...", flush=True)
                if text.strip():
                    f.write(text.strip() + "\n")
                    count += 1
                    if count % 50000 == 0:
                        print(f"    已下载 {count + existing_count} 条...", flush=True)
    except Exception as e:
        print(f"  WARN: wikipedia 下载失败: {e}", flush=True)

    print(f"  维基百科完成: {count} 条", flush=True)
    return count


def download_alpaca(cache_path, existing_count):
    """下载 alpaca-zh 指令数据（多样性补充）。"""
    from datasets import load_dataset

    print(f"  下载 alpaca-zh 指令数据...", flush=True)
    count = 0
    try:
        ds = load_dataset("shibing624/alpaca-zh", split="train")
        with open(cache_path, "a", encoding="utf-8") as f:
            for example in ds:
                parts = []
                for field in ["instruction", "input", "output"]:
                    val = example.get(field, "")
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                if parts:
                    f.write(" ".join(parts) + "\n")
                    count += 1
    except Exception as e:
        print(f"  WARN: alpaca-zh 下载失败: {e}", flush=True)

    print(f"  alpaca-zh 完成: {count} 条", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description="下载大量中文训练数据")
    parser.add_argument("--target_articles", type=int, default=500000,
                        help="目标文章数（默认 50 万条，约 100M 字 ≈ 60M tokens）")
    parser.add_argument("--keep_existing", action="store_true",
                        help="保留现有缓存数据，追加下载（默认重新下载）")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("下载大量中文训练数据", flush=True)
    print(f"  目标: {args.target_articles} 条文本", flush=True)
    print(f"  估计: ~{args.target_articles * TARGET_CHARS_PER_ARTICLE / 1e6:.0f}M 字 ≈ "
          f"~{args.target_articles * TARGET_CHARS_PER_ARTICLE / 1.7 / 1e6:.0f}M tokens", flush=True)
    print("=" * 70, flush=True)

    existing_count = 0
    if args.keep_existing and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
        print(f"保留现有缓存: {existing_count} 条", flush=True)
    else:
        # 重新开始，删除旧缓存
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
            print(f"删除旧缓存，重新下载", flush=True)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    t_start = time.time()

    # 1. 下载维基百科（主要数据源）
    wiki_target = args.target_articles - existing_count
    if wiki_target > 0:
        wiki_count = download_wikipedia(wiki_target, CACHE_PATH, existing_count)
    else:
        wiki_count = 0

    # 2. 下载 alpaca-zh（多样性补充，约 77K 条）
    alpaca_count = download_alpaca(CACHE_PATH, existing_count + wiki_count)

    total_new = wiki_count + alpaca_count
    total_all = existing_count + total_new

    elapsed = time.time() - t_start

    # 统计
    print("\n" + "=" * 70, flush=True)
    print("下载完成", flush=True)
    print(f"  新增: {total_new} 条 (维基 {wiki_count} + alpaca {alpaca_count})", flush=True)
    print(f"  总计: {total_all} 条", flush=True)
    print(f"  耗时: {elapsed:.0f}s", flush=True)

    # 估算 tokens
    total_chars = 0
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_chars += len(line.strip())
    estimated_tokens = total_chars / 1.7  # 中文约 1.7 字/token
    print(f"  总字符: {total_chars/1e6:.1f}M", flush=True)
    print(f"  估计 tokens: {estimated_tokens/1e6:.1f}M", flush=True)
    print(f"  数据/参数比 (491M): {estimated_tokens/491e6:.3f}", flush=True)
    print(f"  Chinchilla 最优 (20:1): {'✅ 达标' if estimated_tokens/491e6 >= 20 else '❌ 仍不足，但已改善'}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
