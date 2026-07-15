"""
下载中文数据集
==============
补充中文对话数据
"""
import os
import json
import requests
import time

def download_file(url, output_path, max_retries=3):
    """下载文件"""
    for attempt in range(max_retries):
        try:
            print(f"  下载中 (尝试 {attempt + 1})...")
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"  下载完成: {len(response.content) / (1024*1024):.1f} MB")
            return True

        except Exception as e:
            print(f"  下载失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return False

def convert_chinese_alpaca(input_file, output_file, max_samples=None):
    """转换中文 Alpaca 格式"""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
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

            messages = [
                {"role": "system", "content": "你是态极，一个有帮助的AI助手。"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text}
            ]

            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    return count

def convert_belle_format(input_file, output_file, max_samples=None):
    """转换 BELLE 格式"""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            if max_samples and count >= max_samples:
                break

            instruction = item.get("instruction", "")
            output_text = item.get("output", "")

            if not instruction or not output_text:
                continue

            messages = [
                {"role": "system", "content": "你是态极，一个有帮助的AI助手。"},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_text}
            ]

            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    return count

def main():
    print("=" * 60)
    print("下载中文数据集")
    print("=" * 60)

    os.makedirs("taiji_data/training_data", exist_ok=True)

    # 中文数据源列表
    datasets = [
        {
            "name": "Alpaca中文 51K",
            "url": "https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/data/alpaca_data_zh_51k.json",
            "output": "taiji_data/training_data/alpaca_zh_51k.jsonl",
            "format": "alpaca",
        },
        {
            "name": "Alpaca中文 105K (扩展版)",
            "url": "https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/data/alpaca_data_zh_105k.json",
            "output": "taiji_data/training_data/alpaca_zh_105k.jsonl",
            "format": "alpaca",
        },
        {
            "name": "BELLE 0.5M (中文)",
            "url": "https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/Belle_open_source_0.5M.json",
            "output": "taiji_data/training_data/belle_0.5m.jsonl",
            "format": "belle",
            "max_samples": 300000,
        },
        {
            "name": "BELLE 1M (中文)",
            "url": "https://huggingface.co/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json",
            "output": "taiji_data/training_data/belle_1m.jsonl",
            "format": "belle",
            "max_samples": 300000,
        },
        {
            "name": "Firefly 中文",
            "url": "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl",
            "output": "taiji_data/training_data/firefly_zh.jsonl",
            "format": "firefly",
            "max_samples": 200000,
        },
    ]

    total_count = 0

    for ds in datasets:
        print()
        print(f"数据集: {ds['name']}")
        print("-" * 40)

        output_file = ds["output"]

        if os.path.exists(output_file):
            with open(output_file, encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            print(f"  已存在: {existing:,} 条")
            total_count += existing
            continue

        # 下载
        temp_file = output_file + ".tmp"
        if not download_file(ds["url"], temp_file):
            continue

        # 转换
        print(f"  转换格式...")
        try:
            if ds["format"] == "alpaca":
                count = convert_chinese_alpaca(temp_file, output_file, ds.get("max_samples"))
            elif ds["format"] == "belle":
                count = convert_belle_format(temp_file, output_file, ds.get("max_samples"))
            elif ds["format"] == "firefly":
                # Firefly 是 JSONL 格式，需要特殊处理
                count = 0
                with open(temp_file, encoding="utf-8") as f_in, \
                     open(output_file, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        if ds.get("max_samples") and count >= ds["max_samples"]:
                            break
                        try:
                            item = json.loads(line)
                            # Firefly 格式: {"kind": "...", "input": "...", "target": "..."}
                            kind = item.get("kind", "")
                            input_text = item.get("input", "")
                            target_text = item.get("target", "")

                            if not input_text or not target_text:
                                continue

                            messages = [
                                {"role": "system", "content": "你是态极，一个有帮助的AI助手。"},
                                {"role": "user", "content": input_text},
                                {"role": "assistant", "content": target_text}
                            ]

                            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                            count += 1
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"  未知格式: {ds['format']}")
                continue

            print(f"  完成: {count:,} 条")
            total_count += count

        except Exception as e:
            print(f"  转换失败: {e}")

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    print()
    print("=" * 60)
    print(f"下载完成！总计: {total_count:,} 条")
    print("=" * 60)

if __name__ == "__main__":
    main()
