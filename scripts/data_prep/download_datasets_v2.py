"""
下载开源数据集 (v2)
==================
使用 requests 直接下载，避免 datasets 库兼容性问题
"""
import os
import json
import requests
from pathlib import Path

# 数据集配置
DATASETS = [
    {
        "name": "Alpaca (英文指令)",
        "url": "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/alpaca_data.json",
        "output": "taiji_data/training_data/alpaca_en.jsonl",
        "format": "alpaca",
    },
    {
        "name": "CodeAlpaca (代码)",
        "url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
        "output": "taiji_data/training_data/code_alpaca.jsonl",
        "format": "alpaca",
    },
    {
        "name": "BELLE (中文指令 1.5M)",
        "url": "https://huggingface.co/datasets/BelleGroup/train_1.5M_CN/resolve/main/Belle_open_source_1.5M.json",
        "output": "taiji_data/training_data/belle_cn.jsonl",
        "format": "belle",
    },
]

def download_file(url, output_path):
    """下载文件"""
    print(f"  下载中: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                if downloaded % (1024*1024) == 0:  # 每MB显示一次
                    print(f"    进度: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)")

    print(f"  下载完成: {output_path}")
    return output_path

def convert_alpaca(input_file, output_file):
    """转换 Alpaca 格式"""
    print(f"  转换格式: Alpaca -> JSONL")

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
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

    print(f"  转换完成: {count:,} 条")
    return count

def convert_belle(input_file, output_file):
    """转换 BELLE 格式"""
    print(f"  转换格式: BELLE -> JSONL")

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
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

    print(f"  转换完成: {count:,} 条")
    return count

def main():
    print("=" * 60)
    print("下载开源数据集 (v2)")
    print("=" * 60)

    # 确保目录存在
    os.makedirs("taiji_data/training_data", exist_ok=True)

    total_count = 0
    temp_dir = Path("taiji_data/training_data/temp")
    temp_dir.mkdir(exist_ok=True)

    for ds_info in DATASETS:
        print()
        print(f"数据集: {ds_info['name']}")
        print("-" * 40)

        if os.path.exists(ds_info["output"]):
            with open(ds_info["output"], encoding="utf-8") as f:
                existing_lines = sum(1 for _ in f)
            print(f"  已存在: {existing_lines:,} 条")
            if existing_lines > 1000:
                print(f"  跳过下载")
                total_count += existing_lines
                continue

        # 下载原始文件
        temp_file = temp_dir / f"{ds_info['name'].split('(')[0].strip()}.json"
        try:
            download_file(ds_info["url"], str(temp_file))
        except Exception as e:
            print(f"  下载失败: {e}")
            continue

        # 转换格式
        if ds_info["format"] == "alpaca":
            count = convert_alpaca(str(temp_file), ds_info["output"])
        elif ds_info["format"] == "belle":
            count = convert_belle(str(temp_file), ds_info["output"])
        else:
            print(f"  未知格式: {ds_info['format']}")
            continue

        total_count += count

        # 删除临时文件
        temp_file.unlink()
        print(f"  清理临时文件")

    # 删除临时目录
    try:
        temp_dir.rmdir()
    except:
        pass

    print()
    print("=" * 60)
    print(f"下载完成！总计: {total_count:,} 条")
    print("=" * 60)

if __name__ == "__main__":
    main()
