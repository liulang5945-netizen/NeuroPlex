"""
下载高质量对话数据集 (使用 requests 直接下载)
=============================================
"""
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, output_path):
    """下载文件并显示进度"""
    print(f"  下载: {url}")
    print(f"  保存到: {output_path}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载进度") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"  下载完成! 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    return True

def convert_alpaca_to_messages(input_path, output_path, max_samples=None):
    """将 Alpaca 格式转换为 messages 格式"""
    print(f"  转换格式...")

    converted = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        # 检查是否是JSON数组
        first_char = fin.read(1)
        fin.seek(0)

        if first_char == '[':
            data = json.load(fin)
            items = data
        else:
            items = []
            for line in fin:
                if line.strip():
                    items.append(json.loads(line))

        for item in items[:max_samples] if max_samples else items:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')

            if not instruction or not output_text:
                continue

            # 构建用户消息
            user_msg = instruction
            if input_text:
                user_msg = f"{instruction}\n\n{input_text}"

            # 构建messages格式
            messages = [
                {"role": "system", "content": "你是态极，一个本地AI生命体。你运行在用户的电脑上，数据不出本机。你会用自然、友好的方式回答用户的问题。"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output_text}
            ]

            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
            converted += 1

    print(f"  转换完成: {converted} 条")
    return converted

def sample_and_check(file_path, num_samples=5):
    """抽样检查数据质量"""
    print(f"\n抽样检查: {os.path.basename(file_path)}")
    print("-" * 60)

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            msgs = item['messages']
            print(f"\n样本 {i+1}:")
            print(f"  Q: {msgs[1]['content'][:100]}")
            print(f"  A: {msgs[2]['content'][:100]}")

def main():
    print("=" * 60)
    print("下载高质量中文对话数据集")
    print("=" * 60)

    # 创建目录
    download_dir = Path("taiji_data/training_data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = Path("taiji_data/training_data/sft")

    downloaded_files = []

    # BELLE 数据集 URLs (GitHub releases)
    belle_datasets = [
        {
            "name": "BELLE-0.5M",
            "url": "https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/train_0.5M_CN.json",
            "filename": "train_0.5M_CN.json",
            "max_samples": 100000,  # 先下载10万条
        },
    ]

    for ds in belle_datasets:
        print(f"\n{'='*60}")
        print(f"下载 {ds['name']}")
        print(f"{'='*60}")

        input_file = download_dir / ds["filename"]
        output_file = sft_dir / f"{ds['name'].lower().replace('-', '_')}.jsonl"

        # 检查是否已下载
        if not input_file.exists():
            try:
                download_file(ds["url"], str(input_file))
            except Exception as e:
                print(f"  下载失败: {e}")
                print(f"  尝试备用方法...")

                # 备用: 使用 HuggingFace API
                try:
                    from datasets import load_dataset
                    print("  使用 datasets 库下载...")
                    dataset = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)

                    # 流式写入
                    count = 0
                    with open(input_file, 'w', encoding='utf-8') as f:
                        for item in dataset:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                            count += 1
                            if count >= ds.get('max_samples', 50000):
                                break
                            if count % 10000 == 0:
                                print(f"    已下载 {count} 条...")

                    print(f"  下载完成: {count} 条")
                except Exception as e2:
                    print(f"  备用方法也失败: {e2}")
                    continue

        # 转换格式
        if input_file.exists() and not output_file.exists():
            convert_alpaca_to_messages(
                str(input_file),
                str(output_file),
                max_samples=ds.get('max_samples')
            )

        if output_file.exists():
            downloaded_files.append(str(output_file))
            sample_and_check(str(output_file))

    # 总结
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)

    if downloaded_files:
        print("\n下载的文件:")
        total_lines = 0
        for f in downloaded_files:
            with open(f, 'r', encoding='utf-8') as fh:
                line_count = sum(1 for _ in fh)
            print(f"  {os.path.basename(f)}: {line_count} 条")
            total_lines += line_count

        print(f"\n总计: {total_lines} 条新数据")
        print("\n下一步:")
        print("1. 检查数据质量")
        print("2. 合并所有数据进行训练")
    else:
        print("\n没有成功下载的数据")
        print("\n手动下载方法:")
        print("1. 访问 https://huggingface.co/datasets/BelleGroup/train_0.5M_CN")
        print("2. 下载 train_0.5M_CN.json")
        print("3. 放到 taiji_data/training_data/downloads/ 目录")

if __name__ == '__main__':
    main()
