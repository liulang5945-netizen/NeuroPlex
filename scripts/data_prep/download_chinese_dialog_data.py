"""
下载高质量中文对话数据集
========================
从 HuggingFace 下载常用的中文对话数据集

数据集来源:
1. BELLE-2M - 中文指令数据 (200万条)
2. Firefly - 中文对话数据 (100万条)
3. ShareGPT-Chinese - 中文ShareGPT数据
4. MOSS-SFT - 中文对话数据
"""
import os
import json
import subprocess
import sys
from pathlib import Path

# 数据集配置
DATASETS = [
    {
        "name": "BELLE-2M",
        "repo": "BelleGroup/train_2M_CN",
        "filename": "train_2M_CN.json",
        "description": "中文指令数据 200万条",
        "size": "~1.5GB",
        "format": "alpaca",  # instruction/input/output
    },
    {
        "name": "BELLE-1M",
        "repo": "BelleGroup/train_1M_CN",
        "filename": "train_1M_CN.json",
        "description": "中文指令数据 100万条",
        "size": "~800MB",
        "format": "alpaca",
    },
    {
        "name": "BELLE-0.5M",
        "repo": "BelleGroup/train_0.5M_CN",
        "filename": "train_0.5M_CN.json",
        "description": "中文指令数据 50万条",
        "size": "~400MB",
        "format": "alpaca",
    },
    {
        "name": "Firefly",
        "repo": "YeungNLP/firefly-train-1.1M",
        "filename": "data.jsonl",
        "description": "中文对话数据 110万条",
        "size": "~1.2GB",
        "format": "messages",
    },
    {
        "name": "MOSS-SFT",
        "repo": "fnlp/moss-003-sft-data",
        "filename": "moss-003-sft-no-tools.jsonl",
        "description": "中文对话数据",
        "size": "~500MB",
        "format": "messages",
    },
]

def download_file_from_hf(repo_id, filename, output_dir):
    """从 HuggingFace 下载文件"""
    try:
        # 使用 huggingface-cli 下载
        cmd = [
            "huggingface-cli", "download",
            repo_id,
            filename,
            "--repo-type", "dataset",
            "--local-dir", output_dir
        ]
        print(f"  执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            print(f"  下载成功!")
            return True
        else:
            print(f"  下载失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  下载超时")
        return False
    except Exception as e:
        print(f"  下载错误: {e}")
        return False

def convert_alpaca_to_messages(input_path, output_path):
    """将 Alpaca 格式转换为 messages 格式"""
    print(f"  转换 Alpaca 格式到 messages 格式...")

    converted = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        # 检查是否是JSON数组
        first_char = fin.read(1)
        fin.seek(0)

        if first_char == '[':
            # JSON数组格式
            data = json.load(fin)
            items = data
        else:
            # JSONL格式
            items = [json.loads(line) for line in fin if line.strip()]

        for item in items:
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

    return converted

def merge_to_training_dir(source_files, target_dir):
    """合并文件到训练目录"""
    target_file = os.path.join(target_dir, "chinese_dialog_merged.jsonl")

    with open(target_file, 'w', encoding='utf-8') as fout:
        total = 0
        for source_file in source_files:
            if not os.path.exists(source_file):
                continue

            print(f"  合并: {os.path.basename(source_file)}")
            with open(source_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
                    total += 1

    print(f"  合并完成: {total} 条 -> {target_file}")
    return target_file

def main():
    print("=" * 60)
    print("下载高质量中文对话数据集")
    print("=" * 60)

    # 创建下载目录
    download_dir = Path("taiji_data/training_data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    target_dir = Path("taiji_data/training_data/sft")
    downloaded_files = []

    # 选择要下载的数据集
    print("\n可用数据集:")
    for i, ds in enumerate(DATASETS):
        print(f"  {i+1}. {ds['name']} - {ds['description']} ({ds['size']})")

    print("\n推荐下载:")
    print("  - BELLE-0.5M (50万条，适合快速验证)")
    print("  - Firefly (110万条，高质量)")

    # 自动下载 BELLE-0.5M (较小的数据集)
    print("\n" + "-" * 60)
    print("下载 BELLE-0.5M (50万条中文指令数据)")
    print("-" * 60)

    belle_ds = DATASETS[2]  # BELLE-0.5M
    belle_file = download_dir / belle_ds["filename"]

    if not belle_file.exists():
        success = download_file_from_hf(
            belle_ds["repo"],
            belle_ds["filename"],
            str(download_dir)
        )
        if not success:
            print("下载失败，请手动下载或检查网络连接")
            print(f"huggingface-cli download {belle_ds['repo']} {belle_ds['filename']} --repo-type dataset --local-dir {download_dir}")
    else:
        print(f"文件已存在: {belle_file}")

    # 转换格式
    if belle_file.exists():
        belle_messages_file = download_dir / "belle_0.5m_messages.jsonl"
        if not belle_messages_file.exists():
            converted = convert_alpaca_to_messages(str(belle_file), str(belle_messages_file))
            print(f"转换完成: {converted} 条")
        else:
            print(f"转换文件已存在: {belle_messages_file}")
        downloaded_files.append(str(belle_messages_file))

    # 合并到训练目录
    if downloaded_files:
        print("\n" + "-" * 60)
        print("合并到训练目录")
        print("-" * 60)
        merged_file = merge_to_training_dir(downloaded_files, str(target_dir))

        # 统计最终数据量
        with open(merged_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"\n最终文件: {merged_file}")
        print(f"数据量: {line_count} 条")

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查下载的数据质量")
    print("2. 继续下载其他数据集 (如需要)")
    print("3. 合并所有数据进行训练")

if __name__ == '__main__':
    main()
