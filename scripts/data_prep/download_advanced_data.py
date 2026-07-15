"""
下载高级/高质量训练数据
=======================
补充高质量的中文逻辑推理 (COIG-CQIA) 和英文多轮对话/推理 (OpenHermes) 数据
"""
import os
import json
import requests
import time
from pathlib import Path

def download_file(url, output_path, max_retries=3):
    """下载文件，带重试功能和进度显示"""
    for attempt in range(max_retries):
        try:
            print(f"  下载中 (尝试 {attempt + 1})...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10*1024*1024) == 0:  # 每10MB打印一次
                        percent = (downloaded / total_size) * 100
                        print(f"    进度: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)")

            print(f"  下载完成: {downloaded / (1024*1024):.1f} MB")
            return True

        except Exception as e:
            print(f"  下载失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return False

def convert_coig_cqia(input_file, output_file, max_samples=None):
    """转换 COIG-CQIA 格式"""
    count = 0
    with open(input_file, encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if max_samples and count >= max_samples:
                break
            try:
                item = json.loads(line)
                
                # COIG-CQIA 通常包含 instruction, input, output
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                
                if not instruction and "query" in item:
                    instruction = item["query"]
                    output_text = item.get("response", "")

                if not instruction or not output_text:
                    continue

                user_content = instruction
                if input_text:
                    user_content += "\n" + input_text

                messages = [
                    {"role": "system", "content": "你是态极，一个严谨且富有逻辑的AI助手。"},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output_text}
                ]

                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
            except json.JSONDecodeError:
                continue
    return count

def convert_sharegpt(input_file, output_file, max_samples=None):
    """转换 ShareGPT/OpenHermes 格式"""
    count = 0
    with open(input_file, encoding="utf-8") as f_in:
        try:
            # 如果是 JSON 数组
            data = json.load(f_in)
        except json.JSONDecodeError:
            # 可能是 JSONL
            f_in.seek(0)
            data = [json.loads(line) for line in f_in]

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in data:
            if max_samples and count >= max_samples:
                break
            
            conversations = item.get("conversations", [])
            if not conversations or len(conversations) < 2:
                continue
                
            messages = [{"role": "system", "content": "你是态极，一个有帮助的AI助手。"}]
            
            valid = True
            for turn in conversations:
                role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                content = turn.get("value", "")
                if not content:
                    valid = False
                    break
                messages.append({"role": role, "content": content})
                
            if valid and len(messages) > 1:
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
                
    return count

def main():
    print("=" * 60)
    print("下载高级/高质量训练数据")
    print("=" * 60)

    output_dir = Path("taiji_data/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    datasets = [
        {
            "name": "COIG-CQIA (弱智吧/逻辑陷阱)",
            "url": "https://huggingface.co/datasets/m-a-p/COIG-CQIA/resolve/main/ruozhiba/ruozhiba_ruozhiba.jsonl",
            "output": str(output_dir / "coig_ruozhiba.jsonl"),
            "format": "coig",
        },
        {
            "name": "COIG-CQIA (逻辑问答)",
            "url": "https://huggingface.co/datasets/m-a-p/COIG-CQIA/resolve/main/logi_qa/logi-qa.jsonl",
            "output": str(output_dir / "coig_logi_qa.jsonl"),
            "format": "coig",
        },
        {
            "name": "COIG-CQIA (考研/考试)",
            "url": "https://huggingface.co/datasets/m-a-p/COIG-CQIA/resolve/main/exam/kaoyan.jsonl",
            "output": str(output_dir / "coig_kaoyan.jsonl"),
            "format": "coig",
        },
        {
            "name": "OpenHermes 2.5 (高质量对话与推理)",
            "url": "https://huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json",
            "output": str(output_dir / "openhermes_2_5.jsonl"),
            "format": "sharegpt",
            "max_samples": 50000, # 限制数量，OpenHermes很大(1M+)，这里取5万条作为补充
        },
    ]

    total_count = 0

    for ds in datasets:
        print()
        print(f"数据集: {ds['name']}")
        print("-" * 40)

        if os.path.exists(ds["output"]):
            with open(ds["output"], encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            print(f"  已存在: {existing:,} 条")
            total_count += existing
            continue

        temp_file = temp_dir / Path(ds["url"]).name
        if not download_file(ds["url"], str(temp_file)):
            continue

        print(f"  转换格式...")
        count = 0
        try:
            if ds["format"] == "coig":
                count = convert_coig_cqia(str(temp_file), ds["output"], ds.get("max_samples"))
            elif ds["format"] == "sharegpt":
                count = convert_sharegpt(str(temp_file), ds["output"], ds.get("max_samples"))
            
            print(f"  完成: {count:,} 条")
            total_count += count
        except Exception as e:
            print(f"  转换失败: {e}")
        finally:
            if temp_file.exists():
                temp_file.unlink()

    try:
        temp_dir.rmdir()
    except:
        pass

    print()
    print("=" * 60)
    print(f"全部处理完成！本次累计准备: {total_count:,} 条高质量数据。")
    print("=" * 60)

if __name__ == "__main__":
    main()
