"""
下载复杂行为训练数据 (工具调用与长上下文推理)
==============================================
解决问题：
1. 工具调用仅停留在"初步学习"（格式学习，缺乏深度规划和纠错）。
2. 长对话仅停留在"学习格式"（简单的多轮问答，缺乏真正的长上下文依赖和逻辑穿透）。

使用高质量开源数据集：
- 工具调用：THUDM/AgentInstruct (包含复杂的Thought-Action-Observation轨迹)
- 长上下文：THUDM/LongAlign-10k (真正的长文档、长依赖多轮对话)
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
                    if total_size > 0 and downloaded % (5*1024*1024) == 0:  # 每5MB打印一次
                        percent = (downloaded / total_size) * 100
                        print(f"    进度: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)")

            print(f"  下载完成: {downloaded / (1024*1024):.1f} MB")
            return True

        except Exception as e:
            print(f"  下载失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return False

def convert_agent_instruct(input_file, output_file):
    """转换 AgentInstruct 格式为标准 messages 格式"""
    count = 0
    with open(input_file, encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                item = json.loads(line)
                conversations = item.get("conversations", [])
                
                messages = [{"role": "system", "content": "你是态极，一个具备高级工具调用(ReAct)和规划能力的AI Agent。"}]
                
                valid = True
                for turn in conversations:
                    role = "user" if turn.get("from") == "user" else "assistant"
                    content = turn.get("value", "")
                    if not content:
                        valid = False
                        break
                    messages.append({"role": role, "content": content})
                    
                if valid and len(messages) > 1:
                    f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    count += 1
            except json.JSONDecodeError:
                continue
    return count

def convert_long_align(input_file, output_file, max_samples=2000):
    """转换 LongAlign 格式，限制数量以控制训练时间"""
    count = 0
    with open(input_file, encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if count >= max_samples:
                break
            try:
                item = json.loads(line)
                messages = item.get("messages", [])
                
                if not messages:
                    continue
                    
                # 确保有 system prompt
                if messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": "你是态极，一个擅长处理超长上下文和复杂多轮逻辑的AI助手。"})
                else:
                    messages[0]["content"] = "你是态极，一个擅长处理超长上下文和复杂多轮逻辑的AI助手。"
                
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
            except json.JSONDecodeError:
                continue
    return count

def main():
    print("=" * 60)
    print("下载复杂行为训练数据 (Tool Use & Long Context)")
    print("=" * 60)

    output_dir = Path("taiji_data/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_complex"
    temp_dir.mkdir(exist_ok=True)

    datasets = [
        {
            "name": "AgentInstruct (复杂ReAct工具调用与规划轨迹)",
            "url": "https://huggingface.co/datasets/THUDM/AgentInstruct/resolve/main/data/agent_instruct_zh.jsonl",
            "output": str(output_dir / "agent_instruct_zh.jsonl"),
            "format": "agent_instruct",
        },
        {
            "name": "LongAlign-10k (真正的长上下文依赖对话)",
            "url": "https://huggingface.co/datasets/THUDM/LongAlign-10k/resolve/main/data/train.jsonl",
            "output": str(output_dir / "long_align_train.jsonl"),
            "format": "long_align",
            "max_samples": 2000, # 长文本非常占显存，先取2000条高质量数据
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
            if ds["format"] == "agent_instruct":
                count = convert_agent_instruct(str(temp_file), ds["output"])
            elif ds["format"] == "long_align":
                count = convert_long_align(str(temp_file), ds["output"], ds.get("max_samples"))
            
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
    print(f"全部处理完成！本次累计准备: {total_count:,} 条复杂行为数据。")
    print("=" * 60)

if __name__ == "__main__":
    main()
