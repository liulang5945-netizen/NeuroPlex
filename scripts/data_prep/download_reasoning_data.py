"""
下载推理和知识数据集
====================
补充推理数学和知识问答数据
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

def main():
    print("=" * 60)
    print("下载推理和知识数据集")
    print("=" * 60)

    os.makedirs("taiji_data/training_data", exist_ok=True)

    datasets = [
        {
            "name": "GSM8K (数学推理)",
            "url": "https://huggingface.co/datasets/gsm8k/resolve/main/main/train.jsonl",
            "output": "taiji_data/training_data/gsm8k.jsonl",
            "format": "gsm8k",
        },
        {
            "name": "Dolly 15K (知识问答)",
            "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
            "output": "taiji_data/training_data/dolly_15k.jsonl",
            "format": "dolly",
        },
        {
            "name": "SQuAD (阅读理解)",
            "url": "https://huggingface.co/datasets/rajpurkar/squad/resolve/main/data/train-v2.0.json",
            "output": "taiji_data/training_data/squad.jsonl",
            "format": "squad",
            "max_samples": 50000,
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

        temp_file = output_file + ".tmp"
        if not download_file(ds["url"], temp_file):
            continue

        print(f"  转换格式...")
        count = 0

        try:
            if ds["format"] == "gsm8k":
                with open(temp_file, encoding="utf-8") as f_in, \
                     open(output_file, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        try:
                            item = json.loads(line)
                            question = item.get("question", "")
                            answer = item.get("answer", "")

                            if not question or not answer:
                                continue

                            if "####" in answer:
                                parts = answer.split("####")
                                reasoning = parts[0].strip()
                                final_answer = parts[1].strip()
                                full_answer = f"{reasoning}\n\n最终答案：{final_answer}"
                            else:
                                full_answer = answer

                            messages = [
                                {"role": "system", "content": "你是态极，一个擅长数学推理的AI助手。"},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": full_answer}
                            ]

                            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                            count += 1
                        except json.JSONDecodeError:
                            continue

            elif ds["format"] == "dolly":
                with open(temp_file, encoding="utf-8") as f_in, \
                     open(output_file, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        try:
                            item = json.loads(line)
                            instruction = item.get("instruction", "")
                            context = item.get("context", "")
                            response = item.get("response", "")

                            if not instruction or not response:
                                continue

                            user_content = instruction
                            if context:
                                user_content = f"背景信息：{context}\n\n问题：{instruction}"

                            messages = [
                                {"role": "system", "content": "你是态极，一个知识渊博的AI助手。"},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": response}
                            ]

                            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                            count += 1
                        except json.JSONDecodeError:
                            continue

            elif ds["format"] == "squad":
                with open(temp_file, encoding="utf-8") as f:
                    data = json.load(f)

                with open(output_file, "w", encoding="utf-8") as f:
                    for article in data.get("data", []):
                        for paragraph in article.get("paragraphs", []):
                            context = paragraph.get("context", "")
                            for qa in paragraph.get("qas", []):
                                question = qa.get("question", "")
                                answers = qa.get("answers", [])

                                if not question or not answers:
                                    continue

                                answer_text = answers[0].get("text", "")

                                messages = [
                                    {"role": "system", "content": "你是态极，一个擅长阅读理解的AI助手。"},
                                    {"role": "user", "content": f"阅读以下文本并回答问题：\n\n{context}\n\n问题：{question}"},
                                    {"role": "assistant", "content": answer_text}
                                ]

                                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                                count += 1

                                if count >= ds.get("max_samples", 999999):
                                    break
                            if count >= ds.get("max_samples", 999999):
                                break
                        if count >= ds.get("max_samples", 999999):
                            break

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
