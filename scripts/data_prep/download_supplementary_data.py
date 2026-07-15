"""
态极训练数据补充脚本 (兼容版)
============================
使用 huggingface_hub 直接下载,避免 datasets 库兼容性问题。

用法:
  python scripts/download_supplementary_data.py --all
  python scripts/download_supplementary_data.py --math --code --safety
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("DataSupplement")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "taiji_data" / "training_data"
SUPPLEMENT_DIR = DATA_DIR / "supplementary"
SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  保存 {len(data)} 条 -> {filepath}")


def to_messages(system, user, assistant):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def download_json_from_hf(repo_id, filename, repo_type="dataset"):
    """从 HuggingFace 下载单个 JSON/JSONL 文件"""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # 尝试按行解析 (JSONL)
        data = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if data:
            return data
        # 尝试整体解析 (JSON)
        return json.loads(content)
    except Exception as e:
        logger.warning(f"  下载失败 {repo_id}/{filename}: {e}")
        return []


# ============================================================
# 1. 数学推理数据
# ============================================================
def download_math(sample_limit=None):
    logger.info("=" * 60)
    logger.info("下载数学推理数据...")
    all_data = []

    # GSM8K - 直接下载 parquet 并转换
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet",
                               repo_type="dataset")
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            q = str(row.get("question", ""))
            a = str(row.get("answer", ""))
            if "####" in a:
                reasoning = a.split("####")[0].strip()
                final_answer = a.split("####")[-1].strip()
            else:
                reasoning = ""
                final_answer = a
            text = f"让我一步步思考这个问题。\n\n{reasoning}\n\n最终答案是：{final_answer}" if reasoning else final_answer
            all_data.append(to_messages("你是一个数学推理助手。", q, text))
        logger.info(f"  GSM8K: {len(df)} 条")
    except Exception as e:
        logger.warning(f"  GSM8K 下载失败: {e}")

    # GSM8K_zh
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="meta-math/GSM8K_zh", filename="train.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            q = item.get("question", "")
            a = item.get("answer", "")
            if q and a:
                all_data.append(to_messages("你是一个数学推理助手，请用中文回答。", q, a))
        logger.info(f"  GSM8K_zh: {len(items)} 条")
    except Exception as e:
        logger.warning(f"  GSM8K_zh 下载失败: {e}")

    # UltraData-Math L1 (网页数学语料)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="openbmb/UltraData-Math",
                               filename="UltraData-Math-L1/train-00000-of-00001.parquet",
                               repo_type="dataset")
        import pandas as pd
        df = pd.read_parquet(path)
        count = 0
        limit = sample_limit or 30000
        for _, row in df.iterrows():
            if count >= limit:
                break
            text = str(row.get("text", row.get("content", "")))
            if len(text) > 100:
                all_data.append({"messages": [
                    {"role": "user", "content": "请阅读并理解以下数学相关内容。"},
                    {"role": "assistant", "content": text[:4000]}
                ]})
                count += 1
        logger.info(f"  UltraData-Math-L1: {count} 条")
    except Exception as e:
        logger.warning(f"  UltraData-Math 下载失败: {e}")

    if sample_limit and len(all_data) > sample_limit:
        import random
        random.shuffle(all_data)
        all_data = all_data[:sample_limit]

    save_jsonl(all_data, SUPPLEMENT_DIR / "math_reasoning.jsonl")
    return all_data


# ============================================================
# 2. 代码数据
# ============================================================
def download_code(sample_limit=None):
    logger.info("=" * 60)
    logger.info("下载代码数据...")
    all_data = []

    # CodeAlpaca-20K
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="sahil2801/CodeAlpaca-20k", filename="data.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            q = f"{inst}\n{inp}".strip() if inp else inst
            if q and out:
                all_data.append(to_messages("你是一个编程助手。", q, out))
        logger.info(f"  CodeAlpaca-20k: {len(items)} 条")
    except Exception as e:
        logger.warning(f"  CodeAlpaca 下载失败: {e}")

    # OpenHermes 代码部分 (已有,补充更多)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="teknium/OpenHermes-2.5",
                               filename="openhermes2_5.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        count = 0
        limit = sample_limit or 30000
        for item in items:
            if count >= limit:
                break
            conv = item.get("conversations", [])
            if len(conv) >= 2:
                user_msg = ""
                asst_msg = ""
                for c in conv:
                    if c.get("from") == "human":
                        user_msg = c.get("value", "")
                    elif c.get("from") == "gpt":
                        asst_msg = c.get("value", "")
                if user_msg and asst_msg and any(kw in user_msg.lower() for kw in
                    ["code", "program", "function", "python", "java", "script", "编程", "代码", "函数"]):
                    all_data.append(to_messages("你是一个编程助手。", user_msg, asst_msg))
                    count += 1
        logger.info(f"  OpenHermes (code): {count} 条")
    except Exception as e:
        logger.warning(f"  OpenHermes 下载失败: {e}")

    if sample_limit and len(all_data) > sample_limit:
        import random
        random.shuffle(all_data)
        all_data = all_data[:sample_limit]

    save_jsonl(all_data, SUPPLEMENT_DIR / "code_supplementary.jsonl")
    return all_data


# ============================================================
# 3. 安全对齐数据
# ============================================================
def download_safety():
    logger.info("=" * 60)
    logger.info("下载安全对齐数据...")
    all_data = []

    # Anthropic HH-RLHF (采样)
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        # 下载前几个parquet分片
        for shard in range(3):
            try:
                filename = f"data/train-{shard:05d}-of-00042.parquet"
                path = hf_hub_download(repo_id="Anthropic/hh-rlhf", filename=filename,
                                       repo_type="dataset")
                df = pd.read_parquet(path)
                for _, row in df.iterrows():
                    chosen = str(row.get("chosen", ""))
                    if "Assistant:" in chosen:
                        parts = chosen.split("\n\nAssistant: ")
                        if len(parts) >= 2:
                            user_q = parts[0].replace("Human: ", "").strip()
                            assistant_a = parts[-1].strip()
                            if user_q and assistant_a:
                                all_data.append(to_messages("你是一个安全、有帮助的AI助手。", user_q, assistant_a))
                logger.info(f"  HH-RLHF shard {shard}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"  HH-RLHF shard {shard} 下载失败: {e}")
                continue
    except Exception as e:
        logger.warning(f"  HH-RLHF 下载失败: {e}")

    # Alpaca-cleaned
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="yahma/alpaca-cleaned", filename="alpaca_data_cleaned.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        count = 0
        for item in items:
            if count >= 15000:
                break
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            q = f"{inst}\n{inp}".strip() if inp else inst
            if q and out and len(out) > 30:
                all_data.append(to_messages("你是一个安全、有帮助的AI助手。", q, out))
                count += 1
        logger.info(f"  Alpaca-cleaned: {count} 条")
    except Exception as e:
        logger.warning(f"  Alpaca-cleaned 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "safety_alignment.jsonl")
    return all_data


# ============================================================
# 4. 科学知识数据
# ============================================================
def download_science():
    logger.info("=" * 60)
    logger.info("下载科学知识数据...")
    all_data = []

    # SciQ
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="allenai/sciq", filename="data/validation-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            q = str(row.get("question", ""))
            correct = str(row.get("correct_answer", ""))
            support = str(row.get("support", ""))
            if q and correct:
                answer = f"{correct}\n\n参考：{support}" if support else correct
                all_data.append(to_messages("你是一个科学知识助手。", q, answer))
        logger.info(f"  SciQ: {len(df)} 条")
    except Exception as e:
        logger.warning(f"  SciQ 下载失败: {e}")

    # ARC
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="allenai/ai2_arc", filename="ARC-Easy/train-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            q = str(row.get("question", ""))
            choices = row.get("choices", {})
            answer = str(row.get("answerKey", ""))
            if q and choices:
                labels = choices.get("label", []) if isinstance(choices, dict) else []
                texts = choices.get("text", []) if isinstance(choices, dict) else []
                choice_text = " ".join([f"({l}) {t}" for l, t in zip(labels, texts)])
                all_data.append(to_messages("你是一个科学推理助手。", f"{q}\n{choice_text}", f"答案是：{answer}"))
        logger.info(f"  ARC-Easy: {len(df)} 条")
    except Exception as e:
        logger.warning(f"  ARC 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "science_knowledge.jsonl")
    return all_data


# ============================================================
# 5. 长文档理解
# ============================================================
def download_long_context():
    logger.info("=" * 60)
    logger.info("下载长文档理解数据...")
    all_data = []

    # CNN/DailyMail
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        for shard in range(2):
            try:
                filename = f"data/train-{shard:05d}-of-00030.parquet"
                path = hf_hub_download(repo_id="ccdv/cnn_dailymail", filename=filename,
                                       repo_type="dataset")
                df = pd.read_parquet(path)
                for _, row in df.iterrows():
                    article = str(row.get("article", ""))
                    summary = str(row.get("highlights", ""))
                    if article and summary and len(article) > 300:
                        all_data.append(to_messages(
                            "你是一个文本摘要助手。",
                            f"请为以下文章生成摘要：\n\n{article[:3000]}",
                            summary
                        ))
                logger.info(f"  CNN/DailyMail shard {shard}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"  CNN shard {shard} 下载失败: {e}")
                continue
    except Exception as e:
        logger.warning(f"  CNN/DailyMail 下载失败: {e}")

    # SQuAD
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="rajpurkar/squad", filename="data/train-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        count = 0
        for _, row in df.iterrows():
            if count >= 30000:
                break
            context = str(row.get("context", ""))
            question = str(row.get("question", ""))
            answers = row.get("answers", {})
            answer_text = ""
            if isinstance(answers, dict):
                texts = answers.get("text", [])
                answer_text = texts[0] if texts else ""
            if context and question and answer_text:
                all_data.append(to_messages(
                    "你是一个阅读理解助手。",
                    f"阅读以下段落并回答问题：\n\n{context}\n\n问题：{question}",
                    answer_text
                ))
                count += 1
        logger.info(f"  SQuAD: {count} 条")
    except Exception as e:
        logger.warning(f"  SQuAD 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "long_context.jsonl")
    return all_data


# ============================================================
# 6. 生命体数据扩充
# ============================================================
def download_lifeform():
    logger.info("=" * 60)
    logger.info("扩充生命体数据...")
    all_data = []

    # EmpatheticDialogues
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="empathetic_dialogues", filename="train.csv",
                               repo_type="dataset")
        import csv
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if count >= 10000:
                    break
                context = str(row.get("context", ""))
                utterance = str(row.get("utterance", ""))
                emotion = str(row.get("emotion", ""))
                if context and utterance:
                    all_data.append(to_messages(
                        f"你是一个有情感的AI生命体。当前用户情绪：{emotion}",
                        context, utterance
                    ))
                    count += 1
        logger.info(f"  EmpatheticDialogues: {count} 条")
    except Exception as e:
        logger.warning(f"  EmpatheticDialogues 下载失败: {e}")

    # DailyDialog (日常对话)
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="daily_dialog", filename="data/train-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        count = 0
        for _, row in df.iterrows():
            if count >= 10000:
                break
            dialog = row.get("dialog", [])
            if isinstance(dialog, list) and len(dialog) >= 2:
                for i in range(0, len(dialog) - 1, 2):
                    if i + 1 < len(dialog):
                        user_q = str(dialog[i])
                        asst_a = str(dialog[i + 1])
                        if user_q and asst_a:
                            all_data.append(to_messages(
                                "你是一个善解人意的AI生命体。",
                                user_q, asst_a
                            ))
                            count += 1
                            if count >= 10000:
                                break
        logger.info(f"  DailyDialog: {count} 条")
    except Exception as e:
        logger.warning(f"  DailyDialog 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "lifeform_expanded.jsonl")
    return all_data


# ============================================================
# 7. 中文高质量数据
# ============================================================
def download_chinese(sample_limit=None):
    logger.info("=" * 60)
    logger.info("下载中文高质量数据...")
    all_data = []

    # BELLE 2M CN (采样)
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        limit = sample_limit or 30000
        for shard in range(5):
            try:
                filename = f"data/train-{shard:05d}-of-00034.parquet"
                path = hf_hub_download(repo_id="BelleGroup/train_2M_CN", filename=filename,
                                       repo_type="dataset")
                df = pd.read_parquet(path)
                count = 0
                for _, row in df.iterrows():
                    if len(all_data) >= limit:
                        break
                    inst = str(row.get("instruction", ""))
                    inp = str(row.get("input", ""))
                    out = str(row.get("output", ""))
                    q = f"{inst}\n{inp}".strip() if inp and inp != "nan" else inst
                    if q and out and out != "nan":
                        all_data.append(to_messages("", q, out))
                        count += 1
                logger.info(f"  BELLE shard {shard}: {count} 条")
                if len(all_data) >= limit:
                    break
            except Exception as e:
                logger.warning(f"  BELLE shard {shard} 下载失败: {e}")
                continue
    except Exception as e:
        logger.warning(f"  BELLE 下载失败: {e}")

    if sample_limit and len(all_data) > sample_limit:
        all_data = all_data[:sample_limit]

    save_jsonl(all_data, SUPPLEMENT_DIR / "chinese_pretrain.jsonl")
    return all_data


# ============================================================
# 8. 多语言数据
# ============================================================
def download_multilingual():
    logger.info("=" * 60)
    logger.info("下载多语言数据...")
    all_data = []

    pairs = ["en-ja", "en-ko", "en-fr", "en-de"]
    for pair in pairs:
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            lang = pair.split("-")[1]
            path = hf_hub_download(repo_id="Helsinki-NLP/opus-100",
                                   filename=f"data/{pair}/train-00000-of-00001.parquet",
                                   repo_type="dataset")
            df = pd.read_parquet(path)
            count = 0
            for _, row in df.iterrows():
                if count >= 5000:
                    break
                translation = row.get("translation", {})
                if isinstance(translation, dict):
                    src = translation.get("en", "")
                    tgt = translation.get(lang, "")
                    if src and tgt:
                        all_data.append(to_messages(
                            f"你是一个翻译助手，请将以下内容翻译为{lang}。",
                            src, tgt
                        ))
                        count += 1
            logger.info(f"  OPUS-100 {pair}: {count} 条")
        except Exception as e:
            logger.warning(f"  OPUS-100 {pair} 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "multilingual.jsonl")
    return all_data


# ============================================================
# 9. Agent/工具调用数据
# ============================================================
def download_agent():
    logger.info("=" * 60)
    logger.info("下载Agent/工具调用数据...")
    all_data = []

    # Gorilla APIBench
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download(repo_id="gorilla-llm/APIBench", filename="data/torch_hub.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        count = 0
        for _, row in df.iterrows():
            if count >= 10000:
                break
            q = str(row.get("prompt", row.get("question", "")))
            a = str(row.get("response", row.get("answer", "")))
            if q and a:
                all_data.append(to_messages("你是一个API调用助手。", q, a))
                count += 1
        logger.info(f"  Gorilla APIBench: {count} 条")
    except Exception as e:
        logger.warning(f"  Gorilla 下载失败: {e}")

    # ToolBench (如果有)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="ToolBench/ToolBench_Instruct",
                               filename="data/train.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        count = 0
        for item in items[:10000]:
            q = item.get("query", item.get("instruction", ""))
            a = item.get("answer", item.get("output", ""))
            if q and a:
                all_data.append(to_messages("你是一个能使用工具的AI助手。", q, a))
                count += 1
        logger.info(f"  ToolBench: {count} 条")
    except Exception as e:
        logger.warning(f"  ToolBench 下载失败: {e}")

    save_jsonl(all_data, SUPPLEMENT_DIR / "agent_tool_calling.jsonl")
    return all_data


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="态极训练数据补充")
    parser.add_argument("--all", action="store_true", help="下载所有补充数据")
    parser.add_argument("--math", action="store_true", help="数学推理数据")
    parser.add_argument("--code", action="store_true", help="代码数据")
    parser.add_argument("--agent", action="store_true", help="Agent/工具调用数据")
    parser.add_argument("--safety", action="store_true", help="安全对齐数据")
    parser.add_argument("--chinese", action="store_true", help="中文预训练数据")
    parser.add_argument("--multilingual", action="store_true", help="多语言数据")
    parser.add_argument("--science", action="store_true", help="科学知识数据")
    parser.add_argument("--long-context", action="store_true", help="长文档理解数据")
    parser.add_argument("--lifeform", action="store_true", help="生命体数据扩充")
    parser.add_argument("--sample", type=int, default=None, help="每个数据源最大采样数")
    args = parser.parse_args()

    if not any([args.all, args.math, args.code, args.agent, args.safety,
                args.chinese, args.multilingual, args.science, args.long_context,
                args.lifeform]):
        parser.print_help()
        return

    total = 0
    if args.all or args.math:
        total += len(download_math(args.sample))
    if args.all or args.code:
        total += len(download_code(args.sample))
    if args.all or args.agent:
        total += len(download_agent())
    if args.all or args.safety:
        total += len(download_safety())
    if args.all or args.chinese:
        total += len(download_chinese(args.sample))
    if args.all or args.multilingual:
        total += len(download_multilingual())
    if args.all or args.science:
        total += len(download_science())
    if args.all or args.long_context:
        total += len(download_long_context())
    if args.all or args.lifeform:
        total += len(download_lifeform())

    logger.info("=" * 60)
    logger.info(f"补充数据下载完成! 共 {total} 条")
    logger.info(f"数据保存在: {SUPPLEMENT_DIR}")


if __name__ == "__main__":
    main()
