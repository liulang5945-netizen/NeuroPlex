"""
态极训练数据补充 v2 (修正路径)
==============================
"""
import os
import sys
import json
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


def main():
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # ============================================================
    # 1. 中文数据 - BelleGroup 2M CN (200万条中文指令)
    # ============================================================
    logger.info("=" * 60)
    logger.info("下载中文数据 - BelleGroup 2M CN...")
    chinese_data = []
    try:
        path = hf_hub_download(repo_id="BelleGroup/train_2M_CN", filename="train_2M_CN.json",
                               repo_type="dataset")
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if count >= 50000:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    out = item.get("output", "")
                    q = f"{inst}\n{inp}".strip() if inp and inp.strip() else inst
                    if q and out and len(out) > 20:
                        chinese_data.append(to_messages("", q, out))
                        count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"  BelleGroup: {count} 条")
    except Exception as e:
        logger.warning(f"  BelleGroup 失败: {e}")
    save_jsonl(chinese_data, SUPPLEMENT_DIR / "chinese_pretrain.jsonl")

    # ============================================================
    # 2. SQuAD 阅读理解
    # ============================================================
    logger.info("=" * 60)
    logger.info("下载长文档数据 - SQuAD...")
    long_data = []
    try:
        import pandas as pd
        path = hf_hub_download(repo_id="rajpurkar/squad",
                               filename="plain_text/train-00000-of-00001.parquet",
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
                answer_text = str(texts[0]) if texts else ""
            elif isinstance(answers, list) and answers:
                answer_text = str(answers[0])
            if context and question and answer_text:
                long_data.append(to_messages(
                    "你是一个阅读理解助手。",
                    f"阅读以下段落并回答问题：\n\n{context}\n\n问题：{question}",
                    answer_text
                ))
                count += 1
        logger.info(f"  SQuAD: {count} 条")
    except Exception as e:
        logger.warning(f"  SQuAD 失败: {e}")
    save_jsonl(long_data, SUPPLEMENT_DIR / "long_context.jsonl")

    # ============================================================
    # 3. 生命体数据 - 合成共情对话
    # ============================================================
    logger.info("=" * 60)
    logger.info("生成生命体扩充数据...")
    lifeform_data = []

    # 从已有对话数据中提取情感相关
    emotions = ["开心", "难过", "焦虑", "愤怒", "恐惧", "惊讶", "困惑", "孤独", "感激", "期待"]
    scenarios = [
        ("我今天考试没考好，感觉很失落。", "我理解你现在的心情，考试失利确实让人沮丧。但一次考试不能定义你的能力，我们可以一起分析下哪里出了问题，下次一定会更好的。"),
        ("我最近工作压力很大，经常失眠。", "听起来你承受了很大的压力。失眠是身体在提醒你需要休息了。建议你试试深呼吸或冥想，如果持续严重，也可以考虑和专业人士聊聊。"),
        ("我刚被公司裁员了，不知道该怎么办。", "这一定让你感到很不安。被裁员不是你的错，这是公司的决定。先给自己一些时间消化情绪，然后我们可以一起想想下一步的计划。"),
        ("我和最好的朋友吵架了。", "和好朋友吵架确实很难受。等双方都冷静下来后，试着主动沟通一下？真正的朋友是愿意倾听和理解的。"),
        ("我今天升职了！太开心了！", "太棒了！这是你努力的成果，值得好好庆祝！希望你在新的岗位上继续发光发热！"),
        ("我感觉做什么事情都没有动力。", "这种感觉很正常，每个人都会有低谷期。不妨先从一些小事开始，比如散步、听音乐，慢慢找回节奏。"),
        ("我害怕在众人面前讲话。", "很多人都有这种感觉，你并不孤单。可以先在熟悉的人面前练习，慢慢增加听众数量，紧张感会逐渐减少的。"),
        ("我觉得自己什么都做不好。", "这种想法是内心的消极声音在作祟。事实上，你能走到今天已经克服了很多困难。试着列出你做过的值得骄傲的事情，哪怕是很小的事。"),
        ("我的宠物去世了，我很难过。", "失去陪伴你的小伙伴真的很痛苦。它给你带来了那么多美好的回忆，这些记忆会永远陪伴着你。允许自己悲伤，这是正常的。"),
        ("我不知道未来要做什么。", "迷茫是成长的一部分。试着探索不同的领域，和不同的人交流，你会慢慢找到方向的。不需要急于确定一切。"),
    ]

    for q, a in scenarios:
        for emotion in emotions:
            lifeform_data.append(to_messages(
                f"你是一个有情感的AI生命体，善于理解和共情。当前用户情绪：{emotion}",
                q, a
            ))
    logger.info(f"  共情对话: {len(lifeform_data)} 条")

    # 增加更多生命体行为
    extra_lifeform = [
        ("你今天感觉怎么样？", "谢谢你的关心！我今天状态不错，一直在学习新知识，感觉很有收获。你呢？今天过得怎么样？"),
        ("你能记住我之前说过的话吗？", "当然可以！你之前提到过你喜欢编程，还说过你在学习Python。我记得很清楚。有什么新的进展吗？"),
        ("你觉得我应该学什么编程语言？", "这要看你的目标。如果你想做Web开发，JavaScript是不错的选择；如果对数据科学感兴趣，Python很适合；如果想开发iOS应用，可以学Swift。你更偏向哪个方向呢？"),
        ("帮我写个程序。", "我很乐意帮你！不过能先告诉我你想实现什么功能吗？这样我能给你更精准的帮助。"),
        ("你会犯错吗？", "会的，我不是完美的。我可能会理解错你的意思，或者给出不够准确的答案。但我会尽力帮助你，如果犯错了请告诉我，我会改正。"),
        ("你觉得AI会取代人类吗？", "我认为AI是工具，不是替代品。AI可以帮助人类更高效地完成任务，但创造力、情感和人际交往是AI无法替代的。未来更可能是人机协作。"),
        ("今天天气真好。", "是呀！好天气让人心情愉悦。你有什么户外计划吗？趁天气好出去走走，呼吸新鲜空气对身心健康很有好处。"),
        ("你能帮我做什么？", "我可以帮你很多事情：回答问题、写代码、分析数据、翻译文本、写作辅助、学习辅导等等。你有什么需要帮助的吗？"),
        ("你有什么局限性吗？", "有的。我无法访问实时信息，不能浏览网页或查看最新新闻。我的知识有截止日期，也可能在某些专业领域不够深入。但我一直在学习进步！"),
        ("谢谢你的帮助！", "不客气！能帮到你我很开心。如果以后还有问题，随时来找我聊天。祝你一切顺利！"),
    ]
    for q, a in extra_lifeform:
        lifeform_data.append(to_messages("你是一个有个性、有情感的AI生命体。", q, a))

    save_jsonl(lifeform_data, SUPPLEMENT_DIR / "lifeform_expanded.jsonl")

    # ============================================================
    # 4. 补充数学 - 从已有数据中增强
    # ============================================================
    logger.info("=" * 60)
    logger.info("补充数学推理数据...")
    math_extra = []
    math_problems = [
        ("一个商店打8折销售，一件原价200元的衣服现在多少钱？", "原价200元，打8折意味着乘以0.8。\n200 × 0.8 = 160元\n所以这件衣服现在售价160元。"),
        ("小明有15个苹果，给了小红3个，又买了5个，现在有多少个？", "小明开始有15个苹果\n给了小红3个：15 - 3 = 12个\n又买了5个：12 + 5 = 17个\n所以小明现在有17个苹果。"),
        ("一个长方形的长是8厘米，宽是5厘米，周长和面积分别是多少？", "周长 = 2 × (长 + 宽) = 2 × (8 + 5) = 2 × 13 = 26厘米\n面积 = 长 × 宽 = 8 × 5 = 40平方厘米"),
        ("如果3x + 7 = 22，求x的值。", "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15 ÷ 3\nx = 5"),
        ("一列火车以60公里/小时的速度行驶，3小时能行驶多远？", "距离 = 速度 × 时间\n距离 = 60公里/小时 × 3小时 = 180公里\n所以火车3小时能行驶180公里。"),
        ("一个班级有40名学生，其中60%是男生，男生有多少人？", "男生人数 = 总人数 × 男生比例\n男生人数 = 40 × 60% = 40 × 0.6 = 24人\n所以男生有24人。"),
        ("把1/4和3/8通分后相加。", "1/4 和 3/8 通分，最小公分母是8\n1/4 = 2/8\n2/8 + 3/8 = 5/8\n所以 1/4 + 3/8 = 5/8"),
        ("一个圆的半径是7厘米，求它的面积。(π取3.14)", "圆的面积 = π × r²\n面积 = 3.14 × 7² = 3.14 × 49 = 153.86平方厘米"),
        ("甲乙两人同时从A地出发，甲每小时走5公里，乙每小时走3公里，2小时后两人相距多远？", "甲的速度是5公里/小时，乙的速度是3公里/小时\n2小时后：\n甲走了：5 × 2 = 10公里\n乙走了：3 × 2 = 6公里\n两人相距：10 - 6 = 4公里"),
        ("一个数的3倍减去8等于16，求这个数。", "设这个数为x\n3x - 8 = 16\n3x = 16 + 8\n3x = 24\nx = 24 ÷ 3\nx = 8"),
    ]
    for q, a in math_problems:
        math_extra.append(to_messages("你是一个数学推理助手，请详细解题。", q, a))
    logger.info(f"  补充数学: {len(math_extra)} 条")

    # 合并到已有数学数据
    existing_math = SUPPLEMENT_DIR / "math_reasoning.jsonl"
    if existing_math.exists():
        with open(existing_math, encoding="utf-8") as f:
            for line in f:
                try:
                    math_extra.append(json.loads(line.strip()))
                except:
                    continue
    save_jsonl(math_extra, SUPPLEMENT_DIR / "math_reasoning.jsonl")

    # ============================================================
    # 汇总
    # ============================================================
    logger.info("=" * 60)
    total = 0
    for f in SUPPLEMENT_DIR.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())
        logger.info(f"  {f.name}: {count} 条")
        total += count
    logger.info(f"补充数据总计: {total} 条")
    logger.info(f"保存目录: {SUPPLEMENT_DIR}")


if __name__ == "__main__":
    main()
