"""清洗对话训练数据：过滤代码/英文密集样本。

背景（2026-08-03）：
  完整重训后 PPL 29.7 / EMERGE 65.7%，但长回复生成崩溃（英文/代码碎片）。
  根因：训练数据 ~31% 含英文、~5% 含代码，模型学会采样英文 token，
  而 zh 50K 词表以中文为主，英文被碎片化 → 输出乱码。

清洗规则：
  1. 代码样本：答案含 ``` 代码块，或 def/import/print/for/while/return/class 关键字 → 过滤
  2. 英文密集：答案英文词数 > MAX_EN_WORDS(8) → 过滤
  3. 保留其余（含 0~8 个英文词的样本，保留合理术语如 AI/Python）

输出：同目录 *_clean.jsonl（保持 {"text": ...} 格式）
"""
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.experiment_config import DIALOGUE_DATA_FILES  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "simple_zh")

# 代码特征正则（中文对话中出现的编程关键字）
CODE_PATTERN = re.compile(
    r"```|\b(def|import|print|for|while|return|class|if __name__|lambda)\b"
)
# 英文单词
EN_WORD = re.compile(r"[A-Za-z]{2,}")
MAX_EN_WORDS = 8  # 答案英文词数上限（保留合理术语）


def is_dirty(text: str) -> tuple:
    """判断样本是否含代码/英文密集。返回 (是否过滤, 原因)。"""
    ans = text.split("答：")[-1] if "答：" in text else text
    if CODE_PATTERN.search(ans):
        return True, "code"
    n_en = len(set(w.lower() for w in EN_WORD.findall(ans)))
    if n_en > MAX_EN_WORDS:
        return True, f"en={n_en}"
    return False, ""


def clean_file(src: str, dst: str) -> dict:
    kept = 0
    dropped = {"code": 0, "en": 0}
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = item.get("text", "")
            bad, reason = is_dirty(text)
            if bad:
                if reason == "code":
                    dropped["code"] += 1
                else:
                    dropped["en"] += 1
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
    return {"kept": kept, "dropped": dropped}


def main():
    print("=" * 60, flush=True)
    print("对话数据清洗 (过滤代码/英文密集样本)", flush=True)
    print("=" * 60, flush=True)
    total_kept = 0
    total_dropped = {"code": 0, "en": 0}
    for fname in DIALOGUE_DATA_FILES:
        src = os.path.join(DATA_DIR, fname)
        if not os.path.exists(src):
            print(f"  ⚠️ 不存在: {fname}", flush=True)
            continue
        stem, ext = os.path.splitext(fname)
        dst = os.path.join(DATA_DIR, f"{stem}_clean{ext}")
        stat = clean_file(src, dst)
        total_kept += stat["kept"]
        total_dropped["code"] += stat["dropped"]["code"]
        total_dropped["en"] += stat["dropped"]["en"]
        print(f"  {fname}: 保留 {stat['kept']} 条, "
              f"过滤 代码={stat['dropped']['code']} 英文密集={stat['dropped']['en']}", flush=True)
    total_in = total_kept + total_dropped["code"] + total_dropped["en"]
    print(f"\n  合计: 输入 {total_in} → 保留 {total_kept} "
          f"({total_kept / total_in * 100:.1f}%), "
          f"过滤 代码={total_dropped['code']} 英文密集={total_dropped['en']}", flush=True)
    print(f"\n  输出文件: {DATA_DIR}/**_clean.jsonl", flush=True)


if __name__ == "__main__":
    main()
