"""扩充对话训练数据：从 BelleGroup/train_2M_CN 下载并转换统一格式。

背景（2026-08-12，zh 对话数据主线）：
  现有对话数据 88.7K 条（alpaca-zh 44.4K + shared_core 13.3K + unique 5×6.2K），
  对话质量受欠训练限制（answer PPL ~70）。扩充数据量是直接提升路径。

流程：
  1. 流式下载 BelleGroup/train_2M_CN（2M 条中文指令，字段 instruction/input/output）
  2. 转换 "问：{instruction[+input]}\n答：{output}" 格式
  3. 与现有 clean 数据去重（text 级）
  4. 清洗（复用 clean_dialogue_data.is_dirty：过滤代码/英文密集答案）
  5. 抽样质量检查 + 输出 dialogue_extended_clean.jsonl

用法：
  python scripts/training/build_dialogue_extended.py [--max-samples 150000]
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.clean_dialogue_data import is_dirty  # noqa: E402
from scripts.training.experiment_config import DIALOGUE_DATA_FILES  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "simple_zh")
OUT_PATH = os.path.join(DATA_DIR, "dialogue_extended_clean.jsonl")
RAW_PATH = os.path.join(DATA_DIR, "dialogue_extended_raw.jsonl")

HF_DATASET = "BelleGroup/train_2M_CN"


def load_existing_texts() -> set:
    """现有 clean 数据（去重用）。"""
    existing = set()
    for fname in DIALOGUE_DATA_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
    print(f"  现有 clean 数据: {len(existing)} 条（去重基准）", flush=True)
    return existing


def build(max_samples: int) -> None:
    existing = load_existing_texts()

    # 1. 流式下载 + 转换
    print(f"\n[1] 下载 {HF_DATASET} (目标 {max_samples})...", flush=True)
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    raw = []
    for ex in ds:
        instruction = ex.get("instruction", "") or ""
        input_ = ex.get("input", "") or ""
        output = ex.get("output", "") or ""
        if not (isinstance(instruction, str) and isinstance(output, str)):
            continue
        if not instruction.strip() or not output.strip():
            continue
        # alpaca 格式：input 非空时拼入问题
        if isinstance(input_, str) and input_.strip():
            question = f"{instruction.strip()}\n{input_.strip()}"
        else:
            question = instruction.strip()
        raw.append(f"问：{question}\n答：{output.strip()}")
        if len(raw) >= max_samples:
            break
    print(f"  下载完成: {len(raw)} 条", flush=True)

    # 2. 与现有数据去重（保持插入序确定性）
    new_texts = []
    for t in raw:
        if t not in existing and t not in new_texts:
            new_texts.append(t)
    print(f"  去重（vs 现有+自身）: {len(raw)} → {len(new_texts)} 条", flush=True)

    # 3. 清洗（代码/英文密集答案）
    kept = []
    dropped = {"code": 0, "en": 0}
    for t in new_texts:
        bad, reason = is_dirty(t)
        if bad:
            dropped[reason if reason == "code" else "en"] += 1
            continue
        kept.append(t)
    print(f"  清洗: {len(new_texts)} → {len(kept)} 条 "
          f"(过滤 代码={dropped['code']} 英文密集={dropped['en']})", flush=True)

    # 4. 输出
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"\n  输出: {OUT_PATH} ({len(kept)} 条)", flush=True)

    # 5. 抽样质量检查
    print(f"\n[5] 抽样质量检查（5 条）:", flush=True)
    for t in kept[:5]:
        print(f"  --- {t[:100]!r}", flush=True)

    print(f"\n  合计（现有 + 新增）: {len(existing)} + {len(kept)} = {len(existing) + len(kept)} 条", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=150000)
    args = ap.parse_args()
    print("=" * 60, flush=True)
    print("对话数据扩充（BelleGroup/train_2M_CN）", flush=True)
    print("=" * 60, flush=True)
    build(args.max_samples)
