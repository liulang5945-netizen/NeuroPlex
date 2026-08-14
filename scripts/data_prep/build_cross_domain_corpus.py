#!/usr/bin/env python3
"""跨域平行语料构建（hub neuron 缺口 L，2026-08-14）。

背景：hub neuron 的跨域对比 loss（草案决策 4C：同义跨域输入对 field_vector
cosine 最大化 / 不同义对最小化）需要 zh↔code 同义对。自动构建最可靠来源：
**alpaca-zh 中含代码块的样本**——中文指令（自然语言语义）↔ 代码实现
（符号语义）是同一语义的两种表达，构成天然平行对。

产物：data/cross_domain_pairs.jsonl
  每行 {"zh": 中文指令, "code": 代码块, "source": 来源样本 index}
  字段语义（供跨域对比 loss 消费）：
  - zh 侧：中文自然语言表达（tokenizer: zh）
  - code 侧：代码实现（tokenizer: code）

过滤规则：
- 样本 output 含 ``` 代码块（fence 包裹），取第一个代码块
- 代码块长度 >= 15 字符（剔除空/占位块）
- 指令含中文 >= 4 字（剔除纯英文指令，确保 zh 侧是中文语义）
- 指令长度 >= 4（短指令语义不足）

运行：python -u scripts/data_prep/build_cross_domain_corpus.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "data", "cache")
# alpaca-zh（shibing624）：中文指令数据集，含大量代码问答样本
ALPACA_ZH_ARROW = os.path.join(
    CACHE_ROOT, "shibing624___alpaca-zh", "shibing624___alpaca-zh",
    "default", "0.0.0", "f39db019a94f8dbea48ab30d2bdc090703284559",
    "alpaca-zh-train.arrow")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_pairs.jsonl")

CODE_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.S)
ZH_CHARS = re.compile(r"[\u4e00-\u9fff]")

MIN_CODE_LEN = 15      # 代码块最小字符数（剔除占位块）
MIN_ZH_INSTR = 4       # 指令最少中文数（保证 zh 侧是中文语义）


def load_alpaca_zh() -> list:
    """alpaca-zh 本地 arrow 直读（与 build_domain_sft_v2 同模式）。"""
    from pyarrow import ipc
    with open(ALPACA_ZH_ARROW, "rb") as f:
        table = ipc.open_stream(f).read_all()
    return table.to_pylist()


def main():
    print("=" * 60, flush=True)
    print("跨域平行语料构建（zh↔code 同义对，hub neuron 缺口 L 地基）", flush=True)
    print("=" * 60, flush=True)
    print(f"  输入: {ALPACA_ZH_ARROW}", flush=True)

    samples = load_alpaca_zh()
    print(f"  原始样本: {len(samples)} 条", flush=True)

    pairs = []
    skipped = {"no_fence": 0, "code_short": 0, "zh_short": 0, "inst_short": 0}
    for i, row in enumerate(samples):
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        outp = str(row.get("output", "")).strip()
        if not inst or "```" not in outp:
            skipped["no_fence"] += 1
            continue
        blocks = CODE_FENCE.findall(outp)
        if not blocks:
            skipped["no_fence"] += 1
            continue
        code = blocks[0].strip()
        if len(code) < MIN_CODE_LEN:
            skipped["code_short"] += 1
            continue
        # zh 侧 = 指令（+ input 补充上下文）
        zh_text = (inst + ("\n" + inp) if inp and len(inp) < 200 else inst).strip()
        if len(ZH_CHARS.findall(zh_text)) < MIN_ZH_INSTR:
            skipped["zh_short"] += 1
            continue
        if len(inst) < 4:
            skipped["inst_short"] += 1
            continue
        pairs.append({
            "zh": zh_text,
            "code": code,
            "source": i,
        })

    print(f"  提取同义对: {len(pairs)} 条", flush=True)
    print(f"  跳过统计: {skipped}", flush=True)

    # 去重（同指令保留第一条）
    seen_zh = set()
    dedup = []
    for p in pairs:
        key = p["zh"][:50]
        if key in seen_zh:
            continue
        seen_zh.add(key)
        dedup.append(p)
    pairs = dedup
    print(f"  去重后: {len(pairs)} 条", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 抽样展示
    print("\n  抽样 3 对:", flush=True)
    for p in pairs[:3]:
        print(f"    zh : {p['zh'][:60]}", flush=True)
        print(f"    code: {p['code'][:80].replace(chr(10), ' | ')}", flush=True)

    # 长度统计
    if pairs:
        zh_lens = [len(p["zh"]) for p in pairs]
        code_lens = [len(p["code"]) for p in pairs]
        print(f"\n  zh 长度 avg={sum(zh_lens)//len(zh_lens)} max={max(zh_lens)}", flush=True)
        print(f"  code 长度 avg={sum(code_lens)//len(code_lens)} max={max(code_lens)}", flush=True)
    print(f"\n  产物: {OUT_PATH}（{len(pairs)} 对）", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if pairs else 1)


if __name__ == "__main__":
    main()
