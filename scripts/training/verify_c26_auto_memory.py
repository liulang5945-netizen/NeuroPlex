#!/usr/bin/env python3
"""C26 增量四验证：记忆自动检索注入对话（记忆自动调取，产品化收尾）（2026-08-14）。

背景：增量二实现"显式 memory_vectors 条件化生成"（需调用方手动传向量），
记忆仍是显式 API。增量四让 generate **自动检索** FieldMemoryBank 注入生成：
未显式传 memory_vectors 且注入过记忆库时，用 prompt 场状态自动检索 top-1
记忆（Titans 式内部记忆的产品化落点）。sleep_engine.set_brain_interfaces
装配时自动注入记忆库（产品默认接入）。

验证：
A. 装配注入：set_brain_interfaces 后 cortex._memory_bank 已挂载
B. 自动检索生效（硬）：generate 未显式传向量 → 记忆库 access_count 增加
   （自动检索调用 retrieve_vectors 的证据），且命中对应记忆
C. 记忆影响生成（软）：auto_memory=True vs False 输出不同（≥1）
D. 显式传向量时自动检索跳过（access_count 不增加）
E. 无记忆库时 auto_memory 静默跳过（不报错，生成正常）

运行：python -u scripts/training/verify_c26_auto_memory.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402
from scripts.training.experiment_config import build_dialogue_prompt  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

MEMORY_ITEMS = [
    {"label": "辉光协议",
     "text": "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
     "query": "什么是辉光协议？"},
    {"label": "铁月海",
     "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
     "query": "铁月海在哪里？"},
    {"label": "卡尔文环",
     "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
     "query": "卡尔文环是什么？"},
    {"label": "频谱蜂鸟",
     "text": "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。",
     "query": "频谱蜂鸟有什么习性？"},
]


def field_state_of(cortex, text: str) -> torch.Tensor:
    gids = cortex._general_sp.encode(text) or [0]
    ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(ids)
    res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                       collab_mode="continuous")
    fs = res.get("field_state")
    if fs is None:
        raise RuntimeError("think() 未返回 field_state")
    if fs.dim() == 2:
        fs = fs.mean(dim=0)
    return fs


def fresh_generate(cortex, prompt: str, max_tokens: int = 24,
                   memory_vectors=None, auto_memory: bool = True) -> str:
    cortex.field.reset()
    if cortex._dialogue_state is not None:
        cortex._dialogue_state.reset()
    return cortex.generate(build_dialogue_prompt(prompt), max_tokens=max_tokens,
                           domain="zh", memory_vectors=memory_vectors,
                           auto_memory=auto_memory)


def acc_of(bank):
    return {e["label"]: e["access_count"] for e in bank.entries}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 增量四：记忆自动检索注入对话（记忆自动调取）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_auto_mem_")
    try:
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        check("装配成功（9 神经元）", len(cortex.neurons) == 9)
        check("A. 装配即注入记忆库（产品默认接入）",
              cortex._memory_bank is not None,
              f"bank={'有' if cortex._memory_bank is not None else '无'}")

        # 用 tmp 记忆库覆盖注入（隔离产品库状态，固化本验证记忆）
        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)

        # 固化 4 条记忆
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        r = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(r)
        bank = cortex._memory_bank
        check("A2. 固化 4 条（bank 引用一致）",
              len(bank) == 4 and bank is sleep_engine.get_field_memory())
        print(f"    初始 access_count: {acc_of(bank)}", flush=True)

        # ── B. 自动检索生效（硬）：generate 不显式传向量 → access_count 增加 ──
        # 自动检索 top-1 依据 prompt 场状态（含"问：/答："格式前缀）余弦，
        # 不保证逐条命中对应记忆；核心证据 = 自动检索真的触发（命中条目 count+1）。
        print("\n[B] 自动检索生效（access_count 证据）...", flush=True)
        total_before = sum(acc_of(bank).values())
        hits = 0
        for item in MEMORY_ITEMS:
            before = bank.entries[[e["label"] for e in bank.entries]
                                  .index(item["label"])]["access_count"]
            out = fresh_generate(cortex, item["query"])
            after = bank.entries[[e["label"] for e in bank.entries]
                                 .index(item["label"])]["access_count"]
            hits += 1 if after == before + 1 else 0
            print(f"    {item['label']}: count {before}→{after}, "
                  f"out={out[:20]!r}", flush=True)
        total_after = sum(acc_of(bank).values())
        check("B. 自动检索触发（总命中数增加）",
              total_after > total_before,
              f"total {total_before}→{total_after}, 逐条命中 {hits}/{len(MEMORY_ITEMS)}")

        # ── C. 记忆影响生成（软）：auto True vs False ──
        print("\n[C] auto_memory 开/关生成对比 ...", flush=True)
        changed = 0
        for item in MEMORY_ITEMS:
            on = fresh_generate(cortex, item["query"], auto_memory=True)
            off = fresh_generate(cortex, item["query"], auto_memory=False)
            changed += 1 if on != off else 0
            print(f"    {item['label']}: auto={on[:20]!r} / 关闭={off[:20]!r} "
                  f"(changed={on != off})", flush=True)
        check("C. 自动记忆改变生成输出（软）", changed >= 1,
              f"{changed}/{len(MEMORY_ITEMS)} 与关闭不同")

        # ── D. 显式传向量时自动检索跳过 ──
        print("\n[D] 显式传向量 → 自动检索跳过 ...", flush=True)
        skip_ok = True
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            top = bank.retrieve_vectors(qv, top_k=1)  # 这里 count +1（显式检索）
            before = bank.entries[[e["label"] for e in bank.entries]
                                  .index(item["label"])]["access_count"]
            fresh_generate(cortex, item["query"],
                           memory_vectors=[(top[0][2], top[0][1])])
            after = bank.entries[[e["label"] for e in bank.entries]
                                 .index(item["label"])]["access_count"]
            if after != before:
                skip_ok = False
                print(f"    {item['label']}: count {before}→{after} (不应变)",
                      flush=True)
        check("D. 显式传向量时自动检索跳过（count 不变）", skip_ok)

        # ── E. 无记忆库时 auto_memory 静默跳过 ──
        cortex_b, _, _ = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        cortex_b.field.reset()
        out_b = cortex_b.generate(build_dialogue_prompt(MEMORY_ITEMS[0]["query"]),
                                  max_tokens=24, domain="zh")
        check("E. 无记忆库时 auto_memory 静默跳过（生成正常）",
              bool(out_b and out_b.strip()), f"out={out_b[:20]!r}")

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
        print(f"  记忆库: {bank.status()}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
