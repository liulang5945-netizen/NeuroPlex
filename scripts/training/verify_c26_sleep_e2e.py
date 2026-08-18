#!/usr/bin/env python3
"""C26 sleep() 端到端验证：记忆三格机制在真实产品入口编排下工作（2026-08-14）。

此前增量一/二/三都是各自调用单 phase 方法验证，本验证走**完整 sleep()
主流程**（record → sleep() → 场固化 Phase 1.5 → 突触沉淀 Phase 1.6 →
LoRA 写回保留），确认 C26 三格机制在真实编排下工作，暴露 phase 顺序/
配置类问题。

配置：SleepConfig(training_enabled=False)——Phase 2 模型训练与 C26 无关，
跳过以聚焦记忆管线（其余 phase 照常执行，均有 try/except 保护）。

流程：
1. record 4 条场记忆（3 高频候选 + 1 低频对照）→ sleep() #1
   → Phase 1.5 固化 4 条（无高频候选，沉淀 0）
2. 检索：高频 ×3 / 低频 ×1（累计访问计数）
3. sleep() #2 → Phase 1.5（无新 pending）+ Phase 1.6 沉淀高频 3 条
   → LoRA 写回 live + 条目标记 consolidated + 持久化
4. 回归：记忆条件化生成（memory_vectors）非空不破坏

断言：
A. sleep() 完整编排：phases_completed 含 field_consolidation/synaptic_consolidation
B. #1 固化 4 条（field_memories_consolidated=4）
C. #2 沉淀 3 条（synaptic_consolidated=3），低频 1 条不沉淀
D. LoRA 写回 live（B 非零 5/5）
E. consolidated 标记持久化（重启恢复）
F. 记忆条件化生成回归（向量通道非空）

运行：python -u scripts/training/verify_c26_sleep_e2e.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig  # noqa: E402
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
     "query": "什么是辉光协议？",
     "high_freq": True},
    {"label": "铁月海",
     "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
     "query": "铁月海在哪里？",
     "high_freq": True},
    {"label": "卡尔文环",
     "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
     "query": "卡尔文环是什么？",
     "high_freq": True},
    {"label": "频谱蜂鸟",
     "text": "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。",
     "query": "频谱蜂鸟有什么习性？",
     "high_freq": False},
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


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 sleep() 端到端：场固化 → 突触沉淀 → LoRA 写回 全编排验证", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_sleep_e2e_")
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
        target_ids = [nid for nid in cortex.neurons
                      if nid.startswith("zh_") and "dialogue" in nid]
        check("装配成功（9 神经元）", len(cortex.neurons) == 9)
        check("装配后无 LoRA（产品 ckpt 未沉淀）",
              all(len(cortex.neurons[n].lora_adapters) == 0 for n in target_ids))

        sleep_engine = SleepEngine(config=SleepConfig(training_enabled=False),
                                   data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)

        # ── 1. record 4 条记忆 → sleep() #1（Phase 1.5 固化）──
        print("\n[sleep #1] 固化 4 条场记忆 ...", flush=True)
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        r1 = sleep_engine.sleep(reason="test_consolidation")
        check("A. sleep 编排：含 field_consolidation",
              "field_consolidation" in r1.phases_completed,
              f"phases={r1.phases_completed}")
        check("A2. sleep 编排：synaptic_consolidation 已挂载",
              "synaptic_consolidation" in r1.phases_completed,
              f"phases={r1.phases_completed}")
        check("B. #1 固化 4 条", r1.field_memories_consolidated == 4,
              f"consolidated={r1.field_memories_consolidated}")
        check("B2. #1 无高频候选 → 沉淀 0", r1.synaptic_consolidated == 0,
              f"synaptic={r1.synaptic_consolidated}")
        bank = sleep_engine.get_field_memory()
        check("B3. 记忆库 4 条 + 持久化文件",
              len(bank) == 4
              and os.path.exists(os.path.join(tmp_dir, "field_memory.pt")))

        # ── 2. 会话检索：高频 ×3 / 低频 ×1 ──
        print("\n[会话] 检索累计访问计数 ...", flush=True)
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            hits = 3 if item["high_freq"] else 1
            for _ in range(hits):
                bank.retrieve_vectors(qv, top_k=1)
        acc = {e["label"]: e["access_count"] for e in bank.entries}
        check("检索计数：高频 3 / 低频 1", all(
            acc[i["label"]] == (3 if i["high_freq"] else 1)
            for i in MEMORY_ITEMS), str(acc))

        # ── 3. sleep() #2（Phase 1.5 无新 pending + Phase 1.6 沉淀）──
        print("\n[sleep #2] 突触沉淀 3 条高频记忆 ...", flush=True)
        r2 = sleep_engine.sleep(reason="test_synaptic")
        check("A3. sleep 编排：synaptic_consolidation 再次执行",
              "synaptic_consolidation" in r2.phases_completed)
        check("C. #2 沉淀 3 条", r2.synaptic_consolidated == 3,
              f"synaptic={r2.synaptic_consolidated}, lora_loss={r2.synaptic_lora_loss}")
        marks = {e["label"]: e["consolidated"] for e in bank.entries}
        check("C2. 标记正确（高频 True / 低频 False）",
              all(marks[i["label"]] == i["high_freq"] for i in MEMORY_ITEMS),
              str(marks))

        # ── 4. LoRA 写回 live（B 非零）──
        nonzero = 0
        for nid in target_ids:
            neuron = cortex.neurons[nid]
            if len(neuron.lora_adapters) == 0:
                continue
            b_max = max(float(p.abs().max().item())
                        for k, p in neuron.lora_adapters.named_parameters()
                        if ".b." in k)
            nonzero += 1 if b_max > 1e-6 else 0
        check("D. LoRA 写回 live（B 非零）", nonzero == len(target_ids),
              f"{nonzero}/{len(target_ids)}")

        # ── 5. consolidated 标记持久化（重启恢复）──
        from neuroplex.resonance.field_memory import FieldMemoryBank
        bank2 = FieldMemoryBank()
        ok5 = bank2.load(os.path.join(tmp_dir, "field_memory.pt"))
        check("E. 磁盘恢复：consolidated 标记保留",
              ok5 and all(e["consolidated"] == i["high_freq"]
                          for i, e in zip(MEMORY_ITEMS, bank2.entries)))

        # ── 6. 记忆条件化生成回归（向量通道非空不破坏）──
        print("\n[回归] 记忆条件化生成 ...", flush=True)
        nonempty = 0
        for item in [i for i in MEMORY_ITEMS if i["high_freq"]]:
            qv = field_state_of(cortex, item["query"])
            top = bank.retrieve_vectors(qv, top_k=1)
            if not top:
                continue
            cortex.field.reset()
            if cortex._dialogue_state is not None:
                cortex._dialogue_state.reset()
            out = cortex.generate(build_dialogue_prompt(item["query"]),
                                  max_tokens=24, domain="zh",
                                  memory_vectors=[(top[0][2], top[0][1])])
            nonempty += 1 if out and out.strip() else 0
            print(f"    {item['key'] if 'key' in item else item['label']}: "
                  f"{out[:24]!r}", flush=True)
        check("F. 记忆向量条件化生成非空（不破坏生成）",
              nonempty == len([i for i in MEMORY_ITEMS if i["high_freq"]]),
              f"{nonempty}/3")

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
        print(f"  记忆库: {bank.status()}", flush=True)
        print(f"  sleep #2 phases: {r2.phases_completed}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
