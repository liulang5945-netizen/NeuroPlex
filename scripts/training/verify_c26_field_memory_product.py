#!/usr/bin/env python3
"""C26 产品闭环验证：WriteGate + AnchorProjector 接入 sleep 场固化（2026-08-11）。

链路（与产品路径一致）：
1. 训练两组件（AnchorProjector + WriteGate，复用 train_field_memory_components 函数）
2. 保存到 sleep data_dir → SleepEngine.get_field_memory **自动装配**
3. sleep 场固化：可学习写门控生效（4 主题写入 + 重复被拒）
4. 检索：跨域语义锚点空间（挂 projector）命中不降
5. 重启：新 SleepEngine 同 data_dir → 组件自动加载 + 记忆恢复 + 检索仍命中
6. 向后兼容回归：空 data_dir 的 SleepEngine → 无 gate/projector（硬阈值 + 场空间）

运行：python -u scripts/training/verify_c26_field_memory_product.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402
from taiji.resonance.field_alignment import AnchorProjector  # noqa: E402
from taiji.resonance.field_memory import WriteGate  # noqa: E402
import train_field_memory_components as tfm  # noqa: E402

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


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("C26 产品闭环：WriteGate + AnchorProjector 接入 sleep 场固化", flush=True)
    print("=" * 64, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=tfm.COLLAB_NAME,
        extra_neurons_dir=tfm.EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=tfm.DIALOGUE_IDS,
    )
    dim = int(cortex.field.dim)
    print(f"装配 {len(cortex.neurons)} 神经元, 场维度 {dim}", flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="field_memory_product_")
    try:
        # ── 1. 训练两组件 + 保存到 sleep data_dir ──
        n = len(tfm.TERM_PAIRS)
        pos_pairs = tfm.TERM_PAIRS
        neg_pairs = [(tfm.TERM_PAIRS[i][0], tfm.TERM_PAIRS[(i + 7) % n][1])
                     for i in range(n)]
        texts = set()
        for a, b in pos_pairs + neg_pairs:
            texts.add(a)
            texts.add(b)
        texts.update(tfm.TOPICS)
        vectors = {}
        print(f"\n[1] 冻结场向量采集（{len(texts)} 个文本）...", flush=True)
        for txt in sorted(texts):
            vectors[txt] = tfm.field_state_of(cortex, txt)

        from taiji.resonance.field_alignment import train_anchor_projector
        proj = train_anchor_projector(vectors, pos_pairs, neg_pairs)
        gate = tfm.train_write_gate([vectors[t] for t in tfm.TOPICS])
        proj_path = os.path.join(tmp_dir, "anchor_projector.pt")
        gate_path = os.path.join(tmp_dir, "write_gate.pt")
        proj.save(proj_path)
        gate.save(gate_path)
        check("训练产物已保存", os.path.exists(proj_path) and os.path.exists(gate_path))

        # ── 2. SleepEngine 自动装配 ──
        print("\n[2] SleepEngine 装配 ...", flush=True)
        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)
        bank = sleep_engine.get_field_memory()
        check("自动装配可学习写门控", bank.gate is not None)
        check("自动装配跨域语义锚点投影", bank.projector is not None)

        # ── 3. sleep 场固化（学习门控生效）──
        print("\n[3] 场固化（4 主题 + 重复）...", flush=True)
        topic_vecs = [vectors[t] for t in tfm.TOPICS]
        topic_labels = [f"T{i}" for i in range(len(tfm.TOPICS))]
        for v, lbl in zip(topic_vecs, topic_labels):
            sleep_engine.record_field_memory(v, lbl)
        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report)
        check("门控固化：4 新主题全部写入", report.field_memories_consolidated == 4
              and len(bank) == 4, f"bank={len(bank)}")
        for v, lbl in zip(topic_vecs, topic_labels):
            sleep_engine.record_field_memory(v, lbl)
        report2 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                              duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report2)
        check("门控固化：重复主题被拒", report2.field_memories_consolidated == 0,
              f"added={report2.field_memories_consolidated}")

        # ── 4. 锚点空间检索 ──
        print("\n[4] 锚点空间检索 ...", flush=True)
        hit = 0
        for lbl, v in zip(topic_labels, topic_vecs):
            top = bank.retrieve(v, top_k=1)
            hit += 1 if top and top[0][0] == lbl else 0
        check("锚点空间检索命中 4/4", hit == len(topic_vecs), f"{hit}/4")

        # ── 5. 重启恢复（新 SleepEngine 同 data_dir）──
        print("\n[5] 重启恢复 ...", flush=True)
        se2 = SleepEngine(data_dir=tmp_dir)
        se2.set_brain_interfaces(cortex=cortex)
        bank2 = se2.get_field_memory()
        check("重启后组件自动加载", bank2.gate is not None
              and bank2.projector is not None)
        check("重启后记忆库恢复（4 条）", len(bank2) == 4, f"bank2={len(bank2)}")
        top2 = bank2.retrieve(topic_vecs[0], top_k=1)
        check("重启后锚点检索仍命中", bool(top2) and top2[0][0] == topic_labels[0])

        # ── 6. 向后兼容回归（空 data_dir）──
        print("\n[6] 向后兼容回归（无产物回退）...", flush=True)
        tmp_empty = tempfile.mkdtemp(prefix="field_memory_empty_")
        try:
            se3 = SleepEngine(data_dir=tmp_empty)
            se3.set_brain_interfaces(cortex=cortex)
            bank3 = se3.get_field_memory()
            check("无产物时无 gate/projector（回退）",
                  bank3.gate is None and bank3.projector is None)
            check("无产物时硬阈值场固化可用",
                  bank3.consolidate(topic_vecs, topic_labels) == 4)
        finally:
            shutil.rmtree(tmp_empty, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 64, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 64, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
