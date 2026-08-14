#!/usr/bin/env python3
"""C27 增量五验证：振荡器节奏训练接入睡眠重放（Phase 1.8，2026-08-14）。

背景：增量四打通振荡器梯度路径（ω/coupling/gaba_amp 可微 + osc_rhythm_loss 作
gaba_amp 梯度源），但尚无训练脚本实际更新参数。增量五在 sleep_engine 新增
Phase 1.8：以记忆/场状态文本为样本，continuous 模式 forward_train，
loss = osc_rhythm_loss + phase_loss，optimizer 只含振荡器参数（内容层不参与）
——o 型节奏控制器随睡眠经验真正学习。训练后参数随 cortex.save_state 持久化。

验证层次：
A. sleep 端到端振荡器训练（样本来自场状态重放）：
   A1 振荡器被训练（osc_trained>0）、A2 loss 有限、
   A3 振荡器参数实际更新（训练生效）、A4 内容层零破坏（neuron 参数不变）
B. 无振荡器静默跳过（osc_trained=0 零回归）
C. 生产零回归：训练后 generate 非空不退化 + 振荡器仍正常牵引

运行：python -u scripts/training/verify_c27_osc_sleep.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402
from taiji.resonance.neuro_modulation import SleepConsolidator  # noqa: E402
# 口径契约：zh/dialogue 域 prompt 必须走训练格式
from taiji.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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
]


def field_state_of(cortex, text: str) -> torch.Tensor:
    """对文本做一次共振前向，取场状态快照（Phase 1.8 样本向量）。"""
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


def osc_params_snapshot(oscs) -> dict:
    return {o.nid: (float(o.omega.item()), float(o.coupling.item()),
                    float(o.gaba_amp.item())) for o in oscs}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C27 增量五：振荡器节奏训练接入睡眠重放（Phase 1.8）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c27_osc_sleep_")
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
        print(f"  装配 {len(cortex.neurons)} 神经元", flush=True)
        check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        from taiji.resonance.neuro_modulation import SleepConsolidator
        sc = SleepConsolidator(replay_buffer_size=50)
        sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

        oscs = list(cortex.ensemble.oscillators)
        check("A0. 装配含双层振荡器（Phase 1.8 前置）",
              len(oscs) == 2, f"oscs={[o.nid for o in oscs]}")

        # ── 样本注入：场状态重放（带 text）──
        for item in MEMORY_ITEMS:
            fs = field_state_of(cortex, item["text"])
            sc.record_high_resonance_state(
                field_state=fs, resonance_score=0.9,
                step=0, active_nids=target_ids,
                threshold=0.5, text=item["query"])
        check("样本注入：3 条场状态带 text 进重放缓冲区",
              len(sc._replay_buffer) == 3)

        # ── A. sleep 端到端振荡器训练（Phase 1.8）──
        print("\n[A] Phase 1.8 振荡器节奏训练 ...", flush=True)
        before = osc_params_snapshot(oscs)
        # 内容层代表参数（零破坏检查）
        nid0 = target_ids[0]
        first_param = next(cortex.neurons[nid0].parameters()).detach().clone()
        r8 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_osc_train(r8)
        after = osc_params_snapshot(oscs)
        check("A1. 振荡器被训练（osc_trained>0）",
              r8.osc_trained == len(oscs),
              f"osc_trained={r8.osc_trained}")
        check("A2. osc_train_loss 有限",
              r8.osc_train_loss is not None
              and r8.osc_train_loss == r8.osc_train_loss
              and abs(r8.osc_train_loss) < 1e3,
              f"loss={r8.osc_train_loss}")
        _changed = any(
            abs(after[o.nid][i] - before[o.nid][i]) > 1e-6
            for o in oscs for i in range(3))
        check("A3. 振荡器参数实际更新（训练生效）", _changed,
              f"before={ {k: tuple(round(v, 6) for v in val) for k, val in before.items()} } "
              f"after={ {k: tuple(round(v, 6) for v in val) for k, val in after.items()} }")
        _nparam = next(cortex.neurons[nid0].parameters()).detach()
        check("A4. 内容层零破坏（neuron 参数不变）",
              torch.equal(first_param, _nparam),
              "optimizer 只含振荡器参数")

        # ── B. 无振荡器静默跳过 ──
        print("\n[B] 无振荡器静默跳过 ...", flush=True)
        saved_oscs = list(cortex.ensemble.oscillators)
        cortex.ensemble.set_oscillators([])
        r8b = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                          duration_seconds=0)
        sleep_engine._sleep_phase_osc_train(r8b)
        check("B1. 无振荡器时 osc_trained=0（静默跳过零回归）",
              r8b.osc_trained == 0, f"osc_trained={r8b.osc_trained}")
        cortex.ensemble.set_oscillators(saved_oscs)

        # ── C. 生产零回归 ──
        print("\n[C] 生产零回归（训练后推理正常）...", flush=True)
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        check("C1. 训练后生成非空不退化", isinstance(out, str)
              and len(out.strip()) > 0 and not cortex._is_degenerate_text(out),
              f"out={out[:30]!r}")
        check("C2. 振荡器相位仍正常推进（推理兼容）",
              all(o.phase >= 0.0 and o.phase < 2 * 3.1416 for o in oscs),
              f"phases={[round(o.phase, 3) for o in oscs]}")
    except Exception as e:
        check("A1. 振荡器被训练", False, f"err={e}")
        check("A2. loss 有限", False, f"err={e}")
        check("A3. 参数更新", False, f"err={e}")
        check("A4. 内容层零破坏", False, f"err={e}")
        check("B1. 无振荡器跳过", False, f"err={e}")
        check("C1. 生产零回归", False, f"err={e}")
        check("C2. 振荡器兼容", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
