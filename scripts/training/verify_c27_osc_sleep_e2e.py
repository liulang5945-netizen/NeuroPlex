#!/usr/bin/env python3
"""C27 增量五 sleep() 端到端：Phase 1.8 振荡器节奏训练在主流程中共存（2026-08-14）。

此前增量四/五验证均为单 phase 调用（verify_c27_osc_train / verify_c27_osc_sleep）。
本验证走**完整 sleep() 主流程**，确认增量五插入的 Phase 1.8 与既有睡眠链路
（1.5 场固化 / 1.6 突触沉淀 / 1.7 前向重放 / 2(跳过) / 3 知识整合 / 3.5 蒸馏 /
4 评估 / 5 递归改进）在真实编排下共存：
- sleep #1：场固化 + 振荡器训练（样本来自场状态重放）
- sleep #2：突触沉淀（3 高频）→ consolidated 记忆成为振荡器训练样本源

配置：SleepConfig(training_enabled=False)——Phase 2 模型训练跳过（与振荡器
训练无关，聚焦睡眠管线；其余 phase 均有 try/except 保护）。

断言：
A. sleep() 完整编排：phases_completed 含 osc_train（Phase 1.8 挂载）
B. sleep #1 振荡器训练生效（osc_trained=2 + 三参数实际更新）
C. sleep #1 内容层零破坏（neuron 参数不变——optimizer 只含振荡器）
D. sleep #2（沉淀 3 条）后振荡器再次训练（osc_trained=2、loss 有限）
E. 生产零回归：两轮完整睡眠后 generate 非空不退化 + 振荡器兼容

运行：python -u scripts/training/verify_c27_osc_sleep_e2e.py
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
from taiji.life.sleep_engine import SleepEngine, SleepConfig  # noqa: E402
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


def osc_params_snapshot(oscs) -> dict:
    return {o.nid: (float(o.omega.item()), float(o.coupling.item()),
                    float(o.gaba_amp.item())) for o in oscs}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C27 增量五 sleep() 端到端：Phase 1.8 振荡器训练主流程共存", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c27_osc_sleep_e2e_")
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

        sleep_engine = SleepEngine(config=SleepConfig(training_enabled=False),
                                   data_dir=tmp_dir)
        sc = SleepConsolidator(replay_buffer_size=50)
        sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)
        oscs = list(cortex.ensemble.oscillators)
        check("前置：双层振荡器已装配", len(oscs) == 2,
              f"oscs={[o.nid for o in oscs]}")

        # ── 样本注入：场状态重放（带 text）──
        for item in MEMORY_ITEMS:
            fs = field_state_of(cortex, item["text"])
            sc.record_high_resonance_state(
                field_state=fs, resonance_score=0.9,
                step=0, active_nids=target_ids,
                threshold=0.5, text=item["query"])

        # ── sleep #1：场固化 + 振荡器训练 ──
        print("\n[sleep #1] 场固化 + Phase 1.8 振荡器训练 ...", flush=True)
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        params0 = osc_params_snapshot(oscs)
        first_param = next(cortex.neurons[target_ids[0]].parameters()).detach().clone()
        r1 = sleep_engine.sleep(reason="test_osc_e2e_1")
        check("A. sleep #1 完整编排（Phase 1.8 挂载）",
              "osc_train" in r1.phases_completed
              and "field_consolidation" in r1.phases_completed
              and "evaluation" in r1.phases_completed,
              f"phases={r1.phases_completed}")
        check("B1. #1 振荡器训练执行（osc_trained>0）",
              r1.osc_trained == len(oscs),
              f"osc_trained={r1.osc_trained}, loss={r1.osc_train_loss}")
        params1 = osc_params_snapshot(oscs)
        _changed1 = any(abs(params1[o.nid][i] - params0[o.nid][i]) > 1e-6
                        for o in oscs for i in range(3))
        check("B2. #1 振荡器参数实际更新（训练生效）", _changed1,
              f"before={ {k: tuple(round(v, 6) for v in val) for k, val in params0.items()} } "
              f"after={ {k: tuple(round(v, 6) for v in val) for k, val in params1.items()} }")
        check("C. #1 内容层零破坏（neuron 参数不变）",
              torch.equal(first_param,
                          next(cortex.neurons[target_ids[0]].parameters()).detach()),
              "optimizer 只含振荡器参数")

        # ── 会话检索（高频 ×3 / 低频 ×1）→ sleep #2 沉淀 ──
        print("\n[会话] 检索累计访问计数 ...", flush=True)
        bank = sleep_engine.get_field_memory()
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            hits = 3 if item["high_freq"] else 1
            for _ in range(hits):
                bank.retrieve_vectors(qv, top_k=1)
        acc = {e["label"]: e["access_count"] for e in bank.entries}
        check("检索计数：高频 3 / 低频 1", all(
            acc[i["label"]] == (3 if i["high_freq"] else 1)
            for i in MEMORY_ITEMS), str(acc))

        # ── sleep #2：突触沉淀 + 前向重放 + 振荡器再训练 ──
        print("\n[sleep #2] 突触沉淀 + 重放 + Phase 1.8 再训练 ...", flush=True)
        r2 = sleep_engine.sleep(reason="test_osc_e2e_2")
        check("D1. #2 沉淀 3 条高频记忆", r2.synaptic_consolidated == 3,
              f"synaptic={r2.synaptic_consolidated}")
        check("D2. #2 振荡器再次训练", r2.osc_trained == len(oscs)
              and r2.osc_train_loss is not None
              and r2.osc_train_loss == r2.osc_train_loss,
              f"osc_trained={r2.osc_trained}, loss={r2.osc_train_loss}")
        params2 = osc_params_snapshot(oscs)
        _changed2 = any(abs(params2[o.nid][i] - params1[o.nid][i]) > 1e-6
                        for o in oscs for i in range(3))
        check("D3. #2 后振荡器参数继续演化（连续学习）", _changed2,
              f"after2={ {k: tuple(round(v, 6) for v in val) for k, val in params2.items()} }")

        # ── 生产零回归 ──
        print("\n[回归] 两轮完整睡眠后生成 ...", flush=True)
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        check("E1. 生成非空不退化", isinstance(out, str)
              and len(out.strip()) > 0 and not cortex._is_degenerate_text(out),
              f"out={out[:30]!r}")
        check("E2. 振荡器相位兼容", all(
            0.0 <= o.phase < 2 * 3.1416 for o in oscs),
            f"phases={[round(o.phase, 3) for o in oscs]}")
    except Exception as e:
        check("A. sleep 完整编排", False, f"err={e}")
        check("B1. #1 振荡器训练", False, f"err={e}")
        check("B2. #1 参数更新", False, f"err={e}")
        check("C. #1 内容层零破坏", False, f"err={e}")
        check("检索计数", False, f"err={e}")
        check("D1. #2 沉淀", False, f"err={e}")
        check("D2. #2 振荡器训练", False, f"err={e}")
        check("D3. #2 参数演化", False, f"err={e}")
        check("E1. 生产零回归", False, f"err={e}")
        check("E2. 振荡器兼容", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
