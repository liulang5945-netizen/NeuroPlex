#!/usr/bin/env python3
"""C27 增量二验证：场向量相位编码（KoPE，2026-08-14）。

背景：相位此前只驱动"激活强度/融合权重"（纯动力学），记忆/路由读到的
field_state 不含相位语义。增量二把相位编码为显式表征（result 附加字段 +
记忆条目扩展，用户决策），并让记忆注入按记忆相位对齐 theta（相位归属记忆，
用户决策）——相位从动力学走向表征的一等公民。

层次：
A. 相位编码：continuous/forward 路径 result 含 phase_code [2N] / phase_mean /
   phase_lock（合法维度、锁相度 ∈ [0,1]）
B. 记忆带相位：record_field_memory(phase) → pending 4 元组；consolidate →
   entry["phase"]；retrieve_with_phase 返回相位；retrieve_vectors 3 元组兼容
C. 相位注入：entrain_memory(target_phase) → theta_phase_at 返回目标相位；
   seed_memories 3 元组 → ct._entrain_phase 生效；2 元组回退 0 峰值（零回归）
D. 生产：generate memory_vectors 3 元组可用、get_last_phase 截获、
   task_chain record 带相位

运行：python -u scripts/training/verify_c27_kope.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import random  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.brain.cortex import TaskSet  # noqa: E402
from taiji.resonance.continuous import ContinuousResonance  # noqa: E402
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


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C27 增量二：场向量相位编码（KoPE）", flush=True)
    print("=" * 60, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)
    check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

    # judge EMA 预热（executive 判定主信号）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成", flush=True)

    zh_domain_all = ["zh_aug0_dialogue", "zh_aug1_dialogue",
                     "zh_aug2_dialogue", "zh_aug3_dialogue",
                     "zh_std0_dialogue", "zh"]
    emb_in = torch.tensor(
        [cortex._general_sp.encode(build_dialogue_prompt("相位编码测试"))],
        dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(emb_in)

    # ── A. 相位编码（forward/continuous_forward 输出）──
    print("\n[A] 相位编码（phase_code / phase_mean / phase_lock）...", flush=True)
    try:
        res_c = cortex.think(emb, active_nids=zh_domain_all,
                             collab_mode="continuous", fusion_mode="soft")
        pc = res_c.get("phase_code")
        pm = res_c.get("phase_mean")
        pl = res_c.get("phase_lock")
        # C27 增量三（BioOSS）：phase_code 含振荡段 → 维度 = 2N + 2M
        _M = len(getattr(cortex.ensemble, "oscillators", []))
        ok = (pc is not None and pc.numel() == 2 * len(zh_domain_all) + 2 * _M
              and isinstance(pm, float) and -math.pi <= pm <= math.pi
              and isinstance(pl, float) and 0.0 <= pl <= 1.0 + 1e-6)
        check("A1. continuous 路径相位编码合法",
              ok, f"code_dim={None if pc is None else tuple(pc.shape)} mean={pm:.3f} lock={pl:.3f}")
    except Exception as e:
        check("A1. continuous 路径相位编码合法", False, f"err={e}")
    try:
        res_d = cortex.think(emb, active_nids=zh_domain_all,
                             collab_mode="fusion", fusion_mode="soft")
        pc2 = res_d.get("phase_code")
        pl2 = res_d.get("phase_lock")
        check("A2. 离散 forward 路径相位编码合法",
              pc2 is not None and pc2.numel() > 0
              and isinstance(pl2, float) and 0.0 <= pl2 <= 1.0 + 1e-6,
              f"code_dim={None if pc2 is None else tuple(pc2.shape)} lock={pl2}")
    except Exception as e:
        check("A2. 离散 forward 路径相位编码合法", False, f"err={e}")

    # ── B. 记忆带相位（相位归属记忆）──
    print("\n[B] 记忆带相位（record/consolidate/retrieve）...", flush=True)
    try:
        from taiji.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        bank = engine.get_field_memory()
        vec = torch.randn(3072)  # field.dim（5 dialogue 装配 = 3072）
        engine.record_field_memory(vec, "相位记忆A", text="记忆内容A", phase=0.7)
        pend = list(engine.pending_field_memories)
        check("B1. record_field_memory 带相位（pending 4 元组）",
              len(pend) == 1 and len(pend[0]) == 4 and pend[0][3] == 0.7,
              f"pending_len={len(pend[0]) if pend else 0}")
        engine.pending_field_memories = []
        added = bank.consolidate([vec], ["相位记忆A"], texts=["记忆内容A"],
                                 phases=[0.7])
        entry = bank.entries[-1]
        check("B2. consolidate 保存记忆相位", added == 1 and entry["phase"] == 0.7,
              f"entry_phase={entry.get('phase')}")
        q = vec / vec.norm()
        rw = bank.retrieve_with_phase(q, top_k=1)
        rv = bank.retrieve_vectors(q, top_k=1)
        check("B3. retrieve_with_phase 返回相位 + retrieve_vectors 兼容",
              len(rw) == 1 and len(rw[0]) == 4 and rw[0][3] == 0.7
              and len(rv) == 1 and len(rv[0]) == 3,
              f"with_phase={rw[0][3] if rw else None}, vec_n={len(rv[0]) if rv else 0}")
    except Exception as e:
        check("B1. record_field_memory 带相位", False, f"err={e}")
        check("B2. consolidate 保存记忆相位", False, f"err={e}")
        check("B3. retrieve 相位返回/兼容", False, f"err={e}")

    # ── C. 相位注入（按记忆相位对齐 theta）──
    print("\n[C] 相位注入（entrain_memory 目标相位）...", flush=True)
    try:
        ct = ContinuousResonance()
        ct.entrain_memory(target_phase=0.7)
        ph = ct.theta_phase_at(5.0)
        check("C1. entrain 目标相位生效（theta_phase_at 返回目标）",
              abs(ph - 0.7) < 1e-6, f"ph={ph:.4f}")
        ct.reset_entrain()
        ph2 = ct.theta_phase_at(5.0)
        check("C2. reset_entrain 恢复自由演化",
              not ct._memory_entrained and ct._entrain_phase == 0.0
              and abs(ph2 - (ct.theta_init + ct.theta_omega * 5.0)) < 1e-6,
              f"ph={ph2:.4f}")
    except Exception as e:
        check("C1. entrain 目标相位生效", False, f"err={e}")
        check("C2. reset_entrain 恢复自由演化", False, f"err={e}")
    try:
        # 3 元组 seed_memories → continuous_forward 按记忆相位对齐（entrain 在
        # forward 末尾 reset 防泄漏，故用 spy 捕获 forward 期间的目标相位）
        my_ct = ContinuousResonance()
        calls3 = {}

        def spy_entrain(target_phase=0.0):
            calls3["phase"] = float(target_phase)
            my_ct._memory_entrained = True
            my_ct._entrain_phase = float(target_phase)

        my_ct.entrain_memory = spy_entrain  # 实例属性替换（forward 内部调用点）
        mem_vec = torch.randn(3072)
        cortex.ensemble.continuous_forward(
            shared_embeddings=emb, active_nids=zh_domain_all,
            seed_memories=[(mem_vec, 0.8, 0.7)], ct=my_ct,
        )
        check("C3. seed_memories 3 元组 → 按记忆相位对齐",
              calls3.get("phase") == 0.7,
              f"entrain_target={calls3.get('phase')}")
    except Exception as e:
        check("C3. seed_memories 3 元组 → 按记忆相位对齐", False, f"err={e}")
    try:
        my_ct2 = ContinuousResonance()
        calls4 = {}

        def spy_entrain2(target_phase=0.0):
            calls4["phase"] = float(target_phase)
            my_ct2._memory_entrained = True
            my_ct2._entrain_phase = float(target_phase)

        my_ct2.entrain_memory = spy_entrain2
        cortex.ensemble.continuous_forward(
            shared_embeddings=emb, active_nids=zh_domain_all,
            seed_memories=[(mem_vec, 0.8)], ct=my_ct2,
        )
        check("C4. 2 元组 seed_memories 回退峰值（增量五零回归）",
              calls4.get("phase") == 0.0,
              f"entrain_target={calls4.get('phase')}")
    except Exception as e:
        check("C4. 2 元组 seed_memories 回退峰值", False, f"err={e}")

    # ── D. 生产（cortex 相位截获 / 注入 / 任务链）──
    print("\n[D] 生产接线（generate / get_last_phase / task_chain）...", flush=True)
    try:
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是神经网络。"),
            max_tokens=32, domain="zh", temperature=0.55,
            memory_vectors=[(torch.randn(3072), 0.5, 0.3)],
        )
        lp = cortex.get_last_phase()
        check("D1. generate memory_vectors 3 元组可用（非空）",
              isinstance(out, str) and len(out.strip()) > 0,
              f"out={out[:30]!r}")
        check("D2. get_last_phase 截获连续共振相位",
              isinstance(lp, float), f"phase={lp}")
    except Exception as e:
        check("D1. generate memory_vectors 3 元组可用", False, f"err={e}")
        check("D2. get_last_phase 截获连续共振相位", False, f"err={e}")
    try:
        cortex.generate_task_chain([
            TaskSet(prompt=build_dialogue_prompt("记录一条带相位的记忆"),
                    mode="continuous", domain="zh", max_tokens=24,
                    record_memory=True, memory_label="KoPE任务链记忆"),
        ])
        engine2 = get_sleep_engine()
        pend2 = list(engine2.pending_field_memories)
        ok_phase = any(len(p) >= 4 and p[3] is not None for p in pend2)
        check("D3. task_chain 记忆写入带相位",
              ok_phase, f"pending={[(None, p[1], p[3] if len(p) >= 4 else None) for p in pend2]}")
        engine2.pending_field_memories = []
    except Exception as e:
        check("D3. task_chain 记忆写入带相位", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
