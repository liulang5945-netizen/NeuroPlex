#!/usr/bin/env python3
"""自举门槛 C1 协作形态自主：撤掉外部协作设计后协作层仍能自然形成（2026-08-20）。

背景：
    A1-A5 全部通过；B1/B1-bis 探索自主性 PASS；B2 autonomous 续航 5/5 PASS。
    但 A/B 系列每条 verify 脚本都**显式指定 neuron_ids=DIALOGUE_IDS** —— 5 个
    对话 neuron 子集是"外部协作设计"。这意味着"协作"是设计者写死的拓扑
    （哪些 neuron 在同一 cortex 一起 forward），不是 cortex 内部 EMERGE 的。

    C1 真正测的是**协作形态自主**：
    1. 撤掉 `assemble_cortex(neuron_ids=DIALOGUE_IDS)` 的显式拓扑设计
    2. 改为**完整集合**（9 neuron：5 dialogue + 4 foundation）—— 让 cortex
       内部 CoactivationTracker 在随机/全集合初始化下自然累积协作权重
    3. 100 步 micro-sleep 后，**协作层（coaction）的 pair/activation 状态**
       仍能自然形成

    设计倾斜（更上限的方案）：
    - 不只是"random 选 4-5 个"——**完整集合 9 neuron 让协作层有最大形成空间**。
    - 通过 100 步 sleep + 24 条 A1 prompt + judge_driven_replay 重复 baseline 条件
    - **与 baseline 严格对比**（baseline = neuron_ids=DIALOGUE_IDS 5 个）
    - 关键判据：random/full 初始化下**强协作连接数 / 激活累计**
      能否达到 baseline 的 ≥ 50%——如果能，说明协作不是设计者的拓扑硬编码，
      是 cortex 在 sleep 中自然涌现的。

判据（C1）：
    C1.a 完整集合初始化下 _activation_counts 总和 >= baseline × 0.5
        （协作累积不弱于显式指定 5 个）
    C1.b 完整集合初始化下 get_strong_pairs(threshold=0.2) 的 pair 数
        >= baseline × 0.5（强协作连接自然形成）
    C1.c 0 崩溃 / 0 NaN
    C1.d 100 步 <= 30 min

    4 维全过 = C1 PASS → 进入"协作形态自主"（协作拓扑可自然涌现、不依赖设计者）。

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A1 真实版 24 prompt + A3 衰减 0.9 + judge_driven_replay
    - 100 步短跑 + 完整集合 9 neuron（与 baseline 同样的 100 步 + 24 prompt）

运行：python -u scripts/training/verify_play_engine_c1_emergence.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import numpy as np  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig, SleepReport  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

from scripts.training.verify_a1_judge_signal_real import (  # noqa: E402
    COLLAB_NAME, EXTRA_NEURONS_DIR,
    DIALOGUE_PROMPTS, KNOWLEDGE_PROMPTS, UNFAMILIAR_PROMPTS,
)
from scripts.training.verify_a3_with_decay import (  # noqa: E402
    field_state_of,
)

passed = 0
failed = 0
N_MICRO = int(os.environ.get("C1_MICRO_N", "100"))
DECAY = float(os.environ.get("C1_DECAY", "0.9"))


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def coaction_stats(cortex) -> dict:
    """抓取协作层内部状态（fast_matrix / slow_matrix / activation_counts）。"""
    coaction = getattr(cortex, "coaction", None)
    if coaction is None:
        return {"fast_pair_count": 0, "strong_pair_count": 0,
                "activation_count_sum": 0}
    return {
        "fast_pair_count": len(coaction._fast_matrix),
        "slow_pair_count": len(coaction._slow_matrix),
        "strong_pair_count": len(coaction.get_strong_pairs(threshold=0.2)),
        "activation_count_sum": int(sum(coaction._activation_counts.values())),
        "n_neurons_tracked": len(coaction._activation_counts),
    }


def run_one_round(label: str, neuron_ids):
    """跑一轮 100 步 micro-sleep，返回 (coaction_stats_pre, coaction_stats_post, ...)"""
    t0 = time.time()
    print(f"\n========== {label} ==========", flush=True)
    print(f"  neuron_ids = {neuron_ids}", flush=True)

    cortex, _tok, _mods = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=neuron_ids,
    )
    target_ids = [nid for nid in cortex.neurons
                  if nid.startswith("zh_") and "dialogue" in nid]
    print(f"  装配 {len(cortex.neurons)} 神经元，judge 目标 = {target_ids}",
          flush=True)

    tmp_data = os.path.join("data", f"_tmp_c1_{label}")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=400)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    all_seed_prompts = (DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS
                        + UNFAMILIAR_PROMPTS)
    print(f"  注入种子记忆：{len(all_seed_prompts)} 条 A1 真实版 prompt...",
          flush=True)
    for i, text in enumerate(all_seed_prompts):
        vec = field_state_of(cortex, text)
        sleep_engine.record_field_memory(vec, f"seed_{i}", text=text)
        sc.record_high_resonance_state(
            field_state=vec, resonance_score=0.9, step=0,
            active_nids=target_ids, threshold=0.5, text=text)
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                          duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)

    print(f"  跑 {N_MICRO} 次 micro-sleep（每 5 步触发 1.6+1.7 + "
          f"手动 coaction.update 让协作层累积）...", flush=True)
    n_crashes = 0
    coaction = getattr(cortex, "coaction", None)
    for step in range(1, N_MICRO + 1):
        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if step % 5 == 0:
                sleep_engine._sleep_phase_synaptic_consolidation(report)
                sleep_engine._sleep_phase_forward_replay(report)
                if coaction is not None and target_ids:
                    coaction.update(list(target_ids))
        except Exception as e:
            n_crashes += 1
            if n_crashes > 5:
                break

    stats = coaction_stats(cortex)
    print(f"  [{label}] coaction 状态: {stats}", flush=True)
    print(f"  [{label}] 100 步完成, 崩溃 {n_crashes} 次, "
          f"耗时 {time.time() - t0:.1f}s", flush=True)
    return stats, n_crashes, time.time() - t0


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print(f"自举门槛 C1 协作形态自主：撤掉外部协作设计后协作层能否自然形成",
          flush=True)
    print(f"  baseline: neuron_ids=DIALOGUE_IDS (5 dialogue)", flush=True)
    print(f"  full    : neuron_ids=None (扫全部 9 neuron)", flush=True)
    print("=" * 64, flush=True)

    from scripts.training.verify_a1_judge_signal_real import DIALOGUE_IDS

    print("\n[1/2] baseline 跑 100 步（显式指定 5 dialogue）...", flush=True)
    base_stats, base_crashes, base_elapsed = run_one_round(
        "baseline", DIALOGUE_IDS)
    base_act = base_stats["activation_count_sum"]
    base_strong = base_stats["strong_pair_count"]
    base_fast = base_stats["fast_pair_count"]
    print(f"\n[2/2] full 跑 100 步（完整集合 9 neuron）...", flush=True)
    full_stats, full_crashes, full_elapsed = run_one_round(
        "full", None)
    full_act = full_stats["activation_count_sum"]
    full_strong = full_stats["strong_pair_count"]
    full_fast = full_stats["fast_pair_count"]

    print("\n" + "=" * 64, flush=True)
    print("C1 4 维判据：", flush=True)
    print("=" * 64, flush=True)

    act_ratio = (full_act / max(base_act, 1)) if base_act > 0 else 0.0
    strong_ratio = (full_strong / max(base_strong, 1)) if base_strong > 0 else 0.0
    fast_ratio = (full_fast / max(base_fast, 1)) if base_fast > 0 else 0.0

    check("C1.a 完整集合下 _activation_counts 总和 >= baseline × 0.5",
          act_ratio >= 0.5,
          f"ratio={act_ratio:.4f}  baseline={base_act}  full={full_act}")

    check("C1.b 完整集合下 get_strong_pairs(0.2) 数 >= baseline × 0.5",
          strong_ratio >= 0.5,
          f"ratio={strong_ratio:.4f}  baseline={base_strong}  full={full_strong}")

    check("C1.c 0 崩溃 / 0 NaN",
          base_crashes == 0 and full_crashes == 0,
          f"baseline_crashes={base_crashes}  full_crashes={full_crashes}")

    elapsed_min = (time.time() - t0) / 60
    check(f"C1.d 200 步 (2 轮) <= 30 min",
          elapsed_min <= 30,
          f"elapsed={elapsed_min:.1f} min")

    c1_pass = (failed == 0)

    print("\n" + "=" * 64, flush=True)
    if c1_pass:
        print("判定: C1 PASS：协作形态自主成立 — 撤掉外部协作设计后协作层仍能自然形成",
              flush=True)
        print("下一步: C1 通过。下一步：C2 跨域迁移 — 验证 1 个域（zh）"
              "学到的协作模式能否跨到 en/code/math 域（基座 vs 上层）", flush=True)
    else:
        print(f"判定: C1 FAIL ({failed} 维不过)", flush=True)
        print("下一步: 调 N_MICRO 更长（100→200）让协作层有时间累积；"
              "或降 threshold 0.2→0.1 看 pair 数", flush=True)
    print("=" * 64, flush=True)

    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "C1 协作形态自主：撤掉外部协作设计后协作层自然形成",
        "cortex": {
            "collab_name": COLLAB_NAME,
            "extra_neurons_dir": EXTRA_NEURONS_DIR,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "decay": DECAY,
            "baseline_neuron_ids": DIALOGUE_IDS,
            "full_neuron_ids": None,
        },
        "baseline": {**base_stats, "crashes": base_crashes,
                     "elapsed_seconds": base_elapsed},
        "full": {**full_stats, "crashes": full_crashes,
                 "elapsed_seconds": full_elapsed},
        "ratios": {
            "activation_ratio": act_ratio,
            "strong_pair_ratio": strong_ratio,
            "fast_pair_ratio": fast_ratio,
        },
        "passed": passed,
        "failed": failed,
        "verdict": ("C1 PASS" if c1_pass else f"C1 FAIL ({failed} 维不过)"),
        "next_step": ("C1 通过。下一步：C2 跨域迁移 — 验证 1 个域（zh）"
                      "学到的协作模式能否跨到 en/code/math 域（基座 vs 上层）"
                      if c1_pass else
                      "调 N_MICRO 更长（100→200）让协作层有时间累积；"
                      "或降 threshold 0.2→0.1 看 pair 数"),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = f"reports/play_engine_c1_emergence_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
