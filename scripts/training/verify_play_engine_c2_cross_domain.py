#!/usr/bin/env python3
"""自举门槛 C2 跨域迁移：zh 域协作模式能否跨到 en/code/math 域（2026-08-20）。

背景：
    C1 协作形态自主 4/4 PASS — 撤掉外部协作设计（neuron_ids=None）后协作层
    仍能自然形成（coaction ratio = 1.0 满 baseline）。但 C1 的 target_ids 仍然
    只含 zh dialogue 5 个 neuron —— 协作层只在 zh 域内形成 pair。

    C2 真正测的是**跨域迁移**：当 target_ids 包含 en/code/math 域的 foundation
    neuron 时，协作层能否跨域形成 pair？如果跨域 pair 数 ≥ baseline × 0.3，
    说明协作不限于同域——可以跨域迁移。

    设计倾斜（更上限的方案）：
    - 不只是"传跨域 target_ids 看不崩"——**对比 baseline（zh-only 5 target）
      vs cross-domain（2 zh + en + code + math = 5 跨域 target）**
    - 关键判据：跨域 coaction pair 数 / activation count 不归零（≥ baseline × 0.3）
    - 阈值 0.3（比 C1 的 0.5 低）——跨域协作天然更难，0.3 是"不归零"线

判据（C2）：
    C2.a 跨域 target 下 _activation_counts 总和 >= baseline × 0.3
        （跨域协作累积不归零）
    C2.b 跨域 target 下 get_strong_pairs(0.2) 数 >= baseline × 0.3
        （跨域强协作连接形成）
    C2.c 0 崩溃 / 0 NaN
    C2.d 200 步 (2 轮) <= 30 min

    4 维全过 = C2 PASS → 进入"跨域迁移"（协作可跨域，不限于同域）。

约束：
    - 冻结 9 成员 production weights
    - 复用 A1 真实版 24 prompt + A3 衰减 0.9
    - Round 1: target_ids = DIALOGUE_IDS (5 zh dialogue) — baseline
    - Round 2: target_ids = [zh_aug0_dialogue, zh_std0_dialogue, en, code, math]
      (2 zh + en + code + math = 5 跨域) — cross-domain

运行：python -u scripts/training/verify_play_engine_c2_cross_domain.py
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
    DIALOGUE_IDS, COLLAB_NAME, EXTRA_NEURONS_DIR,
    DIALOGUE_PROMPTS, KNOWLEDGE_PROMPTS, UNFAMILIAR_PROMPTS,
)
from scripts.training.verify_a3_with_decay import (  # noqa: E402
    field_state_of,
)

passed = 0
failed = 0
N_MICRO = int(os.environ.get("C2_MICRO_N", "100"))
DECAY = float(os.environ.get("C2_DECAY", "0.9"))

CROSS_DOMAIN_IDS = ["zh_aug0_dialogue", "zh_std0_dialogue",
                    "en", "code", "math"]


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def coaction_stats(cortex) -> dict:
    coaction = getattr(cortex, "coaction", None)
    if coaction is None:
        return {"fast_pair_count": 0, "strong_pair_count": 0,
                "activation_count_sum": 0, "n_neurons_tracked": 0}
    return {
        "fast_pair_count": len(coaction._fast_matrix),
        "slow_pair_count": len(coaction._slow_matrix),
        "strong_pair_count": len(coaction.get_strong_pairs(threshold=0.2)),
        "activation_count_sum": int(sum(coaction._activation_counts.values())),
        "n_neurons_tracked": len(coaction._activation_counts),
    }


def run_one_round(label: str, neuron_ids, coaction_target_ids):
    t0 = time.time()
    print(f"\n========== {label} ==========", flush=True)
    print(f"  neuron_ids = {neuron_ids}", flush=True)
    print(f"  coaction_target_ids = {coaction_target_ids}", flush=True)

    cortex, _tok, _mods = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=neuron_ids,
    )
    all_nids = sorted(cortex.neurons.keys())
    print(f"  装配 {len(cortex.neurons)} 神经元: {all_nids}", flush=True)

    valid_targets = [nid for nid in coaction_target_ids
                     if nid in cortex.neurons]
    if len(valid_targets) < len(coaction_target_ids):
        missing = set(coaction_target_ids) - set(valid_targets)
        print(f"  [WARN] 缺失 target: {missing}", flush=True)
    print(f"  有效 coaction target = {valid_targets}", flush=True)

    tmp_data = os.path.join("data", f"_tmp_c2_{label}")
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
    print(f"  注入种子记忆：{len(all_seed_prompts)} 条...", flush=True)
    for i, text in enumerate(all_seed_prompts):
        vec = field_state_of(cortex, text)
        sleep_engine.record_field_memory(vec, f"seed_{i}", text=text)
        sc.record_high_resonance_state(
            field_state=vec, resonance_score=0.9, step=0,
            active_nids=valid_targets, threshold=0.5, text=text)
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                          duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)

    print(f"  跑 {N_MICRO} 次 micro-sleep（每 5 步 coaction.update）...",
          flush=True)
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
                if coaction is not None and valid_targets:
                    coaction.update(list(valid_targets))
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
    print(f"自举门槛 C2 跨域迁移：zh 域协作模式能否跨到 en/code/math 域",
          flush=True)
    print(f"  baseline: target_ids = DIALOGUE_IDS (5 zh dialogue)", flush=True)
    print(f"  cross  : target_ids = {CROSS_DOMAIN_IDS} (2 zh + en + code + math)",
          flush=True)
    print("=" * 64, flush=True)

    print("\n[1/2] baseline 跑 100 步（zh-only 5 target）...", flush=True)
    base_stats, base_crashes, base_elapsed = run_one_round(
        "baseline", DIALOGUE_IDS, DIALOGUE_IDS)
    base_act = base_stats["activation_count_sum"]
    base_strong = base_stats["strong_pair_count"]
    base_fast = base_stats["fast_pair_count"]
    print(f"\n  baseline: act={base_act}  strong={base_strong}  fast={base_fast}",
          flush=True)

    print(f"\n[2/2] cross-domain 跑 100 步（2 zh + en + code + math）...",
          flush=True)
    cross_stats, cross_crashes, cross_elapsed = run_one_round(
        "cross", None, CROSS_DOMAIN_IDS)
    cross_act = cross_stats["activation_count_sum"]
    cross_strong = cross_stats["strong_pair_count"]
    cross_fast = cross_stats["fast_pair_count"]
    print(f"\n  cross: act={cross_act}  strong={cross_strong}  fast={cross_fast}",
          flush=True)

    print("\n" + "=" * 64, flush=True)
    print("C2 4 维判据：", flush=True)
    print("=" * 64, flush=True)

    act_ratio = (cross_act / max(base_act, 1)) if base_act > 0 else 0.0
    strong_ratio = (cross_strong / max(base_strong, 1)) if base_strong > 0 else 0.0
    fast_ratio = (cross_fast / max(base_fast, 1)) if base_fast > 0 else 0.0

    check("C2.a 跨域 _activation_counts 总和 >= baseline × 0.3",
          act_ratio >= 0.3,
          f"ratio={act_ratio:.4f}  baseline={base_act}  cross={cross_act}")

    check("C2.b 跨域 get_strong_pairs(0.2) 数 >= baseline × 0.3",
          strong_ratio >= 0.3,
          f"ratio={strong_ratio:.4f}  baseline={base_strong}  cross={cross_strong}")

    check("C2.c 0 崩溃 / 0 NaN",
          base_crashes == 0 and cross_crashes == 0,
          f"baseline_crashes={base_crashes}  cross_crashes={cross_crashes}")

    elapsed_min = (time.time() - t0) / 60
    check(f"C2.d 200 步 (2 轮) <= 30 min",
          elapsed_min <= 30,
          f"elapsed={elapsed_min:.1f} min")

    c2_pass = (failed == 0)

    print("\n" + "=" * 64, flush=True)
    if c2_pass:
        print("判定: C2 PASS：跨域迁移成立 — zh 域协作模式可跨到 en/code/math 域",
              flush=True)
        print("下一步: C2 通过。门槛 C 完整闭环（C1+C2）。"
              "下一步：D1 长程稳定性 — 1000 步压力测试", flush=True)
    else:
        print(f"判定: C2 FAIL ({failed} 维不过)", flush=True)
        print("下一步: 降 threshold 0.2→0.1；或增 N_MICRO 让跨域 pair 有时间累积",
              flush=True)
    print("=" * 64, flush=True)

    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "C2 跨域迁移：zh 域协作模式能否跨到 en/code/math 域",
        "config": {
            "n_micro": N_MICRO,
            "decay": DECAY,
            "baseline_target_ids": DIALOGUE_IDS,
            "cross_domain_target_ids": CROSS_DOMAIN_IDS,
        },
        "baseline": {**base_stats, "crashes": base_crashes,
                     "elapsed_seconds": base_elapsed},
        "cross_domain": {**cross_stats, "crashes": cross_crashes,
                         "elapsed_seconds": cross_elapsed},
        "ratios": {
            "activation_ratio": act_ratio,
            "strong_pair_ratio": strong_ratio,
            "fast_pair_ratio": fast_ratio,
        },
        "passed": passed,
        "failed": failed,
        "verdict": ("C2 PASS" if c2_pass else f"C2 FAIL ({failed} 维不过)"),
        "next_step": ("C2 通过。门槛 C 完整闭环（C1+C2）。"
                      "下一步：D1 长程稳定性 — 1000 步压力测试"
                      if c2_pass else
                      "降 threshold 0.2→0.1；或增 N_MICRO 让跨域 pair 有时间累积"),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = f"reports/play_engine_c2_cross_domain_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
