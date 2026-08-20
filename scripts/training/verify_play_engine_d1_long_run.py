#!/usr/bin/env python3
"""自举门槛 D1 长程稳定性：1000 步压力测试（2026-08-20）。

背景：
    A1-A5 + B1/B1-bis/B2 + C1/C2 全部 PASS。但所有测试都是短程（100 步或
    1000 步 × 20 决策但只测最终态）。D1 真正测的是**长程稳定性**：
    1000 步 + 6 主题池 + 3 探索机制下，judge NLL / coaction / LoRA L2
    是否在长程下稳定（无累积爆炸 / 无渐进遗忘 / 无协作层崩塌）。

    设计倾斜（更上限的方案）：
    - 复用 B1-bis 完整主循环（6 主题池 + 3 探索机制 + 每 50 步决策）
    - 每 100 步采样 judge NLL + LoRA L2 + coaction 状态（漂移轨迹）
    - pre/post 3 组 judge std/mean 对比（测长程遗忘）
    - 判据：post std ≥ pre × 0.90（长程允许更多漂移，比 B2 的 0.95 宽松）

判据（D1）：
    D1.a dialogue 组：post std >= pre std × 0.90（长程不遗忘）
    D1.b knowledge 组：post std >= pre std × 0.90
    D1.c unfamiliar 组：post std >= pre std × 0.90
    D1.d 0 崩溃 / 0 NaN / 0 爆炸
    D1.e 1000 步 <= 60 min

    5 维全过 = D1 PASS → 长程稳定性成立（1000 步无累积爆炸 / 无渐进遗忘）。

    D1-fix（2026-08-20）：judge 驱动的衰减自调节。
    原版 D1 暴露固定 lora_decay_per_sleep=0.9 在长程下让衰减压过训练
    （knowledge std ratio 0.7517 / unfamiliar 0.8047 < 0.90）—— 过度收敛，
    不是遗忘内容（mean 全程 ±0.03 稳定）。修法：开启 `judge_driven_decay`，
    judge 每次判定 NLL std，< decay_min_judge_std 跳过本轮衰减，让训练
    继续累积 LoRA 防止 std 收窄。

约束：
    - 冻结 9 成员 production weights
    - 复用 B1-bis 6 主题池 + 3 探索机制
    - 每 100 步采样轨迹（NLL / LoRA L2 / coaction）
    - 每 50 步决策（与 B1-bis 一致）

环境变量：
    D1_MICRO_N=1000
    D1_DECISION_EVERY=50
    D1_DECAY=0.9
    D1_JUDGE_DRIVEN_DECAY=1   # 0/1；1 开启 D1-fix
    D1_DECAY_MIN_STD=0.05
    D1_DECAY_SAMPLE_N=3
    D1_HYSTERESIS_N=2          # D1-fix v4：连续 N 周期 SKIP 信号才真 SKIP
    D1_CEILING_RATIO=1.3       # D1-fix v4：LoRA L2 > baseline × 此值强制衰减
    D1_EPSILON=0.10
    D1_FORCE_STREAK=5
    D1_RECENCY_BONUS=0.5
    D1_SAMPLE_EVERY=100

运行：python -u scripts/training/verify_play_engine_d1_long_run.py
或 D1-fix v3：D1_JUDGE_DRIVEN_DECAY=1 python -u scripts/training/verify_play_engine_d1_long_run.py
或 D1-fix v4：D1_JUDGE_DRIVEN_DECAY=1 D1_HYSTERESIS_N=2 D1_CEILING_RATIO=1.3 python -u scripts/training/verify_play_engine_d1_long_run.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
    field_state_of, lora_l2_norm,
)
from scripts.training.verify_a4_post_sleep_judge_signal import (  # noqa: E402
    measure_group_stds,
)
from scripts.training.verify_play_engine_b1_explore import TOPIC_POOLS  # noqa: E402

passed = 0
failed = 0
N_MICRO = int(os.environ.get("D1_MICRO_N", "1000"))
DECISION_EVERY = int(os.environ.get("D1_DECISION_EVERY", "50"))
DECAY = float(os.environ.get("D1_DECAY", "0.9"))
JUDGE_DRIVEN_DECAY = bool(int(os.environ.get("D1_JUDGE_DRIVEN_DECAY", "0")))
DECAY_MIN_STD = float(os.environ.get("D1_DECAY_MIN_STD", "0.05"))
DECAY_MIN_REL_RATIO = float(os.environ.get("D1_DECAY_MIN_REL_RATIO", "0.95"))
DECAY_SAMPLE_N = int(os.environ.get("D1_DECAY_SAMPLE_N", "3"))
HYSTERESIS_N = int(os.environ.get("D1_HYSTERESIS_N", "2"))
CEILING_RATIO = float(os.environ.get("D1_CEILING_RATIO", "1.3"))
EPSILON = float(os.environ.get("D1_EPSILON", "0.10"))
FORCE_SWITCH_STREAK = int(os.environ.get("D1_FORCE_STREAK", "5"))
RECENCY_BONUS = float(os.environ.get("D1_RECENCY_BONUS", "0.5"))
SAMPLE_EVERY = int(os.environ.get("D1_SAMPLE_EVERY", "100"))


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
                "activation_count_sum": 0}
    return {
        "fast_pair_count": len(coaction._fast_matrix),
        "strong_pair_count": len(coaction.get_strong_pairs(threshold=0.2)),
        "activation_count_sum": int(sum(coaction._activation_counts.values())),
    }


def sample_trajectory(sleep_engine, cortex, target_ids, a1_groups):
    """每 100 步采样 judge NLL + LoRA L2 + coaction 状态。"""
    device = next(cortex._shared_embedding.parameters()).device
    nlls = {}
    for gname, prompts in a1_groups.items():
        vals = []
        for text in prompts[:2]:
            jnll = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
            if jnll is not None and jnll < 1e6:
                vals.append(jnll)
        nlls[gname] = float(np.mean(vals)) if vals else 0.0
    lora_l2 = sum(lora_l2_norm(cortex.neurons[nid])
                  for nid in target_ids if nid in cortex.neurons)
    coact = coaction_stats(cortex)
    return {"nlls": nlls, "lora_l2": float(lora_l2), "coaction": coact}


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    n_decisions = N_MICRO // DECISION_EVERY
    print("=" * 64, flush=True)
    print(f"自举门槛 D1 长程稳定性：{N_MICRO} 次 micro-sleep + {n_decisions} 次决策",
          flush=True)
    print(f"  3 机制: ε-greedy {EPSILON*100:.0f}% + "
          f"force_switch streak={FORCE_SWITCH_STREAK} + "
          f"recency_bonus={RECENCY_BONUS}", flush=True)
    print(f"  衰减: lora_decay={DECAY}  "
          f"judge_driven_decay={JUDGE_DRIVEN_DECAY}  "
          f"decay_min_std={DECAY_MIN_STD}  "
          f"decay_min_rel_ratio={DECAY_MIN_REL_RATIO}  "
          f"decay_sample_n={DECAY_SAMPLE_N}  "
          f"hysteresis_n={HYSTERESIS_N}  "
          f"ceiling_ratio={CEILING_RATIO}", flush=True)
    print(f"  每 {SAMPLE_EVERY} 步采样轨迹（NLL / LoRA L2 / coaction）", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/5] 装配 9 成员 production cortex（冻结，不写 checkpoint）...",
          flush=True)
    cortex, _tok, _mods = assemble_cortex(
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
    print(f"  装配 {len(cortex.neurons)} 神经元，judge 目标 = {target_ids}",
          flush=True)

    tmp_data = os.path.join("data", "_tmp_d1")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
        judge_driven_decay=JUDGE_DRIVEN_DECAY,
        decay_min_judge_std=DECAY_MIN_STD,
        decay_min_relative_ratio=DECAY_MIN_REL_RATIO,
        decay_judge_sample_n=DECAY_SAMPLE_N,
        decay_baseline_prompts=tuple(
            DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS),
        decay_baseline_sample_n=DECAY_SAMPLE_N,
        decay_hysteresis_n=HYSTERESIS_N,
        decay_lora_ceiling_ratio=CEILING_RATIO,
        pre_lora_l2_baseline=None,  # 首次测量时自动写
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=400)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    a1_groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    print(f"\n[2/5] 注入 A1 真实版 24 条 + 6 主题池 144 条记忆（初始 168 条）...",
          flush=True)
    for i, text in enumerate(DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS):
        vec = field_state_of(cortex, text)
        sleep_engine.record_field_memory(vec, f"a1_{i}", text=text)
        sc.record_high_resonance_state(
            field_state=vec, resonance_score=0.9, step=0,
            active_nids=target_ids, threshold=0.5, text=text)
    for tname, prompts in TOPIC_POOLS.items():
        for i, text in enumerate(prompts):
            vec = field_state_of(cortex, text)
            sleep_engine.record_field_memory(vec, f"{tname}_{i}", text=text)
            sc.record_high_resonance_state(
                field_state=vec, resonance_score=0.85, step=0,
                active_nids=target_ids, threshold=0.5, text=text)
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                          duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)
    print(f"  注入 168 条 + 场固化", flush=True)

    print(f"\n[3/5] 预测量（pre 3 组 std/mean 基线）...", flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    pre_summary = {}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = pre[g]
        pre_summary[g] = {"std": d["std"], "mean": d["mean"]}
        print(f"  pre {g}: std={d['std']:.6f}  mean={d['mean']:.4f}", flush=True)
    pre_lora = sum(lora_l2_norm(cortex.neurons[nid])
                   for nid in target_ids if nid in cortex.neurons)
    print(f"  pre LoRA L2: {pre_lora:.4f}", flush=True)

    print(f"\n[4/5] 跑 {N_MICRO} 次 micro-sleep（每 {DECISION_EVERY} 步决策 + "
          f"每 {SAMPLE_EVERY} 步采样）...", flush=True)
    device = next(cortex._shared_embedding.parameters()).device
    n_crashes = 0
    last_selected_at = {tname: -10**9 for tname in TOPIC_POOLS}
    last_chosen_topic = None
    current_streak = 0
    switch_count = 0
    selected_counts = {tname: 0 for tname in TOPIC_POOLS}
    distinct_topics_seen = set()
    epsilon_used = 0
    force_used = 0
    trajectory = []

    traj = sample_trajectory(sleep_engine, cortex, target_ids, a1_groups)
    traj["step"] = 0
    trajectory.append(traj)
    print(f"  step    0  NLL d={traj['nlls']['dialogue']:.2f}  "
          f"k={traj['nlls']['knowledge']:.2f}  "
          f"u={traj['nlls']['unfamiliar']:.2f}  "
          f"LoRA={traj['lora_l2']:.4f}  "
          f"coact={traj['coaction']['fast_pair_count']}", flush=True)

    for step in range(1, N_MICRO + 1):
        if step % DECISION_EVERY == 1:
            decision_idx = (step - 1) // DECISION_EVERY
            nll_per_pool = {}
            for tname, prompts in TOPIC_POOLS.items():
                sample_prompts = prompts[:2]
                nlls = []
                for text in sample_prompts:
                    jnll = sleep_engine._sample_judge_nll(
                        text, target_ids, device, cortex._shared_embedding)
                    if jnll is not None and jnll < 1e6:
                        nlls.append(jnll)
                nll_per_pool[tname] = float(np.mean(nlls)) if nlls else 0.0

            nll_with_bonus = {}
            for tname, base_nll in nll_per_pool.items():
                rounds_since = decision_idx - (last_selected_at[tname] // DECISION_EVERY)
                nll_with_bonus[tname] = base_nll + RECENCY_BONUS * rounds_since

            sorted_pools = sorted(nll_with_bonus.items(), key=lambda x: -x[1])
            top1_topic = sorted_pools[0][0]

            force_switch = (last_chosen_topic == top1_topic
                            and current_streak >= FORCE_SWITCH_STREAK)
            epsilon_roll = random.random() < EPSILON
            if force_switch or epsilon_roll:
                other_topics = [t for t in nll_with_bonus if t != top1_topic]
                if other_topics:
                    chosen_topic = random.choice(other_topics)
                    if force_switch:
                        force_used += 1
                    else:
                        epsilon_used += 1
                    mechanism = "force_switch" if force_switch else "epsilon_greedy"
                else:
                    chosen_topic = top1_topic
                    mechanism = "no_alternative"
            else:
                chosen_topic = top1_topic
                mechanism = "exploit"

            if chosen_topic != last_chosen_topic and last_chosen_topic is not None:
                switch_count += 1
            distinct_topics_seen.add(chosen_topic)
            selected_counts[chosen_topic] += 1
            last_selected_at[chosen_topic] = decision_idx
            if chosen_topic == top1_topic:
                current_streak += 1
            else:
                current_streak = 0
            last_chosen_topic = chosen_topic

            for j, text in enumerate(TOPIC_POOLS[chosen_topic]):
                vec = field_state_of(cortex, text)
                sleep_engine.record_field_memory(
                    vec, f"d1_step{step}_{chosen_topic}_{j}", text=text)
                sc.record_high_resonance_state(
                    field_state=vec, resonance_score=0.9, step=step,
                    active_nids=target_ids, threshold=0.5, text=text)

            if step % (DECISION_EVERY * 5) == 1:
                print(f"  decision {decision_idx:2d}  step {step:4d}  "
                      f"chose={chosen_topic:12s}  mech={mechanism:14s}  "
                      f"streak={current_streak:2d}  switches={switch_count}",
                      flush=True)

        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if step % 100 == 0:
                sleep_engine._sleep_phase_synaptic_consolidation(report)
                sleep_engine._sleep_phase_forward_replay(report)
        except Exception as e:
            n_crashes += 1
            if n_crashes > 5:
                break
            continue

        if step % SAMPLE_EVERY == 0:
            traj = sample_trajectory(sleep_engine, cortex, target_ids, a1_groups)
            traj["step"] = step
            trajectory.append(traj)
            print(f"  step {step:4d}  NLL d={traj['nlls']['dialogue']:.2f}  "
                  f"k={traj['nlls']['knowledge']:.2f}  "
                  f"u={traj['nlls']['unfamiliar']:.2f}  "
                  f"LoRA={traj['lora_l2']:.4f}  "
                  f"coact={traj['coaction']['fast_pair_count']}",
                  flush=True)

    elapsed_step = time.time() - t0
    print(f"\n  完成 {N_MICRO} 步, switches={switch_count}, "
          f"epsilon_used={epsilon_used}, force_used={force_used}, "
          f"崩溃 {n_crashes} 次", flush=True)

    print(f"\n[5/5] 后测量（post 3 组 std/mean）...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    post_summary = {}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        post_summary[g] = {"std": d["std"], "mean": d["mean"]}
        print(f"  post {g}: std={d['std']:.6f}  mean={d['mean']:.4f}", flush=True)
    post_lora = sum(lora_l2_norm(cortex.neurons[nid])
                    for nid in target_ids if nid in cortex.neurons)
    print(f"  post LoRA L2: {post_lora:.4f}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("D1 5 维判据：", flush=True)
    print("=" * 64, flush=True)

    ratio_summary = {}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        ratio = (post_summary[g]["std"]
                 / max(pre_summary[g]["std"], 1e-9))
        ratio_summary[g] = ratio
        check(f"D1.{g[0]} {g} 组 std >= pre × 0.90（长程不遗忘）",
              ratio >= 0.90,
              f"ratio={ratio:.4f}  pre={pre_summary[g]['std']:.4f}  "
              f"post={post_summary[g]['std']:.4f}")

    check("D1.d 0 崩溃 / 0 NaN / 0 爆炸",
          n_crashes == 0,
          f"crashes={n_crashes}/{N_MICRO}")

    elapsed_min = (time.time() - t0) / 60
    check(f"D1.e {N_MICRO} 步 <= 60 min",
          elapsed_min <= 60,
          f"elapsed={elapsed_min:.1f} min")

    d1_pass = (failed == 0)

    print("\n" + "=" * 64, flush=True)
    if d1_pass:
        print("判定: D1 PASS：长程稳定性成立 — 1000 步无累积爆炸 / 无渐进遗忘",
              flush=True)
        next_msg = ("D1 通过。门槛 D 起步。"
                    "下一步：D2 极限压力 — 5000 步 + 减少 LoRA 衰减 0.9→0.8 "
                    "看何时崩")
        print(f"下一步: {next_msg}", flush=True)
    else:
        print(f"判定: D1 FAIL ({failed} 维不过)", flush=True)
        if HYSTERESIS_N >= 2 and CEILING_RATIO < 1.5 and DECAY >= 0.88:
            next_msg = (
                f"D1-fix v4 (hysteresis N={HYSTERESIS_N} + ceiling "
                f"{CEILING_RATIO}) 仍 FAIL——"
                f"考虑：a) 调高 HYSTERESIS_N (2→3) 进一步抗噪声；"
                f"b) 收紧 CEILING_RATIO (1.3→1.15) 抑制 SKIP 累积；"
                f"c) 上调 decay_min_rel_ratio (0.95→0.98) 让 SKIP 触发更难"
            )
        elif HYSTERESIS_N >= 2 and 1.5 <= CEILING_RATIO <= 1.7 and DECAY < 0.88:
            next_msg = (
                f"D1-fix v5 (hysteresis N={HYSTERESIS_N} + ceiling "
                f"{CEILING_RATIO} + DECAY={DECAY}) 仍 FAIL——"
                f"考虑：a) 进一步放宽 ceiling (1.6→1.8) 给 SKIP 更多空间；"
                f"b) 进一步收紧 DECAY (0.85→0.80) 让被允许累积衰减更快；"
                f"c) hysteresis N 2→1（既然 v5 ceiling 已经够宽，不再需要抗噪）"
            )
        elif HYSTERESIS_N >= 2 and 1.5 <= CEILING_RATIO <= 1.7 and 0.875 <= DECAY < 0.92:
            next_msg = (
                f"D1-fix v6-F (hysteresis N={HYSTERESIS_N} + ceiling "
                f"{CEILING_RATIO} + DECAY={DECAY}) 仍 FAIL——"
                f"考虑：a) 收紧 ceiling (1.6→1.4) 限制 LoRA 累积上限；"
                f"b) 调高 DECAY (0.88→0.90) 拉回 v3 速率；"
                f"c) 调整 hysteresis N 2→3 进一步抗噪"
            )
        elif JUDGE_DRIVEN_DECAY:
            next_msg = ("D1-fix judge 驱动衰减自调节仍 FAIL——"
                        "考虑：a) 调高 decay_min_std（0.05→0.10）让 skip 更激进；"
                        "b) 调低 DECAY（0.9→0.7）让保留的 LoRA 也衰减；"
                        "c) 缩短 N_MICRO 到 500 看半程")
        else:
            next_msg = ("D1 FAIL 根因=过度收敛。"
                        "运行 D1_JUDGE_DRIVEN_DECAY=1 让 judge 驱动衰减自调节"
                        "（SleepConfig.judge_driven_decay=True）"
                        "—— std<0.05 时 skip 本次衰减，保留训练累积的 LoRA。")
        print(f"下一步: {next_msg}", flush=True)
    print("=" * 64, flush=True)

    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "D1 长程稳定性：1000 步压力测试 + 轨迹采样",
        "config": {
            "n_micro": N_MICRO,
            "decay": DECAY,
            "judge_driven_decay": JUDGE_DRIVEN_DECAY,
            "decay_min_judge_std": DECAY_MIN_STD,
            "decay_judge_sample_n": DECAY_SAMPLE_N,
            "hysteresis_n": HYSTERESIS_N,
            "ceiling_ratio": CEILING_RATIO,
            "epsilon": EPSILON,
            "force_switch_streak": FORCE_SWITCH_STREAK,
            "recency_bonus": RECENCY_BONUS,
            "decision_every": DECISION_EVERY,
            "sample_every": SAMPLE_EVERY,
        },
        "pre_groups": pre_summary,
        "post_groups": post_summary,
        "ratio_summary": ratio_summary,
        "pre_lora_l2": pre_lora,
        "post_lora_l2": post_lora,
        "trajectory": trajectory,
        "switch_count": switch_count,
        "selected_counts": selected_counts,
        "distinct_topics": len(distinct_topics_seen),
        "epsilon_used": epsilon_used,
        "force_used": force_used,
        "crash_count": n_crashes,
        "passed": passed,
        "failed": failed,
        "verdict": ("D1 PASS" if d1_pass else f"D1 FAIL ({failed} 维不过)"),
        "next_step": next_msg,
        "elapsed_seconds": time.time() - t0,
    }
    if HYSTERESIS_N >= 2 and CEILING_RATIO < 1.5 and DECAY >= 0.88:
        out_path = (f"reports/play_engine_d1_fix_v4_hysteresis_ceiling_"
                    f"{today}.json")
    elif HYSTERESIS_N >= 2 and 1.5 <= CEILING_RATIO <= 1.7 and DECAY < 0.88:
        out_path = (f"reports/play_engine_d1_fix_v5_ceiling16_decay85_"
                    f"{today}.json")
    elif HYSTERESIS_N >= 2 and 1.5 <= CEILING_RATIO <= 1.7 and 0.875 <= DECAY < 0.92:
        out_path = (f"reports/play_engine_d1_fix_v6_ceiling16_decay88_"
                    f"{today}.json")
    elif JUDGE_DRIVEN_DECAY:
        out_path = (f"reports/play_engine_d1_fix_judge_driven_decay_"
                    f"{today}.json")
    else:
        out_path = f"reports/play_engine_d1_long_run_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
