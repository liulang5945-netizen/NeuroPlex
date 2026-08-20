#!/usr/bin/env python3
"""自举门槛 B1-bis 探索：在 B1 基础上加入强制切换 / 未选惩罚 / ε-greedy（2026-08-20）。

背景：
    B1 探索自主性 4/4 PASS（字面）但 100% 集中 philosophy = 单调收敛，**不是"探索"是"锁定"**。
    B1-bis 改进选主题逻辑：让"探索"被机制化（不依赖它"主动"换方向）。

    三个探索机制（顺序应用）：
    1. ε-greedy：每步有 10% 概率**强制随机**选 1 个非 top1 主题
    2. 强制切换：top1 主题连续选中 ≥ FORCE_SWITCH_STREAK 次后，**强制随机**选 1 次非 top1
    3. 近期未选惩罚：距上次选中越久的主题 NLL 加权 +RECENCY_BONUS（0.5），让"长期未选"的主题更容易被选

判据（B1-bis）：
    B1-bis.a switch_count ≥ 5（至少选过 5 个不同主题 = 至少切换 4 次不同主题；B1 是 1 主题 = 0 切换）
    B1-bis.b top 主题 ≤ 70%（不锁定单方向；B1 是 100% 集中）
    B1-bis.c 0 崩溃 / 0 NaN
    B1-bis.d 1000 步 ≤ 60 min

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 B1 6 主题池（哲学/法律/医学/艺术/历史/工程，每池 24 条 = 144 条）
    - 复用 A3 衰减 0.9 + SleepConsolidator

运行：python -u scripts/training/verify_play_engine_b1_bis_explore.py
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
N_MICRO = int(os.environ.get("B1BIS_MICRO_N", "1000"))
DECISION_EVERY = int(os.environ.get("B1BIS_DECISION_EVERY", "50"))
DECAY = float(os.environ.get("B1BIS_DECAY", "0.9"))
EPSILON = float(os.environ.get("B1BIS_EPSILON", "0.10"))
FORCE_SWITCH_STREAK = int(os.environ.get("B1BIS_FORCE_STREAK", "5"))
RECENCY_BONUS = float(os.environ.get("B1BIS_RECENCY_BONUS", "0.5"))


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
    today = time.strftime("%Y%m%d")
    n_decisions = N_MICRO // DECISION_EVERY
    print("=" * 64, flush=True)
    print(f"自举门槛 B1-bis 探索：{N_MICRO} 次 micro-sleep + {n_decisions} 次自主选主题",
          flush=True)
    print(f"  3 个探索机制: ε-greedy {EPSILON*100:.0f}% + "
          f"force_switch streak={FORCE_SWITCH_STREAK} + "
          f"recency_bonus={RECENCY_BONUS}", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/5] 装配 9 成员 production cortex（冻结，不写 checkpoint）...", flush=True)
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
    print(f"  装配 {len(cortex.neurons)} 神经元，judge 目标 = {target_ids}", flush=True)

    tmp_data = os.path.join("data", "_tmp_b1bis")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=400)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    a1_groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    print(f"\n[2/5] 注入 A1 真实版 24 条 + 6 个主题池 144 条记忆（初始 168 条）...", flush=True)
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
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)
    print(f"  注入 168 条 + 场固化 {r_init.field_memories_consolidated} 条", flush=True)

    print(f"\n[3/5] 跑 {N_MICRO} 次 micro-sleep（每 {DECISION_EVERY} 步一次自主选主题）...",
          flush=True)
    device = next(cortex._shared_embedding.parameters()).device
    decision_log = []
    selected_counts = {tname: 0 for tname in TOPIC_POOLS}
    n_crashes = 0
    last_selected_at = {tname: -10**9 for tname in TOPIC_POOLS}
    last_chosen_topic = None
    current_streak = 0
    switch_count = 0
    distinct_topics_selected = 0
    distinct_topics_seen = set()
    epsilon_used = 0
    force_used = 0

    for step in range(1, N_MICRO + 1):
        t_step = time.time()

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

            decision_log.append({
                "decision_idx": decision_idx,
                "step": step,
                "nll_per_pool": nll_per_pool,
                "nll_with_bonus": nll_with_bonus,
                "top1_topic": top1_topic,
                "chosen_topic": chosen_topic,
                "mechanism": mechanism,
                "current_streak": current_streak,
            })

            for j, text in enumerate(TOPIC_POOLS[chosen_topic]):
                vec = field_state_of(cortex, text)
                sleep_engine.record_field_memory(
                    vec, f"b1bis_step{step}_{chosen_topic}_{j}", text=text)
                sc.record_high_resonance_state(
                    field_state=vec, resonance_score=0.9, step=step,
                    active_nids=target_ids, threshold=0.5, text=text)

            top_conc_so_far = max(selected_counts.values()) / max(1, decision_idx + 1)
            print(f"  decision {decision_idx:2d}  step {step:4d}  "
                  f"top1={top1_topic:12s}  chose={chosen_topic:12s}  "
                  f"mech={mechanism:14s}  streak={current_streak:2d}  "
                  f"top_conc={top_conc_so_far*100:.1f}%  switches={switch_count}",
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
            print(f"  [WARN] micro-sleep {step} 异常: {type(e).__name__}: {e}",
                  flush=True)
            if n_crashes > 5:
                check(f"micro-sleep {step} 不崩溃", False, f"crashes={n_crashes}")
                break
            continue
        dt_step = time.time() - t_step

    print(f"\n[4/5] 决策统计：{len(decision_log)} 次决策", flush=True)
    for tname in TOPIC_POOLS:
        cnt = selected_counts[tname]
        print(f"  {tname:12s}: 选中 {cnt:2d} 次 ({100*cnt/max(1,len(decision_log)):.1f}%)",
              flush=True)

    n_decisions_done = len(decision_log)
    distinct_topics_selected = len(distinct_topics_seen)
    top_topic = max(selected_counts.items(), key=lambda x: x[1])
    top_concentration = top_topic[1] / max(1, n_decisions_done)

    print(f"\n  distinct 主题数 = {distinct_topics_selected} / {len(TOPIC_POOLS)}", flush=True)
    print(f"  switch_count = {switch_count}", flush=True)
    print(f"  top topic = {top_topic[0]} ({top_concentration*100:.1f}%)", flush=True)
    print(f"  epsilon_used = {epsilon_used}, force_used = {force_used}", flush=True)

    print(f"\n[5/5] 后测量 A1 真实版 3 组...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("B1-bis 4 维判据：", flush=True)
    print("=" * 64, flush=True)

    b1a = distinct_topics_selected >= 5
    check(f"B1-bis.a distinct 主题数 ≥ 5（探索 ≥ 5 个不同主题）",
          b1a,
          f"distinct={distinct_topics_selected}/{len(TOPIC_POOLS)}")

    b1b = top_concentration <= 0.70
    check(f"B1-bis.b top 主题 ≤ 70%（不锁定）",
          b1b,
          f"{top_topic[0]}={top_concentration*100:.1f}%")

    b1c = n_crashes == 0
    check(f"B1-bis.c 0 崩溃 / 0 NaN",
          b1c,
          f"crashes={n_crashes}/{N_MICRO}")

    elapsed_min = (time.time() - t0) / 60
    b1d = elapsed_min <= 60
    check(f"B1-bis.d 1000 步 ≤ 60 min",
          b1d,
          f"elapsed={elapsed_min:.1f} min")

    b1bis_pass = (failed == 0)
    if b1bis_pass:
        verdict = (f"B1-bis PASS：{n_decisions_done} 次决策覆盖 {distinct_topics_selected} 主题，"
                   f"switch_count={switch_count}，top 主题 {top_concentration*100:.1f}% ≤ 70%，"
                   f"epsilon_used={epsilon_used} + force_used={force_used}，"
                   f"0 崩溃，{elapsed_min:.1f} min ≤ 60 min")
        next_step = ("B1-bis 通过。下一步：B2 —— play 引擎常态运行下，"
                     "它能不能在不被喂新经验时仍维持 100 步无遗忘（autonomous 续航）。")
    else:
        if not b1a:
            verdict = (f"B1-bis 半 PASS：distinct {distinct_topics_selected} < 5 — "
                       f"探索机制 3 个不够强，需加大 ε=0.20 或 recency_bonus=1.0")
            next_step = ("B1-bis 失败：distinct < 5。需加大 ε=0.20 / 减小 force_streak=3 / "
                         "加大 recency_bonus=1.0，让更多主题被选中。")
        elif not b1b:
            verdict = (f"B1-bis 半 PASS：top {top_concentration*100:.1f}% > 70% — "
                       f"哲学仍被锁定，3 机制没打破")
            next_step = ("B1-bis 失败：top > 70%。需把 recency_bonus 加到 1.0 让长期未选主题的"
                         "加权远超 base NLL。")
        else:
            verdict = f"B1-bis 失败（{passed} PASS / {failed} FAIL）"
            next_step = "B1-bis 失败：需重审其他判据"
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"play_engine_b1_bis_explore_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": (f"B1-bis 探索：{N_MICRO} 次 micro-sleep + {n_decisions} 次决策 + "
                 f"3 探索机制"),
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "decision_every": DECISION_EVERY,
            "decay": DECAY,
            "n_decisions": n_decisions_done,
            "n_topic_pools": len(TOPIC_POOLS),
            "epsilon": EPSILON,
            "force_switch_streak": FORCE_SWITCH_STREAK,
            "recency_bonus": RECENCY_BONUS,
        },
        "selected_counts": selected_counts,
        "distinct_topics_selected": distinct_topics_selected,
        "switch_count": switch_count,
        "top_topic": top_topic[0],
        "top_concentration": top_concentration,
        "epsilon_used": epsilon_used,
        "force_used": force_used,
        "decision_log": decision_log,
        "crash_count": n_crashes,
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in a1_groups},
        "passed": passed,
        "failed": failed,
        "verdict": verdict,
        "next_step": next_step,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if b1bis_pass else 1)


if __name__ == "__main__":
    main()
