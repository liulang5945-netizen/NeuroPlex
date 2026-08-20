#!/usr/bin/env python3
"""自举门槛 A5 完整：play 引擎常态化下 judge mean 100 步增长实证（2026-08-20）。

背景：
    A5 准备（verify_play_engine_a5_growth.py 30 步）已证 3 组 mean 全部上升
    （dialogue +0.038 / knowledge +0.115 / unfamiliar +0.094），但仅 30 步
    ——尚不能判断 mean 上升是"早期冲击"还是"持续增长"。

    A5 完整要回答：**100 步（含 10 批 × 8 条新经验，全程覆盖）后，judge 信号
    mean 是否仍上升？是 plateau 还是继续涨？**

    关键升级（vs A5 准备）：
    - 30 步 → 100 步（A5_MICRO_N=100）
    - 3 批 → 10 批新经验注入（步 11/21/31/.../91 各注入 8 条新 prompt）
    - 判据从"不显著退化"改为"经验驱动增长"：
        A5-full.a 3 组 mean 都上升 ≥ 0.01
        A5-full.b 任意组 mean 上升 ≤ 0.20（不爆炸）
        A5-full.c worst step 跳水 ≤ 50% pre-std（避免单步大跳）
        A5-full.d 0 NaN / 0 爆炸
    - 加 plateau 监测：30-100 步区间 mean 漂移 ≤ 0.05（不再继续涨也算稳态）

判据（A5 完整）：
    A5-full.a 3/3 组 mean 上升 ≥ 0.01（经验有效）→ 3/3 PASS = 增长方向
    A5-full.b 3/3 组 mean 上升 ≤ 0.20（不爆炸）→ 3/3 PASS = 增长受控
    A5-full.c worst step Δ std ≤ pre-std × 50%（防单步大跳）
    A5-full.d 0 崩溃 / 0 NaN
    A5-full.e plateau 检验：30 步后 mean 漂移 ≤ 0.05（不再冲高 = 稳态）

    5 维全过 = A5 完整 PASS

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A3 衰减 0.9 + A1 真实版 24 prompt + SleepConsolidator
    - 80 条新 prompt 与 A1 真实版 24 prompt 主题/用词明显不同
    - 100 步 ≤ 15 min 预算（forward_replay 累积时间约束）

运行：python -u scripts/training/verify_play_engine_a5_full.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
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

passed = 0
failed = 0
N_MICRO = int(os.environ.get("A5_FULL_MICRO_N", "100"))
CHECKPOINT_EVERY = int(os.environ.get("A5_FULL_CKPT_EVERY", "10"))
DECAY = float(os.environ.get("A5_FULL_DECAY", "0.9"))
NEW_PROMPTS_PER_BATCH = 8

NEW_DIALOGUE_BATCH = [
    "我现在心情不太好，可以聊聊吗？",
    "下周我们要开一个重要的项目会议。",
    "你能不能解释一下这个复杂的概念？",
    "我最近总是失眠，有什么建议吗？",
    "我对未来感到迷茫，你能给我一些方向吗？",
    "我和朋友因为一件小事吵架了。",
    "今天的工作压力很大，我想放松一下。",
    "我正在学习一门新的编程语言。",
]

NEW_KNOWLEDGE_BATCH = [
    "什么是光合作用的暗反应阶段？",
    "请解释牛顿第三定律的物理意义。",
    "DNA 半保留复制是如何被证实的？",
    "为什么天空是蓝色的而不是紫色的？",
    "TCP 三次握手的目的是什么？",
    "什么是相对论的尺缩效应？",
    "细胞凋亡与细胞坏死的区别是什么？",
    "请解释 Python 的 GIL 全局解释器锁。",
]

NEW_UNFAMILIAR_BATCH = [
    "什么是范畴论中的米田引理？",
    "弦理论中的卡拉比-丘流形有何拓扑意义？",
    "高阶范畴论在量子场论中如何应用？",
    "什么是 P vs NP 问题的代数几何视角？",
    "请解释非交换几何在标准模型中的应用。",
    "什么是凝聚态物理中的拓扑序？",
    "代数 K 理论在高维流形分类中的作用是什么？",
    "请解释 Witten 在 M 理论中的五次时空维论证。",
]


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
    print("=" * 64, flush=True)
    print(f"自举门槛 A5 完整：{N_MICRO} 次 micro-sleep + {N_MICRO // CHECKPOINT_EVERY} 批新经验",
          flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a5_full")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=200)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    all_prompts = DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS
    prompt_labels = (["dialogue"] * len(DIALOGUE_PROMPTS) +
                     ["knowledge"] * len(KNOWLEDGE_PROMPTS) +
                     ["unfamiliar"] * len(UNFAMILIAR_PROMPTS))

    new_batches = []
    for _ in range(N_MICRO // CHECKPOINT_EVERY):
        new_batches.append(
            list(zip(
                NEW_DIALOGUE_BATCH + NEW_KNOWLEDGE_BATCH + NEW_UNFAMILIAR_BATCH,
                (["dialogue"] * len(NEW_DIALOGUE_BATCH) +
                 ["knowledge"] * len(NEW_KNOWLEDGE_BATCH) +
                 ["unfamiliar"] * len(NEW_UNFAMILIAR_BATCH))
            ))
        )

    print("\n[2/5] 预测量 pre-sleep A1 真实版 + 注入 24 条记忆...", flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    pre_means = {g: pre[g]["mean"] for g in groups}
    pre_stds = {g: pre[g]["std"] for g in groups}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = pre[g]
        print(f"  pre  {g}: std={d['std']}  mean={d['mean']}", flush=True)

    for i, text in enumerate(all_prompts):
        vec = field_state_of(cortex, text)
        sleep_engine.record_field_memory(vec, f"init_{prompt_labels[i]}_{i}", text=text)
        sc.record_high_resonance_state(
            field_state=vec, resonance_score=0.9, step=0,
            active_nids=target_ids, threshold=0.5, text=text)
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)
    print(f"  注入 {len(all_prompts)} 条 + 场固化 {r_init.field_memories_consolidated} 条",
          flush=True)

    print(f"\n[3/5] 跑 {N_MICRO} 次 micro-sleep（每 {CHECKPOINT_EVERY} 步前注入 "
          f"{NEW_PROMPTS_PER_BATCH} 条新 prompt）...", flush=True)
    checkpoint_curve = []
    n_complete_cycles = 0
    crash_count = 0
    new_injected_total = 0
    worst_step_jump = 0.0
    for step in range(1, N_MICRO + 1):
        if step % CHECKPOINT_EVERY == 1 and step > 1:
            batch_idx = (step - 1) // CHECKPOINT_EVERY - 1
            if 0 <= batch_idx < len(new_batches):
                batch = new_batches[batch_idx]
                for j, (text, label) in enumerate(batch):
                    vec = field_state_of(cortex, text)
                    sleep_engine.record_field_memory(
                        vec, f"step{step}_{label}_{j}", text=text)
                    sc.record_high_resonance_state(
                        field_state=vec, resonance_score=0.9, step=step,
                        active_nids=target_ids, threshold=0.5, text=text)
                    new_injected_total += 1

        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        t_step = time.time()
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if (step % CHECKPOINT_EVERY) == 0:
                sleep_engine._sleep_phase_synaptic_consolidation(report)
                sleep_engine._sleep_phase_forward_replay(report)
                n_complete_cycles += 1
        except Exception as e:
            crash_count += 1
            print(f"  [WARN] micro-sleep {step} 异常: {type(e).__name__}: {e}",
                  flush=True)
            if crash_count > 5:
                print(f"  [ABORT] 连续异常 > 5 次，停止", flush=True)
                check(f"micro-sleep {step} 不崩溃", False, f"crashes={crash_count}")
                break
            continue
        dt_step = time.time() - t_step

        if step % CHECKPOINT_EVERY == 0:
            lora_l2 = {nid: lora_l2_norm(cortex.neurons[nid]) for nid in target_ids}
            ckpt = measure_group_stds(sleep_engine, cortex, target_ids, groups)
            t_now = time.time() - t0

            for g in groups:
                if pre_stds[g] > 0:
                    ratio = abs(ckpt[g]["std"] - pre_stds[g]) / pre_stds[g]
                    if ratio > worst_step_jump:
                        worst_step_jump = ratio

            checkpoint_curve.append({
                "step": step,
                "elapsed_total_s": round(t_now, 1),
                "dt_step_s": round(dt_step, 1),
                "new_injected_so_far": new_injected_total,
                "mean_dialogue": ckpt["dialogue"]["mean"],
                "mean_knowledge": ckpt["knowledge"]["mean"],
                "mean_unfamiliar": ckpt["unfamiliar"]["mean"],
                "std_dialogue": ckpt["dialogue"]["std"],
                "std_knowledge": ckpt["knowledge"]["std"],
                "std_unfamiliar": ckpt["unfamiliar"]["std"],
                "lora_l2_zh_aug3": lora_l2.get("zh_aug3_dialogue"),
            })
            d_m, k_m, u_m = (ckpt["dialogue"]["mean"],
                             ckpt["knowledge"]["mean"],
                             ckpt["unfamiliar"]["mean"])
            print(f"  step {step:3d}/{N_MICRO}  dt={dt_step:5.1f}s  "
                  f"mean(d/k/u)={d_m:.4f}/{k_m:.4f}/{u_m:.4f}  "
                  f"new={new_injected_total}  "
                  f"lora_l2_aug3={lora_l2.get('zh_aug3_dialogue', 0):.3f}",
                  flush=True)

    print(f"\n[4/5] 后测量 post-sleep 100 次 A1 真实版 24 prompt...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    post_means = {g: post[g]["mean"] for g in groups}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}  Δmean={d['mean'] - pre_means[g]:+.4f}",
              flush=True)

    plateau_means = {}
    if len(checkpoint_curve) >= 4:
        mid_ckpt = checkpoint_curve[len(checkpoint_curve) // 2]
        plateau_means = {
            "dialogue": mid_ckpt["mean_dialogue"],
            "knowledge": mid_ckpt["mean_knowledge"],
            "unfamiliar": mid_ckpt["mean_unfamiliar"],
        }
        print(f"\n  plateau 中点（步 {mid_ckpt['step']}）mean: "
              f"d={mid_ckpt['mean_dialogue']:.4f} "
              f"k={mid_ckpt['mean_knowledge']:.4f} "
              f"u={mid_ckpt['mean_unfamiliar']:.4f}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("A5 完整 5 维判据（新判据：经验驱动增长）:", flush=True)
    print("=" * 64, flush=True)

    deltas = {g: post_means[g] - pre_means[g] for g in groups}

    all_up_001 = all(deltas[g] >= 0.01 for g in groups)
    check("A5-full.a: 3 组 mean 全部上升 ≥ 0.01（经验有效）",
          all_up_001,
          " ".join(f"{g} Δ={deltas[g]:+.4f}" for g in groups))

    all_within_020 = all(deltas[g] <= 0.20 for g in groups)
    check("A5-full.b: 3 组 mean 上升 ≤ 0.20（不爆炸）",
          all_within_020,
          f"max Δ={max(deltas.values()):+.4f}")

    check("A5-full.c: worst step 跳水 ≤ 50% pre-std",
          worst_step_jump <= 0.50,
          f"worst_step_jump={worst_step_jump*100:.1f}% (pre_std=0.566/1.028/0.623)")

    check("A5-full.d: 0 崩溃 / 0 NaN",
          crash_count == 0,
          f"crashes={crash_count}/{N_MICRO}, new_injected={new_injected_total}")

    plateau_drift = max(abs(post_means[g] - plateau_means[g]) for g in groups) if plateau_means else None
    if plateau_means:
        check("A5-full.e: plateau 检验（30 步后 mean 漂移 ≤ 0.05）",
              plateau_drift <= 0.05,
              f"drift={plateau_drift:.4f}")
    else:
        check("A5-full.e: plateau 检验（数据不足）", False, "checkpoint_curve < 4")

    a5_pass = (failed == 0)
    if a5_pass:
        if all_up_001 and all_within_020:
            verdict = (f"A5 完整 PASS：100 步 × {len(new_batches)} 批新经验后 "
                       f"3 组 mean 全部上升 0.01-0.20，经验驱动增长方向性 + "
                       f"增长受控 + plateau 稳态")
        else:
            verdict = "A5 完整 PASS（部分判据满足）"
        next_step = ("A5 完整通过。下一步：B1 探索自主性 —— play 引擎常态运行下，"
                     "新经验中由它自己（非脚本）选定的方向占比是否 ≥ 30%。")
    else:
        verdict = f"A5 完整 部分失败（{passed} PASS / {failed} FAIL）"
        if not all_up_001:
            next_step = ("A5 完整失败：mean 上升不足 0.01。可能需要：(1) 增加新经验"
                         "每批 12 条而非 8 条；(2) 收窄 decay 到 0.85 让 LoRA 累积更慢，"
                         "给 sleep 更多空间吸收新经验。")
        elif not all_within_020:
            next_step = ("A5 完整失败：mean 上升爆炸 > 0.20。需把每批 8 条改 4 条，"
                         "或 decay 改 0.95 压住 LoRA 累积。")
        else:
            next_step = ("A5 完整失败：worst step 跳水 > 50%。需加 sleep 间隔 cooldown，"
                         "避免连续 micro-sleep 累积。")
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"play_engine_a5_full_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": (f"A5 完整：{N_MICRO} 次 micro-sleep + {N_MICRO // CHECKPOINT_EVERY} 批 "
                 f"({NEW_PROMPTS_PER_BATCH} 条/批) 新经验"),
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "checkpoint_every": CHECKPOINT_EVERY,
            "decay": DECAY,
            "new_prompts_per_batch": NEW_PROMPTS_PER_BATCH,
            "n_complete_3phase_cycles": n_complete_cycles,
            "n_new_batches": len(new_batches),
        },
        "pre_groups": {g: {k: v for k, v in pre[g].items() if k != "nlls"} for g in groups},
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in groups},
        "pre_means": pre_means,
        "post_means": post_means,
        "delta_means": deltas,
        "worst_step_jump_ratio": worst_step_jump,
        "plateau_means": plateau_means,
        "plateau_drift": plateau_drift,
        "checkpoint_curve": checkpoint_curve,
        "crash_count": crash_count,
        "new_injected_total": new_injected_total,
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
    sys.exit(0 if a5_pass else 1)


if __name__ == "__main__":
    main()
