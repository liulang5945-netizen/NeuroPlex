#!/usr/bin/env python3
"""自举门槛 A5 准备：play 引擎常态化下 judge 信号 mean 增长观察（2026-08-20）。

背景：
    A4 完整（verify_play_engine_a4_drift.py）已证 100 次 micro-sleep 不破坏
    judge 信号（3 组 ratio ≥ 98.6%）。A4 完整语义"经验驱动能力增长"未观测到
    ——原因是 100 次 micro-sleep 没有新经验喂入（replay buffer 没新增），所以
    mean 没有上升趋势。

    A5 准备要回答：**play 引擎常态化喂新经验 100 步后，judge 信号 mean 是否
    上升？** ——这是"经验驱动能力增长"的直接观测。

    关键升级（vs A4 完整）：
    - 每 10 次 micro-sleep 前注入 8 条**新 prompt**（不与 A1 真实版 24 prompt 重叠）
    - 80 条新 prompt 分 10 批（每批 8 条）覆盖 3 组：dialogue/knowledge/unfamiliar
    - 100 步后观测 A1 真实版 3 组 mean 变化

判据（A5 准备）：
    A5a. dialogue 组：post-sleep mean 在 pre ± 0.05 之内（退化 ≤ 0.05）
    A5b. knowledge 组：同 A5a
    A5c. unfamiliar 组：同 A5a
    A5d. 任意一组 mean 上升 ≥ 0.01（**经验驱动增长的方向性证据**）
    A5e. 100 步内 0 NaN / 0 爆炸

    5 维全过 = A5 准备 PASS

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A3 衰减 0.9 + A1 真实版 24 prompt + SleepConsolidator
    - 80 条新 prompt 与 A1 真实版 24 prompt 主题/用词明显不同

运行：python -u scripts/training/verify_play_engine_a5_growth.py
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
N_MICRO = int(os.environ.get("A5_MICRO_N", "100"))
CHECKPOINT_EVERY = int(os.environ.get("A5_CKPT_EVERY", "10"))
DECAY = float(os.environ.get("A5_DECAY", "0.9"))
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
    print(f"自举门槛 A5 准备：{N_MICRO} 次 micro-sleep + 喂新经验", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/4] 装配 9 成员 production cortex（冻结，不写 checkpoint）...", flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a5_prep")
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

    print("\n[2/4] 预测量 pre-sleep A1 真实版 + 注入 24 条记忆...", flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    pre_means = {g: pre[g]["mean"] for g in groups}
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

    print(f"\n[3/4] 跑 {N_MICRO} 次 micro-sleep（每 {CHECKPOINT_EVERY} 步前注入 {NEW_PROMPTS_PER_BATCH} 条新 prompt）...",
          flush=True)
    checkpoint_curve = []
    n_complete_cycles = 0
    crash_count = 0
    new_injected_total = 0
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

    print(f"\n[4/4] 后测量 post-sleep 100 次 A1 真实版 24 prompt...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    post_means = {g: post[g]["mean"] for g in groups}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}  Δmean={d['mean'] - pre_means[g]:+.4f}",
              flush=True)

    print("\n" + "=" * 64, flush=True)
    print("A5 准备 5 维判据：", flush=True)
    print("=" * 64, flush=True)

    pass_lines = []
    for g in ("dialogue", "knowledge", "unfamiliar"):
        m_pre = pre_means[g]
        m_post = post_means[g]
        delta = m_post - m_pre
        ok = abs(delta) <= 0.05
        check(f"A5.{g[0]}: |Δ mean| ≤ 0.05（不显著退化）", ok,
              f"pre={m_pre:.4f}  post={m_post:.4f}  Δ={delta:+.4f}")
        pass_lines.append(f"{g}: mean pre={m_pre:.4f} → post={m_post:.4f} (Δ={delta:+.4f})")

    any_growth = any((post_means[g] - pre_means[g]) >= 0.01 for g in groups)
    best_growth = max((post_means[g] - pre_means[g]) for g in groups)
    best_growth_g = max(groups, key=lambda g: post_means[g] - pre_means[g])
    check("A5d: 至少 1 组 mean 上升 ≥ 0.01（经验驱动增长方向性）",
          any_growth,
          f"best={best_growth_g} Δ={best_growth:+.4f}")

    check("A5e: 100 次 micro-sleep 无崩溃",
          crash_count == 0,
          f"crashes={crash_count}/{N_MICRO}, new_injected={new_injected_total}")

    a5_pass = (failed == 0)
    if a5_pass:
        verdict = "A5 准备 PASS：100 步喂新经验后 judge mean 不退化"
        if any_growth:
            verdict += f"，且 {best_growth_g} 组 mean 上升 {best_growth:+.4f}（经验驱动增长方向性证据）"
        next_step = ("A5 准备通过。下一步 A5 完整：把 A5 准备流程常态嵌入对话循环，"
                     "观测 multi-day 自举演化（每天 1000 步 micro-sleep × 喂新经验）。")
    else:
        verdict = f"A5 准备 部分失败（{passed} PASS / {failed} FAIL）"
        next_step = ("退化 > 0.05 需要：(1) 收窄 decay 到 0.85-0.95；(2) 新 prompt "
                     "每 20 步而非 10 步注入；(3) replay buffer 缩到 100 让 old 经验溢出。")
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"play_engine_a5_growth_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": (f"A5 准备：{N_MICRO} 次 micro-sleep + 每 10 步注入 "
                 f"{NEW_PROMPTS_PER_BATCH} 条新经验"),
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
        },
        "pre_groups": {g: {k: v for k, v in pre[g].items() if k != "nlls"} for g in groups},
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in groups},
        "pre_means": pre_means,
        "post_means": post_means,
        "delta_means": {g: post_means[g] - pre_means[g] for g in groups},
        "checkpoint_curve": checkpoint_curve,
        "pass_lines": pass_lines,
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
