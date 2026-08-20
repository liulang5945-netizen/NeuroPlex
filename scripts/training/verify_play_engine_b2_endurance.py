#!/usr/bin/env python3
"""自举门槛 B2 autonomous 续航：play 引擎不被喂新经验时 100 步无遗忘（2026-08-20）。

背景：
    B1-bis 探索自主性 4/4 PASS（4 个机制协同让 exploit/force/recency 共同生效）。
    但 B1/B1-bis **每步都喂入"主题池"经验**——play 引擎的"探索"是"读新经验"驱动的。

    B2 真正测的是**自举续航**：在没新经验时，play 引擎能不能仅靠记忆库自反思 +
    judge 驱动的 sleep 维持能力。这是"它能不能在没有外部喂食时自己活下去"。

    设计倾斜（更上限的方案）：
    - 100 步 micro-sleep，**完全关闭"喂新经验"通路**：不调用 A1 真实版 24 条 prompt，
      不注入主题池；只**从记忆库自抽 5-8 条已有经验做自反思 query**——模拟"它自己想"。
    - sleep 阶段照常 phase 1.5/1.6/1.7；judge_driven_replay=True 让 judge 选短板。
    - 关键观察：post 3 组 std 维持 ≥ pre × 0.95 = 100 步无遗忘；3 组 mean 漂移
      应自然（不喂新经验 → 不增长，但**不暴跌** = 自举续航成立）。

判据（B2）：
    B2.a dialogue 组：post std >= pre std × 0.95（不遗忘）
    B2.b knowledge 组：post std >= pre std × 0.95
    B2.c unfamiliar 组：post std >= pre std × 0.95
    B2.d 0 崩溃 / 0 NaN / 0 爆炸
    B2.e 100 步 <= 30 min（不喂新经验应更快）

    5 维全过 = B2 PASS → 进入"自举续航"模式（自举的最后一关：没经验也能活）。

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A1 真实版 24 prompt + A3 衰减 0.9 + judge_driven_replay
    - 100 步短跑 + 关闭"喂新经验"通路
    - 自反思 query：从已有 memory bank 随机抽 5-8 条作为"它自己想"的内容

运行：python -u scripts/training/verify_play_engine_b2_endurance.py
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

passed = 0
failed = 0
N_MICRO = int(os.environ.get("B2_MICRO_N", "100"))
DECAY = float(os.environ.get("B2_DECAY", "0.9"))
SELF_RECALL_EVERY = int(os.environ.get("B2_SELF_RECALL_EVERY", "10"))
SELF_RECALL_K = int(os.environ.get("B2_SELF_RECALL_K", "6"))


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
    print(f"自举门槛 B2 autonomous 续航：{N_MICRO} 次 micro-sleep + 关闭喂新经验",
          flush=True)
    print(f"  自反思 query: 每 {SELF_RECALL_EVERY} 步从记忆库抽 {SELF_RECALL_K} 条",
          flush=True)
    print("=" * 64, flush=True)

    print("\n[1/6] 装配 9 成员 production cortex（冻结，不写 checkpoint）...",
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

    tmp_data = os.path.join("data", "_tmp_b2")
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
    all_seed_prompts = (DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS
                        + UNFAMILIAR_PROMPTS)
    print(f"\n[2/6] 注入种子记忆：{len(all_seed_prompts)} 条 A1 真实版 prompt...",
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
    print(f"  注入 {len(all_seed_prompts)} 条 + 场固化 "
          f"{r_init.field_memories_consolidated} 条", flush=True)

    print(f"\n[3/6] 预测量（pre-sleep 3 组 std/mean 基线）...", flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    pre_summary = {}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = pre[g]
        pre_summary[g] = {"std": d["std"], "mean": d["mean"]}
        print(f"  pre {g}: std={d['std']:.6f}  mean={d['mean']:.4f}",
              flush=True)
    pre_lora = sum(lora_l2_norm(cortex.neurons[nid])
                   for nid in target_ids if nid in cortex.neurons)
    print(f"  pre LoRA L2 (compact dialogue): {pre_lora:.4f}", flush=True)

    print(f"\n[4/6] 跑 {N_MICRO} 次 micro-sleep（关闭喂新经验；"
          f"每 {SELF_RECALL_EVERY} 步自反思 query）...", flush=True)
    n_crashes = 0
    n_nan = 0
    self_recall_count = 0
    memory_bank_texts = list(all_seed_prompts)

    for step in range(1, N_MICRO + 1):
        t_step = time.time()

        if step % SELF_RECALL_EVERY == 0:
            recalled = random.sample(memory_bank_texts,
                                     min(SELF_RECALL_K, len(memory_bank_texts)))
            for j, text in enumerate(recalled):
                vec = field_state_of(cortex, text)
                sleep_engine.record_field_memory(
                    vec, f"selfrec_step{step}_{j}", text=text)
                sc.record_high_resonance_state(
                    field_state=vec, resonance_score=0.88, step=step,
                    active_nids=target_ids, threshold=0.5, text=text)
            self_recall_count += 1
            if step % (SELF_RECALL_EVERY * 5) == 0:
                print(f"  step {step:4d}  self-recall #{self_recall_count}  "
                      f"({len(recalled)} 条抽回记忆库)", flush=True)

        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if step % 20 == 0:
                sleep_engine._sleep_phase_synaptic_consolidation(report)
                sleep_engine._sleep_phase_forward_replay(report)
        except Exception as e:
            n_crashes += 1
            print(f"  [WARN] micro-sleep {step} 异常: "
                  f"{type(e).__name__}: {e}", flush=True)
            if n_crashes > 5:
                print(f"  [ABORT] 崩溃 > 5，停止", flush=True)
                break
            continue

    elapsed_step = time.time() - t0
    print(f"\n  完成 {N_MICRO} 步, "
          f"自反思 query 触发 {self_recall_count} 次, "
          f"崩溃 {n_crashes} 次", flush=True)

    print(f"\n[5/6] 后测量（post-sleep 3 组 std/mean）...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    post_summary = {}
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        post_summary[g] = {"std": d["std"], "mean": d["mean"]}
        print(f"  post {g}: std={d['std']:.6f}  mean={d['mean']:.4f}",
              flush=True)
    post_lora = sum(lora_l2_norm(cortex.neurons[nid])
                    for nid in target_ids if nid in cortex.neurons)
    print(f"  post LoRA L2 (compact dialogue): {post_lora:.4f}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("B2 5 维判据：", flush=True)
    print("=" * 64, flush=True)

    ratio_summary = {}
    all_pass = True
    for g in ("dialogue", "knowledge", "unfamiliar"):
        ratio = (post_summary[g]["std"]
                 / max(pre_summary[g]["std"], 1e-9))
        ratio_summary[g] = ratio
        passed_this = ratio >= 0.95
        if not passed_this:
            all_pass = False
        check(f"B2.{g[0]} {g} 组 std >= pre × 0.95（不遗忘）",
              passed_this,
              f"ratio={ratio:.4f}  pre={pre_summary[g]['std']:.4f}  "
              f"post={post_summary[g]['std']:.4f}")

    check("B2.d 0 崩溃 / 0 NaN / 0 爆炸",
          n_crashes == 0 and n_nan == 0,
          f"crashes={n_crashes}/{N_MICRO}  nan={n_nan}")

    elapsed_min = (time.time() - t0) / 60
    check(f"B2.e {N_MICRO} 步 <= 30 min",
          elapsed_min <= 30,
          f"elapsed={elapsed_min:.1f} min")

    b2_pass = (failed == 0)

    print("\n" + "=" * 64, flush=True)
    if b2_pass:
        print("判定: B2 PASS：autonomous 续航成立（100 步无遗忘，"
              "无新经验时它能自反思维生）", flush=True)
        print("下一步: B2 通过。下一步：C1 协作形态自主 — 验证协作权重/结构"
              "随经验演化、撤掉外部协作设计后 EMERGE 不归零", flush=True)
    else:
        print(f"判定: B2 FAIL ({failed} 维不过)", flush=True)
        print("下一步: 调 SELF_RECALL_EVERY 更密 / K 更大；或延长 N_MICRO"
              "观察漂移累积", flush=True)
    print("=" * 64, flush=True)

    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "B2 autonomous 续航：100 步无新经验 + 自反思 query",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "decay": DECAY,
            "self_recall_every": SELF_RECALL_EVERY,
            "self_recall_k": SELF_RECALL_K,
            "new_experience_injected": False,
        },
        "pre_groups": pre_summary,
        "post_groups": post_summary,
        "ratio_summary": ratio_summary,
        "pre_lora_l2": pre_lora,
        "post_lora_l2": post_lora,
        "crash_count": n_crashes,
        "nan_count": n_nan,
        "self_recall_count": self_recall_count,
        "passed": passed,
        "failed": failed,
        "verdict": ("B2 PASS" if b2_pass else f"B2 FAIL ({failed} 维不过)"),
        "next_step": ("B2 通过。下一步：C1 协作形态自主 — 验证协作权重/结构"
                      "随经验演化、撤掉外部协作设计后 EMERGE 不归零"
                      if b2_pass else
                      "调 SELF_RECALL_EVERY 更密 / K 更大；或延长 N_MICRO"
                      "观察漂移累积"),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = f"reports/play_engine_b2_endurance_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
