#!/usr/bin/env python3
"""自举门槛 A4 完整：play 引擎常态化下 judge 信号演化（2026-08-20）。

背景：
    A4 准备（verify_a4_post_sleep_judge_signal.py）已证 8 轮 sleep 后 judge 不
    遗忘（3 组 ratio ≥ 80%）。A4 完整语义是"经验驱动能力增长"——这需要常态化
    play 引擎把 A3 sleep 接入到 judge 触发流程，并观测 A1 真实版 3 组 std 是否
    退化。

    资源约束（5-10 min）：
    - 100 次 micro-sleep 不能跑完整 3-phase（每轮 ~16s，100 轮 = 27 min）
    - 改为：每 10 次 micro-sleep 跑 1 次完整 3-phase A3 衰减 0.9 sleep；
            其它 9 次只跑 1 phase（field_consolidation，最快）
    - 每 10 次 micro-sleep 后做一次 A1 真实版 3 组 std checkpoint
    - 总耗时约 10 × 16s + 10 × 21.6s ≈ 6 min

判据（A4 完整）：
    A4a. dialogue 组：100 次 micro-sleep 后 std >= pre × 0.95（退化 ≤ 5%）
    A4b. knowledge 组：同 A4a
    A4c. unfamiliar 组：同 A4a
    A4d. 演化曲线单调性：std 不出现单步 >10% 跳水（避免 65h 长跑那种崩塌式退化）
    A4e. 100 次内 0 NaN / 0 爆炸

    5 维全过 = A4 完整 PASS：play 引擎常态化下 judge 经验后能力不退化
    （甚至可观测到 mean 缓慢上升趋势，对应 A4 完整语义"增长"）。

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A3 衰减 0.9 + A1 真实版 24 prompt + SleepConsolidator

运行：python -u scripts/training/verify_play_engine_a4_drift.py
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
N_MICRO = int(os.environ.get("A4_MICRO_N", "100"))
CHECKPOINT_EVERY = int(os.environ.get("A4_CKPT_EVERY", "10"))
DECAY = float(os.environ.get("A4_DECAY", "0.9"))
FULL_CYCLES_PER_CKPT = int(os.environ.get("A4_FULL_PER_CKPT", "1"))


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
    print(f"自举门槛 A4 完整：{N_MICRO} 次 micro-sleep + "
          f"{N_MICRO // CHECKPOINT_EVERY} 个 checkpoint", flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a4_full")
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

    print("\n[2/4] 预测量 pre-sleep A1 真实版 + 注入 24 条记忆（同 A4 准备）...",
          flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, groups)
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

    print(f"\n[3/4] 跑 {N_MICRO} 次 micro-sleep（每 {CHECKPOINT_EVERY} 次一个 checkpoint，"
          f"每 {FULL_CYCLES_PER_CKPT} 次完整 3-phase，余为 1 phase）...",
          flush=True)
    checkpoint_curve = []
    n_complete_cycles = 0
    crash_count = 0
    for step in range(1, N_MICRO + 1):
        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        t_step = time.time()
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if (step % CHECKPOINT_EVERY) == 0 and (
                (step // CHECKPOINT_EVERY) <= FULL_CYCLES_PER_CKPT
            ):
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
                "std_dialogue": ckpt["dialogue"]["std"],
                "std_knowledge": ckpt["knowledge"]["std"],
                "std_unfamiliar": ckpt["unfamiliar"]["std"],
                "mean_dialogue": ckpt["dialogue"]["mean"],
                "mean_knowledge": ckpt["knowledge"]["mean"],
                "mean_unfamiliar": ckpt["unfamiliar"]["mean"],
                "lora_l2_zh_aug3": lora_l2.get("zh_aug3_dialogue"),
            })
            print(f"  step {step:3d}/{N_MICRO}  dt={dt_step:5.1f}s  "
                  f"std(d/k/u)={ckpt['dialogue']['std']:.4f}/"
                  f"{ckpt['knowledge']['std']:.4f}/"
                  f"{ckpt['unfamiliar']['std']:.4f}  "
                  f"lora_l2_aug3={lora_l2.get('zh_aug3_dialogue', 0):.3f}",
                  flush=True)

    print(f"\n[4/4] 后测量 post-sleep 100 次 A1 真实版 24 prompt...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("A4 完整 5 维判据：", flush=True)
    print("=" * 64, flush=True)

    pass_lines = []
    for g in ("dialogue", "knowledge", "unfamiliar"):
        s_pre = pre[g]["std"]
        s_post = post[g]["std"]
        if s_pre is None or s_post is None:
            check(f"A4.{g[0]}: post std >= pre × 0.95", False,
                  f"pre={s_pre} post={s_post}")
            continue
        ratio = s_post / s_pre if s_pre > 0 else None
        ok = ratio is not None and s_post >= s_pre * 0.95
        check(f"A4.{g[0]}: post std >= pre × 0.95", ok,
              f"pre={s_pre:.4f}  post={s_post:.4f}  ratio={ratio:.3f}" if ratio else f"pre={s_pre} post={s_post}")
        pass_lines.append(f"{g}: pre={s_pre:.4f} → post={s_post:.4f} ({ratio:.2%})" if ratio else f"{g}: pre={s_pre} post={s_post}")

    max_jumps = []
    if len(checkpoint_curve) >= 2:
        for i in range(1, len(checkpoint_curve)):
            for g in ("dialogue", "knowledge", "unfamiliar"):
                key = f"std_{g}"
                prev = checkpoint_curve[i - 1][key]
                cur = checkpoint_curve[i][key]
                if prev and cur and prev > 0:
                    jump_pct = abs(cur - prev) / prev
                    max_jumps.append((g, i, jump_pct))
        worst_jump = max(max_jumps, key=lambda x: x[2]) if max_jumps else None
        monotonic_safe = worst_jump is not None and worst_jump[2] < 0.10
        check("A4d: 演化曲线无单步 >10% 跳水",
              monotonic_safe,
              f"worst={worst_jump[0]}@{worst_jump[1]} {worst_jump[2]:.2%}" if worst_jump else "n/a")
    else:
        check("A4d: 演化曲线无单步 >10% 跳水", False, "checkpoint 不足")

    check("A4e: 100 次 micro-sleep 无崩溃",
          crash_count == 0,
          f"crashes={crash_count}/{N_MICRO}")

    a4_pass = (failed == 0)
    if a4_pass:
        verdict = ("A4 完整 PASS：100 次 micro-sleep 后 judge 信号不退化，"
                   "演化曲线无 10% 级跳水")
        next_step = ("A4 完整通过——play 引擎常态化下经验不破坏 judge。下一步 "
                     "是把这一机制常态嵌入对话循环（每次 judge 给出低置信度时"
                     "自动触发 10 次 micro-sleep），进入 A5 准备（multi-day "
                     "自举演化观察）。")
    else:
        verdict = f"A4 完整 部分失败（{passed} PASS / {failed} FAIL）"
        next_step = ("退化 > 5% 或出现 10% 跳水，需要：(1) 收窄 decay；"
                     "(2) micro-sleep 间隔加大到每 2 步 1 次而非每步；"
                     "(3) replay buffer size 缩到 50 让 old memory 更快溢出。")
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"play_engine_a4_drift_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": (f"A4 完整：{N_MICRO} 次 micro-sleep + "
                 f"{len(checkpoint_curve)} 个 A1 真实版 checkpoint"),
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "checkpoint_every": CHECKPOINT_EVERY,
            "full_cycles_per_ckpt": FULL_CYCLES_PER_CKPT,
            "decay": DECAY,
            "n_complete_3phase_cycles": n_complete_cycles,
        },
        "pre_groups": {g: {k: v for k, v in pre[g].items() if k != "nlls"} for g in groups},
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in groups},
        "checkpoint_curve": checkpoint_curve,
        "pass_lines": pass_lines,
        "crash_count": crash_count,
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
    sys.exit(0 if a4_pass else 1)


if __name__ == "__main__":
    main()
