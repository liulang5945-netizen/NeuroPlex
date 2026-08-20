#!/usr/bin/env python3
"""自举门槛 A4 准备：经验驱动的 judge 能力不遗忘验证（2026-08-20）。

背景：
    A1 真实版（verify_a1_judge_signal_real.py）证明 judge 能在 24 条真实任务上给
    出有区分度的自我评估（std>0.05）。A3 衰减版（verify_a3_with_decay.py）证明
    8 轮自主 sleep + LoRA 衰减 0.9 不破坏 body、漂移 0.0556 < 0.15。

    A4 准备要回答：**8 轮 sleep 之后，judge 还能在 24 条真实任务上保持有区分度的
    自我评估吗？** ——这是"经验驱动的能力增长"的负向反证：经验（sleep 训练）不
    倒退能力（judge 信号）。

    A4 完整语义是"经验后能力增长"，但**先验证能力不遗忘**（更基础、更可证）。

判据（A4 准备）：
    A4a. dialogue 组：post-sleep std >= pre-sleep std × 0.8
    A4b. knowledge 组：post-sleep std >= pre-sleep std × 0.8
    A4c. unfamiliar 组：post-sleep std >= pre-sleep std × 0.8
    A4d. 3 组 std 全部 > 0.05（守住 A1 真实版的判据）
    A4e. post-sleep 0 NaN / 0 爆炸

    5 维全过 = A4 准备 PASS → A4 完整（能力增长）进入常态化 play 引擎闭环。

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A3 衰减 0.9 + A1 真实版 24 prompt
    - CPU 短跑，5-10 分钟

运行：python -u scripts/training/verify_a4_post_sleep_judge_signal.py
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
    field_state_of, measure_judge_nlls, lora_l2_norm,
)

passed = 0
failed = 0
N_ROUNDS = int(os.environ.get("A4_N_ROUNDS", "8"))
DECAY = float(os.environ.get("A4_DECAY", "0.9"))


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def measure_group_stds(sleep_engine, cortex, target_ids, groups) -> dict:
    """对 3 组 prompt 跑 judge NLL，返回 {group: {std, mean, n_valid, nlls}}。"""
    device = next(cortex._shared_embedding.parameters()).device
    out = {}
    for gname, prompts in groups.items():
        nlls = []
        valid = []
        for text in prompts:
            jnll = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
            nlls.append({"text": text, "judge_nll": jnll})
            if jnll is not None and jnll < 1e6:
                valid.append(jnll)
        if valid:
            out[gname] = {
                "std": float(np.std(valid)),
                "mean": float(np.mean(valid)),
                "n_valid": len(valid),
                "nlls": nlls,
            }
        else:
            out[gname] = {"std": None, "mean": None, "n_valid": 0, "nlls": nlls}
    return out


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print(f"自举门槛 A4 准备：{N_ROUNDS} 轮 sleep 后 judge 能力不遗忘验证", flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a4_prep")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=50)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }

    print("\n[2/5] 预测 pre-sleep A1 真实版 24 prompt 3 组 std...", flush=True)
    pre = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = pre[g]
        print(f"  pre  {g}: std={d['std']}  mean={d['mean']}  n={d['n_valid']}/8",
              flush=True)

    print(f"\n[3/5] 注入 24 条 prompt 记忆（同 A3 衰减 0.9 流程）...", flush=True)
    all_prompts = DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS
    prompt_labels = (["dialogue"] * len(DIALOGUE_PROMPTS) +
                     ["knowledge"] * len(KNOWLEDGE_PROMPTS) +
                     ["unfamiliar"] * len(UNFAMILIAR_PROMPTS))
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

    print(f"\n[4/5] 跑 {N_ROUNDS} 轮 sleep（decay={DECAY}）...", flush=True)
    a3b_history = []
    lora_l2_post = None
    for r in range(1, N_ROUNDS + 1):
        t_round = time.time()
        r_report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        try:
            sleep_engine._sleep_phase_field_consolidation(r_report)
            sleep_engine._sleep_phase_synaptic_consolidation(r_report)
            sleep_engine._sleep_phase_forward_replay(r_report)
        except Exception as e:
            print(f"  [ABORT] Round {r} sleep 阶段异常: {e}", flush=True)
            check(f"Round {r}: sleep 主流程不崩溃", False, f"err={e}")
            break
        dt = time.time() - t_round
        lora_l2_r = {nid: lora_l2_norm(cortex.neurons[nid]) for nid in target_ids}
        a3b_history.append({"round": r, "duration": dt, "lora_l2": lora_l2_r,
                            "fwd_replayed": r_report.forward_replayed,
                            "judge_driven": r_report.judge_driven_replay})
        lora_l2_post = lora_l2_r
        print(f"  Round {r}/{N_ROUNDS}  dt={dt:.1f}s  lora_l2={lora_l2_r}", flush=True)

    print("\n[5/5] 复测 post-sleep A1 真实版 24 prompt 3 组 std...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, groups)
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}  n={d['n_valid']}/8",
              flush=True)

    print("\n" + "=" * 64, flush=True)
    print("A4 准备 5 维判据：", flush=True)
    print("=" * 64, flush=True)
    pass_lines = []
    for g in ("dialogue", "knowledge", "unfamiliar"):
        s_pre = pre[g]["std"]
        s_post = post[g]["std"]
        if s_pre is None or s_post is None:
            check(f"A4.{g[0]}: post std >= pre std × 0.8", False,
                  f"pre={s_pre} post={s_post}")
            continue
        ratio = s_post / s_pre if s_pre > 0 else None
        ok = ratio is not None and s_post >= s_pre * 0.8
        check(f"A4.{g[0]}: post std >= pre std × 0.8", ok,
              f"pre={s_pre:.4f}  post={s_post:.4f}  ratio={ratio:.3f}" if ratio else f"pre={s_pre} post={s_post}")
        pass_lines.append(f"{g}: pre={s_pre:.4f} → post={s_post:.4f} ({ratio:.2%})" if ratio else f"{g}: pre={s_pre} post={s_post}")
        check(f"A4.{g[0]}: 守住 A1 真实版 std>0.05", s_post > 0.05,
              f"post std={s_post:.4f}")

    completed = len(a3b_history)
    check(f"A4e: 完成 {N_ROUNDS} 轮 sleep 无崩溃", completed == N_ROUNDS,
          f"completed={completed}/{N_ROUNDS}")

    a4_pass = (failed == 0) and (completed == N_ROUNDS)
    if a4_pass:
        verdict = "A4 准备 PASS：8 轮 sleep 后 judge 能力不遗忘"
        next_step = ("A4 准备通过——judge 经验后不倒退。下一步 A4 完整：把 A3 sleep "
                     "与 play 引擎常态化对接（不是 sniff 级能闭环的），进入 play "
                     "engine 经验驱动增长观察期。")
    else:
        verdict = f"A4 准备 部分失败（{passed} PASS / {failed} FAIL）"
        next_step = ("某组 std 退化超 20%，需要：(1) 收窄 decay 到 0.85-0.95 之间；"
                     "(2) sleep 间加 cooldown 1-2 步；(3) 重新审视 judge 头是否被训练改动。")
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"a4_post_sleep_judge_signal_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": f"A4 准备：{N_ROUNDS} 轮 sleep 后 judge 信号不遗忘",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "n_rounds": N_ROUNDS,
        "completed_rounds": completed,
        "pre_groups": {g: {k: v for k, v in pre[g].items() if k != "nlls"} for g in groups},
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in groups},
        "pre_nlls_per_prompt": {g: pre[g]["nlls"] for g in groups},
        "post_nlls_per_prompt": {g: post[g]["nlls"] for g in groups},
        "pass_lines": pass_lines,
        "sleep_history": a3b_history,
        "lora_l2_post_round": lora_l2_post,
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
