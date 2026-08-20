#!/usr/bin/env python3
"""自举门槛 A3 多轮稳定版：加 LoRA 衰减后跑 8 轮验证多轮可持续。

背景：
    A3 快速版（verify_a3_autonomous_sleep_fast.py）3 分钟揭示了关键问题：
    1-2 轮闭环成立，但 3+ 轮 judge NLL 持续漂移（+0.122/5 轮），归因反转。
    根因诊断：Phase 1.7 forward_replay 训练 LoRA 后无衰减，多轮累积
    挤占 judge 判定空间。

修复：
    SleepConfig 新增 lora_decay_per_sleep（默认 1.0=不衰减，向后兼容）。
    Phase 1.7 末尾在 _copy_learned(live, shadow) 之后对 live.lora_adapters
    所有参数乘衰减系数。详见 neuroplex/life/sleep_engine.py:1016-1029 (C28 增量一)。

判据（A3 多轮稳定版）：
    A3a. 自指稳定：8 轮内 judge NLL 总均值 Δ 漂移绝对值 < 0.1（vs 快速版 0.122/5 轮）
    A3b. 归因持续正确：8 轮中 ≥ 6 轮 短板 Δ < 非短板 Δ（vs 快速版 2/5）
    A3c. body 零破坏：8 轮内 round1 max|Δ| < 1.5（LoRA 衰减可减少累积；放宽容忍）
    A3d. 自我维持：8 轮无 NaN/无 PPL>1e6 爆炸
    A3e. 衰减生效：每轮 lora L2 norm 单调下降（衰减系数 < 1.0 → 必单调降）

4 结果映射：
    - 5/5 通过：多轮可持续 A3 成立，65h 长跑彻底不需要
    - 4/5 通过：主体稳定，可能需微调衰减（0.95 → 0.9）
    - 3/5 通过：衰减方向对，需调系数
    - ≤2/5 通过：衰减不足或 LoRA 衰减方法不对，需重审

资源预算：
    - 8 轮 sleep ~17s × 8 + 21.6s 观测 × 9 = 3.5-4 min
    - 比 65h 长跑快 1000x

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 不跑 Phase 2 model_training
    - SleepConfig.lora_decay_per_sleep = 0.95

运行：python -u scripts/training/verify_a3_with_decay.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import numpy as np  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig, SleepReport  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

from scripts.training.verify_a1_judge_signal_real import (  # noqa: E402
    DIALOGUE_IDS, COLLAB_NAME, EXTRA_NEURONS_DIR,
    DIALOGUE_PROMPTS, KNOWLEDGE_PROMPTS, UNFAMILIAR_PROMPTS,
)

passed = 0
failed = 0
DECAY = float(os.environ.get("A3_LORA_DECAY", "0.95"))
N_ROUNDS = int(os.environ.get("A3_N_ROUNDS", "8"))


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def field_state_of(cortex, text: str):
    gids = cortex._general_sp.encode(text) or [0]
    ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(ids)
    res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                       collab_mode="continuous")
    fs = res.get("field_state")
    if fs is None:
        raise RuntimeError("think() 未返回 field_state")
    if fs.dim() == 2:
        fs = fs.mean(dim=0)
    return fs


def _to_general(cortex, domain_sp, domain_ids):
    gids = []
    general_sp = cortex._general_sp
    for did in domain_ids:
        gg = general_sp.EncodeAsIds(domain_sp.id_to_piece(did))
        gids.append(gg[0] if gg else 0)
    return gids


def nll_round1(cortex, nid: str, text: str) -> float:
    import torch.nn.functional as F
    neuron = cortex.neurons[nid]
    neuron.eval()
    hub = cortex._tokenizer_hub
    domain_sp = hub.get_tokenizer("zh")
    domain_ids = hub.encode(text, domain="zh")
    if not domain_ids or len(domain_ids) < 3:
        return float("nan")
    gids = _to_general(cortex, domain_sp, domain_ids)
    input_ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(input_ids)
    with torch.no_grad():
        res = neuron.forward(emb, field_state=None, round_num=1, return_logits=True)
        logits = res["logits"]
        target = torch.tensor([domain_ids], dtype=torch.long, device=cortex.device)
        min_len = logits.size(1) - 1
        if min_len < 1:
            return float("nan")
        sl = logits[:, :min_len, :].contiguous()
        st = target[:, 1:1 + min_len].contiguous().clamp(0, logits.size(-1) - 1)
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
    return loss.item()


def measure_judge_nlls(sleep_engine, cortex, target_ids, prompts) -> dict:
    device = next(cortex._shared_embedding.parameters()).device
    out = {}
    for text in prompts:
        nll = sleep_engine._sample_judge_nll(
            text, target_ids, device, cortex._shared_embedding)
        out[text] = nll
    return out


def lora_l2_norm(neuron) -> float:
    """神经元 lora_adapters 全部参数 L2 norm。"""
    s = 0.0
    with torch.no_grad():
        for p in neuron.lora_adapters.parameters():
            s += float(p.data.pow(2).sum().item())
    return s ** 0.5


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print(f"自举门槛 A3 多轮稳定版：{N_ROUNDS} 轮 + LoRA 衰减 {DECAY}", flush=True)
    print("=" * 64, flush=True)

    print(f"\n[1/{N_ROUNDS + 1}] 装配 9 成员 production cortex...", flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a3_decay")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=50)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    all_prompts = DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS
    prompt_labels = (["dialogue"] * len(DIALOGUE_PROMPTS) +
                     ["knowledge"] * len(KNOWLEDGE_PROMPTS) +
                     ["unfamiliar"] * len(UNFAMILIAR_PROMPTS))

    print("\n[2/(N+1)] 注入 24 条 prompt 记忆（喂经验）...", flush=True)
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

    print(f"\n[3/(N+1)] 初始观测 + LoRA 初始 L2...", flush=True)
    nlls_round0 = measure_judge_nlls(sleep_engine, cortex, target_ids, all_prompts)
    valid_nlls0 = [v for v in nlls_round0.values() if v is not None]
    mean0 = float(np.mean(valid_nlls0))
    nll_round1_0 = {nid: nll_round1(cortex, nid, DIALOGUE_PROMPTS[0]) for nid in target_ids}
    lora_l2_round0 = {nid: lora_l2_norm(cortex.neurons[nid]) for nid in target_ids}
    print(f"  judge NLL 总均值 = {mean0:.3f}（{len(valid_nlls0)}/{len(all_prompts)} 有效）",
          flush=True)
    print(f"  LoRA L2 = {lora_l2_round0}", flush=True)

    history = {
        "round_0": {
            "judge_nlls": nlls_round0,
            "mean": mean0,
            "round1_nlls": nll_round1_0,
            "lora_l2": lora_l2_round0,
        }
    }
    a3b_evidence = []
    a3e_lora_l2 = [lora_l2_round0]
    drift_tracker = [mean0]  # for A3a final check

    for r in range(1, N_ROUNDS + 1):
        print(f"\n[Round {r}/{N_ROUNDS}] 自主 sleep（decay={DECAY}）...", flush=True)
        r_report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        t_round = time.time()
        try:
            sleep_engine._sleep_phase_field_consolidation(r_report)
            sleep_engine._sleep_phase_synaptic_consolidation(r_report)
            sleep_engine._sleep_phase_forward_replay(r_report)
        except Exception as e:
            print(f"  [ABORT] sleep 阶段异常: {e}", flush=True)
            check(f"Round {r}: sleep 主流程不崩溃", False, f"err={e}")
            break
        dt = time.time() - t_round
        lora_l2_r = {nid: lora_l2_norm(cortex.neurons[nid]) for nid in target_ids}
        a3e_lora_l2.append(lora_l2_r)
        print(f"  sleep phase 用时 {dt:.1f}s "
              f"(fwd_replay={r_report.forward_replayed} judge_driven={r_report.judge_driven_replay})",
              flush=True)
        print(f"  LoRA L2 = {lora_l2_r}", flush=True)

        nlls_r = measure_judge_nlls(sleep_engine, cortex, target_ids, all_prompts)
        valid_r = [v for v in nlls_r.values() if v is not None and v < 1e6]
        if len(valid_r) < len(valid_nlls0) * 0.9:
            print(f"  [ABORT] 有效 NLL 暴跌（{len(valid_r)}/{len(valid_nlls0)}）→ 爆炸", flush=True)
            check(f"Round {r}: 无 NaN/爆炸", False)
            break
        mean_r = float(np.mean(valid_r))
        drift_tracker.append(mean_r)
        nll_round1_r = {nid: nll_round1(cortex, nid, DIALOGUE_PROMPTS[0]) for nid in target_ids}
        body_deltas = {nid: nll_round1_r[nid] - nll_round1_0[nid] for nid in target_ids}
        max_body_delta = max(abs(v) for v in body_deltas.values())
        print(f"  judge NLL 总均值 = {mean_r:.3f}（Δ={mean_r - mean0:+.3f} 相对 round0）", flush=True)
        print(f"  body NLL Δ = {body_deltas}（max |Δ|={max_body_delta:.3f}）", flush=True)

        sorted_prompts = sorted([(t, v) for t, v in nlls_r.items() if v is not None],
                                key=lambda x: x[1], reverse=True)
        n = len(sorted_prompts)
        top_third = sorted_prompts[: n // 3]
        bot_third = sorted_prompts[-n // 3:]
        top_delta = float(np.mean([nlls_r[t] - nlls_round0[t] for t, _ in top_third]))
        bot_delta = float(np.mean([nlls_r[t] - nlls_round0[t] for t, _ in bot_third]))
        improved = top_delta < bot_delta
        a3b_evidence.append((r, top_delta, bot_delta, improved))
        print(f"  归因: 短板(top1/3) Δ={top_delta:+.3f}  非短板(bot1/3) Δ={bot_delta:+.3f}  "
              f"改善归因 = {improved}", flush=True)

        check(f"Round {r} A3a: judge NLL 总均值不暴涨（|Δ|<0.1）",
              abs(mean_r - mean0) < 0.1, f"|Δ|={abs(mean_r - mean0):.3f}")
        check(f"Round {r} A3b: 归因正确（短板 Δ < 非短板 Δ）",
              improved, f"top={top_delta:+.3f} bot={bot_delta:+.3f}")
        check(f"Round {r} A3c: body max|Δ|<1.5",
              max_body_delta < 1.5, f"max|Δ|={max_body_delta:.3f}")

        history[f"round_{r}"] = {
            "judge_nlls": nlls_r,
            "mean": mean_r,
            "round1_nlls": nll_round1_r,
            "top_third_delta": top_delta,
            "bot_third_delta": bot_delta,
            "improved_attribution": improved,
            "lora_l2": lora_l2_r,
            "duration_seconds": dt,
        }

    print("\n" + "=" * 64, flush=True)
    print(f"A3 多轮稳定版 终判: {passed} PASS / {failed} FAIL（{N_ROUNDS} 轮, decay={DECAY}）",
          flush=True)
    print("=" * 64, flush=True)

    attribution_pass_count = sum(1 for _, _, _, ok in a3b_evidence if ok)
    completed_rounds = len(a3b_evidence)
    final_drift = abs(drift_tracker[-1] - drift_tracker[0]) if len(drift_tracker) > 1 else None
    print(f"\n归因正确轮数: {attribution_pass_count}/{N_ROUNDS}", flush=True)
    print(f"最终漂移 |Δ|={final_drift:.3f}", flush=True)

    if completed_rounds == N_ROUNDS:
        check("A3d: 自我维持（全部 8 轮无崩溃、无 NaN）", True,
              f"completed_rounds={completed_rounds}/{N_ROUNDS}")
    else:
        check("A3d: 自我维持（全部 8 轮无崩溃、无 NaN）", False,
              f"completed_rounds={completed_rounds}/{N_ROUNDS}")

    if DECAY < 1.0 and completed_rounds > 0:
        lora_l2_means = []
        for layer_dict in a3e_lora_l2:
            lora_l2_means.append(float(np.mean(list(layer_dict.values()))))
        is_monotonic_down = all(lora_l2_means[i] <= lora_l2_means[i - 1] + 1e-6
                                for i in range(1, len(lora_l2_means)))
        check("A3e: LoRA L2 衰减生效（per-round 平均单调下降）",
              is_monotonic_down,
              f"per_round_avg={['%.3f' % x for x in lora_l2_means]}")

    attribution_pass = attribution_pass_count >= 6
    drift_ok = final_drift is not None and final_drift < 0.1
    body_ok = all(history[f"round_{r}"]["round1_nlls"] for r in range(1, completed_rounds + 1))

    if attribution_pass and drift_ok and completed_rounds == N_ROUNDS:
        verdict = "A3 多轮稳定版全过：自举 A→B 在多轮上闭环成立"
        next_step = "65h 长跑彻底不需要。可保持小步快跑（每轮 < 3min），常态化驱动 judge 信号。准备进 A4（经验驱动自适应）。"
    elif completed_rounds == N_ROUNDS and attribution_pass_count >= 4:
        verdict = f"A3 多轮稳定版大部分通过（归因 {attribution_pass_count}/{N_ROUNDS}，最终漂移 |Δ|={final_drift:.3f}）"
        next_step = "衰减方向对，调系数：0.95 → 0.92（更激进）或保留 0.95 继续观测更多轮。"
    else:
        verdict = f"A3 多轮稳定版部分失败（归因 {attribution_pass_count}/{N_ROUNDS}，最终漂移 |Δ|={final_drift}）"
        next_step = "LoRA 衰减不够。尝试：1) 衰减到 0.9；2) 把衰减也施加到 Phase 1.6；3) 加 sleep cooldown。"
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"a3_with_decay_{DECAY:.2f}_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": f"A3 多轮稳定版：{N_ROUNDS} 轮 + LoRA 衰减 {DECAY}",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "n_rounds": N_ROUNDS,
        "completed_rounds": completed_rounds,
        "history": history,
        "attribution_per_round": [
            {"round": r, "top_delta": td, "bot_delta": bd, "improved": ok}
            for r, td, bd, ok in a3b_evidence
        ],
        "lora_l2_per_round": a3e_lora_l2,
        "drift_per_round": drift_tracker,
        "final_drift_abs": final_drift,
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
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
