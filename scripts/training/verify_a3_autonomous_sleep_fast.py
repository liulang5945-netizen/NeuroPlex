#!/usr/bin/env python3
"""自举门槛 A3 快速版：3-5 轮自主 sleep 验证（②驱动③，多轮不崩溃）。

背景：
    上一轮 A3 设计思路是"全量训练 50h+ 观察 PPL 下降"——66h 长跑已证明性价比
    极低。本验证用**局部闭环**思路绕过长跑：

    1. 复用 A1 真实版的 24 条 prompt（8 对话 + 8 知识 + 8 陌生领域）作为观测探针
    2. 每轮 sleep 只调**最便宜的改进 phase**：Phase 1.7 forward_replay
       （judge_driven_replay=True + judge NLL 选短板 + 只动读路径/LoRA，body 不动）
       + Phase 1.6 synaptic_consolidation（高频场记忆重放进 LoRA）
       + Phase 1.5 field_consolidation（高频场状态沉淀）
    3. 跳过 Phase 2 model_training（重头，66h 长跑所在）——本次只验证
       "自指信号 → 局部行动 → 自指信号改善"闭环，**不验证长期 PPL 绝对下降**
    4. 跑 3-5 轮 sleep，监控：
       - 每轮前/后 judge NLL（24 条）总均值
       - "判定的短板"是否真的被优先改善（高 NLL 组 Δ更负）
       - "非短板"是否未被破坏（低 NLL 组 Δ ≥ 0 或小幅）
       - round1 NLL（body 零破坏）
       - 任意轮 sleep 后 NaN/爆炸 → 立即退出

判据（A3 快速版）：
    A3a. 自指闭环：每轮 sleep 后，judge NLL 总均值 Δ ≤ 0.5（自身判定"更不擅长"不暴涨）
    A3b. 归因正确：每轮 sleep 后，judge 判定的"短板"（高 NLL）top-1/3 组 Δ 比
         "非短板"（低 NLL）bottom-1/3 组 Δ 更负（短板改善 > 非短板，至少 2/3 轮满足）
    A3c. body 零破坏：每轮 sleep 后，round1 NLL（无条件化）Δ < 1.0（body 不动，
         LoRA 累积允许，body 本质是冻结的）
    A3d. 自我维持：连续 3 轮 sleep 后无 NaN/无 PPL>1e6 爆炸，judge NLL 不单调升

4 结果映射：
    - 3/4 通过：自举 A3 局部闭环成立 → 写 50h 长跑已不必要（小步快跑）
    - 2/4 通过：部分闭环 → 排查哪一相失败
    - ≤1/4 通过：自举 A→B 路径在当前 sleep 设计上不闭环

资源预算：
    - 每轮 sleep ~1-3 min（forward_replay 默认 8 样本 × 5 dialogue × shadow forward + 1.6 高频重放）
    - 每轮观测 24 条 prompt 21.6s
    - 3 轮 = ~10-15 min，5 轮 = ~20-25 min

约束：
    - 冻结 9 成员 production weights（只写读路径/LoRA；不写 production checkpoint）
    - 不跑 Phase 2 model_training
    - 不修改 SleepConfig.judge_driven_replay 默认（保持 True）

运行：python -u scripts/training/verify_a3_autonomous_sleep_fast.py
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

# 复用 A1 真实版的 24 条 prompt
from scripts.training.verify_a1_judge_signal_real import (  # noqa: E402
    DIALOGUE_IDS, COLLAB_NAME, EXTRA_NEURONS_DIR,
    DIALOGUE_PROMPTS, KNOWLEDGE_PROMPTS, UNFAMILIAR_PROMPTS,
)

passed = 0
failed = 0


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
    """round1 无条件化 NLL（body 零破坏观测）。"""
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


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    n_rounds = int(os.environ.get("A3_N_ROUNDS", "3"))
    print("=" * 64, flush=True)
    print(f"自举门槛 A3 快速版：{n_rounds} 轮自主 sleep（②→③ 多轮闭环）", flush=True)
    print("=" * 64, flush=True)

    print(f"\n[1/{n_rounds + 1}] 装配 9 成员 production cortex...", flush=True)
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

    tmp_data = os.path.join("data", "_tmp_a3_fast")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=50)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)
    device = next(cortex._shared_embedding.parameters()).device

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

    print(f"\n[3/(N+1)] 初始观测...", flush=True)
    nlls_round0 = measure_judge_nlls(sleep_engine, cortex, target_ids, all_prompts)
    valid_nlls0 = [v for v in nlls_round0.values() if v is not None]
    mean0 = float(np.mean(valid_nlls0))
    nll_round1_0 = {nid: nll_round1(cortex, nid, DIALOGUE_PROMPTS[0]) for nid in target_ids}
    print(f"  judge NLL 总均值 = {mean0:.3f}（{len(valid_nlls0)}/{len(all_prompts)} 有效）",
          flush=True)
    print(f"  round1 NLL (body 探针) = {nll_round1_0}", flush=True)

    history = {
        "round_0": {
            "judge_nlls": nlls_round0,
            "mean": mean0,
            "round1_nlls": nll_round1_0,
        }
    }
    a3b_evidence = []  # 每轮 (短板_Δ均值, 非短板_Δ均值, 改善归因 True/False)

    for r in range(1, n_rounds + 1):
        print(f"\n[Round {r}/{n_rounds}] 自主 sleep（只跑 1.5/1.6/1.7）...", flush=True)
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
        print(f"  sleep phase 用时 {dt:.1f}s "
              f"(field_mem={r_report.field_memories_consolidated} "
              f"syn={r_report.synaptic_consolidated} "
              f"fwd_replay={r_report.forward_replayed} "
              f"judge_driven={r_report.judge_driven_replay})",
              flush=True)

        nlls_r = measure_judge_nlls(sleep_engine, cortex, target_ids, all_prompts)
        valid_r = [v for v in nlls_r.values() if v is not None and v < 1e6]
        if len(valid_r) < len(valid_nlls0) * 0.9:
            print(f"  [ABORT] 有效 NLL 暴跌（{len(valid_r)}/{len(valid_nlls0)}）→ 爆炸", flush=True)
            check(f"Round {r}: 无 NaN/爆炸", False)
            break
        mean_r = float(np.mean(valid_r))
        nll_round1_r = {nid: nll_round1(cortex, nid, DIALOGUE_PROMPTS[0]) for nid in target_ids}
        body_deltas = {nid: nll_round1_r[nid] - nll_round1_0[nid] for nid in target_ids}
        max_body_delta = max(abs(v) for v in body_deltas.values())
        print(f"  judge NLL 总均值 = {mean_r:.3f}（Δ={mean_r - mean0:+.3f} 相对 round0）", flush=True)
        print(f"  body NLL Δ = {body_deltas}（max |Δ|={max_body_delta:.3f}）", flush=True)

        # A3b 归因：分位数法 — 短板 = 本轮 NLL 最高的 1/3，非短板 = 最低 1/3
        sorted_prompts = sorted([(t, v) for t, v in nlls_r.items() if v is not None],
                                key=lambda x: x[1], reverse=True)
        n = len(sorted_prompts)
        top_third = sorted_prompts[: n // 3]
        bot_third = sorted_prompts[-n // 3:]
        # ΔNLL = round_r - round0（负=改善）
        top_delta = float(np.mean([nlls_r[t] - nlls_round0[t] for t, _ in top_third]))
        bot_delta = float(np.mean([nlls_r[t] - nlls_round0[t] for t, _ in bot_third]))
        improved = top_delta < bot_delta
        a3b_evidence.append((r, top_delta, bot_delta, improved))
        print(f"  归因: 短板(top1/3) Δ={top_delta:+.3f}  非短板(bot1/3) Δ={bot_delta:+.3f}  "
              f"改善归因 = {improved}", flush=True)

        check(f"Round {r} A3a: judge NLL 总均值不暴涨（无退化）",
              mean_r - mean0 < 0.5, f"Δ={mean_r - mean0:+.3f}")
        check(f"Round {r} A3b: 归因正确（短板 Δ < 非短板 Δ）",
              improved, f"top={top_delta:+.3f} bot={bot_delta:+.3f}")
        check(f"Round {r} A3c: body 零破坏（round1 max|Δ|<1.0，LoRA 累积允许）",
              max_body_delta < 1.0, f"max|Δ|={max_body_delta:.3f}")

        history[f"round_{r}"] = {
            "judge_nlls": nlls_r,
            "mean": mean_r,
            "round1_nlls": nll_round1_r,
            "top_third_delta": top_delta,
            "bot_third_delta": bot_delta,
            "improved_attribution": improved,
            "duration_seconds": dt,
        }

    # 终判
    print("\n" + "=" * 64, flush=True)
    print(f"A3 快速版 终判: {passed} PASS / {failed} FAIL（{n_rounds} 轮）", flush=True)
    print("=" * 64, flush=True)
    a3d_evidence = len(a3b_evidence) == n_rounds
    check("A3d: 自我维持（全部 N 轮无崩溃、无 NaN）", a3d_evidence,
          f"completed_rounds={len(a3b_evidence)}/{n_rounds}")

    attribution_pass_count = sum(1 for _, _, _, ok in a3b_evidence if ok)
    print(f"\n归因正确轮数: {attribution_pass_count}/{n_rounds}", flush=True)

    if failed == 0 and attribution_pass_count == n_rounds:
        verdict = "A3 快速版全过：自指→行动→自指改善 闭环成立（3-5 轮内）"
        next_step = "65h 长跑已不必要。可保持小步快跑（每轮 < 3min），常态化驱动 judge 信号。"
    elif passed >= failed * 2 and attribution_pass_count >= (n_rounds + 1) // 2:
        verdict = f"A3 快速版大部分通过（{passed} pass / {failed} fail，归因 {attribution_pass_count}/{n_rounds}）"
        next_step = "主体闭环已建立。可把 forward_replay_max_samples 调到 16（更多样本/轮）或加到 5 轮观测稳态；不再跑 65h 长跑。"
    else:
        verdict = "A3 快速版大部分失败：自举 A→B 路径在当前 sleep 设计上不闭环"
        next_step = "需重审 sleep 设计：forward_replay_max_samples 太小？judge 头与 sleep 解耦？"
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"a3_autonomous_sleep_fast_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": f"A3 快速版：{n_rounds} 轮自主 sleep（只跑 1.5/1.6/1.7）",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
        },
        "n_rounds": n_rounds,
        "completed_rounds": len(a3b_evidence),
        "history": history,
        "attribution_per_round": [
            {"round": r, "top_delta": td, "bot_delta": bd, "improved": ok}
            for r, td, bd, ok in a3b_evidence
        ],
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
    sys.exit(0 if failed == 0 and attribution_pass_count == n_rounds else 1)


if __name__ == "__main__":
    main()
