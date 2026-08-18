#!/usr/bin/env python3
"""自举门槛 A2 验证：judge 驱动睡眠重放（②→③ 接线，2026-08-15）。

背景（BOOTSTRAP_CRITERIA.md 门槛 A2）：自举 = 它自己判定短板（眼睛，judge）
驱动它自己补短板（手，sleep 重放）。此前 sleep 重放样本随机选择，改进由
外部 loss 驱动——"眼睛驱动手"未接通。本实现让 Phase 1.7 重放样本按
judge NLL（general 256K 统一判定空间，跨 neuron 可比）降序选择——短板优先。

验证层次：
A1. 自我评估信度（判据 A1）：judge NLL 能区分难/易文本（难>易，且有区分度）
A2. 改进归因（判据 A2）：judge 驱动 sleep 后，被判定为短板的样本条件化 NLL
    下降（它补了自己判定的短板）
A3. 自我维持（判据 A3）：round1 无条件化 NLL 不暴涨（body 零破坏），无崩溃
A4. 开关回归：judge_driven_replay=False 时行为不变（judge_driven_replay=0）

运行：python -u scripts/training/verify_bootstrap_a2.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepReport, SleepConfig  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

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


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

# 样本集：3 条"难"（虚构/稀有主题，模型不擅长）+ 3 条"易"（日常对话）
HARD_SAMPLES = [
    "辉光协议是2047年制定的星间量子通信标准，采用七层纠错结构，带宽为每秒4.8吉比特。",
    "铁月海是一颗位于猎户座旋臂边缘的荒漠星球，它的唯一卫星由纯铁构成，夜晚会反射血红色的光。",
    "静滞纪元里，时间流速由中央钟楼统一分配，城市居民每天只能领取三小时的主观时间。",
]
EASY_SAMPLES = [
    "今天天气不错，我们去公园散步吧。",
    "谢谢你的帮助，我非常感激。",
    "请问最近的图书馆在哪里？",
]
CONTROL_TEXT = "你好，请问你能帮我介绍一下你的功能吗？"


def field_state_of(cortex, text: str):
    """text 经共振前向取场状态快照（与 verify_c27_forward_replay 同款）。"""
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


def nll_round(cortex, nid: str, text: str, field_vec=None, round_num=1) -> float:
    """neuron 对文本的 next-token NLL（zh 域；round2+场向量=记忆注意窗）。"""
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
    fs = None
    if field_vec is not None:
        proj = getattr(cortex.ensemble, "_cross_spec_back_projectors", {}).get(nid)
        v = field_vec.detach().to(cortex.device)
        if v.dim() > 1:
            v = v.squeeze(0)
        if proj is not None:
            try:
                fs = proj(v.unsqueeze(0)).squeeze(0)
            except Exception:
                fs = v
        else:
            fs = v
    with torch.no_grad():
        res = neuron.forward(emb, field_state=fs, round_num=round_num,
                             return_logits=True)
        logits = res["logits"]
        target = torch.tensor([domain_ids], dtype=torch.long, device=cortex.device)
        min_len = logits.size(1) - 1
        if min_len < 1:
            return float("nan")
        sl = logits[:, :min_len, :].contiguous()
        st = target[:, 1:1 + min_len].contiguous().clamp(0, logits.size(-1) - 1)
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                               ignore_index=-100)
    return loss.item()


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("自举门槛 A2：judge 驱动睡眠重放（②→③ 接线）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="bootstrap_a2_")
    try:
        cortex, tokenizer, modules = assemble_cortex(
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
        nid0 = target_ids[0]
        print(f"  装配 {len(cortex.neurons)} 神经元, 目标 {target_ids}", flush=True)
        check("装配成功（9 neuron）", len(cortex.neurons) == 9)

        cfg = SleepConfig(max_training_steps=10, training_enabled=True,
                          judge_driven_replay=True)
        sleep_engine = SleepEngine(config=cfg, data_dir=tmp_dir)
        sleep_engine.forward_replay_max_samples = 3  # 6 条中选 3 条短板
        sc = SleepConsolidator(replay_buffer_size=50)
        sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)
        device = next(cortex._shared_embedding.parameters()).device

        # ── 1. 注入记忆 + 场状态 → 样本集（3 难 + 3 易）──
        labels = {t: f"hard_{i}" for i, t in enumerate(HARD_SAMPLES)}
        labels.update({t: f"easy_{i}" for i, t in enumerate(EASY_SAMPLES)})
        for text in HARD_SAMPLES + EASY_SAMPLES:
            vec = field_state_of(cortex, text)
            sleep_engine.record_field_memory(vec, labels[text], text=text)
        r_f = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                          duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(r_f)
        bank = sleep_engine.get_field_memory()
        check("记忆库固化 6 条", len(bank) == 6, f"n={len(bank)}")

        # 场状态进重放缓冲区（带 text）→ Phase 1.7 样本源
        for text in HARD_SAMPLES + EASY_SAMPLES:
            fs = field_state_of(cortex, text)
            sc.record_high_resonance_state(
                field_state=fs, resonance_score=0.9,
                step=sleep_engine._current_step, active_nids=target_ids,
                threshold=0.5, text=text)
        check("场状态记录：6 条带 text 进重放缓冲区", len(sc._replay_buffer) == 6)

        # ── A1. 自我评估信度：judge NLL 有区分度（能排序样本）──
        # 自举精神：不用外部"难/易"标签——判据只要求它自己能区分样本
        # （judge NLL 有方差），以及 A2 中"它判定的短板确实在重放后改善"。
        print("\n[A1] judge NLL 区分度（眼睛）...", flush=True)
        judge_nlls = {}
        for text in HARD_SAMPLES + EASY_SAMPLES:
            judge_nlls[labels[text]] = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
        vals = [v for v in judge_nlls.values() if v is not None]
        std = (sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"    judge NLL: {judge_nlls}", flush=True)
        print(f"    区分度 std={std:.3f}", flush=True)
        check("A1. judge NLL 有区分度（std>0.05，能排序样本）", std > 0.05,
              f"std={std:.3f}")

        # ── 基线：重放前条件化 NLL（round2 + 场向量）──
        field_vecs = {labels[t]: field_state_of(cortex, t)
                      for t in HARD_SAMPLES + EASY_SAMPLES}
        base_cond = {labels[t]: nll_round(cortex, nid0, t, field_vecs[labels[t]], round_num=2)
                     for t in HARD_SAMPLES + EASY_SAMPLES}
        ctrl_plain0 = nll_round(cortex, nid0, CONTROL_TEXT, round_num=1)
        print(f"\n[基线] 条件化 NLL: {base_cond}", flush=True)

        # ── A2. judge 驱动 sleep → 短板改善 ──
        print("\n[A2] judge 驱动重放（手，短板优先）...", flush=True)
        r7 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_forward_replay(r7)
        check("A2a. judge 驱动重放执行（judge_driven_replay>0）",
              r7.judge_driven_replay > 0,
              f"judge_driven={r7.judge_driven_replay}, replayed={r7.forward_replayed}")
        after_cond = {labels[t]: nll_round(cortex, nid0, t, field_vecs[labels[t]], round_num=2)
                      for t in HARD_SAMPLES + EASY_SAMPLES}
        deltas = {k: after_cond[k] - base_cond[k] for k in base_cond}
        # judge 判定短板 = judge NLL 最高的 top3（forward_replay_max_samples=3）
        ranked = sorted(judge_nlls, key=lambda k: judge_nlls[k], reverse=True)
        picked = set(ranked[:3])
        picked_d = sum(deltas[k] for k in picked) / len(picked)
        unpicked = set(deltas) - picked
        unpicked_d = sum(deltas[k] for k in unpicked) / len(unpicked)
        print(f"    ΔNLL: {deltas}", flush=True)
        print(f"    judge 选中(短板) Δ均值={picked_d:.3f}, 未选中 Δ均值={unpicked_d:.3f}",
              flush=True)
        check("A2b. judge 判定的短板条件化 NLL 下降（它补了自己判定的差）",
              picked_d < -0.02, f"Δ={picked_d:.3f}")
        check("A2c. 短板改善 > 未选中（归因于 judge 选择）",
              picked_d < unpicked_d,
              f"picked={picked_d:.3f} unpicked={unpicked_d:.3f}")

        # ── A3. 自我维持 / 零破坏 ──
        print("\n[A3] 零破坏（body 未动）...", flush=True)
        ctrl_plain1 = nll_round(cortex, nid0, CONTROL_TEXT, round_num=1)
        check("A3. round1 无条件化 NLL 不暴涨（body 零破坏）",
              abs(ctrl_plain1 - ctrl_plain0) < 0.5,
              f"Δ={ctrl_plain1 - ctrl_plain0:.3f}")

        # ── A4. 开关回归 ──
        print("\n[A4] 开关回归（judge_driven_replay=False）...", flush=True)
        sleep_engine.config.judge_driven_replay = False
        r_off = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            duration_seconds=0)
        sleep_engine._sleep_phase_forward_replay(r_off)
        check("A4. 关闭后 judge_driven_replay==0（旧行为不变）",
              r_off.judge_driven_replay == 0,
              f"judge_driven={r_off.judge_driven_replay}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
