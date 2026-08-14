#!/usr/bin/env python3
"""C26 增量六验证：真正睡眠重放（记忆向量场条件化 forward 重放，2026-08-14）。

背景：增量三（Phase 1.6）把高频记忆文本做**无场条件化**的纯文本 SFT 进 LoRA
（round1, field_state=None）——神经元"记住内容"，但记忆向量从未参与条件化；
推理路径的记忆注意窗（round2+ 场条件化 + 增量五 theta entrain）依赖的是
**随机初始化的 field_read_layers**（R2 审计发现：全仓库无训练路径）。增量六让
睡眠重放真正驱动 forward——以记忆向量/白天场状态作 field_state（round2+ 读路径）
重放文本，把"记忆注意窗下如何生成"固化为可学习权重。

样本源（用户决策：高频记忆 + 场状态混合）：
1. 已沉淀记忆（consolidated=True，内容已在皮层）——补上条件化读取（读路径）
2. SleepConsolidator 重放缓冲区场状态（带 text 的高共振经验）——白天最活跃
   场状态快照条件化重放触发文本

训练：读路径（field_read_layers + field_read_gate）+ LoRA 双训（用户决策），
影子 COW，body 不放进 optimizer → 零破坏起点。

验证层次：
A. 样本源混合：已沉淀记忆 + 场状态（带 text）都进入重放样本
B. 条件化 NLL 下降（硬）：重放后记忆向量条件化 forward（round2）的记忆文本
   NLL 下降——记忆注意窗下生成被固化
C. 读路径写回：live neuron 的 field_read_layers 权重变化（读路径已学习）
D. 零破坏：无条件化（round1）NLL 不暴涨（body 未动）
E. 场状态样本生效：replay buffer 场状态样本被消费（forward_replay_loss 记录）
F. 持久化：重放后的读路径权重随 neuron state_dict 保存/恢复（跨重启保留）

运行：python -u scripts/training/verify_c27_forward_replay.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
# N2（REMEDIATION_PLAN R7）：固定 seed 保证可复现
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn.functional as F  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport, _clone_module  # noqa: E402

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

MEMORY_ITEMS = [
    {
        "label": "辉光协议",
        "text": "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
        "query": "什么是辉光协议？",
    },
    {
        "label": "铁月海",
        "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
        "query": "铁月海在哪里？",
    },
    {
        "label": "卡尔文环",
        "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
        "query": "卡尔文环是什么？",
    },
]
# 对照组：不沉淀、不重放的文本（零破坏验证）
CONTROL_TEXT = "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。"


def field_state_of(cortex, text: str) -> torch.Tensor:
    """对文本做一次共振前向，取归一化场状态快照（think 返回的 field_state）。"""
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
    """neuron 对文本的 next-token NLL（zh 域）。

    round_num=1, field_vec=None → 无条件化（round1 独立路径）
    round_num=2, field_vec=记忆向量 → 记忆注意窗条件化 forward（round2 读路径）
    """
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
        # 与重放训练同一投影契约：back-project 到 neuron.field_dim
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
    print("C26 增量六：真正睡眠重放（记忆向量场条件化 forward 重放）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_fwd_replay_")
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
        print(f"  装配 {len(cortex.neurons)} 神经元", flush=True)
        print(f"  目标神经元: {target_ids}", flush=True)
        check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        # 注入 cortex + sleep_consolidator（重放缓冲区来自 consolidate）
        from taiji.resonance.neuro_modulation import SleepConsolidator
        sc = SleepConsolidator(replay_buffer_size=50)
        sleep_engine.set_brain_interfaces(cortex=cortex,
                                          sleep_consolidator=sc)

        # ── 1. 固化记忆（带内容文本）──
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report)
        bank = sleep_engine.get_field_memory()
        check("记忆库固化 3 条（含内容文本）",
              len(bank) == 3 and all(e["text"] for e in bank.entries))

        # ── 2. 模拟检索高频 3 次 → 全部成为沉淀候选 ──
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            for _ in range(3):
                bank.retrieve_vectors(qv, top_k=1)
        acc = {e["label"]: e["access_count"] for e in bank.entries}
        print(f"    access_count: {acc}", flush=True)
        check("高频判定：3 条 count=3", all(v == 3 for v in acc.values()),
              str(acc))

        # ── 3. 先跑增量三突触沉淀 → 3 条全 consolidated（内容已进 LoRA）──
        r2 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_synaptic_consolidation(r2)
        check("增量三沉淀 3 条（内容入 LoRA，consolidated 标记）",
              r2.synaptic_consolidated == 3,
              f"synaptic_consolidated={r2.synaptic_consolidated}")

        # ── 4. 记录场状态到重放缓冲区（带 text）──
        field_states = []
        for item in MEMORY_ITEMS:
            fs = field_state_of(cortex, item["text"])
            field_states.append(fs)
            sc.record_high_resonance_state(
                field_state=fs, resonance_score=0.9,
                step=sleep_engine._current_step, active_nids=target_ids,
                threshold=0.5, text=item["query"])
        check("场状态记录：3 条带 text 进重放缓冲区", len(sc._replay_buffer) == 3)

        # ── 5. 重放前基线：条件化 NLL（round2 + 记忆向量）vs 无条件化 ──
        nid0 = target_ids[0]
        print("\n[基线] 重放前 NLL ...", flush=True)
        base_cond = {item["label"]: nll_round(cortex, nid0, item["text"],
                                              field_states[i], round_num=2)
                     for i, item in enumerate(MEMORY_ITEMS)}
        base_plain = {item["label"]: nll_round(cortex, nid0, item["text"],
                                               round_num=1)
                      for item in MEMORY_ITEMS}
        ctrl_plain0 = nll_round(cortex, nid0, CONTROL_TEXT, round_num=1)
        print(f"    条件化(round2): {base_cond}", flush=True)
        print(f"    无条件化(round1): {base_plain}", flush=True)
        print(f"    对照(round1): {ctrl_plain0}", flush=True)

        # ── 6. 读路径权重基线（field_read_layers）──
        w_before = {k: v.clone() for k, v in
                    cortex.neurons[nid0].field_read_layers.state_dict().items()}

        # ── 7. Phase 1.7 真正睡眠重放 ──
        print("\n[重放] Phase 1.7 forward replay ...", flush=True)
        r7 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_forward_replay(r7)
        check("A. 重放完成：5 个 dialogue neuron 被重放",
              r7.forward_replayed == len(target_ids),
              f"replayed={r7.forward_replayed}, loss={r7.forward_replay_loss}")
        check("E. 重放 loss 已记录（场状态样本被消费）",
              r7.forward_replay_loss is not None
              and r7.forward_replay_loss < 20.0,
              f"loss={r7.forward_replay_loss}")

        # ── 8. 读路径写回：field_read_layers 权重变化 ──
        w_after = cortex.neurons[nid0].field_read_layers.state_dict()
        read_delta = sum(
            float((w_after[k] - v).abs().max().item())
            for k, v in w_before.items()
        )
        read_max_abs = max(float(v.abs().max().item()) for v in w_after.values())
        print(f"    读路径 delta={read_delta:.4f} max_abs={read_max_abs:.4f}",
              flush=True)
        check("C. 读路径已学习：field_read_layers 权重变化（非零 delta）",
              read_delta > 1e-4, f"delta={read_delta:.4f}")

        # ── 9. 重放后 NLL：条件化下降（硬）+ 零破坏 ──
        after_cond = {item["label"]: nll_round(cortex, nid0, item["text"],
                                               field_states[i], round_num=2)
                      for i, item in enumerate(MEMORY_ITEMS)}
        after_plain = {item["label"]: nll_round(cortex, nid0, item["text"],
                                                round_num=1)
                       for item in MEMORY_ITEMS}
        ctrl_plain1 = nll_round(cortex, nid0, CONTROL_TEXT, round_num=1)
        cond_drops = {k: base_cond[k] - after_cond[k] for k in base_cond}
        print(f"    条件化 NLL 下降: {cond_drops}", flush=True)
        print(f"    无条件化: {base_plain} → {after_plain}", flush=True)
        print(f"    对照: {ctrl_plain0} → {ctrl_plain1}", flush=True)
        check("B. 重放生效（硬）：记忆注意窗下条件化 NLL 下降",
              all(cond_drops[k] > 0.02 for k in cond_drops),
              f"drops={cond_drops}")
        check("D. 零破坏：无条件化 round1 NLL 不暴涨（body 未动）",
              all(after_plain[k] < base_plain[k] + 1.0 for k in base_plain)
              and ctrl_plain1 < ctrl_plain0 + 1.0,
              f"{base_plain}→{after_plain}, ctrl {ctrl_plain0}→{ctrl_plain1}")

        # ── 10. 持久化：读路径随 neuron state_dict 保存/恢复 ──
        saved = cortex.neurons[nid0].state_dict()
        lora_keys = [k for k in saved if k.startswith("lora_adapters.")]
        read_keys = [k for k in saved if k.startswith("field_read_layers.")]
        check("F. 持久化：读路径 + LoRA 都在 neuron state_dict 中",
              len(read_keys) >= len(cortex.neurons[nid0].layers)
              and len(lora_keys) >= 3,
              f"read_keys={len(read_keys)}, lora_keys={len(lora_keys)}")
        # 恢复验证：clone 重建 + load_state_dict → 读路径权重保留
        restored = _clone_module(cortex.neurons[nid0])
        restored.load_state_dict(saved, strict=False)
        rw = {k: v.clone() for k, v in
              restored.field_read_layers.state_dict().items()}
        persist_ok = all(
            float((rw[k] - w_after[k]).abs().max().item()) < 1e-6
            for k in w_after
        )
        check("F2. 重启恢复：重建 neuron 读路径权重与重放后一致",
              persist_ok)

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
        print(f"  记忆库: {bank.status()}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
