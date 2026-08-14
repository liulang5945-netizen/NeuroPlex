#!/usr/bin/env python3
"""C26 增量三验证：记忆 → 突触沉淀（海马→皮层两层记忆，LoRA）（2026-08-14）。

背景：C26 写（固化）✅ 读（条件化生成）✅ 已打通，增量三让高频场记忆
通过睡眠重放**沉淀进神经元权重**——海马（FieldMemoryBank 短期存储）→ 皮层
（神经元 LoRA 增量长期权重）。反复重放（高频检索）的记忆才值得迁移；
沉淀后条目标记 consolidated 防重复重放。

机制：高频未沉淀条目（access_count ≥ 2）内容作 SFT 重放样本（问答对 + 原文
混合），冻结 neuron body 只训尾层 LoRA 增量（enable_lora B 初始 0 → 零破坏
起点），影子权重 COW 训练后只写回 lora 参数，域内全部 dialogue neuron 协作。

验证层次：
A. 高频判定：3 条高频（检索 3 次）沉淀，1 条低频（检索 1 次）不沉淀
B. 沉淀生效（硬）：沉淀后高频记忆文本 NLL 下降（LoRA 记住了）
C. LoRA 写回：live neuron 的 lora_adapters 出现非零权重
D. 零破坏：未沉淀文本（对照组）NLL 不暴涨（容差内）
E. 防重复重放：沉淀条目标记 consolidated + 计数清零，二次沉淀跳过
F. 持久化：field_memory.pt 保存 consolidated 标记，重启恢复

运行：python -u scripts/training/verify_c26_synaptic_consolidation.py
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
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402

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
        "high_freq": True,
    },
    {
        "label": "铁月海",
        "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
        "query": "铁月海在哪里？",
        "high_freq": True,
    },
    {
        "label": "卡尔文环",
        "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
        "query": "卡尔文环是什么？",
        "high_freq": True,
    },
    # 低频对照：只检索 1 次 → 不沉淀
    {
        "label": "频谱蜂鸟",
        "text": "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。",
        "query": "频谱蜂鸟有什么习性？",
        "high_freq": False,
    },
]


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


def nll_of(cortex, nid: str, text: str) -> float:
    """neuron 对文本的 next-token NLL（zh 域对齐，无场条件化 round1）。"""
    neuron = cortex.neurons[nid]
    neuron.eval()
    hub = cortex._tokenizer_hub
    general_sp = cortex._general_sp
    domain_sp = hub.get_tokenizer("zh")
    domain_ids = hub.encode(text, domain="zh")
    if not domain_ids or len(domain_ids) < 3:
        return float("nan")
    gids = []
    for did in domain_ids:
        gg = general_sp.EncodeAsIds(domain_sp.id_to_piece(did))
        gids.append(gg[0] if gg else 0)
    input_ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(input_ids)
    with torch.no_grad():
        res = neuron.forward(emb, field_state=None, round_num=1,
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
    print("C26 增量三：记忆 → 突触沉淀（海马→皮层，LoRA 权重沉淀）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_synapse_")
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
        sleep_engine.set_brain_interfaces(cortex=cortex)

        # ── 1. 固化记忆（带内容文本 text）──
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report)
        bank = sleep_engine.get_field_memory()
        check("记忆库固化 4 条（含内容文本）",
              len(bank) == 4 and all(e["text"] for e in bank.entries))

        # ── 2. 模拟检索：高频 3 次 / 低频 1 次 ──
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            hits = 3 if item["high_freq"] else 1
            for _ in range(hits):
                bank.retrieve_vectors(qv, top_k=1)
        acc = {e["label"]: e["access_count"] for e in bank.entries}
        print(f"    access_count: {acc}", flush=True)
        check("高频判定：3 条 count=3、1 条 count=1",
              all(acc[i["label"]] == (3 if i["high_freq"] else 1)
                  for i in MEMORY_ITEMS),
              str(acc))

        # ── 3. 沉淀前 NLL 基线（高频文本）──
        print("\n[基线] 沉淀前 NLL ...", flush=True)
        hi_items = [i for i in MEMORY_ITEMS if i["high_freq"]]
        lo_items = [i for i in MEMORY_ITEMS if not i["high_freq"]]
        nll0_hi = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                   for i in hi_items}
        nll0_lo = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                   for i in lo_items}
        print(f"    {nll0_hi} (高频) / {nll0_lo} (低频)", flush=True)

        # ── 4. 突触沉淀（Phase 1.6）──
        print("\n[沉淀] Phase 1.6 突触沉淀 ...", flush=True)
        r2 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_synaptic_consolidation(r2)
        check("A. 高频 3 条沉淀、低频 1 条不沉淀",
              r2.synaptic_consolidated == 3,
              f"consolidated={r2.synaptic_consolidated}, lora_loss={r2.synaptic_lora_loss}")
        bank2 = sleep_engine.get_field_memory()
        marks = {e["label"]: e["consolidated"] for e in bank2.entries}
        print(f"    consolidated: {marks}", flush=True)
        check("A2. 沉淀标记正确（高频 True / 低频 False）",
              all(marks[i["label"]] == i["high_freq"] for i in MEMORY_ITEMS))

        # ── 5. 沉淀后 NLL：记住（下降）+ 零破坏（低频不暴涨）──
        nll1_hi = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                   for i in hi_items}
        nll1_lo = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                   for i in lo_items}
        drops = {k: nll0_hi[k] - nll1_hi[k] for k in nll0_hi}
        print(f"    NLL 下降（高频）: {drops}", flush=True)
        print(f"    NLL（低频对照）: {nll0_lo} → {nll1_lo}", flush=True)
        check("B. 沉淀生效：高频记忆文本 NLL 下降（LoRA 记住）",
              all(drops[k] > 0.05 for k in drops), f"drops={drops}")
        check("D. 零破坏：未沉淀文本 NLL 不暴涨",
              all(nll1_lo[k] < nll0_lo[k] + 1.0 for k in nll0_lo),
              f"{nll0_lo} → {nll1_lo}")

        # ── 6. LoRA 写回 live：出现非零权重（B 参数——A 初始 kaiming 非零不算）──
        nonzero = 0
        for nid in target_ids:
            neuron = cortex.neurons[nid]
            if len(neuron.lora_adapters) == 0:
                continue
            b_max = max(
                float(p.abs().max().item())
                for k, p in neuron.lora_adapters.named_parameters()
                if ".b." in k
            )
            nonzero += 1 if b_max > 1e-6 else 0
        check("C. LoRA 增量写回 live（B 非零）",
              nonzero == len(target_ids), f"{nonzero}/{len(target_ids)}")

        # ── 7. 防重复重放：二次沉淀跳过（计数已清零）──
        r3 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_synaptic_consolidation(r3)
        check("E. 防重复重放：二次沉淀跳过",
              r3.synaptic_consolidated == 0,
              f"second_consolidated={r3.synaptic_consolidated}")

        # ── 8. 持久化 + 重启恢复 ──
        from taiji.resonance.field_memory import FieldMemoryBank
        mem_path = os.path.join(tmp_dir, "field_memory.pt")
        bank3 = FieldMemoryBank()
        check("F. 磁盘恢复：consolidated 标记保留",
              bank3.load(mem_path)
              and all(e["consolidated"] == i["high_freq"]
                      for i, e in zip(MEMORY_ITEMS, bank3.entries)))

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
