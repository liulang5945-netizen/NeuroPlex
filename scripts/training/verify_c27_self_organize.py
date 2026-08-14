#!/usr/bin/env python3
"""C26 增量七验证：自组织新生（从经验生长，非 teacher 蒸馏）（2026-08-14）。

背景：审计 S9 缺口——"新生依赖外部 teacher → 自组织新生（从经验生长）"。
现有 IntegrateEngine（C17）新生整合依赖 FeedEngine 样本 + 邻居蒸馏；feed 为空时
新生直接 skipped（no_feed_samples），未利用 C26 记忆库积累的经验。增量七让新生
neuron **从记忆经验生长**：
1. **记忆注意窗预训练**（_memory_pretrain）：用记忆库经验（向量+文本，问答对+
   原文）在 round2+ 场条件化下预热新 neuron 读路径 + LoRA——从经验出生
2. **样本源混合**（用户决策：三者全加入）：feed 样本 + 记忆问答对 + 记忆原文
3. **邻居蒸馏保留降权 0.3**（用户决策）：记忆生长为主，蒸馏辅助融入共振场

验证层次：
A. feed 为空 + 记忆库有经验 → 新生不 skipped（从经验生长，修复 no_feed_samples）
B. 读路径已学习：新 neuron field_read_layers 权重变化（记忆注意窗预训练生效）
C. 记忆条件化可用（软）：记忆向量条件化 forward（round2）对记忆文本 NLL 不劣于
   无条件化（读路径对齐）
D. 邻居蒸馏保留降权：DISTILL_WEIGHT == 0.3（代码契约）
E. 零破坏：成熟 dialogue neuron 参数不变（只有新 neuron 被训练）
F. 持久化：新 neuron 读路径随 state_dict 保存（跨重启保留）

运行：python -u scripts/training/verify_c27_self_organize.py
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
from taiji.life.integrate_engine import IntegrateEngine  # noqa: E402

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
    },
    {
        "label": "铁月海",
        "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
    },
    {
        "label": "卡尔文环",
        "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
    },
]


def field_state_of(cortex, text: str) -> torch.Tensor:
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


def cond_nll(cortex, nid: str, text: str, vec=None, round_num=1) -> float:
    """neuron 对记忆文本的条件化/无条件化 NLL（zh 域）。"""
    neuron = cortex.neurons[nid]
    neuron.eval()
    hub = cortex._tokenizer_hub
    domain_sp = hub.get_tokenizer("zh")
    general_sp = cortex._general_sp
    domain_ids = hub.encode(text, domain="zh")
    if not domain_ids or len(domain_ids) < 3:
        return float("nan")
    gids = []
    for did in domain_ids:
        gg = general_sp.EncodeAsIds(domain_sp.id_to_piece(did))
        gids.append(gg[0] if gg else 0)
    ids_t = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(ids_t)
    fs = None
    if vec is not None:
        proj = getattr(cortex.ensemble, "_cross_spec_back_projectors", {}).get(nid)
        v = vec.detach().to(cortex.device)
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
    print("C26 增量七：自组织新生（从经验生长，非 teacher 蒸馏）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c27_selforg_")
    new_ckpt = None
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
        lifecycle = modules.get("lifecycle")
        print(f"  装配 {len(cortex.neurons)} 神经元", flush=True)
        check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex, lifecycle=lifecycle)

        # ── 1. 固化记忆并模拟检索（access_count ≥ 1 → 成为生长经验）──
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report)
        bank = sleep_engine.get_field_memory()
        check("记忆库固化 3 条", len(bank) == 3)
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["text"])
            bank.retrieve_vectors(qv, top_k=1)
        acc = {e["label"]: e["access_count"] for e in bank.entries}
        print(f"    access_count: {acc}", flush=True)
        check("记忆经验就绪：3 条 access_count ≥ 1",
              all(v >= 1 for v in acc.values()))

        # ── 2. 创建新生 neuron（from_split 继承 + 噪声分化，真实路径）──
        split_parent = "zh_aug0_dialogue"  # 选 compact（512 hidden）与 ensemble 一致
        new_nid = cortex.add_neuron("zh", lifecycle=lifecycle,
                                    from_split=split_parent)
        new_ckpt = os.path.join(cortex.neurons_dir, f"neuron_{new_nid}.pt")
        print(f"  新生 neuron: {new_nid}（split from {split_parent}）", flush=True)
        check("A0. 新生 neuron 创建并注入", new_nid in cortex.neurons)

        # 父/子 读路径基线（from_split 已继承父权重）
        parent_read = {k: v.clone() for k, v in
                       cortex.neurons[split_parent].field_read_layers.state_dict().items()}
        new_read_before = {k: v.clone() for k, v in
                           cortex.neurons[new_nid].field_read_layers.state_dict().items()}
        # 成熟 neuron 基线（零破坏验证）
        mature_before = {k: v.clone() for k, v in
                         cortex.neurons[split_parent].state_dict().items()}

        # ── 3. 集成：feed 为空 + 记忆库有经验 → 从经验生长 ──
        print("\n[整合] IntegrateEngine（feed 为空，仅记忆经验）...", flush=True)
        ie = IntegrateEngine(cortex, lifecycle=lifecycle,
                             feed_engine=None,   # feed 为空 → 纯记忆生长
                             memory_bank=bank)
        result = ie.integrate(new_nid)
        status = result.get("status")
        print(f"    结果: {result}", flush=True)
        check("A. feed 空 + 记忆经验 → 新生不 skipped（从经验生长）",
              status in ("training", "committed"),
              f"status={status}")

        # ── 4. 读路径已学习（记忆注意窗预训练生效）──
        new_read_after = cortex.neurons[new_nid].field_read_layers.state_dict()
        read_delta = sum(
            float((new_read_after[k] - new_read_before[k]).abs().max().item())
            for k in new_read_before
        )
        print(f"    读路径 delta（vs 继承基线）={read_delta:.4f}", flush=True)
        check("B. 读路径已学习：field_read_layers 权重变化（记忆注意窗预训练生效）",
              read_delta > 1e-4, f"delta={read_delta:.4f}")

        # ── 5. 记忆条件化可用（软）：条件化 NLL ≤ 无条件化 NLL + 容差 ──
        item0 = MEMORY_ITEMS[0]
        vec0 = field_state_of(cortex, item0["text"])
        nll_cond = cond_nll(cortex, new_nid, item0["text"], vec0, round_num=2)
        nll_plain = cond_nll(cortex, new_nid, item0["text"], None, round_num=1)
        print(f"    条件化 NLL={nll_cond:.3f}, 无条件化 NLL={nll_plain:.3f}",
              flush=True)
        check("C. 记忆条件化可用：条件化 NLL ≤ 无条件化 + 1.0（读路径对齐经验）",
              nll_cond <= nll_plain + 1.0,
              f"cond={nll_cond:.3f} plain={nll_plain:.3f}")

        # ── 6. 邻居蒸馏保留降权（代码契约）──
        check("D. 邻居蒸馏保留降权：DISTILL_WEIGHT == 0.3",
              IntegrateEngine.DISTILL_WEIGHT == 0.3,
              f"DISTILL_WEIGHT={IntegrateEngine.DISTILL_WEIGHT}")

        # ── 7. 零破坏：成熟 neuron 参数不变 ──
        mature_after = cortex.neurons[split_parent].state_dict()
        mature_delta = sum(
            float((mature_after[k] - mature_before[k]).abs().max().item())
            for k in mature_before
            if k in mature_after and mature_before[k].shape == mature_after[k].shape
        )
        print(f"    成熟 neuron delta={mature_delta:.6f}", flush=True)
        check("E. 零破坏：成熟 neuron（父本）参数不变",
              mature_delta < 1e-8, f"delta={mature_delta:.2e}")

        # ── 8. 持久化：新 neuron 读路径随 state_dict 保存 ──
        saved_sd = cortex.neurons[new_nid].state_dict()
        read_keys = [k for k in saved_sd if k.startswith("field_read_layers.")]
        lora_keys = [k for k in saved_sd if k.startswith("lora_adapters.")]
        check("F. 持久化：新 neuron 读路径 + LoRA 随 state_dict 保存",
              len(read_keys) >= len(cortex.neurons[new_nid].layers)
              and len(lora_keys) >= 3,
              f"read_keys={len(read_keys)}, lora_keys={len(lora_keys)}")

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
        print(f"  记忆库: {bank.status()}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # 清理新生测试 ckpt（不污染 data/neurons）
        if new_ckpt and os.path.exists(new_ckpt):
            try:
                os.remove(new_ckpt)
            except Exception:
                pass

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
