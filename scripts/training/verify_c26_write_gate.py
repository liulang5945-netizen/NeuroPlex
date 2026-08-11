#!/usr/bin/env python3
"""缺口 K 落地验证：可学习写策略（WriteGate）（2026-08-11）。

背景：Titans 的写是梯度驱动（memory-as-model）；态极 C26 第 0 格是朴素场快照 +
硬阈值去重（cosine_threshold=0.92）。缺口 K = 把"是否值得写入"变成可学习的。

本验证（field_memory.py WriteGate 集成）：
1. 门控可学习：BCE 训练后对"新信息（sim 低）"放行、"冗余（sim 高）"拒绝
2. 门控优于硬阈值：硬阈值（0.92）会误收 sim=0.9 的模糊重复；门控（训练负样本
   覆盖 0.85-0.95）学会拒绝——学习写策略的直接收益
3. consolidate(gate) 集成：首次固化 4 主题 + 检索命中 4/4 不降
4. 硬阈值路径回归（无 gate 时 C26 行为不变）
5. 门控持久化 save/load

运行：python -u scripts/training/verify_c26_write_gate.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.resonance.field_memory import FieldMemoryBank, WriteGate  # noqa: E402

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

TOPICS = [
    ("辉光协议", "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。"),
    ("铁月海", "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。"),
    ("卡尔文环", "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。"),
    ("频谱蜂鸟", "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。"),
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
    return (fs / (fs.norm() + 1e-8)).detach()


def train_gate(vectors: list, steps: int = 400) -> WriteGate:
    """训练写门控：正样本 = 新信息（sim 0.4-0.75，实测不同主题场基线 0.57-0.72），
    负样本 = 冗余（sim 0.88-0.98，覆盖重复/模糊重复区）。"""
    import random
    dim = vectors[0].shape[-1]
    torch.manual_seed(42)
    random.seed(42)
    gate = WriteGate(dim)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    # 数据：每主题向量 × 正/负 sim 特征
    xs, ys = [], []
    for v in vectors:
        for _ in range(3):
            xs.append((v, torch.tensor(random.uniform(0.4, 0.75),
                                       dtype=torch.float)))
            ys.append(1.0)
        for _ in range(3):
            xs.append((v, torch.tensor(random.uniform(0.88, 0.98),
                                       dtype=torch.float)))
            ys.append(0.0)
    for _ in range(steps):
        opt.zero_grad()
        tot = 0.0
        for (v, s), y in zip(xs, ys):
            p = gate(v, s)
            loss = torch.nn.functional.binary_cross_entropy(
                p.squeeze(-1), torch.tensor(y, dtype=torch.float))
            loss.backward()
            tot += loss.item()
        opt.step()
    return gate


def gate_decision(gate: WriteGate, v: torch.Tensor, sim: float) -> bool:
    return float(gate(v, torch.tensor(sim, dtype=torch.float)).item()) > 0.5


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("缺口 K 落地：可学习写策略（WriteGate）验证", flush=True)
    print("=" * 64, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    dim = int(cortex.field.dim)
    print(f"装配 {len(cortex.neurons)} 神经元, 场维度 {dim}", flush=True)

    # ── 1. 主题场向量（4 主题）──
    vecs = {}
    for label, text in TOPICS:
        vecs[label] = field_state_of(cortex, text)
    v_list = list(vecs.values())
    print(f"  4 主题场向量采集完成", flush=True)

    # ── 2. 训练写门控（正 sim 0.2-0.7 / 负 sim 0.85-0.98）──
    gate = train_gate(v_list)
    # 判别准确率（训练分布代表点）
    acc_ok = 0
    total = 0
    for v in v_list:
        for sim, want in [(0.5, True), (0.7, True), (0.9, False)]:
            got = gate_decision(gate, v, sim)
            acc_ok += 1 if got == want else 0
            total += 1
    print(f"  门控判别: {acc_ok}/{total} 与标签一致", flush=True)
    check("写门控可学习（判别准确率 ≥ 95%）", acc_ok / total >= 0.95,
          f"{acc_ok}/{total}")

    # ── 3. 门控优于硬阈值：sim=0.9 的模糊重复 ──
    v0 = v_list[0]
    hard_threshold = 0.92
    hard_accepts = 0.9 <= hard_threshold  # 硬阈值漏判（0.9 < 0.92 → 接受）
    gate_rejects = not gate_decision(gate, v0, 0.9)  # 门控训练覆盖 0.85-0.95 → 拒绝
    print(f"  sim=0.9 模糊重复: 硬阈值接受={hard_accepts}, 门控接受={not gate_rejects}",
          flush=True)
    check("门控优于硬阈值（拒绝 0.9 模糊重复，硬阈值漏判）",
          hard_accepts and gate_rejects)

    # ── 4. consolidate(gate) 集成：首次固化 4 主题 + 重复被拒 ──
    bank = FieldMemoryBank(dim=dim, gate=gate)
    first = bank.consolidate(list(vecs.values()), list(vecs.keys()))
    check("门控固化：4 个新主题全部写入", first == 4, f"added={first}")
    # 重复固化（相同主题 → sim≈1 → 门控拒绝）
    dup = bank.consolidate(list(vecs.values()), list(vecs.keys()))
    check("门控固化：重复主题全部拒绝", dup == 0, f"added={dup}")
    # 检索命中不降（门控写入的记忆仍可检索）
    hit = 0
    for label, v in vecs.items():
        top = bank.retrieve(v, top_k=1)
        hit += 1 if top and top[0][0] == label else 0
    check("门控写入后检索命中 4/4", hit == len(vecs), f"{hit}/4")

    # ── 5. 硬阈值路径回归（无 gate 时 C26 行为不变）──
    bank2 = FieldMemoryBank(dim=dim)
    bank2.consolidate(list(vecs.values()), list(vecs.keys()))
    dup2 = bank2.consolidate(list(vecs.values()), list(vecs.keys()))
    check("硬阈值路径回归（重复去重 added=0）", dup2 == 0, f"added={dup2}")

    # ── 6. 门控持久化 ──
    tmp_dir = tempfile.mkdtemp(prefix="write_gate_")
    try:
        path = os.path.join(tmp_dir, "write_gate.pt")
        gate.save(path)
        gate2 = WriteGate(dim)
        ok = gate2.load(path)
        check("写门控持久化 + 恢复", ok,
              f"path={os.path.basename(path)}")
        check("恢复后门控决策一致",
              gate_decision(gate2, v0, 0.9) == gate_decision(gate, v0, 0.9))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 64, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 64, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
