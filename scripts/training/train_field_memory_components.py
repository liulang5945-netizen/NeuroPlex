#!/usr/bin/env python3
"""训练场记忆产品组件并保存到 sleep 数据目录（2026-08-11）。

产物（sleep_engine.get_field_memory 自动装配，存在即用）：
- {sleep_data}/anchor_projector.pt — 跨域语义锚点投影（缺口 L：检索在锚点空间）
- {sleep_data}/write_gate.pt      — 可学习写门控（缺口 K：睡眠固化用学习门控替代硬阈值）

训练数据：
- AnchorProjector：30 对双语术语（zh↔en）冻结场向量 + 对比 margin loss
- WriteGate：4 个知识主题场向量，正样本 sim 0.4-0.75（新信息，实测主题基线
  0.57-0.72）/ 负样本 sim 0.88-0.98（冗余/模糊重复）

运行：python -u scripts/training/train_field_memory_components.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.resonance.field_alignment import (  # noqa: E402
    AnchorProjector,
    evaluate_alignment,
    train_anchor_projector,
)
from taiji.resonance.field_memory import WriteGate  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

TERM_PAIRS = [
    ("函数", "function"), ("数组", "array"), ("神经网络", "neural network"),
    ("循环", "loop"), ("变量", "variable"), ("数据结构", "data structure"),
    ("机器学习", "machine learning"), ("递归", "recursion"),
    ("排序算法", "sorting algorithm"), ("类的继承", "class inheritance"),
    ("梯度下降", "gradient descent"), ("线性回归", "linear regression"),
    ("卷积", "convolution"), ("注意力机制", "attention mechanism"),
    ("词嵌入", "word embedding"), ("损失函数", "loss function"),
    ("学习率", "learning rate"), ("过拟合", "overfitting"),
    ("正则化", "regularization"), ("激活函数", "activation function"),
    ("反向传播", "backpropagation"), ("批归一化", "batch normalization"),
    ("强化学习", "reinforcement learning"), ("决策树", "decision tree"),
    ("聚类", "clustering"), ("特征提取", "feature extraction"),
    ("时间序列", "time series"), ("概率分布", "probability distribution"),
    ("矩阵乘法", "matrix multiplication"), ("量子比特", "qubit"),
]

TOPICS = [
    "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
    "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
    "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
    "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。",
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


def train_write_gate(vectors: list, steps: int = 400) -> WriteGate:
    """训练写门控（正 sim 0.4-0.75 / 负 sim 0.88-0.98，见 verify_c26_write_gate）。"""
    dim = vectors[0].shape[-1]
    torch.manual_seed(42)
    random.seed(42)
    gate = WriteGate(dim)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
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
        for (v, s), y in zip(xs, ys):
            p = gate(v, s)
            loss = torch.nn.functional.binary_cross_entropy(
                p.squeeze(-1), torch.tensor(y, dtype=torch.float))
            loss.backward()
        opt.step()
    return gate


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("训练场记忆产品组件（AnchorProjector + WriteGate）", flush=True)
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

    # ── 1. 冻结场向量采集（30 对术语 + 4 主题）──
    n = len(TERM_PAIRS)
    pos_pairs = TERM_PAIRS
    neg_pairs = [(TERM_PAIRS[i][0], TERM_PAIRS[(i + 7) % n][1]) for i in range(n)]
    texts = set()
    for a, b in pos_pairs + neg_pairs:
        texts.add(a)
        texts.add(b)
    texts.update(TOPICS)
    vectors = {}
    print(f"\n[采集] {len(texts)} 个文本的冻结场向量 ...", flush=True)
    for txt in sorted(texts):
        vectors[txt] = field_state_of(cortex, txt)

    # ── 2. 训练 AnchorProjector（跨域语义锚点）──
    print(f"\n[训练] AnchorProjector（{n} 对同义 + {n} 对错配）...", flush=True)
    proj = train_anchor_projector(vectors, pos_pairs, neg_pairs)
    syn, mis, gap = evaluate_alignment(proj, vectors, pos_pairs, neg_pairs)
    print(f"  锚点空间: 同义 {syn:.3f} vs 错配 {mis:.3f}, 幅度 {gap:+.3f}", flush=True)

    # ── 3. 训练 WriteGate（可学习写门控）──
    topic_vecs = [vectors[t] for t in TOPICS]
    print(f"\n[训练] WriteGate（4 主题 × 正/负 sim 分布）...", flush=True)
    gate = train_write_gate(topic_vecs)
    # 抽查门控决策
    v0 = topic_vecs[0]
    def dec(sim):
        return float(gate(v0, torch.tensor(sim, dtype=torch.float)).item()) > 0.5
    print(f"  门控决策抽查: sim=0.5→{dec(0.5)}（新信息应写）, sim=0.9→{dec(0.9)}（冗余应拒）",
          flush=True)

    # ── 4. 保存到 sleep 数据目录（sleep_engine 自动装配）──
    try:
        from taiji.config import get_taiji_data_path
        data_dir = get_taiji_data_path("sleep_data")
    except ImportError:
        data_dir = "taiji/sleep_data"
    os.makedirs(data_dir, exist_ok=True)
    proj_path = os.path.join(data_dir, "anchor_projector.pt")
    gate_path = os.path.join(data_dir, "write_gate.pt")
    proj.save(proj_path)
    gate.save(gate_path)
    print(f"\n[保存]", flush=True)
    print(f"  AnchorProjector → {proj_path}", flush=True)
    print(f"  WriteGate       → {gate_path}", flush=True)

    # ── 5. 回读校验 ──
    proj2 = AnchorProjector(dim)
    gate2 = WriteGate(dim)
    ok_p = proj2.load(proj_path)
    ok_g = gate2.load(gate_path)
    print(f"  回读: projector={ok_p}, gate={ok_g}", flush=True)
    if not (ok_p and ok_g):
        print("  [FAIL] 产物回读失败", flush=True)
        sys.exit(1)

    print("\n" + "=" * 64, flush=True)
    print(f"完成 ({time.time() - t0:.1f}s)——sleep 场固化将自动装配上述组件", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
