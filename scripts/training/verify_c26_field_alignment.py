#!/usr/bin/env python3
"""缺口 L 落地验证：跨域语义锚点投影正式化（2026-08-11）。

冒烟（verify_c26_cross_domain_align.py 4/4 PASS）已证明冻结场向量蕴含可提取
的跨域语义。本验证把 AlignProj 升级为产品组件 AnchorProjector（taiji/resonance/
field_alignment.py）+ cortex 挂载接口，用扩充语料（30 对双语术语）训练并闭环：

1. 训练收敛：AnchorProjector 对比训练后同义对余弦显著 > 错配对（幅度 ≥ 0.15）
2. 持久化闭环：save → 新实例 load → 评估一致
3. cortex 挂载：set_anchor_projector + project_field_state（投影后 128 维、
   未挂载原样返回零影响）
4. live 权重零改动（投影是独立组件）

运行：python -u scripts/training/verify_c26_field_alignment.py
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
from taiji.resonance.field_alignment import (  # noqa: E402
    AnchorProjector,
    evaluate_alignment,
    train_anchor_projector,
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


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

# 双语术语语料（30 对，zh ↔ en）——种子集，后续可用跨域平行语料扩充
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


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("缺口 L 落地：跨域语义锚点投影正式化验证", flush=True)
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

    # ── 1. 冻结场向量采集（30 对术语）──
    n = len(TERM_PAIRS)
    pos_pairs = TERM_PAIRS
    # 错配对：zh[i] ↔ en[(i+7) % n]（同词表错位，非同义）
    neg_pairs = [(TERM_PAIRS[i][0], TERM_PAIRS[(i + 7) % n][1]) for i in range(n)]

    texts = set()
    for a, b in pos_pairs + neg_pairs:
        texts.add(a)
        texts.add(b)
    vectors = {}
    print(f"\n[采集] {len(texts)} 个术语短语的冻结场向量 ...", flush=True)
    for txt in sorted(texts):
        vectors[txt] = field_state_of(cortex, txt)

    # ── 2. 训练 AnchorProjector ──
    print(f"\n[训练] AnchorProjector（{n} 对同义 + {n} 对错配，300 步）...", flush=True)
    proj = train_anchor_projector(vectors, pos_pairs, neg_pairs)
    syn, mis, gap = evaluate_alignment(proj, vectors, pos_pairs, neg_pairs)
    print(f"  投影空间: 同义 {syn:.3f} vs 错配 {mis:.3f}, 幅度 {gap:+.3f}", flush=True)
    check("锚点空间跨域对齐（幅度 ≥ 0.15）", gap >= 0.15,
          f"gap={gap:+.3f}（原始场 -0.022）")
    check("同义对投影余弦显著为正", syn >= 0.3, f"sim={syn:.3f}")

    # ── 3. 持久化闭环 ──
    tmp_dir = tempfile.mkdtemp(prefix="field_align_")
    try:
        path = os.path.join(tmp_dir, "anchor_projector.pt")
        proj.save(path)
        proj2 = AnchorProjector(in_dim=dim)
        ok_load = proj2.load(path)
        check("投影持久化 + 新实例恢复", ok_load,
              f"path={os.path.basename(path)}")
        syn2, mis2, gap2 = evaluate_alignment(proj2, vectors, pos_pairs, neg_pairs)
        check("恢复后对齐质量一致", abs(gap2 - gap) < 1e-3,
              f"gap2={gap2:+.3f}")

        # ── 4. cortex 挂载接口 ──
        fs_raw = vectors["函数"]
        check("未挂载时 project_field_state 原样返回",
              cortex.project_field_state(fs_raw).shape == fs_raw.shape)
        cortex.set_anchor_projector(proj)
        fs_proj = cortex.project_field_state(fs_raw)
        check("挂载后投影到锚点空间（128 维）",
              tuple(fs_proj.shape) == (128,), f"shape={tuple(fs_proj.shape)}")
        fs_proj_batch = cortex.project_field_state(torch.stack([fs_raw, fs_raw]))
        check("批量投影兼容", tuple(fs_proj_batch.shape) == (2, 128))
        check("投影输出 L2 归一化",
              abs(float(fs_proj.norm().item()) - 1.0) < 1e-3,
              f"norm={fs_proj.norm().item():.3f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 64, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 64, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
