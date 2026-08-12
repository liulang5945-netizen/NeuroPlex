#!/usr/bin/env python3
"""跨域语义对齐现状诊断（2026-08-11）——hub neuron（缺口 L）上马前的数据决策。

背景：HUB_NEURON_DESIGN.md（草案）要解决"跨域语义对齐"——显式约束
zh/code/en 的 field_vector 对齐（跨域对比 loss）。现状：共振场（独立于词表）
已在隐式扮演跨域锚点，但从未验证"自发对齐"到底对齐得怎么样。

诊断三件事：
1. 装配后共振场真实维度（设计文档写 4096，实测由 neuron unified_field_dim 决定）
2. 跨域同义对（zh"函数"↔en"function"）的场余弦 vs 错配对（zh"函数"↔en"array"）
   ——同义对显著更高 = 场已自发对齐语义；无差异 = 无对齐，hub 有上马必要
3. 对齐幅度 = 同义对均值 − 错配对均值（>0.1 视为已对齐，<0.03 视为无对齐）

运行：python -u scripts/training/diag_cross_domain_alignment.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

# 跨域同义对（zh ↔ en/code，词级短语，场输入越短噪声越大→用短语）
SYNONYM_PAIRS = [
    ("函数", "function"),
    ("数组", "array"),
    ("神经网络", "neural network"),
    ("循环", "loop"),
    ("变量", "variable"),
    ("数据结构", "data structure"),
    ("机器学习", "machine learning"),
    ("递归", "recursion"),
    ("排序算法", "sorting algorithm"),
    ("类的继承", "class inheritance"),
    ("梯度下降", "gradient descent"),
    ("线性回归", "linear regression"),
]

# 错配对照（同词表内错位配对，非同义）
MISMATCH_PAIRS = [
    ("函数", "array"),
    ("神经网络", "variable"),
    ("循环", "data structure"),
    ("变量", "recursion"),
    ("数组", "sorting algorithm"),
    ("机器学习", "class inheritance"),
    ("递归", "gradient descent"),
    ("排序算法", "function"),
    ("梯度下降", "loop"),
    ("线性回归", "array"),
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
    return fs / (fs.norm() + 1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b).item())


def main():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("跨域语义对齐现状诊断（hub neuron 上马前数据决策）", flush=True)
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
    nids = list(cortex.neurons.keys())
    print(f"装配 {len(nids)} 神经元: {nids}", flush=True)

    # ── 1. 维度核实 ──
    dim = int(cortex.field.dim)
    print(f"\n[维度] 共振场实测 dim={dim}（设计文档写 4096）", flush=True)
    dims = {}
    for nid in nids:
        n = cortex.neurons[nid]
        fd = n.config.unified_field_dim if n.config.unified_field_dim is not None else n.config.field_dim
        dims[nid] = fd
    print(f"  neuron field 配置: {dims}", flush=True)
    print(f"  → max = {dim}（cortex 取 effective 最大值）", flush=True)

    # ── 2. 场向量采集（去重缓存，减少前向次数）──
    texts = set()
    for a, b in SYNONYM_PAIRS + MISMATCH_PAIRS:
        texts.add(a)
        texts.add(b)
    cache = {}
    print(f"\n[采集] {len(texts)} 个文本短语的场向量 ...", flush=True)
    for txt in sorted(texts):
        cache[txt] = field_state_of(cortex, txt)
    print(f"  采集完成（{len(cache)} 条）", flush=True)

    # ── 3. 同义对 vs 错配对余弦 ──
    syn_sims = [cosine(cache[a], cache[b]) for a, b in SYNONYM_PAIRS]
    mis_sims = [cosine(cache[a], cache[b]) for a, b in MISMATCH_PAIRS]
    syn_mean = sum(syn_sims) / len(syn_sims)
    mis_mean = sum(mis_sims) / len(mis_sims)
    gap = syn_mean - mis_mean

    print("\n[结果] 跨域场余弦（sim ∈ [-1, 1]）", flush=True)
    print(f"  同义对  均值 {syn_mean:.3f}（范围 {min(syn_sims):.3f}~{max(syn_sims):.3f}）", flush=True)
    for (a, b), s in zip(SYNONYM_PAIRS, syn_sims):
        print(f"    {a:10s} ↔ {b:16s}  {s:+.3f}", flush=True)
    print(f"  错配对  均值 {mis_mean:.3f}（范围 {min(mis_sims):.3f}~{max(mis_sims):.3f}）", flush=True)
    for (a, b), s in zip(MISMATCH_PAIRS, mis_sims):
        print(f"    {a:10s} ↔ {b:16s}  {s:+.3f}", flush=True)
    print(f"\n  对齐幅度 = 同义 − 错配 = {gap:+.3f}", flush=True)

    # ── 4. 方向性结论（诊断报告，非硬性断言）──
    print("\n[结论]", flush=True)
    if gap >= 0.10 and syn_mean >= 0.30:
        print(f"  共振场已自发对齐跨域语义（幅度 {gap:+.3f} ≥ 0.10）——", flush=True)
        print("  场作为隐式跨域锚点成立；hub neuron（缺口 L）可降级/简化，", flush=True)
        print("  只需补显式对比约束而非整套 hub 架构", flush=True)
    elif gap >= 0.03:
        print(f"  存在弱对齐趋势（幅度 {gap:+.3f}）——", flush=True)
        print("  场部分捕获跨域语义，但未达显式约束水平；hub 或场对比 loss 需评估", flush=True)
    else:
        print(f"  无有效对齐（幅度 {gap:+.3f} < 0.03）——", flush=True)
        print("  共振场未自发对齐跨域语义，跨域协作依赖词库转译（token 级）；", flush=True)
        print("  hub neuron（跨域对比 loss）有明确上马必要（缺口 L）", flush=True)

    # 自相似基准（同一文本两次前向，应 ≈ 1.0）
    fs1 = field_state_of(cortex, "函数")
    fs2 = field_state_of(cortex, "函数")
    print(f"\n  自相似基准（同文本两次前向）: {cosine(fs1, fs2):.3f}（应≈1.0，验证采集稳定性）", flush=True)

    print("\n" + "=" * 64, flush=True)
    print(f"诊断完成 ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
