#!/usr/bin/env python3
"""C25-E 遗留诊断：continuous leader（round1_scores）与生成质量的错位量化。

背景（plan 2.2 第 4 项）：增量四已用 round1_scores（t=0 场共振分）选 leader，
消除了"时间平均激活均分 → 弱 neuron 独占"的空输出问题。本诊断量化剩余缺口：
- round1_scores（共振强度）与生成质量是否高度一致？
- dialogue neuron 无 judge_lm_head（实测，judge NLL 不可用），其 lm_head 是
  50K zh 词表 → 质量代理 = 用 general→domain 位置对齐后的 prompt
  next-token NLL（与生成前向的位置一致；越低 = 越贴合该域训练分布）。
- 若 leader 常落在 NLL 次优/最差的 neuron 上 → "共振分高但生成差"独占仍存在
  → 值得实施融合（质量信号 × 共振分）；
- 若两者排序高度相关（leader ≈ NLL 最优）→ 增量四已充分，记录收敛不实施。

指标：
- 每 prompt：round1_scores 选出的 leader 在 per-neuron NLL 排序中的位次（0=最优）
- Spearman 秩相关（round1_scores vs -NLL），均值横跨 12 prompt

运行：python -u scripts/training/diag_c25e_leader_quality_gap.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

_QUESTIONS = [
    "请介绍什么是神经网络", "什么是注意力机制", "请解释梯度下降的原理",
    "如何缓解过拟合问题", "什么是自然语言处理", "请介绍 Transformer 架构",
    "什么是词嵌入", "深度学习有哪些应用领域", "你好，请介绍一下你自己",
    "请用一句话介绍态极", "你能帮我做什么", "今天天气不错，聊聊吗",
]


def spearman(a: dict, b: dict) -> float:
    """秩相关（两 dict 同 key）。1=完全一致。"""
    keys = list(a.keys())
    ra = sorted(keys, key=lambda k: a[k])
    rb = sorted(keys, key=lambda k: b[k])
    rank_a = {k: i for i, k in enumerate(ra)}
    rank_b = {k: i for i, k in enumerate(rb)}
    n = len(keys)
    if n < 2:
        return 0.0
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in keys)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main():
    print("=" * 64, flush=True)
    print("C25-E 遗留诊断：round1_scores vs 生成质量（zh lm_head NLL）错位量化", flush=True)
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
    zh_nids = [nid for nid in cortex.neurons if nid.startswith("zh_")]
    print(f"zh dialogue neurons: {zh_nids}", flush=True)

    leader_nll_ranks = []
    fused_nll_ranks = []
    sp_vals = []
    for q in _QUESTIONS:
        prompt = build_dialogue_prompt(q)
        # 判定用 general 编码（neuron 输入），质量用 zh 编码（lm_head target 对齐）
        general_ids = cortex._general_sp.encode(prompt)
        if not general_ids:
            general_ids = [0]
        ids_t = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
        neuron_embeddings = {}
        for nid, emb in cortex._neuron_shared_embeddings.items():
            if nid in zh_nids:
                neuron_embeddings[nid] = emb(ids_t)
        r = cortex.think(
            active_nids=zh_nids,
            neuron_embeddings=neuron_embeddings,
            collab_mode="continuous",
        )
        r1 = r.get("round1_scores") or {}
        quality = cortex._nll_quality_from_round1_logits(r, prompt, "zh")
        nll = {nid: -score for nid, score in quality.items() if nid in zh_nids}
        if not r1 or not nll:
            continue
        leader = max(r1, key=r1.get)
        nll_rank = sorted(nll, key=nll.get)
        leader_nll_rank = nll_rank.index(leader)
        leader_nll_ranks.append(leader_nll_rank)
        fused = cortex._fuse_leader_quality(r1, quality)
        if fused:
            fused_leader = max(fused, key=fused.get)
            fused_nll_ranks.append(nll_rank.index(fused_leader))
        neg_nll = {k: -v for k, v in nll.items()}
        sp = spearman(r1, neg_nll)
        sp_vals.append(sp)
        print(f"  [{q[:14]}...] leader={leader} "
              f"NLL位次={leader_nll_rank}/{len(nll)-1} "
              f"spearman={sp:.3f}  r1=[{min(r1.values()):.3f},{max(r1.values()):.3f}] "
              f"NLL=[{min(nll.values()):.2f},{max(nll.values()):.2f}]", flush=True)

    n = len(leader_nll_ranks)
    if n == 0:
        print("  [FAIL] 无有效样本", flush=True)
        sys.exit(1)
    perfect = sum(1 for rk in leader_nll_ranks if rk == 0)
    fused_perfect = sum(1 for rk in fused_nll_ranks if rk == 0)
    mean_sp = sum(sp_vals) / len(sp_vals)
    print("\n" + "=" * 64, flush=True)
    print(f"leader NLL 位次分布: {sorted(leader_nll_ranks)}", flush=True)
    print(f"leader 恰为 NLL 最优: {perfect}/{n}（{(100*perfect/n):.0f}%）", flush=True)
    print(f"质量融合后 NLL 位次分布: {sorted(fused_nll_ranks)}", flush=True)
    print(f"质量融合后恰为 NLL 最优: {fused_perfect}/{len(fused_nll_ranks)}（{(100*fused_perfect/max(len(fused_nll_ranks), 1)):.0f}%）", flush=True)
    print(f"平均 Spearman(r1, -NLL): {mean_sp:.3f}", flush=True)
    if perfect / n >= 0.75 and mean_sp >= 0.6:
        print("→ round1_scores 与生成质量高度一致，leader 无弱 neuron 独占 → 收敛不实施", flush=True)
    elif perfect / n >= 0.5:
        print("→ 部分错位，leader 偶有 NLL 次优 → 融合收益边际，可暂缓", flush=True)
    else:
        print("→ 错位显著，leader 常选到生成质量差的 neuron → 值得实施融合", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
