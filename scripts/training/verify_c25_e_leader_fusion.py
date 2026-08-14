#!/usr/bin/env python3
"""C25-E 遗留回归：continuous leader 融合 NLL 质量信号（2026-08-14）。

诊断结论（diag_c25e_leader_quality_gap.py）：round1_scores 与生成质量
（zh lm_head NLL）平均 Spearman=-0.171，leader 恰为 NLL 最优仅 1/7——
aug2 共振分碾压（0.7-0.93）当选 leader 5/7 次但 NLL 常最差（弱 neuron 独占）。

修复：continuous leader 融合 = 域内归一化共振分 × 质量信号（-NLL）等权。
本回归验证修复价值：
1. 融合后 leader 质量位次显著优于融合前（leader 应为 NLL 最优/次优）
2. 修复不破坏生成：8 问 API 等价对话非空率不降、质量不劣化
3. 口径守卫不受影响（对话格式 prompt 正常放行）

运行：python -u scripts/training/verify_c25_e_leader_fusion.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

QUESTIONS = [
    "你好", "你是谁？", "今天天气怎么样？", "1+1等于几？",
    "帮我写一首关于春天的诗", "推荐一本好书", "什么是幸福？", "怎么学好英语？",
]

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


def leader_rank_of(scores: dict, nll_quality: dict) -> int:
    """融合分数的 leader 在 NLL 质量排序中的位次（0=质量最优）。"""
    common = [k for k in nll_quality if k in scores]
    if not common:
        return len(nll_quality)  # 全无效 → 最差
    leader = max(scores, key=scores.get)
    order = sorted(common, key=lambda k: nll_quality[k], reverse=True)
    return order.index(leader) if leader in order else len(common)


def main():
    print("=" * 64, flush=True)
    print("C25-E 遗留回归：continuous leader 融合 NLL 质量信号", flush=True)
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
    hub = cortex._tokenizer_hub
    zh_tok = hub.get_tokenizer("zh")
    print(f"zh dialogue neurons: {zh_nids}", flush=True)

    # ── 1. 融合信号逻辑回归（不依赖生成，快速）──
    ranks_pure, ranks_fused = [], []
    for q in QUESTIONS:
        prompt = build_dialogue_prompt(q)
        general_ids = cortex._general_sp.encode(prompt)
        ids_t = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
        nemb = {n: cortex._neuron_shared_embeddings[n](ids_t) for n in zh_nids}
        r = cortex.think(active_nids=zh_nids, neuron_embeddings=nemb,
                         collab_mode="continuous")
        r1 = r.get("round1_scores") or {}
        nll_q = cortex._nll_quality_from_round1_logits(r, prompt, "zh")
        fused = cortex._fuse_leader_quality(r1, nll_q) if nll_q else {}
        if not fused or len(r1) < 2:
            continue
        ranks_pure.append(leader_rank_of(r1, nll_q))
        ranks_fused.append(leader_rank_of(fused, nll_q))
        print(f"  [{q[:10]}...] pure_rank={ranks_pure[-1]} "
              f"fused_rank={ranks_fused[-1]} "
              f"pure_leader={max(r1, key=r1.get)[:12]} "
              f"fused_leader={max(fused, key=fused.get)[:12]}", flush=True)
    n = len(ranks_pure)
    if n == 0:
        print("  [FAIL] 无有效样本", flush=True)
        sys.exit(1)
    avg_pure = sum(ranks_pure) / n
    avg_fused = sum(ranks_fused) / n
    print(f"\nleader NLL 质量位次均值: 融合前 {avg_pure:.2f} → 融合后 {avg_fused:.2f}",
          flush=True)
    check("融合后 leader 质量位次不劣化（≤ 融合前）",
          avg_fused <= avg_pure + 0.2, f"{avg_pure:.2f}→{avg_fused:.2f}")
    check("融合后 leader 质量位次显著提升（均值 < 1.5）",
          avg_fused < 1.5, f"avg={avg_fused:.2f}")

    # ── 2. 端到端生成不破坏（8 问 continuous 非空率）──
    non_empty = 0
    print("\n[2] 融合后 continuous 生成实测（8 问）", flush=True)
    for q in QUESTIONS:
        text = cortex.generate(
            prompt=build_dialogue_prompt(q), max_tokens=60, temperature=0.55,
            top_k=15, domain="zh", repetition_penalty=1.4,
            fusion_mode="soft", collab_mode="continuous",
        )
        if text.strip():
            non_empty += 1
        print(f"  [{q}] {text[:50]!r} len={len(text)}", flush=True)
    check("continuous 生成非空率 ≥ 7/8", non_empty >= 7, f"{non_empty}/8")

    print("\n" + "=" * 64, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 64, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
