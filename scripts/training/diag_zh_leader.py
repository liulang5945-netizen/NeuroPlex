#!/usr/bin/env python3
"""zh 生成 leader 诊断（2026-08-11）。

背景：general zh 基座 51M（hidden 512）是容量瓶颈（培养期平台定性），但升级到
134M 需 spec-up + C24 重训（大任务）。先决策：zh 生成时 leader 是 general zh
（51M）还是 dialogue neuron（zh_std0 134M / zh_aug* 51M）？
- leader = general zh → 升级 general zh 直接提升生成（值得）
- leader = dialogue（zh_std0 134M）→ general zh 主要管判定，升级优先级低

方法：模拟 generate 首轮迭代（编码 → per-neuron embed → think continuous →
应用 _generate_p7 leader 选择逻辑），打印各 neuron round1_scores 与 leader。

运行：python -u scripts/training/diag_zh_leader.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

PROMPTS = [
    "请介绍什么是神经网络",
    "如何缓解过拟合问题",
    "什么是注意力机制",
    "请解释梯度下降的原理",
    "你好，请介绍一下你自己",
]


def main():
    print("=" * 60, flush=True)
    print("zh 生成 leader 诊断（continuous 模式）", flush=True)
    print("=" * 60, flush=True)

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
    print(f"  装配: {nids}", flush=True)
    print(f"  规格: { {n: (getattr(n, 'config', None) and getattr(n.config, 'hidden_size', '?')) for n, nid in [(cortex.neurons[n], n) for n in nids]} }", flush=True)

    domain = "zh"
    domain_nids = [k for k in cortex.neurons if k == domain or k.startswith(domain + "_")]
    print(f"\n  zh 域 neurons: {domain_nids}", flush=True)

    leader_stats = {}
    for prompt in PROMPTS:
        # 模拟 generate 首轮：编码 + per-neuron embed + think continuous
        general_ids = cortex._general_sp.encode(prompt)
        if not general_ids:
            general_ids = [0]
        ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
        neuron_embeddings = {}
        for nid, emb in cortex._neuron_shared_embeddings.items():
            if nid in domain_nids:
                neuron_embeddings[nid] = emb(ids_tensor)
        result = cortex.think(
            active_nids=domain_nids,
            fusion_mode="soft",
            neuron_embeddings=neuron_embeddings,
            collab_mode="continuous",
        )
        qual_scores = result.get("round1_scores") or result.get("final_scores", {})
        domain_scores = {k: v for k, v in qual_scores.items()
                         if k == domain or k.startswith(domain + "_")}
        leader_nid = max(domain_scores, key=domain_scores.get) if domain_scores else "?"
        leader_stats[leader_nid] = leader_stats.get(leader_nid, 0) + 1
        scores_sorted = sorted(domain_scores.items(), key=lambda x: -x[1])
        print(f"\n  prompt: {prompt}", flush=True)
        for nid, s in scores_sorted:
            h = getattr(cortex.neurons[nid].config, "hidden_size", "?")
            mark = " ← leader" if nid == leader_nid else ""
            print(f"    {nid:24s} score={s:7.4f} hidden={h}{mark}", flush=True)

    print("\n" + "=" * 50, flush=True)
    print(f"leader 统计（{len(PROMPTS)} prompts）: {leader_stats}", flush=True)
    gen_leaders = [n for n in leader_stats if n.startswith("zh_")]
    general_zh_leader = leader_stats.get("zh", 0)
    print(f"  general zh(51M) 当选: {general_zh_leader}/{len(PROMPTS)}", flush=True)
    print(f"  dialogue 当选: {sum(leader_stats.get(n, 0) for n in gen_leaders)}/{len(PROMPTS)}", flush=True)


if __name__ == "__main__":
    main()
