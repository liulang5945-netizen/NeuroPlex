"""P2-2.2: 预计算每个 neuron 的 domain anchor。

用本域蒸馏数据计算每个 neuron 的"平均 field_vector"作为 domain anchor，
保存到 data/distill/domain_anchors.pt 供推理时 DomainRouter 使用。

Anchor 计算逻辑：
  对 neuron N 的本域数据 D_N，取前 max_samples 条：
    field_vec_i = neuron_N.forward(shared_emb(D_N[i]))["field_vector"]
    anchor_N = mean(field_vec_i).normalize()

推理时路由：
  当前输入 -> 每个 neuron 生成 field_vector
  weight_N = softmax(cos(field_vector_N, anchor_N) / temperature)

运行：
    python scripts/training/compute_domain_anchors.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


def main():
    print("=" * 72)
    print("P2-2.2: 预计算 DomainRouter 的 domain anchors")
    print("=" * 72)

    # ── 加载 Cortex + teacher ──
    print("\n[1] 加载 Cortex + teacher ...")
    from taiji.loader import create_cortex
    from taiji.training.checkpoint_bridge import load_teacher_model
    from taiji.resonance.shared_embed import SharedEmbedProj
    from taiji.resonance import DomainRouter

    cortex, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        enable_gating=False,
    )
    teacher, shared_embedding = load_teacher_model("checkpoint-481000", device="cpu")
    shared_proj = SharedEmbedProj.load("data/distill/shared_proj.pt", 2048, 512)
    cortex.set_teacher_pipeline(teacher_model=teacher, shared_proj=shared_proj)
    cortex.set_tokenizer(tokenizer)

    # ── 加载 domain 数据 ──
    print("\n[2] 加载 domain_datasets.pt ...")
    domain_data = torch.load("data/distill/domain_datasets.pt", map_location="cpu", weights_only=True)
    for nid, data in domain_data.items():
        print(f"  {nid}: {data.shape}")

    # ── shared_embedding 函数（用于 DomainRouter.compute_anchors） ──
    def shared_emb_fn(input_ids):
        with torch.no_grad():
            return cortex._embed_pipeline(input_ids)

    # ── 计算 anchors ──
    print("\n[3] 计算 domain anchors ...")
    t0 = time.time()
    field_dim = next(iter(cortex.neurons.values())).config.field_dim
    router = DomainRouter(field_dim=field_dim, temperature=0.1)
    router.compute_anchors(
        neurons=cortex.neurons,
        domain_data=domain_data,
        shared_embedding_fn=shared_emb_fn,
        device="cpu",
        batch_size=4,
        max_samples=50,
    )
    print(f"\n  耗时: {time.time()-t0:.1f}s")
    print(f"  anchors: {list(router.anchors.keys())}")

    # ── 验证：计算 anchor 之间的相似度矩阵 ──
    print("\n[4] Anchor 间相似度矩阵（应较低，表示域特化强）...")
    nids = list(router.anchors.keys())
    print(f"  {'':>10}", end="")
    for nid in nids:
        print(f"{nid:>10}", end="")
    print()
    for nid1 in nids:
        print(f"  {nid1:>10}", end="")
        a1 = router.anchors[nid1]
        for nid2 in nids:
            a2 = router.anchors[nid2]
            sim = torch.dot(a1, a2).item()
            print(f"{sim:>10.3f}", end="")
        print()

    # ── 保存 ──
    print("\n[5] 保存 anchors ...")
    router.save("data/distill/domain_anchors.pt")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print("""
使用方式（在 Cortex 创建时注入 DomainRouter）：
    from taiji.resonance import DomainRouter
    router = DomainRouter.load("data/distill/domain_anchors.pt")
    cortex.ensemble.domain_router = router
""")


if __name__ == "__main__":
    main()
