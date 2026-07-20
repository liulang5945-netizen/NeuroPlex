"""Phase 5.1: 计算 ThalamicRouter 的 domain prototypes。

每个 neuron 的 prototype = 教师模型对本域数据 forward 取 last-token hidden state 的平均。

Usage:
    python scripts/training/compute_thalamic_prototypes.py
"""
from __future__ import annotations

import os
import sys

# sentencepiece 安装在 _libs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
# 项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance import ThalamicRouter


# ── 配置 ──
CHECKPOINT_DIR = "checkpoint-481000"
DOMAIN_DATASETS_PATH = "data/distill/domain_datasets.pt"
OUTPUT_PATH = "data/distill/thalamic_prototypes.pt"
TEACHER_HIDDEN_DIM = 2048
MAX_SAMPLES = 50
BATCH_SIZE = 4


def main():
    device = "cpu"
    print(f"[Phase 5.1] Computing ThalamicRouter prototypes")
    print(f"  checkpoint: {CHECKPOINT_DIR}")
    print(f"  domain_data: {DOMAIN_DATASETS_PATH}")
    print(f"  output: {OUTPUT_PATH}")
    print(f"  max_samples/domain: {MAX_SAMPLES}")
    print()

    # 1. 加载教师模型
    print("[1/3] Loading teacher model...")
    teacher, _ = load_teacher_model(CHECKPOINT_DIR, device=device)
    print(f"  Teacher loaded (type={type(teacher).__name__})")

    # 2. 加载域数据集
    print("\n[2/3] Loading domain datasets...")
    if not os.path.exists(DOMAIN_DATASETS_PATH):
        print(f"  ERROR: {DOMAIN_DATASETS_PATH} not found")
        sys.exit(1)
    domain_data = torch.load(DOMAIN_DATASETS_PATH, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(domain_data)} domains:")
    for nid, data in domain_data.items():
        print(f"    {nid}: shape={list(data.shape)}")

    # 3. 计算 prototypes
    print("\n[3/3] Computing prototypes via teacher forward...")
    router = ThalamicRouter(
        hidden_dim=TEACHER_HIDDEN_DIM,
        temperature=0.1,
        hard_route_threshold=0.7,
        soft_route_threshold=0.4,
    )

    # 用 nid 作为 domain key（distill 时存的就是 nid 索引）
    # neurons 这里用 domain_data 的 key 构造 dummy dict
    # router.compute_prototypes 内部只用 neurons 的 nid 和 config.spec
    class _DummyNeuron:
        def __init__(self, nid):
            class _Cfg:
                pass
            self.config = _Cfg()
            self.config.spec = nid

    neurons = {nid: _DummyNeuron(nid) for nid in domain_data.keys()}

    router.compute_prototypes(
        teacher_model=teacher,
        neurons=neurons,
        domain_data=domain_data,
        extract_hidden_fn=extract_hidden_states,
        device=device,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
    )

    # 4. 打印 prototype 矩阵的相似度（验证域之间是否可分）
    print("\n[4/4] Prototype similarity matrix (cosine):")
    nids = list(router.prototypes.keys())
    n = len(nids)
    print(f"  {'':>10}", end="")
    for nid in nids:
        print(f"  {nid[:8]:>10}", end="")
    print()
    for i, ni in enumerate(nids):
        print(f"  {ni[:10]:>10}", end="")
        pi = router.prototypes[ni]
        for j, nj in enumerate(nids):
            pj = router.prototypes[nj]
            sim = torch.dot(pi, pj).item()
            print(f"  {sim:>10.4f}", end="")
        print()

    # 5. 保存
    print(f"\nSaving to {OUTPUT_PATH}...")
    router.save(OUTPUT_PATH)
    print(f"\nDone. {len(router.prototypes)} prototypes saved.")
    print(f"Next: integrate into Cortex.generate() and verify routing on test prompts.")


if __name__ == "__main__":
    main()
