"""快速生成 distill_hidden_proj_512.pt（不依赖完整 cache）.

策略：
  1. 加载 1.5B teacher
  2. 从 domain_datasets.pt 取每个域前 N 条样本（默认 200 条）
  3. teacher forward 取 last-token hidden [total, 2048]
  4. SVD 得到 top-512 奇异向量 [512, 2048]
  5. 保存为 distill_hidden_proj_512.pt

预计耗时：5-10 分钟（vs precompute_teacher_cache.py 的 73 分钟）
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states

CHECKPOINT = "E:/taiji-neuron/checkpoint-481000"
DATA_PATH = "data/distill/domain_datasets.pt"
OUT_PATH = "data/distill/distill_hidden_proj_512.pt"
TARGET_DIM = 512
SAMPLES_PER_DOMAIN = 200  # 总计 5 域 × 200 = 1000 样本，够 SVD


def main():
    print("=" * 70)
    print(f"Fast distill_hidden_proj_{TARGET_DIM} builder")
    print("=" * 70)

    # 1. 加载 teacher
    print(f"[1/4] Loading teacher 1.55B ...")
    t0 = time.time()
    teacher, _ = load_teacher_model(CHECKPOINT, device="cpu")
    teacher.eval()
    print(f"      loaded in {time.time()-t0:.1f}s")

    # 2. 加载 domain datasets，每域取 SAMPLES_PER_DOMAIN 条
    print(f"\n[2/4] Loading domain datasets from {DATA_PATH}")
    datasets = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    all_samples = []
    for domain, ids in datasets.items():
        n = min(len(ids), SAMPLES_PER_DOMAIN)
        # 取前 n 条（不用随机，确保可重现）
        all_samples.append(ids[:n])
        print(f"  {domain}: {n} samples (total {len(ids)})")
    all_ids = torch.cat(all_samples, dim=0)  # [total, L]
    print(f"  total: {all_ids.shape}")

    # 3. teacher forward 取 last-token hidden
    print(f"\n[3/4] Running teacher forward to extract hidden samples...")
    t0 = time.time()
    BATCH = 4
    hidden_samples = []
    with torch.no_grad():
        for i in range(0, len(all_ids), BATCH):
            batch = all_ids[i:i+BATCH]
            h = extract_hidden_states(teacher, batch)  # [B, L, 2048]
            # 取 last-token hidden
            last = h[:, -1, :]  # [B, 2048]
            hidden_samples.append(last)
            if (i // BATCH + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + BATCH) * (len(all_ids) - i - BATCH)
                print(f"    [{i+BATCH}/{len(all_ids)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    hidden_samples = torch.cat(hidden_samples, dim=0).float()  # [N, 2048]
    print(f"  hidden_samples: {tuple(hidden_samples.shape)}, took {time.time()-t0:.1f}s")

    # 4. SVD 得到 top-TARGET_DIM 投影
    print(f"\n[4/4] SVD: {hidden_samples.shape} → top-{TARGET_DIM} components")
    # centered
    mean = hidden_samples.mean(dim=0, keepdim=True)
    centered = hidden_samples - mean
    # SVD: centered [N, 2048] = U @ S @ V^T, V is [2048, 2048]
    # 我们要的投影矩阵 W 使得 h @ W.T → [N, TARGET_DIM]
    # W = V[:, :TARGET_DIM].T 即 [TARGET_DIM, 2048]
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Vh shape: [min(N, 2048), 2048] = [2048, 2048] if N >= 2048
    proj_weight = Vh[:TARGET_DIM, :]  # [TARGET_DIM, 2048]
    print(f"  proj_weight: {tuple(proj_weight.shape)}")
    print(f"  top singular values: {S[:5].tolist()}")
    print(f"  explained variance ratio: {(S[:TARGET_DIM]**2).sum() / (S**2).sum():.4f}")

    # 验证正交性：proj_weight @ proj_weight.T ≈ I
    ortho = proj_weight @ proj_weight.T
    identity = torch.eye(TARGET_DIM)
    ortho_err = (ortho - identity).abs().max().item()
    print(f"  orthogonality check: max|WW^T - I| = {ortho_err:.6f}")

    # 5. 保存为 nn.Linear state_dict（weight shape: [TARGET_DIM, 2048]）
    lin = nn.Linear(2048, TARGET_DIM, bias=False)
    with torch.no_grad():
        lin.weight.copy_(proj_weight)
    for p in lin.parameters():
        p.requires_grad_(False)
    lin.eval()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(lin.state_dict(), OUT_PATH)
    print(f"\n✓ Saved {OUT_PATH}")
    print(f"  weight shape: {tuple(lin.weight.shape)}")
    print(f"  requires_grad: False (frozen)")

    # 6. 验证：投影后样本统计
    projected = lin(hidden_samples)  # [N, TARGET_DIM]
    print(f"\n[Sanity] projected samples stats:")
    print(f"  shape: {tuple(projected.shape)}")
    print(f"  mean: {projected.mean().item():.4f}")
    print(f"  std:  {projected.std().item():.4f}")
    print(f"  per-dim variance: mean={projected.var(dim=0).mean().item():.4f}, "
          f"min={projected.var(dim=0).min().item():.4f}, "
          f"max={projected.var(dim=0).max().item():.4f}")

    print()
    print("=" * 70)
    print(f"✓ distill_hidden_proj_{TARGET_DIM}.pt 已就绪")
    print("=" * 70)


if __name__ == "__main__":
    main()
