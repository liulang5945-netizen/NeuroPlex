"""Precompute teacher hidden states cache for Phase 2 field conditioning.

For each domain in data/distill/domain_datasets.pt, runs the 1.55B teacher
forward pass on all samples and saves:
  data/distill/cache/{domain}_cached.pt  =  {"input_ids": [N,L], "embeddings": [N,L,2048]}
  data/distill/fixed_proj.pt             =  copy of data/shared_proj.pt

This lets distill_neurons.py Phase 2 use real teacher embeddings instead of
random noise (which would train field_read_layers on garbage).

Disk cost: ~5 GB total.  Compute cost: ~30 min on CPU.
"""
from __future__ import annotations
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states

CHECKPOINT = "E:/taiji-neuron/checkpoint-481000"
DATA_PATH  = "data/distill/domain_datasets.pt"
CACHE_DIR  = "data/distill/cache"
FIXED_PROJ = "data/distill/fixed_proj.pt"
SRC_PROJ   = "data/shared_proj.pt"
BATCH      = 4  # matches distill_neurons.py default


def main() -> None:
    print("=" * 60)
    print("Teacher hidden-state cache pre-compute")
    print("=" * 60)

    # 1) copy shared_proj -> fixed_proj so distill_neurons.py picks it up
    if not os.path.exists(FIXED_PROJ):
        os.makedirs(os.path.dirname(FIXED_PROJ), exist_ok=True)
        shutil.copy(SRC_PROJ, FIXED_PROJ)
        print(f"[1/3] Copied {SRC_PROJ} -> {FIXED_PROJ}")
    else:
        print(f"[1/3] {FIXED_PROJ} already exists")

    # 2) load teacher
    print(f"\n[2/3] Loading teacher 1.55B ...")
    t0 = time.time()
    teacher, _ = load_teacher_model(CHECKPOINT, device="cpu")
    teacher.eval()
    print(f"      loaded in {time.time()-t0:.1f}s")

    # 3) run cache
    print(f"\n[3/3] Computing embeddings ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    datasets = torch.load(DATA_PATH, map_location="cpu", weights_only=True)

    for domain, ids in datasets.items():
        out_path = os.path.join(CACHE_DIR, f"{domain}_cached.pt")
        if os.path.exists(out_path):
            print(f"  {domain}: {out_path} exists, skipping")
            continue

        N, L = ids.shape
        embeddings = torch.zeros(N, L, 2048, dtype=torch.float32)

        t0 = time.time()
        with torch.no_grad():
            for i in range(0, N, BATCH):
                b_ids = ids[i:i+BATCH]
                h = extract_hidden_states(teacher, b_ids)  # [B, L, 2048]
                embeddings[i:i+BATCH] = h
                if i % (BATCH * 25) == 0:
                    elapsed = time.time() - t0
                    done = i + BATCH
                    eta = elapsed / max(done, 1) * (N - done)
                    print(f"    {domain} [{done:>4d}/{N}]  {elapsed:5.0f}s elapsed  ETA {eta:5.0f}s")

        torch.save(
            {"input_ids": ids, "embeddings": embeddings},
            out_path,
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    {domain} DONE  {elapsed:5.0f}s  {size_mb:.0f} MB -> {out_path}")

    print()
    print("=" * 60)
    print("Cache ready.  distill_neurons.py Phase 2 will now use real")
    print("teacher embeddings instead of random noise.")
    print("=" * 60)


if __name__ == "__main__":
    main()