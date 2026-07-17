"""Build all three teacher-derived projections in one shot.

Prerequisites:
  data/distill/cache/{zh,en,code,math,general}_cached.pt   (precomputed teacher hidden)
  E:/taiji-neuron/checkpoint-481000                        (teacher weights for embedding)

Produces (all under data/distill/):
  shared_proj.pt                    2048 -> 512  (teacher embedding SVD)
  distill_hidden_proj_384.pt        2048 -> 384  (teacher hidden SVD, FOUNDATION)
  distill_hidden_proj_768.pt        2048 -> 768  (STANDARD, if needed)
  field_proj_4096.pt                2048 -> 4096 (padded teacher hidden basis)
  fixed_proj.pt                     mirror of shared_proj.pt (Phase 2 uses this name)

  Also mirrors shared_proj.pt to data/shared_proj.pt for legacy load paths.
"""
from __future__ import annotations
import os, sys, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from taiji.training.checkpoint_bridge import load_teacher_model
from taiji.resonance.init_from_teacher import (
    get_or_build_shared_proj,
    get_or_build_hidden_proj,
    get_or_build_field_proj,
    sample_teacher_hidden_from_cache,
)

CHECKPOINT = "E:/taiji-neuron/checkpoint-481000"


def main() -> None:
    print("=" * 60)
    print("Teacher-derived SVD projections (H10, H11, H12)")
    print("=" * 60)

    # 1) shared_proj (2048 -> 512) from teacher embedding
    print("\n[1/3] shared_proj: SVD of teacher embedding [256000, 2048]")
    t0 = time.time()
    teacher, embedding = load_teacher_model(CHECKPOINT, device="cpu")
    del teacher   # only need the embedding matrix
    print(f"      teacher embedding: {tuple(embedding.weight.shape)}, "
          f"loaded in {time.time()-t0:.1f}s")

    shared = get_or_build_shared_proj(embedding.weight.detach(), target_dim=512)
    print(f"      done: {shared.weight.shape}")

    # 2) hidden_proj (2048 -> 384) for FOUNDATION neurons distill target
    #    Also 768 for STANDARD, in case we ever revisit.
    print("\n[2/3] distill_hidden_proj: SVD of teacher hidden samples")
    print("      loading a sample of teacher hidden vectors from cache ...")
    t0 = time.time()
    hidden_samples = sample_teacher_hidden_from_cache(max_rows=8000)
    print(f"      hidden samples: {tuple(hidden_samples.shape)}, "
          f"loaded in {time.time()-t0:.1f}s")

    for h_dim in (384, 768):
        hp = get_or_build_hidden_proj(hidden_samples, target_dim=h_dim)
        print(f"      hidden_proj_{h_dim}: {hp.weight.shape}")

    # 3) field_proj (2048 -> 4096) for field-contrastive anchor projection
    print("\n[3/3] field_proj: 2048 -> 4096 (teacher basis + zero padding)")
    fp = get_or_build_field_proj(hidden_samples, field_dim=4096)
    print(f"      field_proj_4096: {fp.weight.shape}")

    # 4) fixed_proj.pt copy for Phase 2 use_cached branch
    src = os.path.join("data", "distill", "shared_proj.pt")
    dst = os.path.join("data", "distill", "fixed_proj.pt")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"\n      mirrored {src} -> {dst}")

    print("\n" + "=" * 60)
    print("All projections ready.  distill_neurons.py can now run deterministic.")
    print("=" * 60)


if __name__ == "__main__":
    main()