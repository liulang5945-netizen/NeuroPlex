"""Teacher-aware SVD initialisation for the shared projections.

Three projections are trained-free but data-driven, initialised once via SVD
of the teacher's embedding matrix so they carry real teacher semantics:

  shared_proj.pt         (2048 -> 512)          teacher embedding basis
  distill_hidden_proj.pt (2048 -> hidden_size)  teacher hidden basis (per hidden dim)
  field_proj.pt          (2048 -> field_dim)    teacher basis padded to 4096

Loaded on demand by distill_neurons.py so every process/run sees the same
projections instead of a fresh random orthogonal each time (H10/H11/H12).

Usage:
    from taiji.resonance.init_from_teacher import (
        get_or_build_shared_proj,      # 2048 -> 512
        get_or_build_hidden_proj,      # 2048 -> H (per hidden dim)
        get_or_build_field_proj,       # 2048 -> D (per field dim)
    )
    proj = get_or_build_shared_proj(teacher_embedding_weight)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn


PROJ_DIR = os.path.join("data", "distill")   # keep all next to shared_proj.pt sibling
LEGACY_SHARED = os.path.join("data", "shared_proj.pt")


def _svd_project(source: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Return a [src_dim, out_dim] projection whose columns are the top-out_dim
    right singular vectors of `source`. This preserves the directions in the
    original space that carry the most variance.

    source: [N, src_dim] matrix (teacher embedding rows, teacher hidden rows, ...)
    """
    assert source.dim() == 2, f"source must be 2D [N, src_dim], got {source.shape}"
    src_dim = source.shape[1]

    with torch.no_grad():
        # Centre so SVD picks variance directions (not the mean direction).
        centred = source - source.mean(dim=0, keepdim=True)
        # torch.linalg.svd is memory-heavy on [256000, 2048]; use lowrank.
        # niter=4 is fine when out_dim << src_dim (we want 512 or 4096 out of 2048).
        # For out_dim >= src_dim we fall back to identity padding below.
        if out_dim <= src_dim:
            # top-k right singular vectors
            q = min(out_dim + 32, src_dim)  # small oversample for stability
            U, S, V = torch.svd_lowrank(centred, q=q, niter=4)  # V: [src_dim, q]
            proj = V[:, :out_dim].contiguous()                   # [src_dim, out_dim]
        else:
            # out_dim > src_dim: full SVD then zero-pad extra columns.
            U, S, V = torch.svd_lowrank(centred, q=src_dim, niter=4)
            base = V[:, :src_dim].contiguous()                   # [src_dim, src_dim]
            pad = torch.zeros(src_dim, out_dim - src_dim, dtype=base.dtype, device=base.device)
            proj = torch.cat([base, pad], dim=1)                 # [src_dim, out_dim]
    return proj


def _make_linear(src_dim: int, out_dim: int, weight: torch.Tensor) -> nn.Linear:
    """weight: [src_dim, out_dim] --> Linear(src_dim, out_dim, bias=False) with weight.T"""
    lin = nn.Linear(src_dim, out_dim, bias=False)
    with torch.no_grad():
        # nn.Linear stores weight as [out, in]; we have [in, out], transpose.
        lin.weight.copy_(weight.T.contiguous())
    for p in lin.parameters():
        p.requires_grad_(False)
    lin.eval()
    return lin


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def get_or_build_shared_proj(
    teacher_embedding_weight: torch.Tensor,
    target_dim: int = 512,
) -> nn.Linear:
    """2048 -> 512 shared embedding projection.

    Persisted at data/distill/shared_proj.pt (and mirrored to data/shared_proj.pt
    for compatibility with the pre-existing SharedEmbedProj.load path).
    """
    os.makedirs(PROJ_DIR, exist_ok=True)
    path = os.path.join(PROJ_DIR, "shared_proj.pt")
    src_dim = teacher_embedding_weight.shape[1]

    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        lin = nn.Linear(src_dim, target_dim, bias=False)
        lin.load_state_dict(sd)
        for p in lin.parameters(): p.requires_grad_(False)
        lin.eval()
        return lin

    print(f"  [SVD] shared_proj {src_dim} -> {target_dim} from teacher embedding "
          f"({teacher_embedding_weight.shape[0]} rows) ...")
    proj = _svd_project(teacher_embedding_weight, target_dim)
    lin = _make_linear(src_dim, target_dim, proj)
    torch.save(lin.state_dict(), path)
    # Legacy mirror is best-effort (sandboxed writes may fail); code paths
    # that need shared_proj should look under data/distill/ first.
    try:
        torch.save(lin.state_dict(), LEGACY_SHARED)
    except (RuntimeError, PermissionError) as e:
        print(f"  [SVD] warning: legacy mirror to {LEGACY_SHARED} failed ({e.__class__.__name__}); primary copy still saved")
    print(f"  [SVD] saved {path}")
    return lin


def get_or_build_hidden_proj(
    teacher_hidden_samples: torch.Tensor,
    target_dim: int,
) -> nn.Linear:
    """2048 -> hidden_size distill target projection (H11).

    teacher_hidden_samples: [N, 2048] flattened teacher last-token or pooled hidden.
    Persisted at data/distill/distill_hidden_proj_{target_dim}.pt.
    """
    os.makedirs(PROJ_DIR, exist_ok=True)
    path = os.path.join(PROJ_DIR, f"distill_hidden_proj_{target_dim}.pt")
    src_dim = teacher_hidden_samples.shape[1]

    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        lin = nn.Linear(src_dim, target_dim, bias=False)
        lin.load_state_dict(sd)
        for p in lin.parameters(): p.requires_grad_(False)
        lin.eval()
        return lin

    print(f"  [SVD] distill_hidden_proj {src_dim} -> {target_dim} "
          f"from {teacher_hidden_samples.shape[0]} teacher hidden samples ...")
    proj = _svd_project(teacher_hidden_samples, target_dim)
    lin = _make_linear(src_dim, target_dim, proj)
    torch.save(lin.state_dict(), path)
    print(f"  [SVD] saved {path}")
    return lin


def get_or_build_field_proj(
    teacher_hidden_samples: torch.Tensor,
    field_dim: int = 4096,
) -> nn.Linear:
    """2048 -> field_dim projection for field-contrastive anchors (H12).

    Uses teacher hidden variance basis (same source as hidden_proj) then pads
    to field_dim with zeros so we never invent extra dimensions.
    Persisted at data/distill/field_proj_{field_dim}.pt.
    """
    os.makedirs(PROJ_DIR, exist_ok=True)
    path = os.path.join(PROJ_DIR, f"field_proj_{field_dim}.pt")
    src_dim = teacher_hidden_samples.shape[1]

    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        lin = nn.Linear(src_dim, field_dim, bias=False)
        lin.load_state_dict(sd)
        for p in lin.parameters(): p.requires_grad_(False)
        lin.eval()
        return lin

    print(f"  [SVD] field_proj {src_dim} -> {field_dim} "
          f"from {teacher_hidden_samples.shape[0]} teacher hidden samples ...")
    proj = _svd_project(teacher_hidden_samples, field_dim)
    lin = _make_linear(src_dim, field_dim, proj)
    torch.save(lin.state_dict(), path)
    print(f"  [SVD] saved {path}")
    return lin


def sample_teacher_hidden_from_cache(
    cache_dir: str = "data/distill/cache",
    max_rows: int = 8000,
) -> torch.Tensor:
    """Load a sample of teacher hidden vectors from precomputed cache.

    Reads each domain's cached embeddings ([N, L, 2048]), flattens to [N*L, 2048]
    and subsamples to at most `max_rows` for SVD input.
    """
    all_rows = []
    remain = max_rows
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"cache dir {cache_dir} not found; run precompute_teacher_cache.py first")
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith("_cached.pt"):
            continue
        d = torch.load(os.path.join(cache_dir, fn), map_location="cpu", weights_only=True)
        emb = d["embeddings"]  # [N, L, 2048]
        flat = emb.reshape(-1, emb.shape[-1])  # [N*L, 2048]
        # take at most 1/n_domains of remaining budget from each domain
        take = min(flat.shape[0], remain // 3 + 1)
        idx = torch.randperm(flat.shape[0])[:take]
        all_rows.append(flat[idx])
        remain -= take
        if remain <= 0:
            break
    out = torch.cat(all_rows, dim=0)[:max_rows].contiguous()
    return out