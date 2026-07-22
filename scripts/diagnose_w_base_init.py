"""一次性检查 1.5B checkpoint 的 lm_head 结构 + distill_hidden_proj 形状。

目的：验证能否构造 W_base = lm_head.weight @ proj.weight.T
"""
import torch
from pathlib import Path

CKPT = "checkpoint-481000/model.pt"
PROJ = "data/distill/distill_hidden_proj_384.pt"

print("=" * 70)
print("[1] 1.5B checkpoint 结构")
print("=" * 70)
state = torch.load(ckpt_path := CKPT, map_location="cpu", weights_only=False)
if hasattr(state, "state_dict"):
    state = state.state_dict()
if not isinstance(state, dict):
    state = {"<root>": state}

# 找 lm_head 相关 key
lm_head_keys = [k for k in state.keys() if "lm_head" in k.lower() or "head" in k.lower()][:20]
print(f"  Total keys: {len(state)}")
print(f"  head-related keys (first 20):")
for k in lm_head_keys:
    v = state[k]
    if hasattr(v, "shape"):
        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"    {k}: type={type(v).__name__}")

# 找 embedding 相关 key
print()
print("  embed-related keys (first 10):")
for k in list(state.keys())[:50]:
    if "embed" in k.lower():
        v = state[k]
        if hasattr(v, "shape"):
            print(f"    {k}: shape={tuple(v.shape)}")

print()
print("=" * 70)
print("[2] distill_hidden_proj_384.pt 结构")
print("=" * 70)
proj_state = torch.load(PROJ, map_location="cpu", weights_only=False)
if isinstance(proj_state, dict):
    for k, v in proj_state.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
else:
    print(f"  Type: {type(proj_state)}")
    if hasattr(proj_state, "shape"):
        print(f"  shape: {tuple(proj_state.shape)}")

print()
print("=" * 70)
print("[3] 当前 _shared_lm_head_base.pt 结构（要替换的）")
print("=" * 70)
W_BASE_PATH = "data/neurons/_shared_lm_head_base.pt"
import os
if os.path.exists(W_BASE_PATH):
    wbase = torch.load(W_BASE_PATH, map_location="cpu", weights_only=False)
    if isinstance(wbase, dict):
        for k, v in wbase.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  Type: {type(wbase)}")
        if hasattr(wbase, "shape"):
            print(f"  shape: {tuple(wbase.shape)}")
else:
    print(f"  not found: {W_BASE_PATH}")

print()
print("=" * 70)
print("[4] 推导：W_base.weight = lm_head.weight @ proj.weight.T")
print("=" * 70)
# 找 lm_head 的实际 tensor
lm_head_weight = None
for k in state:
    if "lm_head" in k.lower() and "weight" in k.lower():
        v = state[k]
        if hasattr(v, "shape") and v.dim() == 2 and v.shape[0] >= 100000:
            lm_head_weight = v
            print(f"  candidate lm_head.weight: key={k}, shape={tuple(v.shape)}")
            break
if lm_head_weight is None:
    for k in state:
        v = state[k]
        if hasattr(v, "shape") and v.dim() == 2 and v.shape[0] == 256000:
            print(f"  candidate 256000-row tensor: key={k}, shape={tuple(v.shape)}")
            lm_head_weight = v
            break

proj_weight = None
if isinstance(proj_state, dict):
    for k, v in proj_state.items():
        if hasattr(v, "shape") and v.dim() == 2 and v.shape == (384, 2048):
            proj_weight = v
            print(f"  proj.weight: key={k}, shape={tuple(v.shape)}")
            break

if lm_head_weight is not None and proj_weight is not None:
    # W_base.weight shape 应该是 [256000, 384]
    # 1.5B lm_head.weight shape [256000, 2048]
    # proj.weight shape [384, 2048]（nn.Linear weight, forward = x @ weight.T = x @ [2048, 384]）
    # h_384 = h_2048 @ proj.weight.T = h_2048 @ [2048, 384]
    # 反投影 h_2048 ≈ h_384 @ proj.weight (因为 SVD 正交) = h_384 @ [384, 2048]
    # W_base(h_384) = h_384 @ W_base.weight.T
    # 让 W_base(h_384) ≈ lm_head(h_2048) = h_2048 @ lm_head.weight.T
    # = (h_384 @ proj.weight) @ lm_head.weight.T = h_384 @ (proj.weight @ lm_head.weight.T)
    # 所以 W_base.weight.T = proj.weight @ lm_head.weight.T
    # 即 W_base.weight = lm_head.weight @ proj.weight.T = [256000, 2048] @ [2048, 384] = [256000, 384] ✓
    new_w_base_weight = lm_head_weight @ proj_weight.T
    print(f"  ✓ Computed W_base.weight: shape={tuple(new_w_base_weight.shape)}")
    print(f"  ✓ Target shape: [256000, 384]")
    print(f"  ✓ Match: {new_w_base_weight.shape == (256000, 384)}")
    print()
    print(f"  Sample stats:")
    print(f"    mean={new_w_base_weight.mean().item():.6f}")
    print(f"    std={new_w_base_weight.std().item():.6f}")
    print(f"    min={new_w_base_weight.min().item():.4f}")
    print(f"    max={new_w_base_weight.max().item():.4f}")
else:
    print(f"  ✗ cannot compute (lm_head_weight={lm_head_weight}, proj_weight={proj_weight})")
