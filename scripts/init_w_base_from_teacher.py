"""W_base 教师投影版初始化（替代随机 W_base）。

数学推导：
    1.5B 教师 hidden h_2048，通过 distill_hidden_proj 投影到 h_H（H=hidden_size）：
        h_H = h_2048 @ proj.weight.T  (proj.weight shape [H, 2048])
    由于 proj.weight 是 SVD 正交投影，反投影近似：
        h_2048 ≈ h_H @ proj.weight = h_H @ [H, 2048]
    
    1.5B lm_head(h_2048) = h_2048 @ lm_head.weight.T  ([256000, 2048].T = [2048, 256000])
    
    代入反投影：
        lm_head(h_2048) ≈ (h_H @ proj.weight) @ lm_head.weight.T
                       = h_H @ (proj.weight @ lm_head.weight.T)
    
    neuron W_base(h_H) = h_H @ W_base.weight.T
    要让两者等价：
        W_base.weight.T = proj.weight @ lm_head.weight.T
        W_base.weight   = lm_head.weight @ proj.weight.T
                       = [256000, 2048] @ [2048, H]
                       = [256000, H] ✓

这意味着新 W_base 在 h_H 空间近似还原了 1.5B 教师的 lm_head 语言能力。
neuron 的 delta_u/delta_v 只需学习"域偏置"，无需重建语言先验。

C5 修复：hidden_size 改为 CLI 参数（--hidden-size），默认走 DEFAULT_NEURON_SPEC，
       避免硬编码 384 与切换 COMPACT 后的 512 不一致。
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

CKPT_PATH = "checkpoint-481000/model.pt"
W_BASE_PATH = "data/neurons/_shared_lm_head_base.pt"
W_BASE_BACKUP_PATH = "data/neurons/_shared_lm_head_base_random_v2.pt"


def parse_args():
    p = argparse.ArgumentParser(description="从 1.5B 教师 lm_head 投影生成 W_base")
    p.add_argument(
        "--hidden-size", type=int, default=None,
        help="目标 neuron hidden_size。None=从 DEFAULT_NEURON_SPEC 推断（推荐）",
    )
    p.add_argument(
        "--spec", type=str, default=None,
        help="目标 spec (foundation/compact/standard/expert)。优先级高于 --hidden-size",
    )
    return p.parse_args()


def resolve_hidden_size(args):
    """从参数或 DEFAULT_NEURON_SPEC 推断 hidden_size。"""
    if args.spec:
        from taiji.resonance import get_default_neuron_config
        cfg = get_default_neuron_config(args.spec)
        return cfg.hidden_size
    if args.hidden_size is not None:
        return args.hidden_size
    # 默认走 DEFAULT_NEURON_SPEC
    from taiji.resonance import get_default_neuron_config
    cfg = get_default_neuron_config()
    return cfg.hidden_size


def main():
    args = parse_args()
    hidden_size = resolve_hidden_size(args)
    vocab_size = 256000  # 全局 tokenizer contract

    print("=" * 70)
    print(f"W_base 教师投影版初始化 (hidden_size={hidden_size})")
    print("=" * 70)

    # ── 1. 备份现有 W_base（若形状不匹配或仍是随机版）──
    if os.path.exists(W_BASE_PATH):
        existing = torch.load(W_BASE_PATH, map_location="cpu", weights_only=False)
        existing_w = existing.get("weight", existing) if isinstance(existing, dict) else existing
        if hasattr(existing_w, "shape") and existing_w.shape == (vocab_size, hidden_size):
            std = existing_w.std().item()
            if std > 0.5:  # 随机 std=0.02，教师投影 std 通常 > 0.5
                print(f"[1] 备份现有 W_base (std={std:.4f}, 看起来是随机版)")
                torch.save(existing, W_BASE_BACKUP_PATH)
            else:
                print(f"[1] 现有 W_base std={std:.4f}, 看起来已是教师投影版，跳过备份")
        else:
            print(f"[1] 现有 W_base shape={getattr(existing_w, 'shape', None)}, 备份")
            torch.save(existing, W_BASE_BACKUP_PATH)

    # ── 2. 加载 1.5B lm_head + distill_hidden_proj_{hidden_size} ──
    proj_path = f"data/distill/distill_hidden_proj_{hidden_size}.pt"
    print()
    print(f"[2] 加载 1.5B 教师权重 + SVD 投影 ({proj_path})...")
    if not os.path.exists(proj_path):
        print(f"  ✗ 投影文件不存在: {proj_path}")
        print(f"    请先运行: python scripts/training/build_shared_projections.py")
        sys.exit(1)
    
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    
    lm_head_weight = state["lm_head.weight"]  # [256000, 2048] bfloat16
    print(f"  lm_head.weight: shape={tuple(lm_head_weight.shape)} dtype={lm_head_weight.dtype}")
    lm_head_weight = lm_head_weight.float()  # 转 float32
    
    proj_state = torch.load(proj_path, map_location="cpu", weights_only=False)
    proj_weight = proj_state["weight"] if isinstance(proj_state, dict) else proj_state
    # proj_weight shape [hidden_size, 2048] (nn.Linear weight for nn.Linear(2048, hidden_size, bias=False))
    print(f"  proj.weight:    shape={tuple(proj_weight.shape)} dtype={proj_weight.dtype}")
    
    if proj_weight.shape != (hidden_size, 2048):
        print(f"  ✗ proj.weight shape 错误: 期望 {(hidden_size, 2048)}, 实际 {tuple(proj_weight.shape)}")
        sys.exit(1)

    # ── 3. 计算新 W_base = lm_head.weight @ proj.weight.T ──
    print()
    print("[3] 计算 W_base = lm_head.weight @ proj.weight.T ...")
    new_w = lm_head_weight @ proj_weight.T  # [256000, 2048] @ [2048, hidden_size] = [256000, hidden_size]
    print(f"  新 W_base.weight: shape={tuple(new_w.shape)} dtype={new_w.dtype}")
    print(f"  stats: mean={new_w.mean().item():.6f}, std={new_w.std().item():.6f}")
    print(f"         min={new_w.min().item():.4f}, max={new_w.max().item():.4f}")
    print(f"         norm(F)={torch.linalg.norm(new_w).item():.2f}")

    # ── 4. 保存 ──
    print()
    print(f"[4] 保存到 {W_BASE_PATH} (frozen, requires_grad=False)")
    new_w_base = nn.Linear(hidden_size, vocab_size, bias=False)
    with torch.no_grad():
        new_w_base.weight.copy_(new_w)
    for p in new_w_base.parameters():
        p.requires_grad = False
    new_w_base.eval()
    torch.save(new_w_base.state_dict(), W_BASE_PATH)
    print(f"  ✓ Saved (frozen, no gradients)")

    # ── 5. 快速合理性检查：W_base 反映教师 token 分布倾向 ──
    print()
    print("[5] 合理性检查：W_base 对常见 token 的偏置...")
    fake_h = torch.zeros(1, hidden_size)
    fake_h[0, 0] = 1.0
    logits = new_w_base(fake_h)  # [1, 256000]
    top5 = torch.topk(logits[0], k=5)
    print(f"  fake h=e_0 → top5 token ids: {top5.indices.tolist()}, logits: {[f'{v:.2f}' for v in top5.values.tolist()]}")
    
    zero_logits = new_w_base(torch.zeros(1, hidden_size))
    print(f"  zero input → logits norm: {torch.linalg.norm(zero_logits).item():.4f} (应接近 0)")

    print()
    print("=" * 70)
    print(f"✓ W_base 教师投影版已就绪 (hidden_size={hidden_size})")
    print("=" * 70)
    print()
    print("下一步：从 neurons_backup_sft_v2/ 恢复蒸馏后版本，然后用新 W_base 重训 SFT v2")
    print("  恢复: copy data/neurons_backup_sft_v2/neuron_*.pt.bak → data/neurons/neuron_*.pt")
    print("  重训: python scripts/training/sft_train_neurons_v2.py --domain zh --steps 1000 --save")


if __name__ == "__main__":
    main()
