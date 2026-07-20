"""P0-1b 深度诊断:核查 checkpoint-481000 的实际结构和加载结果。

回答三个问题:
1. model.pt 到底在哪里? 实际 key 格式是什么?
2. 加载后哪些 key 真正命中, 哪些被静默跳过?
3. 加载后 forward 一条真实数据, 看看 logits 分布是否合理 (vs 随机初始化)?
"""
from __future__ import annotations

import os
import sys
import json
import math
from collections import Counter

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def inspect_checkpoint_dir(ckpt_dir: str):
    """Step 1: 列出 checkpoint 目录的真实内容。"""
    print(f"\n[1] Checkpoint dir: {ckpt_dir}")
    if not os.path.isdir(ckpt_dir):
        print(f"  ❌ 目录不存在!")
        return None
    files = sorted(os.listdir(ckpt_dir))
    print(f"  文件列表 ({len(files)} 个):")
    for f in files:
        full = os.path.join(ckpt_dir, f)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        size_str = f"{size/1e9:.2f}GB" if size > 1e9 else f"{size/1e6:.2f}MB" if size > 1e6 else f"{size/1e3:.1f}KB"
        print(f"    {f}  ({size_str})")
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(model_pt):
        print(f"  ❌ model.pt 不存在!")
        return None
    print(f"  ✓ model.pt 存在")
    return model_pt


def inspect_state_dict_keys(model_pt: str):
    """Step 2: 列出 state_dict 的实际 key 格式。"""
    print(f"\n[2] state_dict key 格式分析")
    sd = torch.load(model_pt, map_location="cpu", weights_only=True)
    print(f"  total keys: {len(sd)}")

    # 按前缀分组
    prefixes = Counter()
    for k in sd.keys():
        # 取前两段
        parts = k.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        prefixes[prefix] += 1
    print(f"  按前两段前缀分组:")
    for p, c in prefixes.most_common():
        print(f"    {p}: {c}")

    # 是否有顶层 model. 前缀
    has_model_prefix = any(k.startswith("model.") for k in sd)
    has_backbone_prefix = any(k.startswith("backbone.") for k in sd)
    has_flat_embed = "embed.weight" in sd
    has_flat_layers = any(k.startswith("layers.") for k in sd)
    print(f"\n  格式判断:")
    print(f"    model.* 前缀: {has_model_prefix}")
    print(f"    backbone.* 前缀: {has_backbone_prefix}")
    print(f"    扁平 embed.weight: {has_flat_embed}")
    print(f"    扁平 layers.*: {has_flat_layers}")

    # 打印前 20 个 key
    print(f"\n  前 20 个 key:")
    for k in list(sd.keys())[:20]:
        print(f"    {k}: shape={tuple(sd[k].shape)}")

    # 重点检查关键 key 是否存在
    print(f"\n  关键 key 检查:")
    critical_keys = [
        "embed.weight",
        "backbone.embedding.weight",
        "model.backbone.embedding.weight",
        "norm.weight",
        "backbone.norm.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "layers.0.attn.wq.weight",
        "backbone.layers.0.attention.wq.weight",
        "model.backbone.layers.0.attention.wq.weight",
        "layers.0.wg.weight",
        "layers.0.w_gate.weight",
        "backbone.layers.0.feed_forward.w_gate.weight",
    ]
    for k in critical_keys:
        exists = k in sd
        if exists:
            print(f"    ✓ {k}: shape={tuple(sd[k].shape)}, std={sd[k].std().item():.6f}")
        else:
            print(f"    ✗ {k}")

    return sd


def inspect_config(ckpt_dir: str):
    """Step 3: 检查 config.json。"""
    print(f"\n[3] config.json")
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"  ❌ config.json 不存在")
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg


def test_load_and_forward(ckpt_dir: str, sd: dict, cfg: dict):
    """Step 4: 用 gen1 architecture 加载并前向,看 PPL。"""
    print(f"\n[4] 加载并前向测试")
    from taiji.training.checkpoint_bridge import load_teacher_model, _remap_legacy_keys

    # 先看 _remap_legacy_keys 的输出
    remapped = _remap_legacy_keys(sd)
    print(f"  remap 后 key 数: {len(remapped)}")
    print(f"  remap 后前 5 key: {list(remapped.keys())[:5]}")

    # 加载 teacher
    print(f"\n  加载 teacher ...")
    teacher, emb = load_teacher_model(ckpt_dir, device="cpu")
    print(f"  teacher loaded, param count: {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B")

    # 权重统计
    layer0 = teacher.backbone.layers[0]
    emb_w = teacher.backbone.embedding.weight.data
    lm_w = teacher.lm_head.weight.data
    print(f"\n  加载后权重统计:")
    print(f"    embedding.weight: std={emb_w.std().item():.6f}, mean={emb_w.mean().item():.6f}")
    print(f"    lm_head.weight:   std={lm_w.std().item():.6f}, mean={lm_w.mean().item():.6f}")
    print(f"    weight tying 共享内存: {emb_w.data_ptr() == lm_w.data_ptr()}")
    for pname, p in layer0.named_parameters():
        print(f"    layer[0].{pname}: std={p.std().item():.6f}, mean={p.mean().item():.6f}")

    # 用真实数据测 PPL
    print(f"\n  前向测试 (尝试 data/real/*.pt):")
    candidates = [
        "data/real/zh.pt",
        "data/distill/zh.pt",
        "data/real_zh.pt",
        "data/distill/domain_datasets.pt",
    ]
    data_path = None
    for c in candidates:
        if os.path.exists(c):
            data_path = c
            break

    if data_path is None:
        print(f"    无可用数据,使用随机 ids 测试 forward 是否崩溃")
        sample = torch.randint(0, cfg.get("vocab_size", 256000), (2, 32))
        print(f"    sample shape: {sample.shape} (random ids)")
    else:
        print(f"    使用 {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            # domain_datasets.pt 是 dict
            for k, v in data.items():
                print(f"      {k}: {v.shape}")
            sample = list(data.values())[0][:2, :64]
        else:
            sample = data[:2, :64] if data.dim() == 2 else data[:2]
        print(f"    sample shape: {sample.shape}")
        print(f"    sample[0, :10] ids: {sample[0, :10].tolist()}")

    with torch.no_grad():
        out = teacher(sample)
        logits = out.logits if hasattr(out, "logits") else out
    print(f"    logits shape: {logits.shape}")
    print(f"    logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, max={logits.max().item():.4f}, min={logits.min().item():.4f}")

    # PPL on this sample
    shift = logits[:, :-1, :].contiguous()
    targets = sample[:, 1:].contiguous()
    loss = F.cross_entropy(shift.view(-1, shift.size(-1)), targets.view(-1))
    ppl = math.exp(min(loss.item(), 20.0))
    print(f"    loss: {loss.item():.4f}, PPL: {ppl:.2f}, log(vocab)={math.log(cfg.get('vocab_size', 256000)):.4f}")

    # Top-1 accuracy (随机应该 ~1/vocab)
    preds = shift.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    print(f"    top-1 accuracy: {correct*100:.4f}% (random baseline: {100/cfg.get('vocab_size', 256000):.6f}%)")

    # 看预测分布
    print(f"\n    第一个 token 预测 top-5:")
    probs = F.softmax(logits[0, 0], dim=-1)
    topv, topi = probs.topk(5)
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"      id={i}, prob={v:.6f}")


def main():
    print("=" * 70)
    print("P0-1b 深度诊断:checkpoint 真实结构 + 加载结果")
    print("=" * 70)

    ckpt_dir = "e:/taiji-neuron/checkpoint-481000"
    model_pt = inspect_checkpoint_dir(ckpt_dir)
    if model_pt is None:
        # 尝试其他位置
        print("\n  尝试 e:/taiji/checkpoint-400000 ...")
        alt_dir = "e:/taiji/checkpoint-400000"
        if os.path.isdir(alt_dir):
            ckpt_dir = alt_dir
            model_pt = inspect_checkpoint_dir(ckpt_dir)
        if model_pt is None:
            print("  ❌ 找不到 model.pt, 退出")
            return

    cfg = inspect_config(ckpt_dir)
    sd = inspect_state_dict_keys(model_pt)
    if sd is None:
        return
    if cfg is None:
        return
    test_load_and_forward(ckpt_dir, sd, cfg)


if __name__ == "__main__":
    main()
