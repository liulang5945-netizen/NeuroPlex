#!/usr/bin/env python3
"""hub 锚定 loss 验证（缺口 L 阶段 3 第三部分·渐进第一步，2026-08-14）。

验证 train_cross_domain_collab.py 的 --hub-anchor-weight（决策 4B→4C 渐进）：
- A1. compute_hub_anchor_loss 计算正确（= 手动 1-cos(proj(v_d), v_hub)，值合理）
- A2. 梯度只流 cross_spec_projectors（域 neuron / hub body 冻结 → grad 全 None）
- A3. 训练后投影域 field 与 hub cosine 提升（hub 成为跨域锚点，锚定生效）
- A4. 零破坏：域 neuron 与 hub body 参数训练前后完全不变
- A5. checkpoint 回读：投影层学习状态持久化（与训练后一致）

运行：python -u scripts/training/verify_hub_anchor.py
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from scripts.training.utils import load_general_tokenizer  # noqa: E402
import scripts.training.train_cross_domain_collab as tcdc  # noqa: E402
from scripts.training.verify_hub_collab_train import (  # noqa: E402
    build_ensemble_with_hub, make_batch,
)

SFT_DIR = "data/sft"

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def cos_alignment(ensemble, neurons, neuron_embeddings, nid: str) -> float:
    """投影后域 field 与 hub field 的 cosine（锚定目标量）。"""
    v_d = neurons[nid].forward(neuron_embeddings[nid], round_num=1)["field_vector"]
    v_h = neurons["hub"].forward(neuron_embeddings["hub"], round_num=1)["field_vector"]
    if nid in ensemble._cross_spec_projectors:
        v_d = ensemble._cross_spec_projectors[nid](v_d)
    if "hub" in ensemble._cross_spec_projectors:
        v_h = ensemble._cross_spec_projectors["hub"](v_h)
    return float(F.cosine_similarity(v_d, v_h, dim=-1).mean().item())


def body_params(neurons):
    """可比较的 body 状态快照（排除可训协作层参数）。"""
    snap = {}
    for nid, n in neurons.items():
        for name, p in n.named_parameters():
            if (name.startswith(("excite_", "inhibit_")) or "scale_" in name
                    or "lora_adapters" in name):
                continue
            snap[f"{nid}.{name}"] = p.data.clone()
    return snap


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("hub 锚定 loss 验证（阶段 3 第三部分·锚定）", flush=True)
    print("=" * 60, flush=True)

    # ── 装配（复用 hub_collab_train 同款阵容，统一空间=装配口径 3072，
    #     hub 4096 也经投影参与锚定——与真实协作层训练/装配一致）──
    neurons, shared_embeddings, ensemble, general_sp, _ = build_ensemble_with_hub(
        field_dim=3072)
    hub = neurons["hub"]

    # 收集可训练参数（域 neuron/hub body 冻结，协作层可训——collab 脚本 LoRA 模式同款）
    proj_ids = [pid for pid in ["code", "hub"] if pid in ensemble._cross_spec_projectors]
    trainable = [p for pid in proj_ids
                 for p in ensemble._cross_spec_projectors[pid].parameters()
                 if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    data = torch.load(os.path.join(SFT_DIR, "code_sft.pt"),
                      map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data][:16]
    neuron_embeddings, targets, mask = make_batch(
        texts, general_sp, shared_embeddings, seq_len=32)

    # ── A1. loss 计算正确性 ──
    print("\n[A1] anchor loss 计算...", flush=True)
    loss = tcdc.compute_hub_anchor_loss(ensemble, neurons, neuron_embeddings, "code")
    v_d = neurons["code"].forward(neuron_embeddings["code"], round_num=1)["field_vector"]
    v_h = hub.forward(neuron_embeddings["hub"], round_num=1)["field_vector"]
    v_dp = ensemble._cross_spec_projectors["code"](v_d)
    # 统一空间非 hub 原生维度（3072 口径）时 hub 也经投影参与锚定——与
    # compute_hub_anchor_loss 内部一致（2026-08-15：此前手动对照漏 hub 投影
    # → 3072 口径下 3072 vs 4096 维度不匹配必崩）。
    if "hub" in ensemble._cross_spec_projectors:
        v_h = ensemble._cross_spec_projectors["hub"](v_h)
    manual = 1.0 - F.cosine_similarity(v_dp, v_h, dim=-1).mean()
    check("A1. anchor loss 与手动计算一致",
          abs(float(loss.item()) - float(manual.item())) < 1e-5,
          f"loss={loss.item():.4f} manual={manual.item():.4f}")
    check("A1b. anchor loss 值合理（cos∈[-1,1] → loss∈[0,2]）",
          0.0 <= float(loss.item()) <= 2.0,
          f"cos={float(v_dp.shape[0] and 1 - loss.item()):.4f}")

    # ── A2 梯度隔离 + A3 锚定生效 + A4 零破坏（同一训练循环，单条 backward 链）──
    print("\n[A2/A3/A4] 训练 6 步锚定 + 梯度隔离 + 零破坏...", flush=True)
    cos_before = cos_alignment(ensemble, neurons, neuron_embeddings, "code")
    body_before = body_params(neurons)
    grads = {}
    for step in range(6):
        opt.zero_grad()
        loss = tcdc.compute_hub_anchor_loss(ensemble, neurons, neuron_embeddings, "code")
        loss.backward()
        if step == 5:  # 最后一步 backward 后（step 前）检查梯度隔离
            proj_grad = ensemble._cross_spec_projectors["code"].linear1.weight.grad
            body_grads_any = any(
                p.grad is not None
                for n in neurons.values()
                for name, p in n.named_parameters()
                if not name.startswith(("excite_", "inhibit_")) and "scale_" not in name
                and "lora_adapters" not in name)
            hub_side_grad = any(
                p.grad is not None for ch in hub.excite_channels.values()
                for p in ch.parameters())
            grads = {"proj": proj_grad, "body": body_grads_any, "hub_side": hub_side_grad}
        opt.step()
    check("A2. 梯度只流 cross_spec_projectors[code]",
          grads["proj"] is not None and float(grads["proj"].abs().sum()) > 0
          and not grads["body"] and not grads["hub_side"],
          f"proj_grad={grads['proj'].abs().sum().item() if grads['proj'] is not None else 'None'}")
    cos_after = cos_alignment(ensemble, neurons, neuron_embeddings, "code")
    check("A3. 锚定生效（投影域 field 对齐 hub cosine 提升）",
          cos_after > cos_before + 1e-4,
          f"cos {cos_before:.4f} → {cos_after:.4f} (Δ={cos_after - cos_before:+.4f})")
    body_after = body_params(neurons)
    body_same = all(
        torch.equal(body_before[k], body_after[k]) for k in body_before)
    check("A4. 零破坏（域 neuron 与 hub body 参数不变）", body_same,
          f"{len(body_before)} 组参数全等")

    # ── A5. checkpoint 回读（投影层学习持久化）──
    print("\n[A5] checkpoint 回读...", flush=True)
    tmp = tempfile.mkdtemp(prefix="hub_anchor_ckpt_")
    ckpt_path = os.path.join(tmp, "hub_anchor_smoke.ckpt.pt")
    tcdc.save_checkpoint(ckpt_path, 1, 6, neurons, ensemble,
                         None, None, None, [{"loss": float(loss.item())}])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cs = ck["cross_spec_state"]
    code_proj_state = cs["forward"]["code"]["linear1.weight"]
    live_state = ensemble._cross_spec_projectors["code"].linear1.weight.data.clone()
    check("A5. checkpoint 回读（投影层学习状态持久化）",
          torch.equal(code_proj_state, live_state),
          f"投影层 weight 与训练后一致")
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
