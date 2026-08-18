#!/usr/bin/env python3
"""跨域对比 loss 验证（缺口 L 阶段 3 第三部分·渐进第二步，2026-08-14）。

验证 train_cross_domain_collab.py 的 --hub-contrastive-weight（决策 4C 最终形态）：
- A1. InfoNCE 计算正确（与手动双向 CE 一致，值有限）
- A2. 梯度只流 zh/code 两侧 cross_spec_projectors（域 neuron/hub body 全 None）
- A3. 训练后同义跨域对 sim 提升（hub 空间成为跨域共享语义空间）
- A4. 零破坏：zh/code/hub body 参数训练前后完全不变
- A5. checkpoint 回读：zh/code 投影层学习状态持久化

运行：python -u scripts/training/verify_hub_contrastive.py
"""
from __future__ import annotations

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

from neuroplex.resonance.translator import batch_align_and_embed  # noqa: E402
import scripts.training.train_cross_domain_collab as tcdc  # noqa: E402
from scripts.training.verify_hub_collab_train import build_ensemble_with_hub  # noqa: E402

ZH_ID = "zh"
CODE_ID = "code"
TAU = 0.1
N_PAIRS = 8

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


def diag_sim(ensemble, neurons, shared_embeddings, general_sp, zh_texts, code_texts) -> float:
    """同义对 diagonal cosine（A3 目标量，no_grad）。"""
    with torch.no_grad():
        zh_emb = batch_align_and_embed(
            zh_texts, general_sp, general_sp, shared_embeddings[ZH_ID], max_seq_len=64)[0]
        code_emb = batch_align_and_embed(
            code_texts, general_sp, general_sp, shared_embeddings[CODE_ID], max_seq_len=64)[0]
        v_zh = neurons[ZH_ID].forward(zh_emb, round_num=1)["field_vector"]
        v_code = neurons[CODE_ID].forward(code_emb, round_num=1)["field_vector"]
        v_zh = ensemble._cross_spec_projectors[ZH_ID](v_zh)
        v_code = ensemble._cross_spec_projectors[CODE_ID](v_code)
        return float(F.cosine_similarity(v_zh, v_code, dim=-1).mean().item())


def body_params(neurons):
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
    print("跨域对比 loss 验证（阶段 3 第三部分·渐进第二步）", flush=True)
    print("=" * 60, flush=True)

    # ── 装配（zh + code + hub，统一空间=装配口径 3072，对比 loss 数据源 zh↔code）──
    neurons, shared_embeddings, ensemble, general_sp, _ = build_ensemble_with_hub(
        nids=[ZH_ID, CODE_ID], field_dim=3072)
    hub = neurons["hub"]
    pairs = tcdc.load_pairs_texts()
    sample = random.sample(pairs, min(N_PAIRS, len(pairs)))
    zh_texts = [p[0] for p in sample]
    code_texts = [p[1] for p in sample]

    # ── A1. InfoNCE 计算正确性 ──
    print("\n[A1] InfoNCE 计算...", flush=True)
    loss = tcdc.compute_hub_contrastive_loss(
        ensemble, neurons, shared_embeddings, general_sp,
        ZH_ID, CODE_ID, zh_texts, code_texts, tau=TAU, max_seq_len=64)
    zh_emb = batch_align_and_embed(
        zh_texts, general_sp, general_sp, shared_embeddings[ZH_ID], max_seq_len=64)[0]
    code_emb = batch_align_and_embed(
        code_texts, general_sp, general_sp, shared_embeddings[CODE_ID], max_seq_len=64)[0]
    v_zh = neurons[ZH_ID].forward(zh_emb.detach(), round_num=1)["field_vector"]
    v_code = neurons[CODE_ID].forward(code_emb.detach(), round_num=1)["field_vector"]
    v_zh = ensemble._cross_spec_projectors[ZH_ID](v_zh)
    v_code = ensemble._cross_spec_projectors[CODE_ID](v_code)
    sim = F.cosine_similarity(v_zh.unsqueeze(1), v_code.unsqueeze(0), dim=-1) / TAU
    labels = torch.arange(sim.size(0))
    manual = 0.5 * F.cross_entropy(sim, labels) + 0.5 * F.cross_entropy(sim.t(), labels)
    check("A1. InfoNCE 与手动双向 CE 一致",
          abs(float(loss.item()) - float(manual.item())) < 1e-5,
          f"loss={loss.item():.4f} manual={manual.item():.4f}")
    check("A1b. 同义对初始 sim（随机投影层，合理波动）",
          -1.0 <= diag_sim(ensemble, neurons, shared_embeddings, general_sp,
                           zh_texts, code_texts) <= 1.0)

    # ── A2-A4. 训练 8 步（梯度隔离 + 同义 sim 提升 + 零破坏）──
    print("\n[A2/A3/A4] 训练 8 步对比 + 梯度隔离 + 零破坏...", flush=True)
    trainable = ([p for p in ensemble._cross_spec_projectors[ZH_ID].parameters()
                  if p.requires_grad]
                 + [p for p in ensemble._cross_spec_projectors[CODE_ID].parameters()
                    if p.requires_grad])
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    sim_before = diag_sim(ensemble, neurons, shared_embeddings, general_sp,
                          zh_texts, code_texts)
    body_before = body_params(neurons)
    grads = {}
    for step in range(8):
        opt.zero_grad()
        loss = tcdc.compute_hub_contrastive_loss(
            ensemble, neurons, shared_embeddings, general_sp,
            ZH_ID, CODE_ID, zh_texts, code_texts, tau=TAU, max_seq_len=64)
        loss.backward()
        if step == 7:
            proj_grads = {
                pid: ensemble._cross_spec_projectors[pid].linear1.weight.grad
                for pid in [ZH_ID, CODE_ID]}
            body_grads_any = any(
                p.grad is not None
                for n in neurons.values()
                for name, p in n.named_parameters()
                if not name.startswith(("excite_", "inhibit_")) and "scale_" not in name
                and "lora_adapters" not in name)
            grads = {"proj": proj_grads, "body": body_grads_any}
        opt.step()
    check("A2. 梯度只流 zh/code 两侧 cross_spec_projectors",
          all(g is not None and float(g.abs().sum()) > 0 for g in grads["proj"].values())
          and not grads["body"],
          f"zh_grad={grads['proj'][ZH_ID].abs().sum().item():.1f} "
          f"code_grad={grads['proj'][CODE_ID].abs().sum().item():.1f} body=None")
    sim_after = diag_sim(ensemble, neurons, shared_embeddings, general_sp,
                         zh_texts, code_texts)
    check("A3. 同义跨域对 sim 提升（hub 空间跨域对齐生效）",
          sim_after > sim_before + 1e-4,
          f"sim {sim_before:.4f} → {sim_after:.4f} (Δ={sim_after - sim_before:+.4f})")
    body_after = body_params(neurons)
    body_same = all(torch.equal(body_before[k], body_after[k]) for k in body_before)
    check("A4. 零破坏（zh/code/hub body 参数不变）", body_same,
          f"{len(body_before)} 组参数全等")

    # ── A5. checkpoint 回读 ──
    print("\n[A5] checkpoint 回读...", flush=True)
    tmp = tempfile.mkdtemp(prefix="hub_contrast_ckpt_")
    ckpt_path = os.path.join(tmp, "hub_contrast_smoke.ckpt.pt")
    tcdc.save_checkpoint(ckpt_path, 1, 8, neurons, ensemble,
                         None, None, None, [{"loss": float(loss.item())}])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cs = ck["cross_spec_state"]
    ok = all(
        "zh" in cs["forward"] and "code" in cs["forward"]
        and torch.equal(
            cs["forward"][pid]["linear1.weight"],
            ensemble._cross_spec_projectors[pid].linear1.weight.data.clone())
        for pid in [ZH_ID, CODE_ID])
    check("A5. checkpoint 回读（zh/code 投影层学习状态持久化）", ok)
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
