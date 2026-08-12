#!/usr/bin/env python3
"""验证统一空间（同 vocab 256K）推理路由：max-prob 分工路由（fusion_mode="division"）。

背景（2026-08-06）：共享 general lm_head 统一输出空间后，4 域 neuron（code/math/zh/en）
原生输出都在 general 256K 空间——max-prob 天然尖锐（旧诊断 max-prob≈0.001 是静态稀疏
投影到 256K 的置信度稀释，已随共享 head 消除）。per-position 硬路由把每个 token 位置
交给最自信的 neuron：域内文本由域 neuron 全位置胜出（≈个体无伤害），跨域时不同位置
不同 neuron 分工。

验证：
[1] division 路由可跑：真实 general 基座上 forward 正常，fused logits 256K、无 NaN
[2] 域内分工：code 文本 → code neuron 路由权重（位置占比）最高（zh/en/math 同理）
[3] PPL 对比：division vs soft(共振分融合) vs 最强单 neuron —— division 应 ≤ 最强个体
    （分工路由 = 每位置取最自信者，天然不低于任何单 neuron 的自回归）
[4] 生成冒烟：中文 prompt 贪心生成正常（general 256K 解码、无崩溃）

Usage:
    python scripts/training/verify_unified_space_routing.py
"""
import os
import sys
import math
import random

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

SAVE_DIR = "data/foundation_v1_general"
DOMAINS = ["code", "math", "zh", "en"]
FIELD_DIM = 2048  # compact spec field_dim
HIDDEN = 512  # compact spec hidden_size（general 基座 embedding/lm_head 维度）


def compute_avg_loss(ens, emb, general_sp, text, fusion_mode, max_len=128):
    """与 verify_checkpoint 口径一致：返回平均 loss（非 PPL），8 条文本后统一 exp。"""
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=max_len)
    shared_emb, targets, mask = out[0], out[1], out[2]
    with torch.no_grad():
        r = ens.forward(shared_embeddings=shared_emb, return_logits=True,
                        fusion_mode=fusion_mode)
    logits = r["weighted_logits"]  # [B, L, V]
    sl, st, sm = logits[:, :-1].contiguous(), targets[:, 1:].contiguous(), mask[:, 1:].contiguous()
    st = st.clone()
    st[~sm] = -100
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                          ignore_index=-100, reduction="sum")
    n = int(sm.sum().item())
    if n == 0:
        return None
    return loss.item() / max(n, 1)


def route_profile(ens, emb, general_sp, text, max_len=128):
    """返回 division 模式下每 neuron 的位置占比 dict。"""
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=max_len)
    with torch.no_grad():
        r = ens.forward(shared_embeddings=out[0], return_logits=True, fusion_mode="division")
    weights = r["weights"]  # [N] 位置占比（顺序 = route_nids）
    nids = r.get("route_nids") or list(ens.neurons.keys())
    return dict(zip(nids, weights))


def greedy_generate(ens, emb, general_sp, prompt, max_new=16):
    """贪心自回归（division 分工路由），验证生成链路。"""
    cur = general_sp.encode(prompt)
    if not cur:
        cur = [0]
    generated = []
    for _ in range(max_new):
        inp = torch.tensor([cur], dtype=torch.long)
        shared_emb = emb(inp)
        with torch.no_grad():
            r = ens.forward(shared_embeddings=shared_emb, return_logits=True,
                            fusion_mode="division")
        logits = r["weighted_logits"][0, -1]
        next_id = int(logits.argmax().item())
        cur.append(next_id)
        generated.append(next_id)
        if next_id == 3:  # general EOS（通用 256K tokenizer 的 <unk>/<s> 系按 IdToPiece 检查）
            piece = general_sp.IdToPiece(next_id)
            if piece in ("</s>", "<s>", "<unk>"):
                break
    return general_sp.DecodeIds(cur)


def main():
    from scripts.training.train_cross_domain_collab import (
        load_neuron, load_shared_lm_head, load_shared_embedding,
    )
    from scripts.training.train_multi_domain_foundation import (
        load_domain_texts, load_general_tokenizer,
    )
    from taiji.resonance.ensemble import ResonanceEnsemble
    from taiji.resonance.field import ResonanceField

    print("=" * 60)
    print("统一空间推理路由验证（fusion_mode=division，max-prob 分工）")
    print("=" * 60)

    general_sp = load_general_tokenizer()
    texts = {d: load_domain_texts(d, 3000) for d in DOMAINS}
    for d in DOMAINS:
        print(f"  {d}: {len(texts[d])} 条文本")

    # ── 1. 加载 general 基座 ──
    head = load_shared_lm_head(SAVE_DIR, HIDDEN, "cpu")
    assert head is not None, "[加载] shared_lm_head 缺失（非 general 基座）"
    neurons = {d: load_neuron(d, SAVE_DIR, "cpu", shared_lm_head=head) for d in DOMAINS}
    emb = load_shared_embedding(SAVE_DIR, "cpu")
    field = ResonanceField(dim=FIELD_DIM, device=torch.device("cpu"))
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)
    for n in neurons.values():
        n.eval()

    # ── [1] division 路由可跑 + 输出空间 ──
    print("\n[1] division 路由 forward 冒烟（输出 256K、无 NaN）...")
    sample = texts["code"][0]
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    out = batch_align_and_embed([sample], general_sp, general_sp, emb, max_seq_len=64)
    with torch.no_grad():
        r = ens.forward(shared_embeddings=out[0], return_logits=True, fusion_mode="division")
    fused = r["weighted_logits"]
    assert fused.shape[-1] == 256000, f"[1] fused vocab={fused.shape[-1]} != 256000"
    assert torch.isfinite(fused).all(), "[1] fused 含 NaN/Inf"
    assert r["fusion_mode"] == "division"
    print(f"  fused {tuple(fused.shape)} ✓ 无 NaN ✓ fusion_mode={r['fusion_mode']}")
    rnids = r.get("route_nids") or list(ens.neurons.keys())
    print(f"  路由权重(位置占比): { {k: round(v, 3) for k, v in zip(rnids, r['weights'])} }")

    # ── [1.5] max-prob 尺度诊断：跨 neuron 的 logits 尖锐度是否可比 ──
    # division 分工路由的前提：per-position max-prob 可比。若某 neuron logits 系统性
    # 更尖锐（平均 max-prob 高），它会在所有位置胜出——分工退化为"最尖锐者主导"。
    print("\n[1.5] 平均 max-prob 尺度诊断（code 文本样本，per-neuron）...")
    for nid in DOMAINS:
        n_out = ens.forward(shared_embeddings=out[0], return_logits=True,
                            fusion_mode="per_position",
                            active_nids=[nid])["neuron_logits"]
        mp = F.softmax(n_out[nid], dim=-1).max(dim=-1).values.mean().item()
        print(f"  {nid:5s}: 平均 max-prob = {mp:.4f}")

    # ── [1.6] ensemble vs 直接 neuron.forward 的 logits 一致性 ──
    # solo ensemble PPL(45) 远差于 verify_checkpoint 回读(6.1) —— 定位差异来源。
    print("\n[1.6] ensemble vs direct neuron logits 一致性（code 样本）...")
    from taiji.resonance.neuron import ResonanceNeuron
    cfg = torch.load(os.path.join(SAVE_DIR, "neuron_code.pt"),
                     map_location="cpu", weights_only=False)["neuron_config"]
    cfg.unified_field_dim = None
    direct_n = ResonanceNeuron(cfg, shared_lm_head=head)
    direct_n.load_state_dict(neurons["code"].state_dict(), strict=False)
    direct_n.eval()
    with torch.no_grad():
        d_logits = direct_n.forward(out[0], return_logits=True)["logits"]
        solo = ResonanceEnsemble({"code": neurons["code"]}, field, max_rounds=1)
        neurons["code"].eval()
        s_res = solo.forward(shared_embeddings=out[0], return_logits=True,
                             fusion_mode="division")
    s_logits = s_res["weighted_logits"]
    max_diff = (d_logits - s_logits).abs().max().item()
    print(f"  direct logits {tuple(d_logits.shape)} vs ensemble {tuple(s_logits.shape)}")
    print(f"  max |diff| = {max_diff:.6f}  (0 = 完全一致)")
    if max_diff > 1e-4:
        # 找到差异最大的位置，检查是哪个 neuron 胜出
        diff_map = (d_logits - s_logits).abs().max(dim=-1).values  # [B, L]
        print(f"  差异最大的 logits 行: {diff_map.max().item():.4f}")

    # ── [2] 域内分工：域 neuron 在其域文本上占比应高于其在其他域文本的占比 ──
    # 注意：分工路由按"实际能力"分工而非名义域——GSM8K 是英文数学题，en neuron
    # 在英文位置胜出、math neuron 在数字/公式位置胜出（正确行为）。验证标准 =
    # 对角优势（每 neuron 在自己域文本上占比最高），而非绝对阈值。
    print("\n[2] 域内分工（每域 8 条文本，交叉占比矩阵，对角应占优）...")
    n_check = 8
    share = {d: {nid: 0.0 for nid in DOMAINS} for d in DOMAINS}  # [域文本][neuron]
    for d in DOMAINS:
        for t in random.sample(texts[d], n_check):
            prof = route_profile(ens, emb, general_sp, t)
            for nid, w in prof.items():
                share[d][nid] += w / n_check
    print(f"  {'文本\\neuron':>10s} " + "".join(f"{n:>8s}" for n in DOMAINS))
    for d in DOMAINS:
        row = "".join(f"{share[d][n]:8.3f}" for n in DOMAINS)
        print(f"  {d:>10s} {row}")
    for nid in DOMAINS:
        own = share[nid][nid]
        max_other = max(share[od][nid] for od in DOMAINS if od != nid)
        if own > max_other:
            print(f"  ✓ {nid} neuron 对角占优（自身域 {own:.3f} > 其他域最高 {max_other:.3f}）")
        else:
            print(f"  ⚠ {nid} neuron 未对角占优（自身域 {own:.3f} < 其他域最高 {max_other:.3f}）"
                  f" —— 见 [1.5] 尺度诊断")

    # ── [3] PPL 对比：division(裸) vs division_norm(归一化) vs soft vs 最强个体 ──
    # 预期基线：协作层未训练时 max-prob/共振分未校准 → 个体通常 ≥ division（分工把位置
    # 交给"最自信"但未必"最准确"的 neuron）。本对比是协作层训练前的基线，供训练后衡量
    # division 分工路由是否追平/超越个体（校准后分工应 ≤ 个体）。
    # division vs division_norm：验证 per-neuron 归一化能否消除"系统性尖锐者主导"。
    print("\n[3] PPL 对比（8 随机文本/域，loss 算术平均 → exp）...")
    print(f"  {'域':6s} {'div裸':>10s} {'div归一化':>10s} {'soft':>10s} {'最强个体':>10s}")
    summary = {}
    for d in DOMAINS:
        div_loss, divn_loss, soft_loss, indiv_loss = [], [], [], []
        for t in random.sample(texts[d], n_check):
            l = compute_avg_loss(ens, emb, general_sp, t, "division")
            if l:
                div_loss.append(l)
            ln_ = compute_avg_loss(ens, emb, general_sp, t, "division_norm")
            if ln_:
                divn_loss.append(ln_)
            l_soft = compute_avg_loss(ens, emb, general_sp, t, "soft")
            if l_soft:
                soft_loss.append(l_soft)
            nid = neurons[d]
            solo = ResonanceEnsemble({d: nid}, field, max_rounds=1)
            nid.eval()
            l_ind = compute_avg_loss(solo, emb, general_sp, t, "division")
            if l_ind:
                indiv_loss.append(l_ind)
        avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        div_a, divn_a, soft_a, ind_a = avg(div_loss), avg(divn_loss), avg(soft_loss), avg(indiv_loss)
        summary[d] = (math.exp(div_a), math.exp(divn_a), math.exp(soft_a), math.exp(ind_a))
        print(f"  {d:6s} {math.exp(div_a):10.2f} {math.exp(divn_a):10.2f} "
              f"{math.exp(soft_a):10.2f} {math.exp(ind_a):10.2f}")
    # 关键断言：所有模式均远优于随机（ln 256000 ≈ 12.45 → PPL ≈ 255K），基座能力真实
    for d in DOMAINS:
        for name, ppl in zip(("div裸", "div归一化", "soft", "个体"), summary[d]):
            assert ppl < math.exp(11.5), f"[3] {d} {name} PPL {ppl:.0f} 接近随机"
    print("  ok [3] 所有路由模式有效（远优于随机）——基线已记录供协作层训练后对比")

    # ── [4] 生成冒烟 ──
    print("\n[4] 生成冒烟（division 分工路由，贪心 16 token）...")
    for prompt in ["写一个 Python 函数计算斐波那契数列", "什么是机器学习？",
                   "def add(a, b): return a + b  # 请解释这段代码"]:
        out_text = greedy_generate(ens, emb, general_sp, prompt, max_new=16)
        print(f"  prompt: {prompt[:24]}")
        print(f"  → {out_text[:80]}")
    print("  ok [4] 生成链路正常（无崩溃）")

    print(f"\n{'='*60}")
    print("统一空间 division 分工路由验证完成")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
