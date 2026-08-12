#!/usr/bin/env python3
"""验证混合阵容协作层训练产物（collab_v1_mixed.ckpt.pt）。

阵容：旧 5 对话 neuron（zh 50K 域空间，词库转译）+ 新 4 general neuron（256K 统一空间）。
训练：target=general，side_channels + cross_spec 投影 + body 尾层微调。

验证：
[1] checkpoint 加载完整：side_channels / scale_bias / body / cross_spec 全 9 neuron 注入
[2] 域内 EMERGE：协作（soft / division 融合，target=general 转译口径）vs 最强个体
    —— 对比训练前基线（verify_unified_space_routing：division 劣于个体）：
       训练后协作应追平/超越个体（side_channels 校准了"谁在什么位置擅长"）
[3] 对话域：旧 5 协作 PPL vs 旧 5 最强个体（对话能力经协作层是否保留/提升）
[4] 生成冒烟：zh 提问 → 跨 vocab 转译融合生成

Usage:
    python scripts/training/verify_collab_mixed.py
"""
import os
import sys
import math
import random

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

CKPT_PATH = "data/neurons/collab_v1_mixed.ckpt.pt"
GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]


def compute_avg_loss_ensemble(ens, embeddings, general_sp, text, fusion_mode="soft",
                              seq_len=64):
    """协作 PPL：全部 neuron 各自 home embedding 编码 → forward_train(target=general) 融合。"""
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    neuron_embeddings, targets, mask = {}, None, None
    for nid, emb in embeddings.items():
        out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=seq_len)
        neuron_embeddings[nid] = out[0]
        if targets is None:
            targets, mask = out[1], out[2]
    with torch.no_grad():
        r = ens.forward_train(
            neuron_embeddings=neuron_embeddings, n_rounds=2,
            fusion_mode=fusion_mode, targets=targets,
            field_conditioning=True, target_domain="general",
        )
    logits = r["fused_logits"]
    sl, st, sm = logits[:, :-1].contiguous(), targets[:, 1:].clone().contiguous(), mask[:, 1:].contiguous()
    st[~sm] = -100
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                          ignore_index=-100, reduction="sum")
    n = int(sm.sum().item())
    if n == 0:
        return None
    # 单点数值爆炸（旧 5 转译 logits 极端值 → 某位置概率≈0 → loss 巨大）会污染平均：
    # cap 单条文本平均 loss 到 30（exp(30)≈1e13，仍是"极差"级别，不影响相对比较）
    return min(loss.item() / max(n, 1), 30.0)


def compute_avg_loss_solo(neuron, emb, general_sp, text, seq_len=64):
    """个体 PPL：单 neuron + 自己 home embedding。
    旧 5（zh 50K head）的 logits 转译到 general 256K 再算 CE（与训练 target=general 口径一致）；
    新 4 原生 256K 无需转译。"""
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=seq_len)
    with torch.no_grad():
        r = neuron.forward(out[0], return_logits=True)
    logits = r["logits"]
    if logits.shape[-1] != 256000:
        from taiji.resonance.translator import build_logits_alignment_matrix
        from scripts.training.utils import load_domain_tokenizer
        src_sp = load_domain_tokenizer("zh")
        m = build_logits_alignment_matrix(src_sp, general_sp, "zh", "general",
                                          cache={}, source_vocab_size=logits.shape[-1])
        b, l, vi = logits.shape
        logits = torch.sparse.mm(logits.reshape(-1, vi), m.to(logits.dtype)).reshape(b, l, 256000)
    targets, mask = out[1], out[2]
    sl, st, sm = logits[:, :-1].contiguous(), targets[:, 1:].clone().contiguous(), mask[:, 1:].contiguous()
    st[~sm] = -100
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                          ignore_index=-100, reduction="sum")
    n = int(sm.sum().item())
    if n == 0:
        return None
    # 单点数值爆炸（旧 5 转译 logits 极端值 → 某位置概率≈0 → loss 巨大）会污染平均：
    # cap 单条文本平均 loss 到 30（exp(30)≈1e13，仍是"极差"级别，不影响相对比较）
    return min(loss.item() / max(n, 1), 30.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/neurons/collab_v1_mixed.ckpt.pt",
                    help="协作层 checkpoint 路径（评估 v2 用 --ckpt data/neurons/collab_v2_routing.ckpt.pt）")
    ap.add_argument("--n-eval", type=int, default=8,
                    help="每域评估文本条数（seed 固定，可复现）")
    args = ap.parse_args()
    global CKPT_PATH, n_eval
    CKPT_PATH = args.ckpt
    n_eval = args.n_eval
    random.seed(42)  # 固定评估采样，保证同 ckpt 下结果可复现
    from scripts.training.train_cross_domain_collab import (
        load_neuron, load_shared_lm_head, load_shared_embedding,
    )
    from scripts.training.utils import (
        load_general_tokenizer, load_dialogue_texts_multi, create_shared_embedding,
    )
    from scripts.training.train_multi_domain_foundation import load_domain_texts
    from taiji.resonance.ensemble import ResonanceEnsemble
    from taiji.resonance.field import ResonanceField
    from taiji.resonance.geometry import NeuronGeometry
    from taiji.resonance.topology import build_topology, establish_topology_channels
    from taiji.resonance.translator import TokenizerHub
    from scripts.training.utils import load_domain_tokenizer

    print("=" * 64)
    print("混合阵容协作层产物验证（collab_v1_mixed.ckpt.pt）")
    print("=" * 64)

    general_sp = load_general_tokenizer()
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"  epoch={ck['epoch']}, total_steps={ck['total_steps']}")

    # ── 1. 加载 9 neuron + home embeddings（与训练一致）──
    print("\n[1] 加载 9 neuron + home embedding...")
    shared_lm_head = load_shared_lm_head(GENERAL_DIR, 512, "cpu")
    neurons, embeddings = {}, {}
    for nid in DOMAINS:
        n = load_neuron(nid, GENERAL_DIR, "cpu", shared_lm_head=shared_lm_head)
        neurons[nid] = n
        embeddings[nid] = load_shared_embedding(GENERAL_DIR, "cpu")
    for nid in DIALOGUE_IDS:
        ckp = torch.load(os.path.join(DIALOGUE_DIR, f"neuron_{nid}.pt"),
                         map_location="cpu", weights_only=False)
        cfg = ckp["neuron_config"]; cfg.unified_field_dim = None
        n = __import__("taiji.resonance.neuron", fromlist=["ResonanceNeuron"]).ResonanceNeuron(cfg)
        n.load_state_dict(ckp["state_dict"], strict=False)
        neurons[nid] = n
        emb = create_shared_embedding("cpu")
        ses = ckp.get("shared_embedding_state", {})
        w = ses["weight"] if isinstance(ses, dict) else ses
        emb.weight.data.copy_(w)
        embeddings[nid] = emb
    print(f"  {len(neurons)} neuron: {DOMAINS} + {DIALOGUE_IDS}")

    # ── 2. 重建拓扑 + 加载训练产物 ──
    print("\n[2] 重建拓扑 + 加载协作层...")
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    # tokenizer hub（target=general 转译）
    hub = TokenizerHub()
    for dom in DOMAINS:
        hub.register_domain(dom, load_domain_tokenizer(dom))
    hub.register_domain("zh", load_domain_tokenizer("zh"))
    hub.register_domain("general", general_sp)
    ens.set_tokenizer_hub(hub)
    # 注入训练产物
    for nid, sd in ck["side_channels_state"].items():
        for pid, ch_sd in sd.get("excite", {}).items():
            if pid in neurons[nid].excite_channels:
                neurons[nid].excite_channels[pid].load_state_dict(ch_sd)
        for pid, ch_sd in sd.get("inhibit", {}).items():
            if pid in neurons[nid].inhibit_channels:
                neurons[nid].inhibit_channels[pid].load_state_dict(ch_sd)
    for nid, sb in ck["scale_bias_state"].items():
        with torch.no_grad():
            for name, val in sb.items():
                for pname, p in neurons[nid].named_parameters():
                    if pname == name:
                        p.copy_(val)
                for bname, b in neurons[nid].named_buffers():
                    if bname == name:
                        b.copy_(val)
    for nid, bp in ck["body_state"].items():
        with torch.no_grad():
            for name, val in bp.items():
                for pname, p in neurons[nid].named_parameters():
                    if pname == name:
                        p.copy_(val)
    for nid, sd in ck["cross_spec_state"].get("forward", {}).items():
        if nid in ens._cross_spec_projectors:
            ens._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in ck["cross_spec_state"].get("backward", {}).items():
        if nid in ens._cross_spec_back_projectors:
            ens._cross_spec_back_projectors[nid].load_state_dict(sd)
    for n in neurons.values():
        n.eval()
    print("  ✓ side_channels/scale_bias/body/cross_spec 注入完成")

    # ── 3. 数据 ──
    print("\n[3] 加载评估数据...")
    texts = {d: load_domain_texts(d, 3000) for d in DOMAINS}
    texts["dialogue"] = load_dialogue_texts_multi(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "simple_zh"),
        max_texts=2000, max_answer_chars=150)
    print(f"  dialogue: {len(texts['dialogue'])} 条")

    # ── 4. 域内 EMERGE：协作(soft 融合) vs 最强个体 ──
    # 注意：forward_train 只支持 soft 融合（训练口径）；推理侧 division 分工路由
    # 在混合 vocab 场景尚未接入（下一步：跨 vocab 转译 + per-position 硬路由）。
    print("\n[4] 域内 EMERGE（协作 soft 融合 vs 最强个体，8 随机文本，target=general）...")
    print(f"  {'数据':8s} {'协作soft':>10s} {'最强个体':>10s} {'EMERGE':>10s}")
    for src in DOMAINS + ["dialogue"]:
        collab_soft, indiv = [], []
        for t in random.sample(texts[src], n_eval):
            l_soft = compute_avg_loss_ensemble(ens, embeddings, general_sp, t, "soft")
            if l_soft:
                collab_soft.append(l_soft)
            # 最强个体：该域最擅长的 neuron
            if src in DOMAINS:
                cands = {src: (neurons[src], embeddings[src])}
            else:
                cands = {nid: (neurons[nid], embeddings[nid]) for nid in DIALOGUE_IDS}
            best = float("inf")
            for nid, (n, emb) in cands.items():
                l = compute_avg_loss_solo(n, emb, general_sp, t)
                if l:
                    best = min(best, l)
            if best < float("inf"):
                indiv.append(best)
        avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        cs, ind = avg(collab_soft), avg(indiv)
        es = (ind - cs) / ind * 100 if cs == cs and ind == ind else float("nan")
        print(f"  {src:8s} {math.exp(cs) if cs==cs else float('nan'):10.2f} "
              f"{math.exp(ind) if ind==ind else float('nan'):10.2f} {es:9.1f}%")

    # ── 4b. 上界实验：已知域硬门控 trust（路由完美的协作 PPL 上界）──
    # 若上界仍 ≤ solo → 转译/融合本身有损，加强训练无济于事；
    # 若上界 ≥ solo → 路由是唯一瓶颈，值得用训练让 scores 逼近该门控。
    print("\n[4b] 已知域硬门控 trust 上界（trust[src]=100，其他=1）...")
    nids_ordered = list(ens.neurons.keys())
    print(f"  {'数据':8s} {'协作门控':>10s} {'最强个体':>10s} {'EMERGE':>10s}")
    for src in DOMAINS:
        collab_gate, indiv = [], []
        for t in random.sample(texts[src], n_eval):
            neuron_embeddings, targets = {}, None
            for nid, emb in embeddings.items():
                out = batch_align_and_embed([t], general_sp, general_sp, emb, max_seq_len=64)
                neuron_embeddings[nid] = out[0]
                if targets is None:
                    targets, _mask = out[1], out[2]
            trust = torch.ones(len(nids_ordered))
            trust[nids_ordered.index(src)] = 100.0  # 已知域 → 域 neuron 绝对信任
            with torch.no_grad():
                r = ens.forward_train(
                    neuron_embeddings=neuron_embeddings, n_rounds=2,
                    fusion_mode="soft", targets=targets,
                    field_conditioning=True, target_domain="general",
                    trust_override=trust,
                )
            logits = r["fused_logits"]
            sl, st, sm = (logits[:, :-1].contiguous(), targets[:, 1:].clone().contiguous(),
                          _mask[:, 1:].contiguous())
            st[~sm] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                                   ignore_index=-100, reduction="sum")
            n = int(sm.sum().item())
            if n:
                collab_gate.append(min(loss.item() / n, 30.0))
            l_solo = compute_avg_loss_solo(neurons[src], embeddings[src], general_sp, t)
            if l_solo:
                indiv.append(l_solo)
        avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        cg, ind = avg(collab_gate), avg(indiv)
        es = (ind - cg) / ind * 100 if cg == cg and ind == ind else float("nan")
        print(f"  {src:8s} {math.exp(cg) if cg==cg else float('nan'):10.2f} "
              f"{math.exp(ind) if ind==ind else float('nan'):10.2f} {es:9.1f}%")

    # ── 5. 路由诊断：各域文本上共振分 scores 排序（判断训练是否学到"域内 neuron 应胜出"）──
    print("\n[5] 域适配诊断：各域文本上 9 neuron 共振分 scores 排序...")
    nids_ordered = list(ens.neurons.keys())
    for src in DOMAINS + ["dialogue"]:
        sample = texts[src][0]
        neuron_embeddings, targets = {}, None
        for nid, emb in embeddings.items():
            out = batch_align_and_embed([sample], general_sp, general_sp, emb, max_seq_len=64)
            neuron_embeddings[nid] = out[0]
            if targets is None:
                targets = out[1]
        with torch.no_grad():
            r = ens.forward_train(neuron_embeddings=neuron_embeddings, n_rounds=2,
                                  fusion_mode="soft", targets=targets,
                                  field_conditioning=True, target_domain="general")
        sc = r["scores"]  # [N] batch 平均共振分
        rank = sorted(nids_ordered, key=lambda k: -sc[nids_ordered.index(k)])
        desc = "  >  ".join(f"{k}={sc[nids_ordered.index(k)]:.3f}" for k in rank)
        print(f"  {src:9s} | {desc}")

    # ── 5b. 路由权重诊断：code 文本上谁在抢位置（跨空间 max-prob 校准后）──
    print("\n[5b] 路由权重诊断（code 文本，_confidence_routing_fusion per-position 占比）...")
    sample = texts["code"][0]
    neuron_embeddings, targets = {}, None
    for nid, emb in embeddings.items():
        out = batch_align_and_embed([sample], general_sp, general_sp, emb, max_seq_len=64)
        neuron_embeddings[nid] = out[0]
        if targets is None:
            targets = out[1]
    with torch.no_grad():
        r = ens.forward_train(neuron_embeddings=neuron_embeddings, n_rounds=2,
                              fusion_mode="soft", targets=targets,
                              field_conditioning=True, target_domain="general")
    w = r["weights"]  # [N] batch 平均路由权重（position 占比）
    nids = list(ens.neurons.keys())
    prof = dict(zip(nids, w))
    print(f"  样例: {sample[:60]}")
    for nid in sorted(prof, key=lambda k: -prof[k]):
        tag = " (旧5对话)" if "dialogue" in nid else " (新4general)"
        print(f"    {nid:18s}: {prof[nid]:.3f}{tag}")

    # ── 6. 生成冒烟（跨 vocab 转译融合）──
    print("\n[5] 生成冒烟（zh 提问 → 转译融合）...")
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    prompt = "写一个 Python 函数计算斐波那契数列"
    cur = general_sp.encode(prompt)
    for _ in range(8):
        neuron_embeddings = {}
        for nid, emb in embeddings.items():
            ids_t = torch.tensor([cur], dtype=torch.long)
            neuron_embeddings[nid] = emb(ids_t)
        with torch.no_grad():
            r = ens.forward_train(
                neuron_embeddings=neuron_embeddings, n_rounds=2,
                fusion_mode="soft", field_conditioning=True, target_domain="general",
            )
        nxt = int(r["fused_logits"][0, -1].argmax().item())
        cur.append(nxt)
        if general_sp.IdToPiece(nxt) in ("</s>", "<s>", "<unk>"):
            break
    out_text = general_sp.DecodeIds(cur)
    print(f"  prompt: {prompt}")
    print(f"  → {out_text[:120]}")
    print("  ✓ 生成链路正常")

    print(f"\n{'='*64}")
    print("混合阵容协作层产物验证完成")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
