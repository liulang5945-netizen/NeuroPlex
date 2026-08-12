#!/usr/bin/env python3
"""诊断 quality_logits 路由错误模式（collab_v3_c15，2026-08-08）。

背景：C15（D 方案：预测质量路由）用 quality_head 替代 domain_score_head 域判别——
quality 监督 = per-neuron NLL 排序（谁能预测好当前文本谁上）。
本脚本逐条文本（seed 42，与 verify_collab_mixed 同批）对比：
  - quality_logits 软路由权重（argmax 是谁？正确吗？）
  - soft 融合 PPL vs 已知域门控 PPL vs 全 9 neuron solo 最优 PPL
逐条分类：
  - 路由错误：门控 ≈ solo 最优、soft 显著差 → quality 没选对人（可训练修复）
  - 融合有损：门控也显著差于 solo → 融合/转译本身有损（zh 案例，训练修不了）
  - 正常：soft ≈ 门控 ≈ solo

Usage:
    python scripts/training/diag_route_errors.py --ckpt data/neurons/collab_v3_c13.ckpt.pt
"""
import os
import sys
import math
import random

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
SEQ_LEN = 64
ROUTER_TEMP = 0.15


def encode_all(text, embeddings, general_sp):
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    neuron_embeddings, targets, mask = {}, None, None
    for nid, emb in embeddings.items():
        out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=SEQ_LEN)
        neuron_embeddings[nid] = out[0]
        if targets is None:
            targets, mask = out[1], out[2]
    return neuron_embeddings, targets, mask


def ppl_from_logits(logits, targets, mask):
    sl, st, sm = logits[:, :-1].contiguous(), targets[:, 1:].clone().contiguous(), mask[:, 1:].contiguous()
    st[~sm] = -100
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                          ignore_index=-100, reduction="sum")
    n = int(sm.sum().item())
    if n == 0:
        return None
    return min(loss.item() / max(n, 1), 30.0)  # cap 与 verify 口径一致


def solo_loss(neuron, emb, general_sp, text):
    """单 neuron + home embedding，旧 5 logits 转译到 general 256K 口径。"""
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    out = batch_align_and_embed([text], general_sp, general_sp, emb, max_seq_len=SEQ_LEN)
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
    return ppl_from_logits(logits, out[1], out[2])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/neurons/collab_v3_c13.ckpt.pt")
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--gate-thresh", type=float, default=1.10,
                    help="门控 PPL / solo 最优 超过该比例 → 判定融合有损")
    ap.add_argument("--soft-thresh", type=float, default=1.10,
                    help="soft PPL / 门控 PPL 超过该比例 → 判定路由错误")
    args = ap.parse_args()

    random.seed(42)
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

    general_sp = load_general_tokenizer()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt: epoch={ck['epoch']}, total_steps={ck['total_steps']}")

    # ── 加载 9 neuron + home embedding ──
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

    # ── 重建拓扑 + 注入协作层产物 ──
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    hub = TokenizerHub()
    for dom in DOMAINS:
        hub.register_domain(dom, load_domain_tokenizer(dom))
    hub.register_domain("zh", load_domain_tokenizer("zh"))
    hub.register_domain("general", general_sp)
    ens.set_tokenizer_hub(hub)
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

    nids = list(neurons.keys())
    print(f"[加载完成] {len(nids)} neuron: {nids}\n")

    # ── 数据 ──
    texts = {d: load_domain_texts(d, 3000) for d in DOMAINS}
    texts["dialogue"] = load_dialogue_texts_multi(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "simple_zh"),
        max_texts=2000, max_answer_chars=150)
    print(f"[数据加载完成] dialogue={len(texts['dialogue'])} 条\n")

    # ── 逐文本诊断 ──
    print("=" * 110)
    print("逐文本路由诊断（gate/solo > %.2f → 融合有损；soft/gate > %.2f → 路由错误）"
          % (args.gate_thresh, args.soft_thresh))
    print("=" * 110)

    stats = {s: {"route_err": 0, "fuse_loss": 0, "ok": 0} for s in DOMAINS + ["dialogue"]}

    for src in DOMAINS + ["dialogue"]:
        print(f"\n── {src} 域（{args.n_eval} 条，seed 42 同批）──")
        for t in random.sample(texts[src], args.n_eval):
            neuron_embeddings, targets, mask = encode_all(t, embeddings, general_sp)

            # 1) soft 融合 + quality_logits
            with torch.no_grad():
                r_soft = ens.forward_train(
                    neuron_embeddings=neuron_embeddings, n_rounds=2,
                    fusion_mode="soft", targets=targets,
                    field_conditioning=True, target_domain="general",
                )
            dl = r_soft.get("quality_logits")  # [N] active_ids 顺序
            dl_rank = sorted(nids, key=lambda k: -dl[nids.index(k)].item())
            dl_argmax = dl_rank[0]
            soft_ppl = ppl_from_logits(r_soft["fused_logits"], targets, mask)

            # 2) 已知域门控
            trust = torch.ones(len(nids))
            if src in DOMAINS:
                trust[nids.index(src)] = 100.0
            else:  # dialogue → 5 个 dialogue neuron 全部 100（对话文本属集体）
                for nid in DIALOGUE_IDS:
                    trust[nids.index(nid)] = 100.0
            with torch.no_grad():
                r_gate = ens.forward_train(
                    neuron_embeddings=neuron_embeddings, n_rounds=2,
                    fusion_mode="soft", targets=targets,
                    field_conditioning=True, target_domain="general",
                    trust_override=trust,
                )
            gate_ppl = ppl_from_logits(r_gate["fused_logits"], targets, mask)

            # 3) 全 9 neuron solo 最优
            best_solo, best_nid = float("inf"), None
            for nid in nids:
                l = solo_loss(neurons[nid], embeddings[nid], general_sp, t)
                if l is not None and l < best_solo:
                    best_solo, best_nid = l, nid

            # 4) 分类
            if gate_ppl is None or best_solo == float("inf"):
                tag = "N/A"
            elif gate_ppl > best_solo * args.gate_thresh:
                tag = "融合有损"
                stats[src]["fuse_loss"] += 1
            elif soft_ppl is not None and soft_ppl > gate_ppl * args.soft_thresh:
                tag = "路由错误"
                stats[src]["route_err"] += 1
            else:
                tag = "正常"
                stats[src]["ok"] += 1

            dl_str = ", ".join(f"{k}={dl[nids.index(k)]:.2f}" for k in dl_rank[:4])
            fmt = lambda x: f"{math.exp(x):7.1f}" if x is not None else "    N/A"
            print(f"  argmax={dl_argmax:16s} soft={fmt(soft_ppl)} gate={fmt(gate_ppl)} "
                  f"solo={fmt(best_solo)}({best_nid}) [{tag}] | {dl_str}")
            print(f"    样例: {t[:70].replace(chr(10), ' ')}")

    # ── 汇总 ──
    print("\n" + "=" * 110)
    print("汇总（残留负 EMERGE 的分类）")
    print("=" * 110)
    print(f"  {'数据':10s} {'正常':>5s} {'路由错误':>8s} {'融合有损':>8s}")
    for src in DOMAINS + ["dialogue"]:
        s = stats[src]
        total = s["ok"] + s["route_err"] + s["fuse_loss"]
        print(f"  {src:10s} {s['ok']:5d} {s['route_err']:8d} {s['fuse_loss']:8d}"
              f"  (共 {total})")
    print("\n  路由错误 → 判别器没选对人（可训练修复：增强 domain head / 加数据）")
    print("  融合有损 → 门控也救不了（训练修不了：查融合/转译路径）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
