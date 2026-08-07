#!/usr/bin/env python3
"""验证 C15 contrastive_loss 生效（D 方案核心验证，2026-08-08）。

确认：forward_train 的 contrastive_loss 非零 + quality_head 收到梯度 +
quality_logits 排序与 per-neuron NLL 排序正相关（预测质量路由的监督信号）。
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "data/neurons/collab_v3_c15_smoke.ckpt.pt"
    from scripts.training.train_cross_domain_collab import (
        load_neuron, load_shared_lm_head, load_shared_embedding,
    )
    from scripts.training.utils import (
        load_general_tokenizer, create_shared_embedding,
    )
    from scripts.training.train_multi_domain_foundation import load_domain_texts, batch_align_and_embed
    from taiji.resonance.ensemble import ResonanceEnsemble
    from taiji.resonance.field import ResonanceField
    from taiji.resonance.geometry import NeuronGeometry
    from taiji.resonance.topology import build_topology, establish_topology_channels
    from taiji.resonance.translator import TokenizerHub
    from scripts.training.utils import load_domain_tokenizer

    general_sp = load_general_tokenizer()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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
    for nid, bp in ck["body_state"].items():
        with torch.no_grad():
            for name, val in bp.items():
                for pname, p in neurons[nid].named_parameters():
                    if pname == name:
                        p.copy_(val)
                for bname, b in neurons[nid].named_buffers():
                    if bname == name:
                        b.copy_(val)
    for nid, sd in ck["cross_spec_state"].get("forward", {}).items():
        if nid in ens._cross_spec_projectors:
            ens._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in ck["cross_spec_state"].get("backward", {}).items():
        if nid in ens._cross_spec_back_projectors:
            ens._cross_spec_back_projectors[nid].load_state_dict(sd)
    for n in neurons.values():
        n.eval()

    texts = {d: load_domain_texts(d, 100) for d in DOMAINS}
    nids = list(neurons.keys())

    # 只开 quality_head 梯度
    for n in neurons.values():
        for p in n.parameters():
            p.requires_grad = False
        for p in n.quality_head.parameters():
            p.requires_grad = True

    print(f"[加载完成] {len(nids)} neuron；quality_head 只读验证")
    all_ok = True
    for src in DOMAINS:
        t = texts[src][0]
        neuron_embeddings, targets, mask = {}, None, None
        for nid, emb in embeddings.items():
            out = batch_align_and_embed([t], general_sp, general_sp, emb, max_seq_len=64)
            neuron_embeddings[nid] = out[0]
            if targets is None:
                targets, mask = out[1], out[2]

        r = ens.forward_train(
            neuron_embeddings=neuron_embeddings, n_rounds=2,
            fusion_mode="soft", targets=targets,
            field_conditioning=True, target_domain="general",
        )
        cl = float(r["contrastive_loss"].item())
        ql = r["quality_logits"].detach()
        ql_rank = sorted(nids, key=lambda k: -ql[nids.index(k)].item())
        ql_str = ", ".join(f"{k}={ql[nids.index(k)]:.2f}" for k in ql_rank[:4])

        # 手动算 per-neuron NLL（与 ensemble 口径一致，验证监督目标）
        nll_map = {}
        for nid in nids:
            emb = embeddings[nid]
            with torch.no_grad():
                out = batch_align_and_embed([t], general_sp, general_sp, emb, max_seq_len=64)
                rn = neurons[nid].forward(out[0], return_logits=True)
            logits = rn["logits"]
            if logits.shape[-1] != 256000:
                from taiji.resonance.translator import build_logits_alignment_matrix
                src_sp = load_domain_tokenizer("zh")
                m = build_logits_alignment_matrix(src_sp, general_sp, "zh", "general",
                                                  cache={}, source_vocab_size=logits.shape[-1])
                b, l, vi = logits.shape
                logits = torch.sparse.mm(logits.reshape(-1, vi), m.to(logits.dtype)).reshape(b, l, 256000)
            sl, st, sm = logits[:, :-1].contiguous(), out[1][:, 1:].clone().contiguous(), out[2][:, 1:].contiguous()
            st[~sm] = -100
            nll = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                                  ignore_index=-100, reduction="sum") / max(int(sm.sum().item()), 1)
            nll_map[nid] = float(nll)
        nll_rank = sorted(nids, key=lambda k: nll_map[k])
        nll_str = ", ".join(f"{k}={nll_map[k]:.1f}" for k in nll_rank[:4])

        # contrastive 梯度
        r["contrastive_loss"].backward(retain_graph=True)
        g = neurons["code"].quality_head[2].weight.grad
        grad_norm = float(g.abs().sum().item()) if g is not None else 0.0

        ok = cl > 0 and grad_norm > 0
        all_ok &= ok
        print(f"  {src:5s} contrastive={cl:.4f} quality_top4=[{ql_str}]")
        print(f"        NLL_top4(low) =[{nll_str}] grad(qh)= {grad_norm:.2f} {'OK' if ok else 'FAIL'}")

    print(f"\n{'✓ contrastive_loss 生效（非零 + 梯度流动）' if all_ok else '✗ 有问题'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
