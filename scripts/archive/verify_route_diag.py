#!/usr/bin/env python3
"""路由权重诊断：混合阵容下 _confidence_routing_fusion 的 per-position 占比。

目的：验证"code/math/en 负 EMERGE = max-prob 跨空间不可比（旧 5 的 zh 50K 原生空间
小 → softmax max-prob 天然更高 → 抢非自身域位置）"假设。

对每域 2 条文本，打印各 neuron 的路由权重（position 占比）+ 各 neuron 的原生/投影 max-prob。
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

CKPT_PATH = "data/neurons/collab_v1_mixed.ckpt.pt"
GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]


def main():
    from scripts.training.train_cross_domain_collab import (
        load_neuron, load_shared_lm_head, load_shared_embedding,
    )
    from scripts.training.utils import (
        load_general_tokenizer, create_shared_embedding, load_domain_tokenizer,
    )
    from scripts.training.train_multi_domain_foundation import (
        load_domain_texts, batch_align_and_embed,
    )
    from taiji.resonance.ensemble import ResonanceEnsemble
    from taiji.resonance.field import ResonanceField
    from taiji.resonance.geometry import NeuronGeometry
    from taiji.resonance.topology import build_topology, establish_topology_channels
    from taiji.resonance.translator import TokenizerHub

    print("=" * 64)
    print("混合阵容路由权重诊断")
    print("=" * 64)
    general_sp = load_general_tokenizer()
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # 加载（与评估脚本一致）
    shared_lm_head = load_shared_lm_head(GENERAL_DIR, 512, "cpu")
    neurons, embeddings = {}, {}
    for nid in DOMAINS:
        neurons[nid] = load_neuron(nid, GENERAL_DIR, "cpu", shared_lm_head=shared_lm_head)
        embeddings[nid] = load_shared_embedding(GENERAL_DIR, "cpu")
    for nid in DIALOGUE_IDS:
        ckp = torch.load(os.path.join(DIALOGUE_DIR, f"neuron_{nid}.pt"),
                         map_location="cpu", weights_only=False)
        cfg = ckp["neuron_config"]; cfg.unified_field_dim = None
        from taiji.resonance.neuron import ResonanceNeuron
        n = ResonanceNeuron(cfg)
        n.load_state_dict(ckp["state_dict"], strict=False)
        neurons[nid] = n
        emb = create_shared_embedding("cpu")
        ses = ckp.get("shared_embedding_state", {})
        emb.weight.data.copy_(ses["weight"] if isinstance(ses, dict) else ses)
        embeddings[nid] = emb

    # 重建 + 注入训练产物
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    field = ResonanceField(dim=max(n.config.field_dim for n in neurons.values()))
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

    texts = {d: load_domain_texts(d, 3000) for d in DOMAINS}

    # 逐域诊断
    import torch.nn.functional as F
    for d in DOMAINS:
        sample = texts[d][0]
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
        w = dict(zip(ens.neurons.keys(), r["weights"]))
        print(f"=== {d} 文本: {sample[:50]}")
        for nid in sorted(w, key=lambda k: -w[k]):
            tag = " 旧5" if "dialogue" in nid else " 新4"
            print(f"  {nid:18s}: {w[nid]:.3f}{tag}")

    print(f"\n{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
