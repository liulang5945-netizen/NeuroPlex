#!/usr/bin/env python3
"""诊断 routing_loss 是否在协作层训练中真正生效（梯度量级对比）。

复用 verify_collab_mixed.py 的加载逻辑重建 v2 阵容，跑一个 code batch 的
forward_train，对比：
1. routing_loss 对 domain neuron (code) body 的梯度范数
2. CE loss（融合路径）对 code body 的梯度范数
3. scores 的 requires_grad / 梯度是否真正流入 neuron

结论输出：routing_loss 有效梯度 vs CE 梯度的量级比。

Usage:
    python scripts/training/diag_routing_gradient.py
"""
import os
import sys
import math

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

CKPT_PATH = "data/neurons/collab_v2_routing.ckpt.pt"
GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.topology import (
    NeuronGeometry, build_topology, establish_topology_channels,
)
from taiji.resonance.translator import TokenizerHub
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer, create_shared_embedding,
)
from scripts.training.train_cross_domain_collab import (
    load_neuron, load_shared_lm_head, load_shared_embedding,
)
from scripts.training.train_multi_domain_foundation import batch_align_and_embed


def main():
    torch.set_grad_enabled(True)
    device = "cpu"

    general_sp = load_general_tokenizer()
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"[ckpt] epoch={ck['epoch']}, total_steps={ck['total_steps']}")

    # ── 1. 加载 9 neuron + home embeddings（与训练一致）──
    shared_lm_head = load_shared_lm_head(GENERAL_DIR, 512, device)
    neurons, embeddings = {}, {}
    for nid in DOMAINS:
        n = load_neuron(nid, GENERAL_DIR, device, shared_lm_head=shared_lm_head)
        neurons[nid] = n
        embeddings[nid] = load_shared_embedding(GENERAL_DIR, device)
    for nid in DIALOGUE_IDS:
        ckp = torch.load(os.path.join(DIALOGUE_DIR, f"neuron_{nid}.pt"),
                         map_location=device, weights_only=False)
        cfg = ckp["neuron_config"]; cfg.unified_field_dim = None
        n = ResonanceNeuron(cfg)
        n.load_state_dict(ckp["state_dict"], strict=False)
        neurons[nid] = n
        emb = create_shared_embedding(device)
        ses = ckp.get("shared_embedding_state", {})
        w = ses["weight"] if isinstance(ses, dict) else ses
        emb.weight.data.copy_(w)
        embeddings[nid] = emb

    # ── 2. 重建拓扑 + 注入训练产物 ──
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
    print(f"[loaded] 9 neuron + collab 产物注入完成")

    # ── 3. 冻结策略（与训练一致：只解冻尾层 + field_write + side_channels）──
    for neuron in neurons.values():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        n_layers = len(neuron.layers)
        unfreeze_from = max(0, n_layers - 2)
        for i in range(unfreeze_from, n_layers):
            for p in neuron.layers[i].parameters():
                p.requires_grad = True
        for p in neuron.norm.parameters():
            p.requires_grad = True
        if hasattr(neuron, 'lm_head') and neuron.lm_head is not None:
            for p in neuron.lm_head.parameters():
                p.requires_grad = True
        for p in neuron.get_field_write_parameters():
            p.requires_grad = True
        neuron.train()

    # ── 4. 加载 code 训练数据（与训练一致，取前 2 条）──
    data_path = os.path.join("data", "sft", "code_sft.pt")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data][:2]
    print(f"[data] code texts: {len(texts)}")

    # ── 5. 构造 code batch 输入（general 空间）──
    emb_g = load_shared_embedding(GENERAL_DIR, device)
    neuron_embeddings = {}
    targets = mask = None
    for nid in neurons.keys():
        out = batch_align_and_embed(texts, general_sp, general_sp, emb_g, max_seq_len=64)
        neuron_embeddings[nid] = out[0]
        if targets is None:
            targets, mask = out[1], out[2]
    print(f"[batch] seq={targets.shape[1]}")

    # ── 6. forward_train（与训练完全一致）──
    result = ens.forward_train(
        neuron_embeddings=neuron_embeddings,
        n_rounds=2,
        fusion_mode="soft",
        targets=targets,
        field_conditioning=True,
        step=0,
        target_domain="general",
    )
    scores = result["scores"]
    print(f"[scores] requires_grad={scores.requires_grad}, shape={scores.shape}")
    print(f"[scores] 排序: " + ", ".join(
        f"{nid}={float(s):.4f}" for nid, s in zip(list(neurons.keys()), scores.detach())
    ))

    # C15: 预测质量 logits（D 方案，监督 = NLL 排序对比，替代 C13 域判别）
    domain_logits = result.get("quality_logits")
    print(f"[quality_logits] requires_grad={domain_logits.requires_grad if domain_logits is not None else 'N/A'}, "
          f"shape={domain_logits.shape if domain_logits is not None else 'N/A'}")
    if domain_logits is not None:
        dl_sorted = sorted(zip(list(neurons.keys()), domain_logits.detach()), key=lambda x: -x[1])
        print(f"[quality_logits] 排序: " + ", ".join(f"{nid}={float(d):.4f}" for nid, d in dl_sorted))

    # ── 7. CE loss（与训练一致）──
    fused_logits = result["fused_logits"]
    shift_logits = fused_logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    shift_targets = shift_targets.clone()
    shift_targets[~shift_mask] = -100
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100, reduction="sum",
    )
    n_tokens = max(shift_mask.sum().item(), 1)
    ce_loss = ce_loss / n_tokens
    print(f"[loss] ce={ce_loss.item():.4f}")

    # ── 8. routing_loss（与训练一致，v2 用 domain_logits）──
    domain = "code"
    nids_all = list(neurons.keys())
    domain_idx = nids_all.index(domain)
    if domain_logits is not None:
        routing_loss = -F.log_softmax(domain_logits / 0.15, dim=0)[domain_idx]
        rl_source = "domain_logits"
    else:
        routing_loss = -F.log_softmax(scores / 0.15, dim=0)[domain_idx]
        rl_source = "scores (fallback)"
    print(f"[loss] routing_loss({rl_source})={routing_loss.item():.4f}, weight=0.5 -> {0.5*routing_loss.item():.4f}")

    # ── 9. 分别反传对比梯度 ──
    def zero_all_grads():
        for n in neurons.values():
            for p in n.parameters():
                p.grad = None
        for mod in [ens.field_score_proj] if getattr(ens, 'field_score_proj', None) else []:
            mod.zero_grad()
        for proj in getattr(ens, '_cross_spec_projectors', {}).values():
            for p in proj.parameters():
                p.grad = None
        for proj in getattr(ens, '_cross_spec_back_projectors', {}).values():
            for p in proj.parameters():
                p.grad = None

    def param_norms(param_filter):
        return [(n, p.grad.detach().norm().item()) for n, p in param_filter
                if p.grad is not None]

    def neuron_body_grads(prefix):
        out = []
        for nid, n in neurons.items():
            if nid == domain:
                for name, p in n.named_parameters():
                    if p.grad is not None:
                        out.append((f"{prefix}:{nid}:{name}", p.grad.norm().item()))
        return out

    # 9a. 只反传 CE
    zero_all_grads()
    ce_loss.backward(retain_graph=True)
    ce_grads = neuron_body_grads("CE")
    ce_body_sum = sum(g for _, g in ce_grads)
    print(f"\n[grad] CE -> code body 梯度: {len(ce_grads)} 参数, L1={ce_body_sum:.6f}")
    for name, g in sorted(ce_grads, key=lambda x: -x[1])[:5]:
        print(f"    {name}: {g:.6f}")

    # 9b. routing_loss 梯度
    zero_all_grads()
    routing_loss.backward(retain_graph=True)
    rl_grads = neuron_body_grads("RL")
    rl_body_sum = sum(g for _, g in rl_grads)
    print(f"[grad] RL -> code body 梯度: {len(rl_grads)} 参数, L1={rl_body_sum:.6f}")
    for name, g in sorted(rl_grads, key=lambda x: -x[1])[:5]:
        print(f"    {name}: {g:.6f}")

    # 9c. 对比量级
    if ce_body_sum > 0 and rl_body_sum > 0:
        ratio = rl_body_sum / ce_body_sum
        print(f"\n=== 结论: routing_loss 梯度 / CE 梯度 = {ratio:.4f} ===")
        if ratio < 0.01:
            print(">>> routing_loss 梯度被 CE 完全淹没（<1%），训练无效")
        elif ratio < 0.1:
            print(">>> routing_loss 有效但偏弱（1-10%），可能被 CE 主导抵消")
        else:
            print(">>> routing_loss 梯度量级正常（>10%），问题在别处（步数/温度/冲突）")

    # 9d. 其他 neuron 是否被 routing_loss 影响（loo 耦合，fix 后应接近 0）
    zero_all_grads()
    routing_loss.backward()
    other_sum = 0.0
    other_names = []
    for nid, n in neurons.items():
        if nid == domain:
            continue
        for name, p in n.named_parameters():
            if p.grad is not None:
                other_sum += p.grad.norm().item()
                other_names.append(f"{nid}:{name}:{p.grad.norm().item():.4f}")
    print(f"[grad] RL -> 其他 8 neuron body 梯度 L1={other_sum:.6f}（fix 后应≈0，泄漏已消除）")
    if other_names:
        print(f"    top5 泄漏: {sorted(other_names, key=lambda x: -float(x.rsplit(':',1)[1]))[:5]}")
    # 自身 code 的梯度（fix 后应保持）
    code_sum = sum(g for _, g in rl_grads)
    print(f"[grad] RL -> code 自身梯度 L1={code_sum:.6f}（与 fix 前 15.2 对比，有效信号应保留）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
