#!/usr/bin/env python3
"""协作层真实生成质量测试（2026-08-08，回应"生成质量没有测试"）。

对比 verify_collab_mixed 的 PPL 评估（forward_train soft 融合）：
本脚本走**推理路径** ensemble.forward（与 cortex._generate_p7 同路径：
跨 vocab confidence_routing 融合），多 prompt 自回归生成 40 token。
这才是真实生成质量（PPL 是代理指标，生成是最终产物）。

Usage:
    python scripts/training/gen_test_collab.py --ckpt data/neurons/collab_v3_c14b.ckpt.pt
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

GENERAL_DIR = "data/foundation_v1_general"
DIALOGUE_DIR = "data/neurons"
DOMAINS = ["code", "math", "zh", "en"]
DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
MAX_TOKENS = 40

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def sample_token(logits, temperature, top_k):
    """temperature=0 → argmax；>0 → top-k 采样（与 cortex 生成一致）。"""
    if temperature <= 0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    if top_k > 0 and top_k < probs.shape[-1]:
        topv, _ = torch.topk(probs, top_k)
        probs[probs < topv[-1].item()] = 0.0
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/neurons/collab_v3_c14b.ckpt.pt")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--solo", default="",
                    help="单 neuron 生成对比（如 code/math/zh/en/zh_std0_dialogue）——"
                         "定位生成崩坏层：solo 也差 → 基座问题；solo 好 ensemble 差 → 融合破坏")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度（0=argmax 贪心；>0 用 top-k 采样，如 0.9）")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    from scripts.training.train_cross_domain_collab import (
        load_neuron, load_shared_lm_head, load_shared_embedding,
    )
    from scripts.training.utils import (
        load_general_tokenizer, create_shared_embedding,
    )
    from taiji.resonance.ensemble import ResonanceEnsemble
    from taiji.resonance.field import ResonanceField
    from taiji.resonance.geometry import NeuronGeometry
    from taiji.resonance.topology import build_topology, establish_topology_channels
    from taiji.resonance.translator import TokenizerHub
    from scripts.training.utils import load_domain_tokenizer

    general_sp = load_general_tokenizer()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt: epoch={ck['epoch']}, total_steps={ck['total_steps']}")

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

    print(f"[加载完成] {len(neurons)} neuron\n")

    def generate(prompt, max_tokens=args.max_tokens):
        cur = general_sp.EncodeAsIds(prompt)
        nids = list(neurons.keys())
        for step in range(max_tokens):
            ids_t = torch.tensor([cur], dtype=torch.long)
            neuron_embeddings = {}
            for nid in nids:
                neuron_embeddings[nid] = embeddings[nid](ids_t)
            with torch.no_grad():
                r = ens.forward(
                    neuron_embeddings=neuron_embeddings,
                    return_logits=True,
                    field_conditioning=True,
                    active_filter=True,  # 真实推理路径（低共振过滤）
                )
            logits = r.get("weighted_logits")
            if logits is None:  # fusion 失败 fallback
                return None, r
            nxt = sample_token(logits[0, -1], args.temperature, args.top_k)
            cur.append(nxt)
            if general_sp.IdToPiece(nxt) in ("</s>", "<s>", "<unk>"):
                break
        return cur, r

    print("=" * 80)
    if args.solo:
        print(f"单 neuron 生成测试（--solo {args.solo}，定位生成崩坏层）")
    else:
        print("真实推理路径生成测试（ensemble.forward，40 token）")
    print("=" * 80)

    if args.solo:
        nid = args.solo
        emb = embeddings[nid]
        for tag, prompt in PROMPTS:
            cur = general_sp.EncodeAsIds(prompt)
            for step in range(args.max_tokens):
                ids_t = torch.tensor([cur], dtype=torch.long)
                with torch.no_grad():
                    r = neurons[nid].forward(emb(ids_t), return_logits=True)
                logits = r["logits"]
                if logits.shape[-1] != 256000:  # 旧 5 转译到 general
                    from taiji.resonance.translator import build_logits_alignment_matrix
                    from scripts.training.utils import load_domain_tokenizer
                    src_sp = load_domain_tokenizer("zh")
                    m = build_logits_alignment_matrix(src_sp, general_sp, "zh", "general",
                                                      cache={}, source_vocab_size=logits.shape[-1])
                    b, l, vi = logits.shape
                    logits = torch.sparse.mm(logits.reshape(-1, vi), m.to(logits.dtype)).reshape(b, l, 256000)
                nxt = sample_token(logits[0, -1], args.temperature, args.top_k)
                cur.append(nxt)
                if general_sp.IdToPiece(nxt) in ("</s>", "<s>", "<unk>"):
                    break
            out = general_sp.DecodeIds(cur)
            print(f"\n── [{tag}] prompt: {prompt}")
            print(f"    → {out}")
        print(f"\n完成。对比 ensemble 生成判断崩坏层。")
        return 0

    for tag, prompt in PROMPTS:
        cur, r = generate(prompt)
        if cur is None:
            print(f"\n── [{tag}] {prompt}")
            print(f"    融合失败: fusion_mode={r.get('fusion_mode')}, error={r.get('fusion_error')}")
            continue
        out = general_sp.DecodeIds(cur)
        weights = r.get("weights", [])
        w_str = ", ".join(f"{k}={w:.2f}" for k, w in zip(list(neurons.keys()), weights)) if weights else "N/A"
        print(f"\n── [{tag}] prompt: {prompt}")
        print(f"    → {out}")
        print(f"    路由权重: {w_str}")

    print(f"\n{'='*80}")
    print("完成。请人工判断生成质量（流畅度/相关性/格式）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
