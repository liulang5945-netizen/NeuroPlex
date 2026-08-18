"""跨域协作层评估：验证 5>5 涌现（分工 > 各自独立）。

评估项：
1. 各域内 EMERGE：code/math/zh/en 数据上，协作 PPL vs 最强个体
   （用训练同路径 forward_train(target_domain=域)）
2. 跨域任务生成：中文提问 → code 输出（zh 理解 + code 表达的组合能力）
   对比：单 code neuron（无 zh 帮助）vs 协作 ensemble

Usage:
    python -u scripts/training/_eval_cross_domain_collab.py \
        --ckpt data/neurons/cross_domain_v1.ckpt.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceField, ResonanceEnsemble
from neuroplex.resonance.geometry import NeuronGeometry
from neuroplex.resonance.topology import build_topology, establish_topology_channels
from neuroplex.resonance.translator import TokenizerHub, batch_align_and_embed
from scripts.training.train_cross_domain_collab import (
    load_neuron, load_shared_embedding, load_shared_lm_head, load_tokenizer_for_vocab,
)
from scripts.training.eval_dialogue import load_cross_spec_weights
from scripts.training.utils import load_general_tokenizer
from scripts.training.experiment_config import SFT_ANSWER_MARKER

DEVICE = "cpu"
DOMAINS = ["code", "math", "zh", "en"]
CKPT = "data/neurons/cross_domain_v1.ckpt.pt"


def load_ensemble(neuron_dir, domains, ckpt_path, no_weights=False):
    neurons = {}
    shared_embeddings = {}
    # 统一输出空间（general 基座）：注入共享 general 256K head
    shared_lm_head = load_shared_lm_head(neuron_dir, 512, DEVICE)
    for dom in domains:
        neurons[dom] = load_neuron(dom, neuron_dir, DEVICE, shared_lm_head=shared_lm_head)
        shared_embeddings[dom] = load_shared_embedding(neuron_dir, DEVICE)
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    if not no_weights and ckpt_path:
        load_cross_spec_weights(ensemble, "dialogue", ckpt_path)
    return neurons, shared_embeddings, ensemble


def domain_ppl(neurons, shared_embeddings, ensemble, hub, general_sp, domain, texts,
               rounds=1):
    """协作 vs 原生 neuron：answer-only PPL（训练同口径，forward_train）。

    基线 = 本域原生 neuron（跨 vocab 的个体对比无意义：code neuron 的 12K
    vocab 无法输出 zh 的 20K token，唯一公平对比是各域自己的原生 neuron）。
    zh 域与训练一致使用 answer marker（只计 "答：" 之后）。
    """
    domain_sp = hub.get_tokenizer(domain)
    answer_marker = SFT_ANSWER_MARKER if domain == "zh" else None
    marker_mode = "last" if answer_marker else "first"
    # 最强个体 = 本域原生 neuron（冻结）
    native = neurons[domain]
    emb = shared_embeddings[domain]
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            out = batch_align_and_embed([text], domain_sp, general_sp, emb,
                                        answer_marker=answer_marker,
                                        answer_marker_mode=marker_mode)
            shared_emb_out, targets, mask = out[0].to(DEVICE), out[1].to(DEVICE), out[2].to(DEVICE)
            sft_mask = out[3].to(DEVICE) if len(out) > 3 else None
            res = native.forward(shared_emb_out, return_logits=True)
            logits = res["logits"]
            sl, st = logits[:, :-1, :].contiguous(), targets[:, 1:].contiguous()
            sm = mask[:, 1:].contiguous()
            st = st.clone()
            if sft_mask is not None:
                ss = sft_mask[:, 1:].contiguous()
                st[~(sm & ss)] = -100
                n_tok = (sm & ss).sum().item()
            else:
                st[~sm] = -100
                n_tok = sm.sum().item()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                                   ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += max(n_tok, 1)
    best_avg = total_loss / max(total_tokens, 1)
    best_id = domain
    # 协作
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            neuron_embeddings = {}
            targets = None
            mask = None
            sft_mask = None
            for nid, e in shared_embeddings.items():
                out = batch_align_and_embed([text], domain_sp, general_sp, e,
                                            answer_marker=answer_marker,
                                            answer_marker_mode=marker_mode)
                neuron_embeddings[nid] = out[0].to(DEVICE)
                if targets is None:
                    targets, mask = out[1].to(DEVICE), out[2].to(DEVICE)
                    if len(out) > 3:
                        sft_mask = out[3].to(DEVICE)
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings, n_rounds=rounds, fusion_mode="soft",
                targets=targets, target_domain=domain,
            )
            fused = result["fused_logits"]
            sl, st = fused[:, :-1, :].contiguous(), targets[:, 1:].contiguous()
            sm = mask[:, 1:].contiguous()
            st = st.clone()
            if sft_mask is not None:
                ss = sft_mask[:, 1:].contiguous()
                st[~(sm & ss)] = -100
                n_tok = (sm & ss).sum().item()
            else:
                st[~sm] = -100
                n_tok = sm.sum().item()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                                   ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += max(n_tok, 1)
    collab_avg = total_loss / max(total_tokens, 1)
    emerge = (best_avg - collab_avg) / best_avg * 100 if best_avg > 0 else 0
    return {
        "domain": domain,
        "best_individual": best_id,
        "best_ppl": math.exp(min(best_avg, 20)),
        "collab_ppl": math.exp(min(collab_avg, 20)),
        "emerge_pct": emerge,
    }


def load_sft_texts(data_dir, domain, max_texts=20):
    path = os.path.join(data_dir, f"{domain}_sft.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return [d["full"] for d in data[:max_texts]]


def cross_domain_generate(neurons, shared_embeddings, ensemble, hub, general_sp,
                          zh_prompt, target_domain="code", max_tokens=30, rounds=1):
    """跨域生成：中文 prompt（general 编码）→ target 域输出（用 code neuron 表达）。"""
    target_sp = hub.get_tokenizer(target_domain)
    ids = torch.tensor([general_sp.encode(zh_prompt)], dtype=torch.long, device=DEVICE)
    generated = []
    eos_id = target_sp.eos_id() if hasattr(target_sp, 'eos_id') else None
    if eos_id is not None and eos_id < 0:
        eos_id = None
    with torch.no_grad():
        for _ in range(max_tokens):
            neuron_embeddings = {}
            for nid, emb in shared_embeddings.items():
                neuron_embeddings[nid] = emb(ids)
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings, n_rounds=rounds, fusion_mode="soft",
                target_domain=target_domain,
            )
            logits = result["fused_logits"][:, -1, :].float()
            probs = F.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()
            generated.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            piece = target_sp.decode([nxt])
            new_ids = general_sp.encode(piece)
            ids = torch.cat([ids, torch.tensor([new_ids], dtype=torch.long, device=DEVICE)], dim=1)
    return target_sp.decode(generated)


def cross_domain_generate_general(neurons, shared_embeddings, ensemble, hub, general_sp,
                                  zh_prompt, max_tokens=40, rounds=2):
    """通用空间生成：zh 提问 → general 256K 空间分工路由（zh 理解 + code 表达共存）。

    各 neuron logits 投影到 general 空间，按位置路由（min(原生,投影) 置信度）：
    zh token → zh neuron（中文片段自信），code token → code neuron（代码片段自信）。
    语义桥接的关键架构探针：目标域路由在结构上阻止 code neuron 参与 zh 域批次，
    通用空间让两种专家在输出空间共存。
    """
    target_sp = general_sp
    ids = torch.tensor([general_sp.encode(zh_prompt)], dtype=torch.long, device=DEVICE)
    generated = []
    eos_id = None
    with torch.no_grad():
        for _ in range(max_tokens):
            neuron_embeddings = {}
            for nid, emb in shared_embeddings.items():
                neuron_embeddings[nid] = emb(ids)
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings, n_rounds=rounds, fusion_mode="soft",
                target_domain="general",
            )
            logits = result["fused_logits"][:, -1, :].float()
            probs = F.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()
            generated.append(nxt)
            piece = target_sp.decode([nxt])
            new_ids = general_sp.encode(piece)
            ids = torch.cat([ids, torch.tensor([new_ids], dtype=torch.long, device=DEVICE)], dim=1)
    return target_sp.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neuron-dir", default="data/verify_v3")
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--n-ppl", type=int, default=10)
    parser.add_argument("--no-weights", action="store_true",
                        help="不加载协作层权重（只测基座 + 路由融合）")
    parser.add_argument("--rounds", type=int, default=1,
                        help="forward_train 共振轮数（无训练权重时用 1 避免 side 扭曲）")
    args = parser.parse_args()

    print("加载多域 neuron + 协作层...", flush=True)
    neurons, shared_embeddings, ensemble = load_ensemble(
        args.neuron_dir, DOMAINS, args.ckpt, no_weights=args.no_weights)
    print("已加载", list(neurons.keys()), flush=True)

    # 与训练同口径注册 tokenizer：zh 用 neuron vocab 匹配的 v20k 变体
    hub = TokenizerHub()
    for dom in DOMAINS:
        sp = load_tokenizer_for_vocab(dom, neurons[dom].config.vocab_size)
        hub.register_domain(dom, sp)
    general_sp = load_general_tokenizer()
    hub.register_domain("general", general_sp)
    ensemble.set_tokenizer_hub(hub)

    print(f"\n{'='*60}\n[1] 各域内 EMERGE（协作 vs 最强个体）\n{'='*60}", flush=True)
    total_emerge = 0.0
    for dom in DOMAINS:
        texts = load_sft_texts(args.data_dir, dom, args.n_ppl)
        r = domain_ppl(neurons, shared_embeddings, ensemble, hub, general_sp, dom, texts,
                       rounds=args.rounds)
        total_emerge += r["emerge_pct"]
        print(f"  [{dom}] 最强个体={r['best_individual']} PPL={r['best_ppl']:.1f} | "
              f"协作 PPL={r['collab_ppl']:.1f} | EMERGE={r['emerge_pct']:+.1f}%", flush=True)
    print(f"  平均 EMERGE: {total_emerge/len(DOMAINS):+.1f}%", flush=True)

    print(f"\n{'='*60}\n[2] 跨域任务生成（中文提问 → code 输出）\n{'='*60}", flush=True)
    zh_prompts = [
        "请写一个 Python 函数，判断一个数是否为偶数。",
        "用 Python 打印 1 到 10。",
        "写一个函数计算两个数的和。",
    ]
    for p in zh_prompts:
        print(f"\n  中文提问：{p}", flush=True)
        try:
            out = cross_domain_generate(neurons, shared_embeddings, ensemble, hub,
                                        general_sp, p, target_domain="code",
                                        rounds=args.rounds)
            print(f"  [code 域] → {out}", flush=True)
        except Exception as e:
            print(f"  生成失败: {e}", flush=True)
        try:
            out_g = cross_domain_generate_general(neurons, shared_embeddings, ensemble,
                                                  hub, general_sp, p, rounds=args.rounds)
            print(f"  [general 域] → {out_g}", flush=True)
        except Exception as e:
            print(f"  通用空间生成失败: {e}", flush=True)


if __name__ == "__main__":
    main()
