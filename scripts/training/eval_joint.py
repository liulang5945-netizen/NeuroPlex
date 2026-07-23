"""评估联合训练效果——个体 vs 协作 PPL + 生成质量对比。

关键判据：
  协作 PPL < min(个体 PPL) → 涌现确认（1+1>2）
  协作 PPL ≈ min(个体 PPL) → 协作无效
  协作 PPL > min(个体 PPL) → 协作有害

Usage:
    python -u scripts/training/eval_joint.py --n_neurons 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_texts, load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR,
)

DOMAIN = "zh"


def load_joint_neurons(n_neurons: int, domain: str, device: str):
    """加载联合训练的神经元 + shared_embedding。"""
    cfg = get_domain_neuron_config(domain)
    neurons = {}
    for i in range(n_neurons):
        nid = f"{domain}_j{i}"
        path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到联合训练神经元: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        neuron = ResonanceNeuron(cfg).to(device)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.eval()
        neurons[nid] = neuron
        result = ckpt.get("result", {})
        print(f"  [{nid}] best_loss={result.get('best_loss', '?')} "
              f"saved={result.get('saved', '?')}", flush=True)

    # 加载 shared_embedding
    emb_path = os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt")
    shared_embedding = nn.Embedding(256000, 512)
    shared_embedding.load_state_dict(torch.load(emb_path, map_location=device))
    shared_embedding.to(device).eval()

    return neurons, shared_embedding, cfg


def compute_ppl(logits, targets, mask):
    """计算 next-token PPL。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    shift_targets = shift_targets.clone()
    shift_targets[~shift_mask] = -100
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_tokens = shift_mask.sum().item()
    avg_loss = loss.item() / max(n_tokens, 1)
    return math.exp(min(avg_loss, 20)), avg_loss


def eval_ppl(neurons, shared_embedding, domain_sp, general_sp, device, n_eval=200):
    """对比个体 vs 协作 PPL。"""
    print("\n" + "=" * 70, flush=True)
    print("[PPL 评估] 个体 vs 协作", flush=True)
    print("=" * 70, flush=True)

    texts = load_domain_texts(DOMAIN, max_texts=n_eval)
    # 用最后 n_eval 条作为评估集（避免和训练数据完全重叠的偏差）
    texts = texts[-n_eval:] if len(texts) > n_eval else texts
    print(f"  评估集: {len(texts)} 条文本", flush=True)

    field = ResonanceField(dim=next(iter(neurons.values())).config.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

    # 个体 PPL
    individual_ppls = {}
    for nid, neuron in neurons.items():
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                shared_emb, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_embedding,
                )
                shared_emb = shared_emb.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                result = neuron.forward(shared_emb, return_logits=True)
                logits = result["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += shift_mask.sum().item()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        individual_ppls[nid] = ppl
        print(f"  个体 [{nid}]: PPL={ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    # 协作 PPL
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            shared_emb, targets, mask = batch_align_and_embed(
                [text], domain_sp, general_sp, shared_embedding,
            )
            shared_emb = shared_emb.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            result = ensemble.forward_train(shared_emb, temperature=1.0)
            fused_logits = result["fused_logits"]
            shift_logits = fused_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    collab_ppl = math.exp(min(avg_loss, 20))
    print(f"  协作 [all]:    PPL={collab_ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    # 判读
    min_individual = min(individual_ppls.values())
    best_individual = min(individual_ppls, key=individual_ppls.get)
    print(f"\n  最强个体: [{best_individual}] PPL={min_individual:.1f}", flush=True)
    print(f"  协作 PPL: {collab_ppl:.1f}", flush=True)
    if collab_ppl < min_individual:
        improvement = (min_individual - collab_ppl) / min_individual * 100
        print(f"  ✅ 涌现确认！协作比最强个体好 {improvement:.1f}%", flush=True)
    elif collab_ppl < sum(individual_ppls.values()) / len(individual_ppls):
        print(f"  ⚠️ 协作优于平均但未超最强个体", flush=True)
    else:
        print(f"  ❌ 协作未优于个体", flush=True)

    return individual_ppls, collab_ppl


def eval_generation(neurons, shared_embedding, domain_sp, general_sp, cfg, device):
    """生成质量对比：个体 vs 协作。"""
    print("\n" + "=" * 70, flush=True)
    print("[生成质量对比] 个体 vs 协作", flush=True)
    print("=" * 70, flush=True)

    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

    PROMPTS = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "深度学习在自然语言处理中的应用",
        "请解释神经网络的工作原理",
    ]

    def generate_collab(prompt, max_tokens=50):
        """协作生成：用 forward_train 聚合 logits。"""
        general_ids = general_sp.EncodeAsIds(prompt)
        ids = torch.tensor([general_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated_domain = []
            for _ in range(max_tokens):
                shared_emb = shared_embedding(ids)  # [1, L, 512]
                result = ensemble.forward_train(shared_emb, temperature=1.0)
                logits = result["fused_logits"][:, -1, :]  # [1, V]
                # 贪婪解码
                next_token = logits.argmax(dim=-1).item()
                if next_token == domain_sp.eos_id() if hasattr(domain_sp, 'eos_id') else False:
                    break
                generated_domain.append(next_token)
                # domain token → general token（简单映射：用 domain_sp decode 再 general_sp encode）
                piece = domain_sp.DecodeIds([next_token])
                gen_ids = general_sp.EncodeAsIds(piece)
                if gen_ids:
                    ids = torch.cat([ids, torch.tensor([gen_ids], dtype=torch.long, device=device)], dim=1)
                else:
                    break
                if ids.shape[1] > 200:
                    break
            text = domain_sp.DecodeIds(generated_domain)
        return text

    def generate_individual(prompt, nid, max_tokens=50):
        """个体生成：单神经元 forward。"""
        neuron = neurons[nid]
        general_ids = general_sp.EncodeAsIds(prompt)
        ids = torch.tensor([general_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated_domain = []
            for _ in range(max_tokens):
                shared_emb = shared_embedding(ids)
                result = neuron.forward(shared_emb, return_logits=True)
                logits = result["logits"][:, -1, :]
                next_token = logits.argmax(dim=-1).item()
                if next_token == domain_sp.eos_id() if hasattr(domain_sp, 'eos_id') else False:
                    break
                generated_domain.append(next_token)
                piece = domain_sp.DecodeIds([next_token])
                gen_ids = general_sp.EncodeAsIds(piece)
                if gen_ids:
                    ids = torch.cat([ids, torch.tensor([gen_ids], dtype=torch.long, device=device)], dim=1)
                else:
                    break
                if ids.shape[1] > 200:
                    break
            text = domain_sp.DecodeIds(generated_domain)
        return text

    for prompt in PROMPTS:
        print(f"\n  prompt: {prompt}", flush=True)
        # 协作生成
        collab_out = generate_collab(prompt)
        print(f"  协作: {collab_out[:150] if collab_out else '(empty)'}", flush=True)
        # 最强个体生成（取第一个）
        first_nid = list(neurons.keys())[0]
        indiv_out = generate_individual(prompt, first_nid)
        print(f"  个体[{first_nid}]: {indiv_out[:150] if indiv_out else '(empty)'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="评估联合训练效果")
    parser.add_argument("--n_neurons", type=int, default=5, help="神经元数量")
    parser.add_argument("--domain", default="zh", help="域")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--n_eval", type=int, default=200, help="PPL评估文本数")
    args = parser.parse_args()

    print(f"加载联合训练的 {args.n_neurons} 个 {args.domain} 神经元...", flush=True)
    neurons, shared_embedding, cfg = load_joint_neurons(
        args.n_neurons, args.domain, args.device
    )
    domain_sp = load_domain_tokenizer(args.domain)
    general_sp = load_general_tokenizer()

    # PPL 评估
    individual_ppls, collab_ppl = eval_ppl(
        neurons, shared_embedding, domain_sp, general_sp, args.device, args.n_eval
    )

    # 生成质量对比
    eval_generation(
        neurons, shared_embedding, domain_sp, general_sp, cfg, args.device
    )

    print("\n" + "=" * 70, flush=True)
    print("评估完成", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
