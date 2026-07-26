"""评估用简单中文数据训练的 compact 神经元（用修复后的生成函数）。

加载 checkpoint，评估 val PPL + 生成中文样本（多个 prompt）。

Usage:
    python -u scripts/training/eval_simple_neuron.py --neuron_id zh_simple0
    python -u scripts/training/eval_simple_neuron.py --neuron_id zh_simple0 --prompts "从前" "小猫" "今天天气"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    load_or_create_shared_embedding,
    OUTPUT_DIR,
)

DATA_PATH = "data/simple_zh/simple_zh_texts.jsonl"


def generate_sample(neuron, domain_sp, general_sp, shared_embedding, device, prompt,
                    max_tokens=100, temperature=0.8, top_k=40, top_p=0.9,
                    repetition_penalty=1.3, no_repeat_ngram=3):
    """生成中文样本（修复：用 general_sp 编码 prompt，和训练一致）。

    训练时：text → general_sp.encode → general_ids（输入）
    生成时：prompt → general_sp.encode → general_ids（输入）← 之前用 domain_sp 错了！

    模型预测 domain token → decode → piece → general_sp.encode → 追加所有 general ids
    """
    neuron.eval()
    with torch.no_grad():
        # 1. prompt → general_ids（用 general tokenizer，和训练一致！）
        general_ids = general_sp.encode(prompt)
        if not general_ids:
            return "(empty)"

        generated_domain_ids = []
        for _ in range(max_tokens):
            emb_input = shared_embedding(torch.tensor([general_ids], device=device))
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :].float() / temperature

            # 重复惩罚
            if len(generated_domain_ids) > 0:
                recent = generated_domain_ids[-20:]
                for prev_id in set(recent):
                    logits[0, prev_id] /= repetition_penalty

            # no-repeat-ngram
            if no_repeat_ngram > 0 and len(generated_domain_ids) >= no_repeat_ngram:
                ngram = tuple(generated_domain_ids[-(no_repeat_ngram-1):])
                banned = set()
                for i in range(len(generated_domain_ids) - no_repeat_ngram + 1):
                    if tuple(generated_domain_ids[i:i+no_repeat_ngram-1]) == ngram:
                        banned.add(generated_domain_ids[i + no_repeat_ngram - 1])
                for banned_id in banned:
                    logits[0, banned_id] = float('-inf')

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_domain_id = torch.multinomial(probs, num_samples=1).item()
            generated_domain_ids.append(next_domain_id)

            # domain token → general tokens（追加所有，不是只取第一个）
            piece = domain_sp.decode([next_domain_id])
            gids = general_sp.encode(piece)
            if gids:
                general_ids.extend(gids)  # 追加所有 general ids
            else:
                general_ids.append(general_sp.unk_id() if hasattr(general_sp, 'unk_id') else 0)

        # 用 domain_sp 解码生成的 domain tokens
        text = domain_sp.decode(generated_domain_ids)
    neuron.train()
    return text


def main():
    parser = argparse.ArgumentParser(description="评估 compact 神经元（修复后的生成函数）")
    parser.add_argument("--neuron_id", default="zh_simple0")
    parser.add_argument("--prompts", nargs='+', default=["从前", "小猫", "今天天气", "大森林里", "妈妈说"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_eval", type=int, default=100, help="评估 PPL 的样本数")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"评估神经元: {args.neuron_id}", flush=True)
    print("=" * 60, flush=True)

    # 加载 checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"neuron_{args.neuron_id}.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint 不存在: {ckpt_path}", flush=True)
        return
    print(f"\n[1] 加载 checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"  best_val_ppl: {ckpt.get('result', {}).get('best_val_ppl', 'N/A')}", flush=True)
    print(f"  best_step: {ckpt.get('result', {}).get('best_step', 'N/A')}", flush=True)
    print(f"  data_source: {ckpt.get('data_source', 'N/A')}", flush=True)

    # tokenizers
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    # shared_embedding
    print(f"\n[3] 加载 shared_embedding...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)
    # 如果 checkpoint 有 shared_embedding_state，加载它
    if ckpt.get('shared_embedding_state'):
        shared_embedding.load_state_dict(ckpt['shared_embedding_state'])
        print(f"  从 checkpoint 加载 shared_embedding_state ✅", flush=True)

    # 创建神经元
    print(f"\n[4] 创建神经元...", flush=True)
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg).to(args.device)
    neuron.load_state_dict(ckpt['state_dict'])
    neuron.eval()
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  参数量: {n_params/1e6:.1f}M", flush=True)

    # 评估 PPL
    print(f"\n[5] 评估 val PPL（{args.n_eval} 条）...", flush=True)
    eval_texts = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[-args.n_eval:]:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            text = d.get('text', '')
            if len(text) >= 20:
                eval_texts.append(text)
        except json.JSONDecodeError:
            continue

    total_ce = 0.0
    n_eval = 0
    with torch.no_grad():
        for text in eval_texts:
            shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_embedding)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits']
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets_flat = shift_targets.clone()
            shift_targets_flat[~shift_mask] = -100
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets_flat.view(-1),
                ignore_index=-100,
            )
            total_ce += ce.item()
            n_eval += 1

    val_ppl = math.exp(min(total_ce / max(n_eval, 1), 20))
    print(f"  val PPL = {val_ppl:.2f} ({n_eval} 条)", flush=True)

    # 生成样本
    print(f"\n[6] 生成样本（修复后的生成函数）...", flush=True)
    print("=" * 60, flush=True)
    for prompt in args.prompts:
        sample = generate_sample(neuron, domain_sp, general_sp, shared_embedding, args.device, prompt)
        print(f"\n提示: {prompt}", flush=True)
        print(f"生成: {sample}", flush=True)
        print("-" * 60, flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"评估完成！", flush=True)
    print(f"  val PPL = {val_ppl:.2f}", flush=True)
    print(f"  目标: PPL < 10 (连贯生成基线), PPL < 6 (良好)", flush=True)
    if val_ppl < 6:
        print(f"  → ✅ PPL 良好，检查生成质量", flush=True)
    elif val_ppl < 10:
        print(f"  → ⚠️ PPL 达标但不够好", flush=True)
    else:
        print(f"  → ❌ PPL 太高，需要更多训练", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
