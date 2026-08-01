"""快速评估单个神经元的对话生成能力。

Usage:
    python -u scripts/training/eval_single_dialogue.py --neuron_id zh_sft_std0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from scripts.training.utils import load_general_tokenizer, OUTPUT_DIR, create_shared_embedding
from scripts.training.experiment_config import (
    DEFAULT_DOMAIN,
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_REPETITION_PENALTY, SAMPLING_MAX_TOKENS,
    DIALOGUE_PROMPTS,
)

DEVICE = "cpu"


def load_neuron(neuron_id: str):
    """加载单个神经元。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{neuron_id}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DEFAULT_DOMAIN, spec="compact")

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.eval()

    shared_emb = nn.Embedding(256000, 512)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE).eval()

    result = ckpt.get("result", {})
    print(f"  [{neuron_id}] spec={cfg.spec}, best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb, cfg


def generate(neuron, shared_emb, domain_sp, general_sp, prompt, max_tokens=SAMPLING_MAX_TOKENS, temperature=SAMPLING_TEMPERATURE, top_k=SAMPLING_TOP_K, repetition_penalty=SAMPLING_REPETITION_PENALTY):
    """生成对话回复。

    关键修复：neuron 的 lm_head 输出是 domain token ID（不是 general token ID）。
    - 输入：general token IDs → shared_embedding → neuron（训练路径一致）
    - 输出：domain token ID → 需要转回 general token IDs 才能追加到输入
    - 解码：用 domain_sp 解码（不是 general_sp）
    """
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return "(empty)"

    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    generated_domain_ids = []  # 收集 domain token IDs 用于解码

    # domain tokenizer 的 EOS
    domain_eos_id = None
    if hasattr(domain_sp, 'eos_id'):
        eid = domain_sp.eos_id()
        if eid is not None and eid >= 0:
            domain_eos_id = int(eid)

    with torch.no_grad():
        for _ in range(max_tokens):
            emb_input = shared_emb(ids)
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :].float()  # [1, domain_vocab_size]

            # Repetition penalty（对 domain token IDs）
            if generated_domain_ids:
                for prev_token in set(generated_domain_ids[-20:]):
                    if prev_token < logits.size(-1):
                        logits[0, prev_token] /= repetition_penalty

            # Temperature + top-k
            logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits[0], top_k)
                threshold = topk_vals[-1]
                logits[0][logits[0] < threshold] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_domain_token = torch.multinomial(probs, num_samples=1).item()
            generated_domain_ids.append(next_domain_token)

            # EOS 检测（domain tokenizer）
            if domain_eos_id is not None and next_domain_token == domain_eos_id:
                break

            # 关键修复：domain token ID → 文本 → general token IDs → 追加到输入
            piece_text = domain_sp.decode([next_domain_token])
            new_general_ids = general_sp.encode(piece_text)
            if not new_general_ids:
                new_general_ids = [general_sp.pad_id()]  # fallback
            ids = torch.cat([ids, torch.tensor([new_general_ids], dtype=torch.long, device=DEVICE)], dim=1)

    # 用 domain tokenizer 解码
    text = domain_sp.DecodeIds(generated_domain_ids)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neuron_id", required=True)
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print("=" * 70, flush=True)
    print(f"单神经元对话评估: {args.neuron_id}", flush=True)
    print("=" * 70, flush=True)

    neuron, shared_emb, cfg = load_neuron(args.neuron_id)
    general_sp = load_general_tokenizer()
    from scripts.training.utils import load_domain_tokenizer
    domain_sp = load_domain_tokenizer("zh")

    PROMPTS = DIALOGUE_PROMPTS

    for prompt in PROMPTS:
        print(f"\n  {prompt}", flush=True)
        response = generate(neuron, shared_emb, domain_sp, general_sp, prompt)
        print(f"  回复: {response}", flush=True)
        print(f"  {'-' * 60}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("评估完成", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
