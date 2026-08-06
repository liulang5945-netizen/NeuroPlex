"""A/B 实验：推理融合方式对生成质量的影响。

假设：训练学的融合（共振分 softmax，forward_train）比推理用的
entropy 路由（forward）生成质量更好——因为训练时模型学的就是共振分融合。

同一 prompt，两种融合各自采样生成，对比输出。

Usage:
    python -u scripts/training/_exp_inference_fusion_ab.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceField, ResonanceEnsemble
from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.topology import build_topology, establish_topology_channels
from scripts.training.eval_dialogue import load_neurons_and_weights, load_cross_spec_weights
from scripts.training.utils import load_domain_tokenizer, load_general_tokenizer
from scripts.training.experiment_config import DEFAULT_DOMAIN

DEVICE = "cpu"
PROMPTS = [
    "问：你好，请介绍一下自己\n答：",
    "问：什么是人工智能？\n答：",
    "问：如何学习编程？\n答：",
    "问：请解释神经网络的工作原理\n答：",
    "问：法国的首都是什么？\n答：",
]
MAX_TOKENS = 40
TEMPERATURE = 0.55
TOP_K = 15
REPETITION_PENALTY = 1.4


def generate_with(ensemble, neurons, shared_embeddings, domain_sp, general_sp,
                  prompt, mode):
    """采样生成。mode: "per_position"（entropy 路由）或 "score"（共振分融合）。"""
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return "(empty)"
    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    generated_domain = []
    domain_eos_id = domain_sp.eos_id() if hasattr(domain_sp, 'eos_id') else None
    if domain_eos_id is not None and domain_eos_id < 0:
        domain_eos_id = None

    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            neuron_embeddings = {}
            for nid, emb in shared_embeddings.items():
                neuron_embeddings[nid] = emb(ids)

            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True, fusion_mode=mode,
            )
            if "weighted_logits" in result:
                logits = result["weighted_logits"][:, -1, :].float()
            else:
                logits = None
            if logits is None:
                return "(no weighted_logits)"

            if generated_domain:
                for prev_token in set(generated_domain[-20:]):
                    if prev_token < logits.size(-1):
                        logits[0, prev_token] /= REPETITION_PENALTY

            logits = logits / TEMPERATURE
            if TOP_K > 0:
                cur_top_k = min(TOP_K, logits.size(-1))
                topk_vals, _ = torch.topk(logits[0], cur_top_k)
                logits[0][logits[0] < topk_vals[-1]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()
            generated_domain.append(nxt)
            if domain_eos_id is not None and nxt == domain_eos_id:
                break

            piece_text = domain_sp.decode([nxt])
            new_general_ids = general_sp.encode(piece_text)
            ids = torch.cat([ids, torch.tensor([new_general_ids], dtype=torch.long, device=DEVICE)], dim=1)

    return domain_sp.decode(generated_domain)


def main():
    domain_sp = load_domain_tokenizer(DEFAULT_DOMAIN)
    general_sp = load_general_tokenizer()
    neurons, shared_embeddings = load_neurons_and_weights("dialogue", "data/neurons/cross_spec_dialogue.ckpt.pt")
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    load_cross_spec_weights(ensemble, "dialogue", "data/neurons/cross_spec_dialogue.ckpt.pt")

    print(f"\n{'='*60}\n推理融合 A/B 实验（{len(PROMPTS)} prompts，同一 forward 推理路径）\n{'='*60}", flush=True)
    for prompt in PROMPTS:
        print(f"\n问：{prompt.replace('问：','').replace('答：','').strip()}", flush=True)
        for mode, label in [("per_position", "A: per_position(entropy 路由)"), ("score", "B: score(共振分融合)")]:
            try:
                out = generate_with(ensemble, neurons, shared_embeddings,
                                    domain_sp, general_sp, prompt, mode)
                print(f"  [{label}] {out}", flush=True)
            except Exception as e:
                print(f"  [{label}] 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
