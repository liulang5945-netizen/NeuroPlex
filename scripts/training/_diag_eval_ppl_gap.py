"""诊断：eval PPL 705 vs 训练 PPL 3.0 的口径验证。

假设：eval_dialogue.py 用全文本 loss（无 answer_marker），训练用 answer-only
（sft_mask）。本脚本用训练同管线（answer-only + EOS）在训练数据上算协作 PPL，
若 ≈3.0 则模型正常学到训练分布，问题在推理路径/泛化而非训练失败。

Usage:
    python -u scripts/training/_diag_eval_ppl_gap.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, get_domain_neuron_config,
    NeuronConfig,
)
from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.topology import build_topology, establish_topology_channels
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.finetune_cross_spec import load_neuron_with_embedding, load_dialogue_texts
from scripts.training.utils import load_dialogue_texts_multi, load_domain_tokenizer, load_general_tokenizer
from scripts.training.experiment_config import (
    ENSEMBLE_DIALOGUE_IDS as NEURON_IDS, DEFAULT_DOMAIN, SFT_ANSWER_MARKER,
)

DEVICE = "cpu"


def compute_answer_ppl(neurons, shared_embeddings, ensemble, domain_sp, general_sp,
                       texts, use_train_path=True):
    """answer-only 协作 PPL（与训练同口径）。"""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            neuron_embeddings = {}
            targets = None
            mask = None
            sft_mask = None
            for nid, shared_emb in shared_embeddings.items():
                out = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                    answer_marker=SFT_ANSWER_MARKER,
                    answer_marker_mode="last",
                )
                neuron_embeddings[nid] = out[0].to(DEVICE)
                if targets is None:
                    targets = out[1].to(DEVICE)
                    mask = out[2].to(DEVICE)
                    sft_mask = out[3].to(DEVICE)

            if use_train_path:
                result = ensemble.forward_train(
                    neuron_embeddings=neuron_embeddings,
                    n_rounds=2, fusion_mode="soft",
                    targets=targets,
                    target_domain=DEFAULT_DOMAIN,
                )
                fused = result["fused_logits"]
            else:
                result = ensemble.forward(
                    neuron_embeddings=neuron_embeddings,
                    return_logits=True, fusion_mode="soft",
                )
                fused = result.get("weighted_logits")

            shift_logits = fused[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_sft = sft_mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~(shift_mask & shift_sft)] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100, reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (shift_mask & shift_sft).sum().item()

    avg = total_loss / max(total_tokens, 1)
    return avg, math.exp(min(avg, 20))


def main():
    domain_sp = load_domain_tokenizer(DEFAULT_DOMAIN)
    general_sp = load_general_tokenizer()

    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)

    # 加载 cross_spec 权重（用完整 ckpt：含 side_channels/scale_bias/body）
    from scripts.training.eval_dialogue import load_cross_spec_weights
    ckpt = os.path.join("data", "neurons", "cross_spec_dialogue.ckpt.pt")
    load_cross_spec_weights(ensemble, "dialogue", ckpt)
    print(f"\n最终模型已加载（dialogue weights）", flush=True)

    # 训练数据（同管线）
    train_texts = load_dialogue_texts_multi("data/simple_zh", max_texts=100, max_answer_chars=150)
    print(f"训练分布样本: {len(train_texts)} 条", flush=True)

    # eval 数据（原始 alpaca-zh，无筛选——eval_dialogue 用）
    eval_path = os.path.join("data", "simple_zh", "alpaca_zh_sft_clean.jsonl")
    eval_texts = load_dialogue_texts(eval_path, max_texts=30)
    print(f"eval 分布样本: {len(eval_texts)} 条", flush=True)

    for name, texts in [("训练分布", train_texts), ("eval 分布", eval_texts)]:
        for path_name, use_train in [("forward_train(同训练路径)", True), ("forward(推理路径)", False)]:
            try:
                avg, ppl = compute_answer_ppl(
                    neurons, shared_embeddings, ensemble, domain_sp, general_sp,
                    texts, use_train_path=use_train,
                )
                print(f"  [{name}] {path_name}: loss={avg:.4f} PPL={ppl:.1f}", flush=True)
            except Exception as e:
                print(f"  [{name}] {path_name} 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
