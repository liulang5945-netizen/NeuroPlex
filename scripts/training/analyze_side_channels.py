"""分析 side_channels 利用率：找出死通道。

检查每条 channel 的：
1. 权重矩阵范数（是否萎缩到接近 0）
2. 投影输出幅度（forward 后的 proj 范数）
3. 梯度更新幅度（如果有 checkpoint）

Usage:
    python -u scripts/training/analyze_side_channels.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config, NeuronGeometry,
)
from taiji.resonance.topology import (
    build_topology, establish_topology_channels,
    infer_topology_from_state, topology_detail,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts,
)
from scripts.training.finetune_side_channels import load_neuron_with_embedding
from scripts.training.experiment_config import ZH_COMPACT_NEURON_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN

DEVICE = "cpu"


def main():
    parser = argparse.ArgumentParser(description="分析 side_channels 利用率")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print("=" * 60)
    print("side_channels 利用率分析")
    print("=" * 60)

    cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid, cfg)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # S7: 建立 side_channels（拓扑驱动，优先从 checkpoint 推断）
    ckpt_path = os.path.join(OUTPUT_DIR, "side_channels_finetuned.ckpt.pt")
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = None
    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        side_state_peek = ckpt.get("side_channels_state", {})
        if side_state_peek and isinstance(side_state_peek, dict):
            topology = infer_topology_from_state(side_state_peek)
            print(f"[topology] 从 checkpoint 推断: {topology_detail(topology, neurons)}")
    if topology is None or not any(topology.values() if topology else []):
        topology = build_topology(neurons, geometry, mode="hybrid")
        print(f"[topology] 回退到 hybrid: {topology_detail(topology, neurons)}")
    establish_topology_channels(neurons, topology, geometry)

    # 加载已训练的 side_channels（如果存在）
    if ckpt is not None:
        side_state = ckpt["side_channels_state"]
        for nid, neuron in neurons.items():
            if nid not in side_state:
                continue
            for pid, ch_state in side_state[nid].get("excite", {}).items():
                if pid in neuron.excite_channels:
                    neuron.excite_channels[pid].load_state_dict(ch_state)
        print(f"已加载训练 checkpoint: epoch={ckpt['epoch']}, steps={ckpt['total_steps']}")
        print(f"最近 PPL 趋势: {ckpt['loss_history'][-3:] if ckpt['loss_history'] else 'N/A'}")
    else:
        print("未找到训练 checkpoint，分析随机初始化的 side_channels")

    # 冻结
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        neuron.eval()
    for emb in shared_embeddings.values():
        emb.eval()

    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=40)

    BATCH_SIZE = 4
    batch_texts = texts[:BATCH_SIZE]
    neuron_embeddings = {}
    for nid, shared_emb in shared_embeddings.items():
        emb_out, _, _ = batch_align_and_embed(batch_texts, domain_sp, general_sp, shared_emb)
        neuron_embeddings[nid] = emb_out.to(DEVICE)

    print("\n--- 1. 权重矩阵分析 ---")
    print(f"{'通道':<20} {'weight_norm':>12} {'weight_max':>12} {'weight_std':>12}")
    print("-" * 60)
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            ch = neurons[post_id].excite_channels[pre_id]
            w = ch.weight.data
            print(f"{pre_id}->{post_id:<12} {w.norm().item():12.4f} {w.abs().max().item():12.4f} {w.std().item():12.4f}")

    print("\n--- 2. 前向投影输出分析 ---")
    print("每条 channel 的 proj 输出范数（4 条样本平均）：")
    print(f"{'通道':<20} {'proj_norm':>12} {'proj_max':>12} {'proj_mean':>12}")
    print("-" * 60)

    with torch.no_grad():
        # 先获取每个神经元的 field_vector
        field_vectors = {}
        for nid in NEURON_IDS:
            r = neurons[nid].forward(neuron_embeddings[nid], return_logits=False)
            field_vectors[nid] = r["field_vector"]

        for post_id in NEURON_IDS:
            for pre_id in NEURON_IDS:
                if pre_id == post_id:
                    continue
                ch = neurons[post_id].excite_channels[pre_id]
                sig = field_vectors[pre_id]
                proj = ch(sig)
                print(f"{pre_id}->{post_id:<12} {proj.norm().item():12.4f} {proj.abs().max().item():12.4f} {proj.abs().mean().item():12.6f}")

    print("\n--- 3. 调制效果分析（gate 值） ---")
    print("gate = 1 + tanh(proj)，如果 gate ≈ 1.0 说明通道无效")
    print(f"{'通道':<20} {'gate_mean':>12} {'gate_max':>12} {'gate_min':>12} {'有效':>6}")
    print("-" * 70)

    with torch.no_grad():
        for post_id in NEURON_IDS:
            for pre_id in NEURON_IDS:
                if pre_id == post_id:
                    continue
                ch = neurons[post_id].excite_channels[pre_id]
                sig = field_vectors[pre_id]
                proj = ch(sig)
                gate = 1.0 + torch.tanh(proj.unsqueeze(1))  # [B, L, hidden]
                g_mean = gate.mean().item()
                g_max = gate.max().item()
                g_min = gate.min().item()
                # gate 偏离 1.0 的程度
                deviation = (gate - 1.0).abs().mean().item()
                effective = "✓" if deviation > 0.01 else "✗"
                print(f"{pre_id}->{post_id:<12} {g_mean:12.4f} {g_max:12.4f} {g_min:12.4f} {effective:>6}")

    print("\n--- 4. 聚合调制效果（所有通道累加） ---")
    print("每神经元接收的总 excite_sum：")
    with torch.no_grad():
        for post_id in NEURON_IDS:
            excite_sum = None
            for pre_id in NEURON_IDS:
                if pre_id == post_id:
                    continue
                ch = neurons[post_id].excite_channels[pre_id]
                sig = field_vectors[pre_id]
                proj = ch(sig)
                excite_sum = proj if excite_sum is None else excite_sum + proj
            gate = 1.0 + torch.tanh(excite_sum.unsqueeze(1))
            dev = (gate - 1.0).abs().mean().item()
            print(f"  {post_id}: excite_sum_norm={excite_sum.norm().item():.4f}, "
                  f"gate_deviation={dev:.4f}, gate_range=[{gate.min().item():.3f}, {gate.max().item():.3f}]")

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
