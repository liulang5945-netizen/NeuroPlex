"""联合微调 side_channels：冻结神经元核心参数，仅训练突触通道。

4 个已训练的 zh_aug0~3 神经元，每对之间有 excite side_channel（随机初始化）。
此脚本端到端训练 side_channels 参数，让突触通道学会正确转译 peer 信号。

策略：
  1. 加载 4 个已训练神经元 + 各自的 shared_embedding
  2. 冻结所有 neuron 参数 + shared_embedding
  3. 仅 side_channels 的 Linear 参数可训练
  4. 用 ensemble.forward(max_rounds=2) 获取协作 logits
  5. CE loss 反向传播更新 side_channels

Usage:
    python -u scripts/training/finetune_side_channels.py
"""
from __future__ import annotations

import math
import os
import sys
import time

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
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR,
)
from scripts.training.train_cortex_joint import load_simple_zh_texts

DOMAIN = "zh"
NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]
DEVICE = "cpu"


def load_neuron_with_embedding(nid, cfg):
    """加载单个神经元及其 shared_embedding。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)

    shared_emb = nn.Embedding(256000, 512)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE)

    result = ckpt.get("result", {})
    print(f"  [{nid}] best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb


def main():
    print("=" * 60, flush=True)
    print("联合微调 side_channels", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载
    print("\n[1] 加载神经元...", flush=True)
    cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid, cfg)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 2. 建立 side_channels
    print("\n[2] 建立 side_channels...", flush=True)
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
        print(f"  [{post_id}] {len(neurons[post_id].excite_channels)} excite channels", flush=True)

    # 3. 冻结核心参数，仅 side_channels 可训练
    print("\n[3] 冻结核心参数...", flush=True)
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        # 解冻 side_channels
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        neuron.train()

    for emb in shared_embeddings.values():
        for p in emb.parameters():
            p.requires_grad = False
        emb.eval()

    # 统计可训练参数
    trainable = 0
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            trainable += sum(p.numel() for p in ch.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,} (side_channels only)", flush=True)

    # 4. 创建 ensemble
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    # 5. 加载训练数据
    print("\n[4] 加载训练数据...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=2000)
    print(f"  训练集: {len(texts)} 条文本", flush=True)

    # 6. 训练循环
    print("\n[5] 开始训练 side_channels...", flush=True)
    lr = 1e-3
    # 只收集 side_channels 参数
    side_params = []
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            side_params.extend(ch.parameters())
        for ch in neuron.inhibit_channels.values():
            side_params.extend(ch.parameters())
    optimizer = torch.optim.Adam(side_params, lr=lr)

    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    LOG_EVERY = 50

    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        import random
        random.shuffle(texts)
        epoch_loss = 0.0
        epoch_tokens = 0

        for i in range(0, len(texts) - BATCH_SIZE, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            # 编码每个神经元的输入
            neuron_embeddings = {}
            targets = None
            mask = None
            valid = True
            for nid, shared_emb in shared_embeddings.items():
                emb_out, tgt, msk = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_emb,
                )
                neuron_embeddings[nid] = emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)

            optimizer.zero_grad()

            # Ensemble forward (max_rounds=2, side_channels 生效)
            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )

            if "weighted_logits" not in result:
                valid = False

            if valid:
                fused_logits = result["weighted_logits"]
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
                n_tokens = shift_mask.sum().item()
                loss = loss / max(n_tokens, 1)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * n_tokens
                epoch_tokens += n_tokens
                total_steps += 1

                if total_steps % LOG_EVERY == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    ppl = math.exp(min(avg_loss, 20))
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} step {total_steps}: "
                          f"loss={avg_loss:.4f} PPL={ppl:.1f}", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f}", flush=True)

    # 7. 保存微调后的 side_channels
    print("\n[6] 保存 side_channels...", flush=True)
    save_path = os.path.join(OUTPUT_DIR, "side_channels_finetuned.pt")
    side_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
    torch.save(side_state, save_path)
    print(f"  已保存: {save_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("微调完成。运行 eval_aug_joint.py 查看效果。", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
