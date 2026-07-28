"""联合微调 side_channels：冻结神经元核心参数，仅训练突触通道。

4 个已训练的 zh_aug0~3 神经元，每对之间有 excite side_channel（随机初始化）。
此脚本端到端训练 side_channels 参数，让突触通道学会正确转译 peer 信号。

策略：
  1. 加载 4 个已训练神经元 + 各自的 shared_embedding
  2. 冻结所有 neuron 参数 + shared_embedding
  3. 仅 side_channels 的 Linear 参数可训练
  4. 用 ensemble.forward(max_rounds=2) 获取协作 logits
  5. CE loss 反向传播更新 side_channels

工程保障：
  - stdout 同时写入日志文件（logs/finetune_side_channels_YYYYMMDD_HHMMSS.log）
  - 每个 epoch 结束保存 checkpoint（含 optimizer + side_channels + loss history）
  - 支持 --resume 从最新 checkpoint 断点续训
  - 训练趋势可监控（loss_history 字段）

Usage:
    # 从头训练
    python -u scripts/training/finetune_side_channels.py

    # 断点续训
    python -u scripts/training/finetune_side_channels.py --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

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

# 日志与 checkpoint 路径
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
CKPT_PATH = os.path.join(OUTPUT_DIR, "side_channels_finetuned.ckpt.pt")  # 训练用 checkpoint
FINAL_PATH = os.path.join(OUTPUT_DIR, "side_channels_finetuned.pt")     # 最终交付产物


class TeeLogger:
    """同时输出到 stdout 和日志文件。"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fp = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg: str):
        sys.__stdout__.write(msg)
        self.fp.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self.fp.flush()

    def close(self):
        self.fp.close()


def save_checkpoint(path, epoch, total_steps, optimizer, neurons, loss_history):
    """保存训练 checkpoint，支持断点续训。"""
    side_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
    torch.save(
        {
            "epoch": epoch,
            "total_steps": total_steps,
            "optimizer_state": optimizer.state_dict(),
            "side_channels_state": side_state,
            "loss_history": loss_history,
            "saved_at": datetime.now().isoformat(),
        },
        path,
    )


def load_checkpoint(path, optimizer, neurons):
    """加载 checkpoint，恢复 side_channels、optimizer、训练进度。"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    side_state = ckpt["side_channels_state"]
    for nid, neuron in neurons.items():
        if nid not in side_state:
            continue
        for pid, ch_state in side_state[nid].get("excite", {}).items():
            if pid in neuron.excite_channels:
                neuron.excite_channels[pid].load_state_dict(ch_state)
        for pid, ch_state in side_state[nid].get("inhibit", {}).items():
            if pid in neuron.inhibit_channels:
                neuron.inhibit_channels[pid].load_state_dict(ch_state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["total_steps"], ckpt.get("loss_history", [])


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 断点续训")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_texts", type=int, default=10000)
    args = parser.parse_args()

    # 1. 设置日志 tee
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"finetune_side_channels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print("联合微调 side_channels", flush=True)
    print(f"日志: {log_path}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 2. 加载神经元
    print("\n[1] 加载神经元...", flush=True)
    cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid, cfg)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 3. 建立 side_channels
    print("\n[2] 建立 side_channels...", flush=True)
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
        print(f"  [{post_id}] {len(neurons[post_id].excite_channels)} excite channels", flush=True)

    # 4. 冻结核心参数，仅 side_channels 可训练
    print("\n[3] 冻结核心参数...", flush=True)
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
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

    trainable = 0
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            trainable += sum(p.numel() for p in ch.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,} (side_channels only)", flush=True)

    # 5. 创建 ensemble
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    # 6. 加载训练数据
    print("\n[4] 加载训练数据...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=args.max_texts)
    print(f"  训练集: {len(texts)} 条文本", flush=True)

    # 7. 训练循环
    print("\n[5] 开始训练 side_channels...", flush=True)
    side_params = []
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            side_params.extend(ch.parameters())
        for ch in neuron.inhibit_channels.values():
            side_params.extend(ch.parameters())
    optimizer = torch.optim.Adam(side_params, lr=args.lr)

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LOG_EVERY = 50

    total_steps = 0
    start_epoch = 0
    loss_history = []  # [{step, epoch, loss, ppl, tokens}]

    # 断点续训
    if args.resume and os.path.exists(CKPT_PATH):
        print(f"\n[resume] 加载 checkpoint: {CKPT_PATH}", flush=True)
        start_epoch, total_steps, loss_history = load_checkpoint(
            CKPT_PATH, optimizer, neurons,
        )
        # start_epoch 是上次完成的 epoch 编号，从下一个开始
        print(f"  已恢复: epoch={start_epoch} (从 epoch {start_epoch+1} 继续), "
              f"total_steps={total_steps}, loss_history={len(loss_history)} 条", flush=True)
        start_epoch = start_epoch + 1
    elif args.resume:
        print(f"\n[resume] 未找到 checkpoint ({CKPT_PATH})，从头开始", flush=True)

    import random
    random.seed(42)

    for epoch in range(start_epoch, NUM_EPOCHS):
        random.shuffle(texts)
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start_time = time.time()

        for i in range(0, len(texts) - BATCH_SIZE, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

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
                    elapsed = time.time() - epoch_start_time
                    steps_done = (i + BATCH_SIZE) / BATCH_SIZE
                    steps_total = (len(texts) - BATCH_SIZE) / BATCH_SIZE
                    eta = elapsed / max(steps_done, 1) * (steps_total - steps_done)
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} step {total_steps}: "
                          f"loss={avg_loss:.4f} PPL={ppl:.1f} "
                          f"[{steps_done:.0f}/{steps_total:.0f} ETA {eta/60:.1f}min]",
                          flush=True)
                    loss_history.append({
                        "step": total_steps,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "ppl": ppl,
                        "tokens": epoch_tokens,
                    })

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        epoch_elapsed = time.time() - epoch_start_time
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f} "
              f"耗时 {epoch_elapsed/60:.1f} min", flush=True)

        # 每 epoch 保存 checkpoint（断点续训用，含 optimizer state）
        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons, loss_history)
        print(f"  [checkpoint 已保存] {CKPT_PATH}", flush=True)

        # 同步保存最终产物（纯 side_state 格式，下游 eval 直接加载）
        # 即使后续 epoch 中断也有可用模型
        side_state = {}
        for nid, neuron in neurons.items():
            side_state[nid] = {
                "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
                "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
            }
        torch.save(side_state, FINAL_PATH)
        print(f"  [final 已保存] {FINAL_PATH}", flush=True)

        # 趋势分析：最近 5 个 log 点
        recent = loss_history[-5:]
        if len(recent) >= 2:
            first_ppl = recent[0]["ppl"]
            last_ppl = recent[-1]["ppl"]
            delta = last_ppl - first_ppl
            print(f"  [趋势] 最近 5 点 PPL: {first_ppl:.1f} -> {last_ppl:.1f} "
                  f"(Δ={delta:+.1f}, {'下降' if delta < 0 else '上升/停滞'})", flush=True)

    # 8. 最终保存
    print("\n[6] 训练完成，最终保存...", flush=True)
    side_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
    torch.save(side_state, FINAL_PATH)
    print(f"  已保存: {FINAL_PATH}", flush=True)

    # 保存 loss_history 为 JSON 供分析
    history_path = os.path.join(LOG_DIR, "finetune_side_channels_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path} ({len(loss_history)} 条记录)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("微调完成。运行 eval_aug_joint.py 查看效果。", flush=True)
    print("=" * 60, flush=True)

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
