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

from neuroplex.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config, NeuronGeometry,
)
from neuroplex.resonance.topology import (
    build_topology, establish_topology_channels, topology_detail,
)
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts, create_shared_embedding,
    make_wsd_scheduler, build_muon_adamw_optimizers,
    load_dialogue_texts_multi,
)
from scripts.training.experiment_config import ZH_COMPACT_NEURON_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER
from scripts.training.data_augmentation import DialogueAugmenter

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


def save_checkpoint(path, epoch, total_steps, optimizer, neurons, loss_history,
                    adamw_optimizer=None, scheduler=None,
                    body_optimizer=None, body_scheduler=None,
                    shared_embeddings=None, ensemble=None):
    """保存训练 checkpoint，支持断点续训。"""
    side_state = {}
    scale_bias_state = {}
    body_state = {}  # S8: unfrozen neuron body params
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
        # 保存 scale 和 bias 参数
        sb = {}
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                sb[name] = p.data.clone()
        for name, buf in neuron.named_buffers():
            if "bias_" in name:
                sb[name] = buf.clone()
        scale_bias_state[nid] = sb
        # S8: 保存 body 参数（非 side_channels，非 scale，requires_grad=True）
        bp = {}
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            bp[name] = p.data.clone()
        if bp:
            body_state[nid] = bp
    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer_state": optimizer.state_dict(),
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    if body_state:
        ckpt["body_state"] = body_state
    # R1: 场门控权重随产物保存（W_cond 训练闭环）
    if ensemble is not None and hasattr(ensemble._field, "W_cond"):
        ckpt["field_w_cond"] = ensemble._field.W_cond.data.clone()
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if body_optimizer is not None:
        ckpt["body_optimizer_state"] = body_optimizer.state_dict()
    if body_scheduler is not None:
        ckpt["body_scheduler_state"] = body_scheduler.state_dict()
    # S8: 保存 shared_embedding（如果训练）
    if shared_embeddings is not None:
        emb_state = {}
        for nid, emb in shared_embeddings.items():
            if any(p.requires_grad for p in emb.parameters()):
                emb_state[nid] = {k: v.detach().clone() for k, v in emb.state_dict().items()}
        if emb_state:
            ckpt["shared_embedding_state"] = emb_state
    torch.save(ckpt, path)


def load_checkpoint(path, optimizer, neurons, adamw_optimizer=None, scheduler=None,
                    body_optimizer=None, body_scheduler=None,
                    shared_embeddings=None):
    """加载 checkpoint，恢复 side_channels、scale/bias、body、optimizer、训练进度。"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    side_state = ckpt["side_channels_state"]
    scale_bias_state = ckpt.get("scale_bias_state", {})
    body_state = ckpt.get("body_state", {})  # S8: 可能不存在（旧 ckpt）
    for nid, neuron in neurons.items():
        if nid not in side_state:
            continue
        for pid, ch_state in side_state[nid].get("excite", {}).items():
            if pid in neuron.excite_channels:
                neuron.excite_channels[pid].load_state_dict(ch_state)
        for pid, ch_state in side_state[nid].get("inhibit", {}).items():
            if pid in neuron.inhibit_channels:
                neuron.inhibit_channels[pid].load_state_dict(ch_state)
        # 恢复 scale 和 bias 参数
        if nid in scale_bias_state:
            sb = scale_bias_state[nid]
            for name, p in neuron.named_parameters():
                if name in sb and "scale_" in name:
                    p.data.copy_(sb[name])
            for name, buf in neuron.named_buffers():
                if name in sb and "bias_" in name:
                    buf.copy_(sb[name])
        # S8: 恢复 body 参数
        if nid in body_state:
            for name, p in neuron.named_parameters():
                if name in body_state[nid]:
                    p.data.copy_(body_state[nid][name])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if adamw_optimizer is not None and "adamw_optimizer_state" in ckpt:
        adamw_optimizer.load_state_dict(ckpt["adamw_optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if body_optimizer is not None and "body_optimizer_state" in ckpt:
        body_optimizer.load_state_dict(ckpt["body_optimizer_state"])
    if body_scheduler is not None and "body_scheduler_state" in ckpt:
        body_scheduler.load_state_dict(ckpt["body_scheduler_state"])
    # S8: 恢复 shared_embedding
    if shared_embeddings is not None and "shared_embedding_state" in ckpt:
        emb_state = ckpt["shared_embedding_state"]
        for nid, emb in shared_embeddings.items():
            if nid in emb_state:
                emb.load_state_dict(emb_state[nid])
    return ckpt["epoch"], ckpt["total_steps"], ckpt.get("loss_history", [])


def load_neuron_with_embedding(nid, cfg, debug=False):
    """加载单个神经元及其 shared_embedding。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    missing, unexpected = neuron.load_state_dict(ckpt["state_dict"], strict=False)
    if debug and (missing or unexpected):
        print(f"  [{nid}] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}", flush=True)
        print(f"  [{nid}] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}", flush=True)

    shared_emb = create_shared_embedding(DEVICE)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE)

    result = ckpt.get("result", {})
    print(f"  [{nid}] best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb


def build_final_artifact(neurons, shared_embeddings) -> dict:
    """S8: 构建最终交付产物，含 side_channels + body + emb。

    下游 eval 脚本加载此产物后，应用全部微调结果（不再丢失 body/emb）。
    """
    side_state = {}
    body_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
        bp = {}
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            bp[name] = p.data.clone()
        if bp:
            body_state[nid] = bp
    artifact = {"side_channels": side_state}
    if body_state:
        artifact["body_state"] = body_state
    if shared_embeddings is not None:
        emb_state = {}
        for nid, emb in shared_embeddings.items():
            if any(p.requires_grad for p in emb.parameters()):
                emb_state[nid] = {k: v.detach().clone() for k, v in emb.state_dict().items()}
        if emb_state:
            artifact["shared_embedding_state"] = emb_state
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 断点续训")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_texts", type=int, default=100000)
    parser.add_argument("--data", type=str, default="dialogue",
                        choices=["dialogue", "simple_zh"],
                        help="S5: dialogue=多文件合并对话数据, simple_zh=作文数据")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--topology", default="hybrid",
                        choices=["full", "knn", "hub_spoke", "hybrid"],
                        help="S7: side_channels 拓扑模式 (default: hybrid)")
    parser.add_argument("--topology_k", type=int, default=3,
                        help="k-NN 拓扑的 k 值 (仅 knn 模式)")
    parser.add_argument("--unfreeze_layers", type=int, default=2,
                        help="S8: 解冻最后 N 层 transformer + norm + lm_head + field_write (0=全冻结)")
    parser.add_argument("--train_embedding", action="store_true",
                        help="S8: 训练 shared_embedding（默认冻结）")
    parser.add_argument("--body_lr_ratio", type=float, default=0.1,
                        help="S8: body 参数学习率比例 (相对 args.lr)")
    parser.add_argument("--augment", action="store_true",
                        help="T4: 启用数据增强（模板改写 + 多轮拼接）")
    parser.add_argument("--aug_rewrite_prob", type=float, default=0.5,
                        help="T4: 模板改写概率")
    parser.add_argument("--aug_multi_turn_prob", type=float, default=0.4,
                        help="T4: 多轮拼接概率")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

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

    # 3. 建立 side_channels（S7: 拓扑驱动替代全连接 mesh）
    print(f"\n[2] 建立 side_channels (topology={args.topology})...", flush=True)
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(
        neurons, geometry, mode=args.topology, k=args.topology_k,
    )
    print(f"  {topology_detail(topology, neurons)}", flush=True)
    stats = establish_topology_channels(neurons, topology, geometry)
    for nid, n_ch in stats.items():
        print(f"  [{nid}] {n_ch} excite channels", flush=True)

    # 4. 冻结核心参数，仅 side_channels + scale + (S8: 最后N层) 可训练
    print(f"\n[3] 冻结核心参数 (unfreeze_layers={args.unfreeze_layers})...", flush=True)
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        # side_channels 可训练
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        # scale 参数可训练
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        # S8: 解冻最后 N 层 transformer + norm + lm_head + field_write
        if args.unfreeze_layers > 0:
            n_layers = len(neuron.layers)
            unfreeze_from = max(0, n_layers - args.unfreeze_layers)
            for i in range(unfreeze_from, n_layers):
                for p in neuron.layers[i].parameters():
                    p.requires_grad = True
            # final norm
            for p in neuron.norm.parameters():
                p.requires_grad = True
            # lm_head（如果存在）
            if hasattr(neuron, 'lm_head') and neuron.lm_head is not None:
                for p in neuron.lm_head.parameters():
                    p.requires_grad = True
            # field_write（让场写入适配协作动态，C6 多头兼容）
            for p in neuron.get_field_write_parameters():
                p.requires_grad = True
            # R2（REMEDIATION_PLAN 2026-08-14）：field_read 解冻训练——
            # round2+ 场条件化读取路径此前恒为随机初始化（审计发现），
            # 解冻使其成为可学习路径（落入 body 低 lr，温柔更新）。
            for p in neuron.get_field_read_parameters():
                p.requires_grad = True
        neuron.train()

    # S8: shared_embedding 可选训练
    for emb in shared_embeddings.values():
        if args.train_embedding:
            for p in emb.parameters():
                p.requires_grad = True
            emb.train()
        else:
            for p in emb.parameters():
                p.requires_grad = False
            emb.eval()

    # 统计可训练参数
    trainable_side = 0
    trainable_body = 0
    trainable_emb = 0
    for nid, neuron in neurons.items():
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]) or "scale_" in name:
                trainable_side += p.numel()
            else:
                trainable_body += p.numel()
    for emb in shared_embeddings.values():
        trainable_emb += sum(p.numel() for p in emb.parameters() if p.requires_grad)
    print(f"  可训练: side_channels={trainable_side:,}, body={trainable_body:,}, emb={trainable_emb:,}", flush=True)

    # 5. 创建 ensemble
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    # R1（REMEDIATION_PLAN 2026-08-14）：场门控 W_cond 参与训练
    # （训练-推理评分口径统一后，W_cond 需要梯度才能成为可学习门控）
    ensemble._field.W_cond.requires_grad = True

    # 6. 加载训练数据（S5: 支持对话数据扩充）
    print("\n[4] 加载训练数据...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    if args.data == "dialogue":
        dialogue_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "simple_zh",
        )
        texts = load_dialogue_texts_multi(dialogue_dir, max_texts=args.max_texts)
        print(f"  训练集(多文件合并对话): {len(texts)} 条对话", flush=True)
    else:
        texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=args.max_texts)
        print(f"  训练集(simple_zh): {len(texts)} 条文本", flush=True)

    # T4: 数据增强器（在线模板改写 + 多轮拼接）
    augmenter = None
    if args.augment:
        augmenter = DialogueAugmenter(
            rewrite_prob=args.aug_rewrite_prob,
            multi_turn_prob=args.aug_multi_turn_prob,
        )
        augmenter.set_context_pool(texts)  # 多轮拼接的上下文池
        print(f"  [T4] 数据增强启用: 改写={args.aug_rewrite_prob}, "
              f"多轮拼接={args.aug_multi_turn_prob}", flush=True)

    # 7. 训练循环
    print("\n[5] 开始训练 side_channels...", flush=True)
    # Muon + AdamW 混合优化器（借鉴 DeepSeek V4 / GLM-5.2）
    # - 2D 权重矩阵（Linear weight）用 Muon：Newton-Schulz 正交化突破 Adam 局部最小值
    # - 1D 参数（bias/LayerNorm）用 AdamW：Muon 仅适用于 2D
    # - scale 参数（0D scalar）用 AdamW
    # S8: body 参数（最后N层）用单独 AdamW，lr = args.lr * body_lr_ratio
    muon_params = []   # 2D weight (side_channels only)
    adamw_params = []  # 1D bias/norm + 0D scale (side_channels only)
    body_params = []   # S8: unfrozen neuron body params
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
        # scale 参数（0D scalar，可学习）
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if "scale_" in name and p.ndim == 0:
                adamw_params.append(p)
        # S8: body 参数（非 side_channels，非 scale）
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            body_params.append(p)

    # R1: 场门控 W_cond（2D → Muon）
    if hasattr(ensemble._field, "W_cond") and ensemble._field.W_cond.requires_grad:
        muon_params.append(ensemble._field.W_cond)

    # S8: shared_embedding 参数（如果训练）
    emb_params = []
    if args.train_embedding:
        for emb in shared_embeddings.values():
            emb_params.extend([p for p in emb.parameters() if p.requires_grad])

    # Muon + AdamW 混合优化器（配置抽取到 utils.build_muon_adamw_optimizers）
    muon_lr = args.lr
    optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, lr=muon_lr,
    )
    print(f"  Muon 参数: {sum(p.numel() for p in muon_params):,} (2D weight, lr={muon_lr})", flush=True)
    if adamw_optimizer is not None:
        print(f"  AdamW 参数: {sum(p.numel() for p in adamw_params):,} (1D bias/scale, lr={muon_lr})", flush=True)
    else:
        print(f"  AdamW 参数: 0 (无 1D 参数)", flush=True)

    # S8: body + emb 优化器（低 lr 温柔微调，避免破坏预训练表示）
    body_optimizer = None
    body_scheduler = None
    all_body_params = body_params + emb_params
    if all_body_params:
        body_lr = args.lr * args.body_lr_ratio
        body_optimizer = torch.optim.AdamW(all_body_params, lr=body_lr, weight_decay=0.1)
        print(f"  Body+Emb 参数: {sum(p.numel() for p in all_body_params):,} (lr={body_lr}, ratio={args.body_lr_ratio})", flush=True)

    # 学习率调度：WSD（warmup + stable + cosine decay）
    # 修复 Playbook #4(warmup) 和 #5(decay) 合规项，公式抽取到 utils.make_wsd_scheduler
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    total_est_steps = NUM_EPOCHS * ((len(texts) - BATCH_SIZE) // BATCH_SIZE)
    warmup_steps = 100
    decay_ratio = 0.8
    scheduler = make_wsd_scheduler(
        optimizer, num_steps=total_est_steps,
        warmup_steps=warmup_steps, decay_ratio=decay_ratio,
    )
    # S8: body scheduler（与主调度同步）
    if body_optimizer is not None:
        body_scheduler = make_wsd_scheduler(
            body_optimizer, num_steps=total_est_steps,
            warmup_steps=warmup_steps, decay_ratio=decay_ratio,
        )
    decay_start = max(warmup_steps + 1, int(total_est_steps * decay_ratio))
    print(f"  LR 调度: warmup={warmup_steps}步, decay 从 {decay_start}/{total_est_steps} 步开始", flush=True)

    LOG_EVERY = 50
    BIAS_UPDATE_EVERY = 50  # Auxiliary-loss-free balancing bias 更新频率
    BIAS_UPDATE_RATE = 0.1  # bias 更新步长

    total_steps = 0
    start_epoch = 0
    loss_history = []  # [{step, epoch, loss, ppl, tokens}]

    # 断点续训
    if args.resume and os.path.exists(CKPT_PATH):
        print(f"\n[resume] 加载 checkpoint: {CKPT_PATH}", flush=True)
        start_epoch, total_steps, loss_history = load_checkpoint(
            CKPT_PATH, optimizer, neurons, adamw_optimizer, scheduler,
            body_optimizer, body_scheduler, shared_embeddings,
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
        # T4: 每 epoch 重置增强种子（epoch 间变体不同，epoch 内可复现）
        if augmenter is not None:
            augmenter.set_epoch(epoch)

        for i in range(0, len(texts) - BATCH_SIZE, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            # T4: 在线数据增强（模板改写 + 多轮拼接）
            if augmenter is not None:
                batch_texts = [augmenter.augment(t) for t in batch_texts]

            neuron_embeddings = {}
            targets = None
            mask = None
            sft_mask = None
            valid = True
            for nid, shared_emb in shared_embeddings.items():
                # S3: 传入 answer_marker，获取 sft_mask（只对 answer 部分计算 loss）
                emb_out, tgt, msk, sft = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_emb,
                    answer_marker=SFT_ANSWER_MARKER,
                    answer_marker_mode="last",  # T4: 多轮精确 masking
                )
                neuron_embeddings[nid] = emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)
                    sft_mask = sft.to(DEVICE)

            optimizer.zero_grad()
            if adamw_optimizer is not None:
                adamw_optimizer.zero_grad()
            # S8: body_optimizer 也要 zero_grad（否则 body 参数梯度会无限累积）
            if body_optimizer is not None:
                body_optimizer.zero_grad()

            # S1: 改用 forward_train（全可微多轮共振路径）
            # 让 side_channels + field_state 在训练中真正生效
            # 注：field_conditioning=False 是推理路径选项，forward_train 中
            # round 2+ 默认传 field_state，但 field_read_layers 是否被使用
            # 取决于 neuron.forward 内 round_num>1 的判断
            # C12: 传入 targets 启用 contrastive_loss（共振分与 NLL 排序对齐）
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings,
                n_rounds=2,
                fusion_mode="soft",
                return_individual_logits=(total_steps == 0),  # 首步返回用于 debug
                targets=targets,
            )

            # Debug: 首次 forward 打印 fusion 权重和各神经元单独 PPL
            if total_steps == 0 and "individual_logits" in result:
                print("\n  [debug] Fusion 权重和各神经元单独 PPL:", flush=True)
                weights = result.get("weights")
                if weights is not None:
                    for i, nid in enumerate(NEURON_IDS):
                        w_val = weights[i].item() if weights.dim() == 1 else weights[i]
                        print(f"    {nid}: fusion_weight={w_val:.4f}", flush=True)
                # 计算各神经元单独 PPL
                for nid, logits in result["individual_logits"].items():
                    shift_l = logits[:, :-1, :].contiguous()
                    shift_t = targets[:, 1:].contiguous()
                    shift_m = mask[:, 1:].contiguous()
                    shift_t = shift_t.clone()
                    shift_t[~shift_m] = -100
                    n_tok = shift_m.sum().item()
                    if n_tok > 0:
                        loss_nid = F.cross_entropy(
                            shift_l.view(-1, shift_l.size(-1)),
                            shift_t.view(-1),
                            ignore_index=-100,
                            reduction="sum",
                        ) / n_tok
                        print(f"    {nid}: solo_ppl={math.exp(min(loss_nid.item(), 20)):.1f}", flush=True)
                # 协作 PPL
                fused = result["fused_logits"]
                shift_l = fused[:, :-1, :].contiguous()
                shift_t = targets[:, 1:].contiguous()
                shift_m = mask[:, 1:].contiguous()
                shift_t = shift_t.clone()
                shift_t[~shift_m] = -100
                n_tok = shift_m.sum().item()
                if n_tok > 0:
                    loss_fused = F.cross_entropy(
                        shift_l.view(-1, shift_l.size(-1)),
                        shift_t.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ) / max(n_tok, 1)
                    print(f"    协作 PPL={math.exp(min(loss_fused.item(), 20)):.1f}", flush=True)
                print(f"    n_rounds={result.get('n_rounds', '?')}", flush=True)
                print()

            valid = "fused_logits" in result

            if valid:
                fused_logits = result["fused_logits"]
                shift_logits = fused_logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                # S3: SFT answer masking — 只对 answer 部分计算 loss
                shift_sft_mask = sft_mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~(shift_mask & shift_sft_mask)] = -100
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_tokens = (shift_mask & shift_sft_mask).sum().item()
                ce_loss = ce_loss / max(n_tokens, 1)

                # 多任务 loss：CE + 负载均衡 + 多样性 + 对比约束
                balance_loss = result["balance_loss"]
                diversity_loss = result["diversity_loss"]
                contrastive_loss = result.get("contrastive_loss", torch.tensor(0.0))
                balance_weight = 0.01
                diversity_weight = 0.05
                contrastive_weight = 0.1  # C12: 弱约束，让共振分学习公平性
                loss = (
                    ce_loss
                    + balance_weight * balance_loss
                    + diversity_weight * diversity_loss
                    + contrastive_weight * contrastive_loss
                )

                loss.backward()
                optimizer.step()
                if adamw_optimizer is not None:
                    adamw_optimizer.step()
                # S8: body 参数低 lr 微调（之前漏调 step，body 永不更新且梯度无限累积）
                if body_optimizer is not None:
                    body_optimizer.step()
                scheduler.step()
                if body_scheduler is not None:
                    body_scheduler.step()

                epoch_loss += ce_loss.item() * n_tokens
                epoch_tokens += n_tokens
                total_steps += 1

                # Auxiliary-loss-free balancing bias 更新（每 BIAS_UPDATE_EVERY 步）
                # 借鉴 DeepSeek V3：非梯度更新，根据 channel usage 启发式调整 bias
                # 必须在 total_steps 增量后，与 LOG_EVERY 对齐
                if total_steps % BIAS_UPDATE_EVERY == 0:
                    total_delta = 0.0
                    for nid, neuron in neurons.items():
                        deltas = neuron.update_channel_bias(update_rate=BIAS_UPDATE_RATE)
                        total_delta += sum(abs(d) for d in deltas.values())
                    if total_delta > 0:
                        n_channels = sum(len(n.get_channel_usage_stats()) for n in neurons.values())
                        print(f"  [bias update] step {total_steps}: "
                              f"{n_channels} channels, total_delta={total_delta:.4f}",
                              flush=True)

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

                    # Channel usage 诊断（每 LOG_EVERY 步输出，监控死通道）
                    all_usages = []
                    for nid, neuron in neurons.items():
                        stats = neuron.get_channel_usage_stats()
                        for ch_key, usage in stats.items():
                            all_usages.append(usage)
                    if all_usages:
                        avg_usage = sum(all_usages) / len(all_usages)
                        min_usage = min(all_usages)
                        max_usage = max(all_usages)
                        # 死通道判定：usage < avg * 0.1
                        dead_count = sum(1 for u in all_usages if u < avg_usage * 0.1)
                        print(f"    [channels] usage avg={avg_usage:.4f} "
                              f"min={min_usage:.4f} max={max_usage:.4f} "
                              f"dead={dead_count}/{len(all_usages)}",
                              flush=True)

                # 中途 checkpoint（每 500 步保存，防止崩溃丢失进度）
                if total_steps % 500 == 0:
                    save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons, loss_history,
                                    adamw_optimizer, scheduler,
                                    body_optimizer, body_scheduler, shared_embeddings,
                                    ensemble=ensemble)
                    print(f"  [中途 checkpoint] step {total_steps} 已保存", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        epoch_elapsed = time.time() - epoch_start_time
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f} "
              f"耗时 {epoch_elapsed/60:.1f} min", flush=True)

        # 每 epoch 保存 checkpoint（断点续训用，含 optimizer state）
        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons, loss_history,
                        adamw_optimizer, scheduler,
                        body_optimizer, body_scheduler, shared_embeddings,
                        ensemble=ensemble)
        print(f"  [checkpoint 已保存] {CKPT_PATH}", flush=True)

        # 同步保存最终产物（含 S8 body + emb，下游 eval 直接加载）
        # 即使后续 epoch 中断也有可用模型
        torch.save(build_final_artifact(neurons, shared_embeddings), FINAL_PATH)
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
    torch.save(build_final_artifact(neurons, shared_embeddings), FINAL_PATH)
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
