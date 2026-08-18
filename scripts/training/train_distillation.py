"""R7: 代际迁移蒸馏训练脚本（teacher → student 三联蒸馏）。

老神经元（teacher）已成熟，通过三联蒸馏（Logits KL + 中间层对齐 + 注意力转移）
把领域知识迁移给新神经元（student），让新神经元快速获得基础能力再个性化发展。
与"小神经元协作匹配大模型"的核心理念一致：先继承、再创新。

支持混合规格：
- teacher/student hidden_size / 层数 / heads 可不同（DistillationLoss 内部投影对齐）
- 同域蒸馏（默认）：teacher/student 用相同 domain tokenizer，KL 直接计算
- 跨域蒸馏：--vocab_alignment 提供 {student_id: teacher_id} 映射 JSON

Usage:
    # 同域蒸馏（teacher=zh_std0 已训练 standard, student=zh_aug0 已训练 compact）
    python -u scripts/training/train_distillation.py \
        --teacher zh_std0 --student zh_aug0 \
        --epochs 4 --lr 3e-4

    # 新建 student（从 teacher 复制匹配权重初始化）
    python -u scripts/training/train_distillation.py \
        --teacher zh_std0 --student zh_new0 --init_from_teacher \
        --epochs 4 --lr 3e-4

    # 断点续训
    python -u scripts/training/train_distillation.py \
        --teacher zh_std0 --student zh_aug0 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.resonance import (
    ResonanceNeuron,
    DistillationLoss,
    get_domain_neuron_config,
)
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, create_shared_embedding,
    make_wsd_scheduler, build_muon_adamw_optimizers,
    load_dialogue_texts_multi,
)
from scripts.training.experiment_config import (
    DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER,
)
from scripts.training.data_augmentation import DialogueAugmenter

DEVICE = "cpu"
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
CKPT_PATH = os.path.join(OUTPUT_DIR, "distillation.ckpt.pt")
FINAL_PATH = os.path.join(OUTPUT_DIR, "distilled_student.pt")


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


def load_neuron(nid: str) -> ResonanceNeuron:
    """加载已训练神经元（从 checkpoint 读取 config + state_dict）。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neuron checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    result = ckpt.get("result", {})
    print(f"  [{nid}] spec={cfg.spec}, hidden={cfg.hidden_size}, "
          f"layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}, "
          f"best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron


def copy_matching_weights(teacher: ResonanceNeuron, student: ResonanceNeuron) -> int:
    """从 teacher 复制形状匹配的参数到 student（权重初始化）。

    相同维度的参数（如 embed_adapter、部分 body 层）直接复制，
    维度不匹配的参数保持 student 自身初始化（蒸馏会逐步对齐）。
    Returns: 复制的参数数量
    """
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied = 0
    with torch.no_grad():
        for name, s_param in student.named_parameters():
            if name not in t_sd:
                continue
            t_param = t_sd[name]
            if t_param.shape == s_param.shape:
                s_param.data.copy_(t_param.data)
                copied += 1
    print(f"  [init] 从 teacher 复制 {copied} 个匹配参数到 student", flush=True)
    return copied


def save_checkpoint(path, epoch, total_steps, optimizer, student, distill,
                    adamw_optimizer=None, scheduler=None, loss_history=None):
    """保存蒸馏训练 checkpoint，支持断点续训。"""
    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer_state": optimizer.state_dict(),
        "student_state": student.state_dict(),
        "distill_state": distill.state_dict(),
        "loss_history": loss_history or [],
        "saved_at": datetime.now().isoformat(),
    }
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, optimizer, student, distill,
                    adamw_optimizer=None, scheduler=None):
    """加载 checkpoint，恢复 student/distill/optimizer/进度。"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt["student_state"], strict=False)
    distill.load_state_dict(ckpt["distill_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if adamw_optimizer is not None and "adamw_optimizer_state" in ckpt:
        adamw_optimizer.load_state_dict(ckpt["adamw_optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["total_steps"], ckpt.get("loss_history", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, help="teacher neuron id（已训练）")
    parser.add_argument("--student", required=True, help="student neuron id（已训练或新建）")
    parser.add_argument("--init_from_teacher", action="store_true",
                        help="student 不存在时从 teacher 复制匹配权重初始化")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_texts", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="KL 蒸馏温度（软化分布）")
    parser.add_argument("--w_ce", type=float, default=1.0, help="CE loss 权重")
    parser.add_argument("--w_kl", type=float, default=1.0, help="KL 蒸馏权重")
    parser.add_argument("--w_hidden", type=float, default=0.5, help="中间层对齐权重")
    parser.add_argument("--w_attn", type=float, default=0.3, help="注意力转移权重")
    parser.add_argument("--hidden_mode", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--attn_align_mode", default="mean", choices=["mean", "proj"])
    parser.add_argument("--vocab_alignment", default=None,
                        help="跨域蒸馏对齐表 JSON（{student_id: teacher_id}），同域可省略")
    parser.add_argument("--augment", action="store_true",
                        help="T4: 启用数据增强（模板改写 + 多轮拼接）")
    parser.add_argument("--aug_rewrite_prob", type=float, default=0.5,
                        help="T4: 模板改写概率")
    parser.add_argument("--aug_multi_turn_prob", type=float, default=0.4,
                        help="T4: 多轮拼接概率")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR, f"train_distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print("R7 代际迁移蒸馏（三联: KL + Hidden + Attention）", flush=True)
    print(f"teacher={args.teacher}, student={args.student}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载 teacher（完全冻结）
    print("\n[1] 加载 teacher...", flush=True)
    teacher = load_neuron(args.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. 加载/创建 student
    print("\n[2] 加载/创建 student...", flush=True)
    student_path = os.path.join(OUTPUT_DIR, f"neuron_{args.student}.pt")
    if os.path.exists(student_path):
        student = load_neuron(args.student)
    else:
        print(f"  student {args.student} 不存在，创建新神经元", flush=True)
        cfg = get_domain_neuron_config(DOMAIN, spec="compact")
        cfg.unified_field_dim = None
        student = ResonanceNeuron(cfg).to(DEVICE)
        if args.init_from_teacher:
            copy_matching_weights(teacher, student)
    student.train()

    # 3. 构建蒸馏 loss
    print("\n[3] 构建 DistillationLoss...", flush=True)
    vocab_alignment = None
    if args.vocab_alignment:
        with open(args.vocab_alignment, "r", encoding="utf-8") as f:
            vocab_alignment = {int(k): int(v) for k, v in json.load(f).items()}
        print(f"  跨域蒸馏: vocab 对齐表 {len(vocab_alignment)} 条", flush=True)
    else:
        print(f"  同域蒸馏: teacher/student 共享 tokenizer，KL 直接计算", flush=True)

    distill = DistillationLoss(
        student_hidden=student.config.hidden_size,
        teacher_hidden=teacher.config.hidden_size,
        student_layers=len(student.layers),
        teacher_layers=len(teacher.layers),
        student_heads=student.config.num_attention_heads,
        teacher_heads=teacher.config.num_attention_heads,
        temperature=args.temperature,
        vocab_alignment=vocab_alignment,
        attn_align_mode=args.attn_align_mode,
    ).to(DEVICE)
    print(f"  层映射: {distill.layer_map}", flush=True)

    # 4. 优化器（Muon for 2D weights + AdamW for 1D + AdamW for distill heads）
    print("\n[4] 设置优化器...", flush=True)
    muon_params = []
    adamw_params = []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, lr=args.lr,
    )
    # distill 投影头参数（AdamW）
    distill_params = list(distill.parameters())
    print(f"  Student 可训练: muon={sum(p.numel() for p in muon_params):,}, "
          f"adamw={sum(p.numel() for p in adamw_params):,}", flush=True)
    print(f"  Distill heads: {sum(p.numel() for p in distill_params):,}", flush=True)

    # 5. 加载数据
    print("\n[5] 加载训练数据...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    dialogue_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh",
    )
    texts = load_dialogue_texts_multi(dialogue_dir, max_texts=args.max_texts)
    print(f"  训练集: {len(texts)} 条对话", flush=True)

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

    # 6. 调度器
    total_est_steps = args.epochs * ((len(texts) - args.batch_size) // args.batch_size)
    warmup_steps = 50
    decay_ratio = 0.8
    scheduler = make_wsd_scheduler(
        optimizer, num_steps=total_est_steps,
        warmup_steps=warmup_steps, decay_ratio=decay_ratio,
    )
    print(f"  LR 调度: warmup={warmup_steps}步, total≈{total_est_steps}步", flush=True)

    LOG_EVERY = 25
    total_steps = 0
    start_epoch = 0
    loss_history = []

    if args.resume and os.path.exists(CKPT_PATH):
        print(f"\n[resume] 加载 checkpoint: {CKPT_PATH}", flush=True)
        start_epoch, total_steps, loss_history = load_checkpoint(
            CKPT_PATH, optimizer, student, distill, adamw_optimizer, scheduler,
        )
        print(f"  已恢复: epoch={start_epoch} (从 epoch {start_epoch+1} 继续), "
              f"total_steps={total_steps}, loss_history={len(loss_history)} 条", flush=True)
        start_epoch = start_epoch + 1
    elif args.resume:
        print(f"\n[resume] 未找到 checkpoint ({CKPT_PATH})，从头开始", flush=True)

    import random
    random.seed(42)

    # 7. 训练循环
    print("\n[6] 开始蒸馏训练...", flush=True)
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(texts)
        epoch_start_time = time.time()
        # T4: 每 epoch 重置增强种子（epoch 间变体不同，epoch 内可复现）
        if augmenter is not None:
            augmenter.set_epoch(epoch)

        for i in range(0, len(texts) - args.batch_size, args.batch_size):
            batch_texts = texts[i:i + args.batch_size]

            # T4: 在线数据增强（模板改写 + 多轮拼接）
            if augmenter is not None:
                batch_texts = [augmenter.augment(t) for t in batch_texts]

            # shared embedding（teacher/student 共享同一外部嵌入）
            shared_emb = create_shared_embedding(DEVICE)
            emb_out, targets, mask, sft_mask = batch_align_and_embed(
                batch_texts, domain_sp, general_sp, shared_emb,
                answer_marker=SFT_ANSWER_MARKER,
                answer_marker_mode="last",  # T4: 多轮精确 masking（前序轮次为纯上下文）
            )
            emb = emb_out.to(DEVICE)
            mask = mask.to(DEVICE)

            # teacher forward（冻结，no_grad）
            with torch.no_grad():
                t_result = teacher.forward(
                    emb, return_logits=True, return_intermediate=True,
                )

            # student forward（可训练）
            s_result = student.forward(
                emb, return_logits=True, return_intermediate=True,
            )

            # 三联蒸馏 loss
            d = distill(
                student_logits=s_result["logits"],
                teacher_logits=t_result["logits"],
                student_hiddens=s_result["intermediate_hidden"],
                teacher_hiddens=t_result["intermediate_hidden"],
                student_attns=s_result.get("attn_weights"),
                teacher_attns=t_result.get("attn_weights"),
                mask=mask,
                hidden_mode=args.hidden_mode,
            )

            # CE loss（student 对真实目标）
            shift_logits = s_result["logits"][:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = (mask[:, 1:] & sft_mask[:, 1:]).to(DEVICE)
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            ) / max(shift_mask.sum().item(), 1)

            loss = (
                args.w_ce * ce_loss
                + args.w_kl * d["kl"]
                + args.w_hidden * d["hidden"]
                + args.w_attn * d["attn"]
            )

            optimizer.zero_grad()
            if adamw_optimizer is not None:
                adamw_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if adamw_optimizer is not None:
                adamw_optimizer.step()
            scheduler.step()
            total_steps += 1

            if total_steps % LOG_EVERY == 0:
                elapsed = time.time() - epoch_start_time
                print(
                    f"  Epoch {epoch+1}/{args.epochs} step {total_steps}: "
                    f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"kl={d['kl'].item():.4f} hidden={d['hidden'].item():.4f} "
                    f"attn={d['attn'].item():.4f} "
                    f"[{elapsed:.0f}s]",
                    flush=True,
                )
                loss_history.append({
                    "step": total_steps, "epoch": epoch + 1,
                    "loss": loss.item(), "ce": ce_loss.item(),
                    "kl": d["kl"].item(), "hidden": d["hidden"].item(),
                    "attn": d["attn"].item(),
                })

            if total_steps % 500 == 0:
                save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, student, distill,
                                adamw_optimizer, scheduler, loss_history)
                print(f"  [中途 checkpoint] step {total_steps} 已保存", flush=True)

        print(f"  [Epoch {epoch+1} 完成] 耗时 {(time.time()-epoch_start_time)/60:.1f} min", flush=True)
        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, student, distill,
                        adamw_optimizer, scheduler, loss_history)
        print(f"  [checkpoint 已保存] {CKPT_PATH}", flush=True)

        # 同步保存最终产物（student state_dict + distill heads）
        artifact = {
            "student_state": student.state_dict(),
            "distill_state": distill.state_dict(),
            "teacher": args.teacher,
            "saved_at": datetime.now().isoformat(),
        }
        torch.save(artifact, FINAL_PATH)
        print(f"  [final 已保存] {FINAL_PATH}", flush=True)

    # 8. 最终保存
    artifact = {
        "student_state": student.state_dict(),
        "distill_state": distill.state_dict(),
        "teacher": args.teacher,
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(artifact, FINAL_PATH)
    print(f"\n蒸馏完成。student 已保存: {FINAL_PATH}", flush=True)

    history_path = os.path.join(LOG_DIR, "train_distillation_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path} ({len(loss_history)} 条记录)", flush=True)

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
