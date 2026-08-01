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
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts, create_shared_embedding,
    make_wsd_scheduler, build_muon_adamw_optimizers,
)
from scripts.training.experiment_config import ZH_COMPACT_NEURON_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER

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


def save_checkpoint(path, epoch, total_steps, optimizer, neurons, loss_history, adamw_optimizer=None, scheduler=None):
    """保存训练 checkpoint，支持断点续训。"""
    side_state = {}
    scale_bias_state = {}
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
    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer_state": optimizer.state_dict(),
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, optimizer, neurons, adamw_optimizer=None, scheduler=None):
    """加载 checkpoint，恢复 side_channels、scale/bias、optimizer、训练进度。"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    side_state = ckpt["side_channels_state"]
    scale_bias_state = ckpt.get("scale_bias_state", {})
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
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if adamw_optimizer is not None and "adamw_optimizer_state" in ckpt:
        adamw_optimizer.load_state_dict(ckpt["adamw_optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 断点续训")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_texts", type=int, default=10000)
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
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

    # 3. 建立 side_channels
    print("\n[2] 建立 side_channels...", flush=True)
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
        print(f"  [{post_id}] {len(neurons[post_id].excite_channels)} excite channels", flush=True)

    # 4. 冻结核心参数，仅 side_channels + scale 可训练
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
        # scale 参数可训练（Auxiliary-loss-free balancing 的可学习部分）
        for name, p in neuron.named_parameters():
            if "scale_" in name:
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
    # Muon + AdamW 混合优化器（借鉴 DeepSeek V4 / GLM-5.2）
    # - 2D 权重矩阵（Linear weight）用 Muon：Newton-Schulz 正交化突破 Adam 局部最小值
    # - 1D 参数（bias/LayerNorm）用 AdamW：Muon 仅适用于 2D
    # - scale 参数（0D scalar）用 AdamW
    muon_params = []   # 2D weight
    adamw_params = []  # 1D bias/norm + 0D scale
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

    # Muon + AdamW 混合优化器（配置抽取到 utils.build_muon_adamw_optimizers）
    # Muon 学习率：side_channels 是 12.58M 小参数，lr 不宜过大。
    # 之前 lr=0.01 (args.lr*10) 导致 step 150 后 PPL 反弹震荡，降为 args.lr=1e-3。
    muon_lr = args.lr
    optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, lr=muon_lr,
    )
    print(f"  Muon 参数: {sum(p.numel() for p in muon_params):,} (2D weight, lr={muon_lr})", flush=True)
    if adamw_optimizer is not None:
        print(f"  AdamW 参数: {sum(p.numel() for p in adamw_params):,} (1D bias/scale, lr={muon_lr})", flush=True)
    else:
        print(f"  AdamW 参数: 0 (无 1D 参数)", flush=True)

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
            sft_mask = None
            valid = True
            for nid, shared_emb in shared_embeddings.items():
                # S3: 传入 answer_marker，获取 sft_mask（只对 answer 部分计算 loss）
                emb_out, tgt, msk, sft = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_emb,
                    answer_marker=SFT_ANSWER_MARKER,
                )
                neuron_embeddings[nid] = emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)
                    sft_mask = sft.to(DEVICE)

            optimizer.zero_grad()
            if adamw_optimizer is not None:
                adamw_optimizer.zero_grad()

            # S1: 改用 forward_train（全可微多轮共振路径）
            # 让 side_channels + field_state 在训练中真正生效
            # 注：field_conditioning=False 是推理路径选项，forward_train 中
            # round 2+ 默认传 field_state，但 field_read_layers 是否被使用
            # 取决于 neuron.forward 内 round_num>1 的判断
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings,
                n_rounds=2,
                fusion_mode="soft",
                return_individual_logits=(total_steps == 0),  # 首步返回用于 debug
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

                # 多任务 loss：CE + 负载均衡 + 多样性
                balance_loss = result["balance_loss"]
                diversity_loss = result["diversity_loss"]
                balance_weight = 0.01
                diversity_weight = 0.05
                loss = ce_loss + balance_weight * balance_loss + diversity_weight * diversity_loss

                loss.backward()
                optimizer.step()
                if adamw_optimizer is not None:
                    adamw_optimizer.step()
                scheduler.step()

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
                    save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons, loss_history, adamw_optimizer, scheduler)
                    print(f"  [中途 checkpoint] step {total_steps} 已保存", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        epoch_elapsed = time.time() - epoch_start_time
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f} "
              f"耗时 {epoch_elapsed/60:.1f} min", flush=True)

        # 每 epoch 保存 checkpoint（断点续训用，含 optimizer state）
        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons, loss_history, adamw_optimizer, scheduler)
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
