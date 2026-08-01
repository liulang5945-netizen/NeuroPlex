"""微调跨规格 side_channels + 投影层：4×compact + 1×standard 协作。

基于 finetune_side_channels.py，增加：
  1. 加载 zh_std0 (standard) + zh_aug0~3 (compact)
  2. ensemble 自动创建跨规格投影层（field_dim -> unified_dim）
  3. 将跨规格投影层加入可训练参数
  4. 保存跨规格投影层权重

冻结：neuron 核心参数 + shared_embedding
可训练：side_channels + scale + 跨规格投影层（正向 + 反向）

Usage:
    python -u scripts/training/finetune_cross_spec.py
    python -u scripts/training/finetune_cross_spec.py --resume
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
    load_dialogue_texts_multi,
)

from scripts.training.experiment_config import ENSEMBLE_DIALOGUE_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER

DEVICE = "cpu"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
# 对话训练产物（独立于 simple_zh 训练产物）
CKPT_PATH = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.ckpt.pt")
FINAL_PATH = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.pt")


class TeeLogger:
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


def save_checkpoint(path, epoch, total_steps, optimizer, neurons, ensemble,
                    loss_history, adamw_optimizer=None, scheduler=None):
    """保存训练 checkpoint，含 side_channels + 跨规格投影层。"""
    side_state = {}
    scale_bias_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
        sb = {}
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                sb[name] = p.data.clone()
        for name, buf in neuron.named_buffers():
            if "bias_" in name:
                sb[name] = buf.clone()
        scale_bias_state[nid] = sb

    # 保存跨规格投影层
    cross_spec_state = {
        "forward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_projectors.items()},
        "backward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_back_projectors.items()},
    }

    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer_state": optimizer.state_dict(),
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "cross_spec_state": cross_spec_state,
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, optimizer, neurons, ensemble, adamw_optimizer=None, scheduler=None):
    """加载 checkpoint。"""
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
        if nid in scale_bias_state:
            sb = scale_bias_state[nid]
            for name, p in neuron.named_parameters():
                if name in sb and "scale_" in name:
                    p.data.copy_(sb[name])
            for name, buf in neuron.named_buffers():
                if name in sb and "bias_" in name:
                    buf.copy_(sb[name])

    # 恢复跨规格投影层
    cross_spec_state = ckpt.get("cross_spec_state", {})
    for nid, sd in cross_spec_state.get("forward", {}).items():
        if nid in ensemble._cross_spec_projectors:
            ensemble._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in cross_spec_state.get("backward", {}).items():
        if nid in ensemble._cross_spec_back_projectors:
            ensemble._cross_spec_back_projectors[nid].load_state_dict(sd)

    optimizer.load_state_dict(ckpt["optimizer_state"])
    if adamw_optimizer is not None and "adamw_optimizer_state" in ckpt:
        adamw_optimizer.load_state_dict(ckpt["adamw_optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["total_steps"], ckpt.get("loss_history", [])


def load_neuron_with_embedding(nid):
    """加载单个神经元及其 shared_embedding（支持混合规格）。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # 优先使用 checkpoint 中的 neuron_config
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)

    shared_emb = create_shared_embedding(DEVICE)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE)

    result = ckpt.get("result", {})
    print(f"  [{nid}] spec={cfg.spec}, best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb


def load_dialogue_texts(jsonl_path: str, max_texts: int = 10000) -> list:
    """加载对话训练数据（alpaca-zh SFT 格式）。

    每条格式: "问：{instruction}\n答：{output}"
    """
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            if len(texts) >= max_texts:
                break
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_texts", type=int, default=10000)
    parser.add_argument("--data", type=str, default="dialogue",
                        choices=["dialogue", "simple_zh"],
                        help="dialogue=alpaca-zh SFT, simple_zh=作文数据")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"finetune_cross_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print("微调跨规格 side_channels + 投影层（对话训练）", flush=True)
    print(f"神经元: {NEURON_IDS}", flush=True)
    print(f"日志: {log_path}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载神经元
    print("\n[1] 加载神经元...", flush=True)
    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 2. 建立 side_channels（per-pair，跨规格自动适配）
    print("\n[2] 建立 side_channels...", flush=True)
    for post_id in NEURON_IDS:
        for pre_id in NEURON_IDS:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
        print(f"  [{post_id}] {len(neurons[post_id].excite_channels)} excite channels", flush=True)

    # 3. 冻结核心参数，仅 side_channels + scale 可训练
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
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        neuron.train()

    for emb in shared_embeddings.values():
        for p in emb.parameters():
            p.requires_grad = False
        emb.eval()

    # 4. 创建 ensemble（用最大 field_dim，自动创建跨规格投影层）
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    print(f"\n  field.dim={max_field_dim}, 跨规格投影层: "
          f"{len(ensemble._cross_spec_projectors)} 正向 + "
          f"{len(ensemble._cross_spec_back_projectors)} 反向", flush=True)

    # 跨规格投影层设为可训练
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True

    # 统计可训练参数
    trainable_side = 0
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            trainable_side += sum(p.numel() for p in ch.parameters() if p.requires_grad)
    trainable_proj = sum(
        sum(p.numel() for p in proj.parameters() if p.requires_grad)
        for proj in ensemble._cross_spec_projectors.values()
    ) + sum(
        sum(p.numel() for p in proj.parameters() if p.requires_grad)
        for proj in ensemble._cross_spec_back_projectors.values()
    )
    print(f"  可训练参数: side_channels={trainable_side:,}, 跨规格投影={trainable_proj:,}", flush=True)

    # 5. 加载训练数据（S5: 多文件合并扩充）
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

    # 6. 训练循环
    print("\n[5] 开始训练...", flush=True)
    muon_params = []
    adamw_params = []
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
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if "scale_" in name and p.ndim == 0:
                adamw_params.append(p)

    # 跨规格投影层（2D weight → Muon）
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)

    # Muon + AdamW 混合优化器（配置抽取到 utils.build_muon_adamw_optimizers）
    muon_lr = args.lr
    optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, lr=muon_lr,
    )
    print(f"  Muon 参数: {sum(p.numel() for p in muon_params):,} (2D weight, lr={muon_lr})", flush=True)
    if adamw_optimizer is not None:
        print(f"  AdamW 参数: {sum(p.numel() for p in adamw_params):,} (1D bias/scale, lr={muon_lr})", flush=True)

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
    BIAS_UPDATE_EVERY = 50
    BIAS_UPDATE_RATE = 0.1

    total_steps = 0
    start_epoch = 0
    loss_history = []

    if args.resume and os.path.exists(CKPT_PATH):
        print(f"\n[resume] 加载 checkpoint: {CKPT_PATH}", flush=True)
        start_epoch, total_steps, loss_history = load_checkpoint(
            CKPT_PATH, optimizer, neurons, ensemble, adamw_optimizer, scheduler,
        )
        print(f"  已恢复: epoch={start_epoch}, total_steps={total_steps}, "
              f"loss_history={len(loss_history)} 条", flush=True)
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
            # 让 side_channels + 跨规格投影层 + field_state + 调质在训练中真正生效
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings,
                n_rounds=2,
                fusion_mode="soft",
                return_individual_logits=False,
            )

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
            # balance_loss 鼓励神经元均衡贡献（防死通道）
            # diversity_loss 鼓励 field_vector 差异化（防退化相同）
            balance_loss = result["balance_loss"]
            diversity_loss = result["diversity_loss"]
            balance_weight = 0.01   # 弱约束，避免压制主任务
            diversity_weight = 0.05  # 弱约束，鼓励差异但不强制正交
            loss = ce_loss + balance_weight * balance_loss + diversity_weight * diversity_loss

            loss.backward()
            optimizer.step()
            if adamw_optimizer is not None:
                adamw_optimizer.step()
            scheduler.step()

            epoch_loss += ce_loss.item() * n_tokens
            epoch_tokens += n_tokens
            total_steps += 1

            if total_steps % BIAS_UPDATE_EVERY == 0:
                for nid, neuron in neurons.items():
                    neuron.update_channel_bias(update_rate=BIAS_UPDATE_RATE)

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

            if total_steps % 500 == 0:
                save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons,
                                ensemble, loss_history, adamw_optimizer, scheduler)
                print(f"  [中途 checkpoint] step {total_steps} 已保存", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        epoch_elapsed = time.time() - epoch_start_time
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f} "
              f"耗时 {epoch_elapsed/60:.1f} min", flush=True)

        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons,
                        ensemble, loss_history, adamw_optimizer, scheduler)
        print(f"  [checkpoint 已保存] {CKPT_PATH}", flush=True)

        # 保存最终产物
        side_state = {}
        for nid, neuron in neurons.items():
            side_state[nid] = {
                "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
                "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
            }
        cross_spec_state = {
            "forward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_projectors.items()},
            "backward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_back_projectors.items()},
        }
        torch.save({"side_channels": side_state, "cross_spec": cross_spec_state}, FINAL_PATH)
        print(f"  [final 已保存] {FINAL_PATH}", flush=True)

        recent = loss_history[-5:]
        if len(recent) >= 2:
            first_ppl = recent[0]["ppl"]
            last_ppl = recent[-1]["ppl"]
            delta = last_ppl - first_ppl
            print(f"  [趋势] 最近 5 点 PPL: {first_ppl:.1f} -> {last_ppl:.1f} "
                  f"(Δ={delta:+.1f}, {'下降' if delta < 0 else '上升/停滞'})", flush=True)

    print("\n[6] 训练完成，最终保存...", flush=True)
    side_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
    cross_spec_state = {
        "forward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_projectors.items()},
        "backward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_back_projectors.items()},
    }
    torch.save({"side_channels": side_state, "cross_spec": cross_spec_state}, FINAL_PATH)
    print(f"  已保存: {FINAL_PATH}", flush=True)

    history_path = os.path.join(LOG_DIR, "finetune_cross_spec_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path} ({len(loss_history)} 条记录)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("微调完成。运行 eval_dialogue.py 查看交流效果。", flush=True)
    print("=" * 60, flush=True)

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
