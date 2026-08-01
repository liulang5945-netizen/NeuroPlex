"""Fine-tune 已有神经元用对话数据：加载 zh_std0，用 alpaca-zh SFT 继续训练。

策略：
  1. 加载 zh_std0（百科训练，val PPL=34，已有语言能力）
  2. 用 alpaca-zh SFT 对话数据继续训练（fine-tune）
  3. 学习率 5e-4（比从头训练低）
  4. 4000 步（fine-tune 不需要太多步）

工程保障：
  - stdout 同时写入日志文件
  - 每次刷新 best val PPL 立即保存 checkpoint
  - 支持 --resume 断点续训

Usage:
    python -u scripts/training/finetune_neuron_dialogue.py --base_id zh_std0 --target_id zh_std0_dialogue
    python -u scripts/training/finetune_neuron_dialogue.py --base_id zh_std0 --target_id zh_std0_dialogue --resume
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

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, SequentialSampler,
)

DOMAIN = "zh"
DEVICE = "cpu"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)


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


def load_dialogue_texts(jsonl_path: str, max_texts: int = 100000) -> list:
    """加载对话训练数据。"""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if len(text) >= 20:
                texts.append(text)
            if len(texts) >= max_texts:
                break
    return texts


def load_base_neuron(base_id: str):
    """加载基础神经元（已训练）。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{base_id}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DOMAIN, spec="standard")

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)

    shared_emb = nn.Embedding(256000, 512)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE)

    result = ckpt.get("result", {})
    print(f"  [{base_id}] spec={cfg.spec}, best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb, cfg


def generate_sample(neuron, domain_sp, general_sp, shared_emb, prompt="问：你好\n答："):
    """生成样本用于训练监控。

    关键修复：neuron 输出 domain token ID，需转回 general token IDs 才能追加到输入，
    解码用 domain_sp（不是 general_sp）。
    """
    neuron.eval()
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return "(empty)"
    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    generated_domain_ids = []

    domain_eos_id = None
    if hasattr(domain_sp, 'eos_id'):
        eid = domain_sp.eos_id()
        if eid is not None and eid >= 0:
            domain_eos_id = int(eid)

    with torch.no_grad():
        for _ in range(80):
            emb_input = shared_emb(ids)
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :].float()
            # 简单 top-k sampling
            top_k = min(40, logits.size(-1))
            topk_vals, _ = torch.topk(logits[0], top_k)
            logits[0][logits[0] < topk_vals[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_domain_token = torch.multinomial(probs, num_samples=1).item()
            generated_domain_ids.append(next_domain_token)

            if domain_eos_id is not None and next_domain_token == domain_eos_id:
                break

            # domain token ID → 文本 → general token IDs
            piece_text = domain_sp.decode([next_domain_token])
            new_general_ids = general_sp.encode(piece_text)
            if not new_general_ids:
                new_general_ids = [general_sp.pad_id()]
            ids = torch.cat([ids, torch.tensor([new_general_ids], dtype=torch.long, device=DEVICE)], dim=1)
    text = domain_sp.DecodeIds(generated_domain_ids)
    neuron.train()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_id", required=True, help="基础神经元 ID（如 zh_std0）")
    parser.add_argument("--target_id", required=True, help="目标神经元 ID（如 zh_std0_dialogue）")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_embedding", action="store_true",
                        help="训练 shared_embedding（默认冻结，防止 token 映射被破坏）")
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_texts", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    # 日志
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"finetune_dialogue_{args.target_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print(f"Fine-tune 神经元用对话数据", flush=True)
    print(f"  base: {args.base_id} -> target: {args.target_id}", flush=True)
    print(f"  steps={args.steps}, lr={args.lr}, batch={args.batch_size}×{args.grad_accum}", flush=True)
    print(f"  日志: {log_path}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载基础神经元
    print("\n[1] 加载基础神经元...", flush=True)
    neuron, shared_emb, cfg = load_base_neuron(args.base_id)
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  参数: {n_params/1e6:.1f}M, spec={cfg.spec}", flush=True)

    # 2. 加载训练数据
    print("\n[2] 加载对话训练数据...", flush=True)
    dialogue_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh", "alpaca_zh_sft.jsonl",
    )
    texts = load_dialogue_texts(dialogue_path, max_texts=args.max_texts)
    print(f"  训练集: {len(texts)} 条对话", flush=True)

    # 3. tokenizer
    print("\n[3] tokenizer...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    # 4. 评估数据（用训练数据的最后 30 条作为 val）
    eval_texts = texts[-30:]
    train_texts = texts[:-30]

    # 5. sampler
    sampler = SequentialSampler(train_texts, args.batch_size, seed=42)

    # 6. 优化器 + 调度器
    # fine-tune: 默认冻结 shared_embedding 防止 token 映射被破坏
    if not args.train_embedding:
        for p in shared_emb.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(neuron.parameters(), lr=args.lr, weight_decay=0.1)
        print(f"  shared_embedding: FROZEN（保留原有 token 映射）", flush=True)
    else:
        all_params = list(neuron.parameters()) + list(shared_emb.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.1)
        print(f"  shared_embedding: TRAINABLE", flush=True)

    # WSD 调度
    decay_start = max(args.warmup_steps + 1, int(args.steps * 0.85))
    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if step >= decay_start:
            progress = (step - decay_start) / max(1, args.steps - decay_start)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 7. 断点续训
    save_path = os.path.join(OUTPUT_DIR, f"neuron_{args.target_id}.pt")
    start_step = 0
    best_val_loss = float('inf')
    best_step = 0

    if args.resume and os.path.exists(save_path):
        print(f"\n[resume] 加载 checkpoint: {save_path}", flush=True)
        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"]:
            shared_emb.load_state_dict(ckpt["shared_embedding_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        result = ckpt.get("result", {})
        best_val_loss = result.get("best_val_ppl", float('inf'))
        best_step = result.get("best_step", 0)
        start_step = result.get("steps", 0)
        print(f"  已恢复: step={start_step}, best_val_ppl={best_val_loss:.2f}@step{best_step}", flush=True)

    # 8. 训练循环
    print(f"\n[4] 开始 fine-tune...", flush=True)
    neuron.train()
    step = start_step
    epoch_start_time = time.time()

    while step < args.steps:
        batch_texts = sampler.sample_batch()

        # grad accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                shared_emb_out, targets, mask = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_emb,
                )
            except Exception:
                accum_loss = 0
                break

            result = neuron.forward(shared_emb_out, return_logits=True)
            logits = result["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets_flat = shift_targets.clone()
            shift_targets_flat[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets_flat.view(-1),
                ignore_index=-100,
            ) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.log_every == 0:
            avg_loss = accum_loss
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"  [{args.target_id}] step {step}/{args.steps} "
                  f"loss={avg_loss:.4f} PPL={ppl:.1f} lr={current_lr:.2e} "
                  f"elapsed={elapsed/60:.0f}min", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            neuron.eval()
            total_ce = 0.0
            n_eval = 0
            with torch.no_grad():
                for text in eval_texts:
                    shared_emb_out, targets, mask = batch_align_and_embed(
                        [text], domain_sp, general_sp, shared_emb,
                    )
                    result = neuron.forward(shared_emb_out, return_logits=True)
                    logits = result["logits"]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = targets[:, 1:].contiguous()
                    shift_mask = mask[:, 1:].contiguous()
                    shift_targets_flat = shift_targets.clone()
                    shift_targets_flat[~shift_mask] = -100
                    ce = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets_flat.view(-1),
                        ignore_index=-100,
                    )
                    total_ce += ce.item()
                    n_eval += 1
            val_ppl = math.exp(min(total_ce / max(n_eval, 1), 20))
            print(f"\n  [{args.target_id}] [EVAL] step {step}: val PPL={val_ppl:.2f}", flush=True)

            if val_ppl < best_val_loss:
                best_val_loss = val_ppl
                best_step = step
                print(f"    [SAVE] best (val PPL={best_val_loss:.2f})", flush=True)

            # 每次 eval 都保存 latest checkpoint（避免 resume 回退到旧 best）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "neuron_config": neuron.config,
                "state_dict": {k: v.detach().clone() for k, v in neuron.state_dict().items()},
                "shared_embedding_state": {k: v.detach().clone() for k, v in shared_emb.state_dict().items()},
                "domain": DOMAIN,
                "data_source": "alpaca_zh_sft_finetune",
                "result": {
                    "best_val_ppl": best_val_loss,
                    "best_step": best_step,
                    "steps": step,
                    "base_id": args.base_id,
                    "finetune": True,
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, save_path)

            # 生成样本
            sample = generate_sample(neuron, domain_sp, general_sp, shared_emb, prompt="问：你好\n答：")
            print(f"    生成: {sample[:200]}", flush=True)
            neuron.train()

    # 9. 最终保存
    print(f"\n[5] 训练完成", flush=True)
    print(f"  best_val_PPL={best_val_loss:.2f}@step{best_step}", flush=True)
    print(f"  Checkpoint: {save_path}", flush=True)

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
