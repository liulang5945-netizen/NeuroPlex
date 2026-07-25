"""长时间训练单个 compact 神经元（修复采样器问题）。

核心改进（对比 train_individual_neurons.py）：
1. 顺序 epoch 采样：shuffle → 顺序遍历 → 重洗，100% 利用率（vs 随机采样 0.8%）
2. 全量数据：4.3M 文本不分割，最大化数据多样性
3. 长时间训练：32000 步（4x 原始），看到 128K 唯一文本（vs 32K）
4. 用户洞察：CPU 对单个 compact 足够，不要太在意时间成本

Usage:
    python -u scripts/training/train_single_long.py --steps 32000
    python -u scripts/training/train_single_long.py --steps 64000 --batch_size 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    load_or_create_shared_embedding,
    OUTPUT_DIR,
)
from scripts.training.train_standard_leader import SequentialSampler, load_all_texts

DATA_PATH = "data/distill/zh_texts.jsonl"


def train_single_long(
    neuron: ResonanceNeuron,
    texts: list[str],
    neuron_id: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 32000,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    log_every: int = 500,
    save_path: str = None,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    eval_every: int = 4000,
) -> dict:
    """长时间训练单个神经元（顺序采样 + 定期评估 argmax）。"""
    sampler = SequentialSampler(texts, batch_size, seed=42)

    all_params = list(neuron.parameters())
    all_params += list(shared_embedding.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # WSD 学习率调度（长训练用更长 decay 阶段）
    decay_start = max(warmup_steps + 1, int(num_steps * 0.85))
    def _wsd_lr(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        elif step < decay_start:
            return 1.0
        else:
            progress = (step - decay_start) / max(1, num_steps - decay_start)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _wsd_lr)

    neuron.train()
    shared_embedding.train()

    total_loss = 0.0
    step, t_start = 0, time.time()
    best_loss = float("inf")
    best_step = 0
    best_state = None
    recent_losses = []

    # 加载评估文本（用于定期检查 argmax）
    eval_texts = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    for line in all_lines[-100:]:
        line = line.strip()
        if len(line) >= 20:
            eval_texts.append(line)

    print(f"\n  [{neuron_id}] 开始训练: {num_steps} 步, batch={batch_size}, lr={lr}", flush=True)
    print(f"  预计每步 ~0.7s, 总计 ~{num_steps * 0.7 / 3600:.1f}h", flush=True)
    print(f"  评估: 每 {eval_every} 步检查 argmax（100 条测试集）", flush=True)

    for _ in range(num_steps):
        batch_texts = sampler.sample_batch()
        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding,
        )
        shared_emb = shared_emb.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        shift_targets = shift_targets.clone()
        shift_targets[~shift_mask] = -100

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step += 1

        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        if len(recent_losses) >= 50:
            recent_avg = sum(recent_losses) / len(recent_losses)
            if recent_avg < best_loss:
                best_loss = recent_avg
                best_step = step
                best_state = {k: v.detach().clone() for k, v in neuron.state_dict().items()}

        if step % log_every == 0:
            avg_loss = total_loss / step
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]
            unique_pct = sampler.unique_seen / sampler.n_texts * 100
            print(
                f"  [{neuron_id}] step {step}/{num_steps} "
                f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                f"PPL={ppl:.1f} lr={current_lr:.2e} "
                f"best={best_loss:.4f}@{best_step} "
                f"unique={unique_pct:.1f}% "
                f"elapsed={elapsed/60:.0f}min",
                flush=True,
            )

        # 定期评估 argmax
        if step % eval_every == 0:
            neuron.eval()
            correct, total_eval = 0, 0
            with torch.no_grad():
                for text in eval_texts[:50]:  # 50 条快速评估
                    shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
                    result = neuron.forward(shared, return_logits=True)
                    logits = result['logits']
                    shift_logits = logits[:, :-1, :]
                    shift_targets = targets[:, 1:]
                    shift_mask = mask[:, 1:]
                    preds = shift_logits.argmax(dim=-1)
                    valid = shift_mask.bool()
                    correct += (preds[valid] == shift_targets[valid]).sum().item()
                    total_eval += valid.sum().item()
            eval_argmax = correct / max(total_eval, 1) * 100
            print(f"  [{neuron_id}] ★ argmax 评估: {eval_argmax:.1f}% (目标 85%)", flush=True)
            neuron.train()

    # 最终评估
    neuron.eval()
    correct, total_eval = 0, 0
    total_ce = 0.0
    with torch.no_grad():
        for text in eval_texts:
            shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits']
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            valid = shift_mask.bool()
            correct += (preds[valid] == shift_targets[valid]).sum().item()
            total_eval += valid.sum().item()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )
            total_ce += ce.item()
    final_argmax = correct / max(total_eval, 1) * 100
    final_ppl = math.exp(min(total_ce / len(eval_texts), 20))

    avg_loss = total_loss / max(step, 1)
    elapsed = time.time() - t_start
    print(
        f"\n  [{neuron_id}] Done. {step} steps, "
        f"avg_loss={avg_loss:.4f}, PPL={final_ppl:.2f}, "
        f"argmax={final_argmax:.1f}%, "
        f"best_loss={best_loss:.4f}@step{best_step}, "
        f"time={elapsed/60:.1f}min",
        flush=True,
    )

    save_state = best_state if best_state is not None else neuron.state_dict()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "neuron_config": neuron.config,
        "state_dict": save_state,
        "domain": "zh",
        "result": {
            "final_loss": avg_loss,
            "final_ppl": final_ppl,
            "final_argmax": final_argmax,
            "steps": step,
            "best_loss": best_loss,
            "best_step": best_step,
            "saved": "best" if best_state is not None else "final",
            "spec": "compact",
            "sampler": "sequential_epoch",
        },
    }, save_path)
    print(f"  Saved: {save_path} (best@step{best_step}, loss={best_loss:.4f}, argmax={final_argmax:.1f}%)", flush=True)

    return {
        "neuron_id": neuron_id,
        "final_argmax": final_argmax,
        "final_ppl": final_ppl,
        "best_loss": best_loss,
        "best_step": best_step,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="长时间训练单个 compact 神经元（修复采样器）")
    parser.add_argument("--steps", type=int, default=32000, help="训练步数（默认 32000 = 4x 原始）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=4000, help="argmax 评估间隔")
    parser.add_argument("--max_texts", type=int, default=10000000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neuron_id", default="zh_long0", help="神经元 ID")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"长时间训练单个 compact 神经元（修复采样器问题）", flush=True)
    print(f"  规格: compact (hidden=512, layers=6, ~36M params)", flush=True)
    print(f"  步数: {args.steps} (4x 原始 8000)", flush=True)
    print(f"  采样: 顺序 epoch（100% 利用率，vs 随机 0.8%）", flush=True)
    print(f"  数据: 全量 4.3M 文本不分割", flush=True)
    print(f"  评估: 每 {args.eval_every} 步检查 argmax（目标 85%）", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载全部数据
    print(f"\n[1] 加载全部训练数据...", flush=True)
    all_texts = load_all_texts(DATA_PATH, max_texts=args.max_texts)

    # 2. 加载 tokenizers
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    # 3. 加载 shared_embedding（可训练，因为这是第一个神经元）
    print(f"\n[3] 加载 shared_embedding（可训练）...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)

    # 4. 创建 compact 神经元
    print(f"\n[4] 创建 compact 神经元...", flush=True)
    cfg = get_domain_neuron_config("zh", spec="compact")
    cfg.dropout = args.dropout
    neuron = ResonanceNeuron(cfg).to(args.device)
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  {args.neuron_id}: spec=compact, params={n_params/1e6:.1f}M", flush=True)

    # 5. 训练
    print(f"\n[5] 开始长时间训练...", flush=True)
    save_path = os.path.join(OUTPUT_DIR, f"neuron_{args.neuron_id}.pt")
    result = train_single_long(
        neuron=neuron,
        texts=all_texts,
        neuron_id=args.neuron_id,
        shared_embedding=shared_embedding,
        domain_sp=domain_sp,
        general_sp=general_sp,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        save_path=save_path,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
    )

    print(f"\n{'='*70}", flush=True)
    print(f"训练完成！{args.neuron_id}", flush=True)
    print(f"  argmax={result['final_argmax']:.1f}%", flush=True)
    print(f"  PPL={result['final_ppl']:.2f}", flush=True)
    print(f"  best_loss={result['best_loss']:.4f}@step{result['best_step']}", flush=True)
    print(f"  time={result['elapsed_s']/60:.1f}min", flush=True)
    print(f"  Checkpoint: {save_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
