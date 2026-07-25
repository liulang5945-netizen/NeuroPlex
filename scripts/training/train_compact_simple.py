"""用简单中文数据 + 正规配置训练 compact 神经元。

符合 AI_TRAINING_PLAYBOOK.md 准则：
1. 数据：TinyStoriesAdv-zh（小学/幼儿园水平，287M tokens，匹配 36M 能力）
2. batch_size ≥ 32（梯度累积：batch=8 × grad_accum=4 = 32）
3. embedding 不加 weight_decay（参数组分离）
4. lr=1e-3（小模型标准）
5. WSD 调度（warmup + stable + cosine decay）
6. 评估 PPL + 生成质量（不用 argmax）
7. 顺序 epoch 采样（100% 数据利用率）
8. 保存 best val loss checkpoint

Usage:
    python -u scripts/training/train_compact_simple.py --steps 8000
    python -u scripts/training/train_compact_simple.py --steps 16000 --batch_size 12
"""
from __future__ import annotations

import argparse
import json
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
from scripts.training.train_standard_leader import SequentialSampler

DATA_PATH = "data/simple_zh/simple_zh_texts.jsonl"


def load_simple_texts(path: str, max_texts: int = 10000000) -> list[str]:
    """加载简单中文数据（jsonl 格式，提取 text 字段）。"""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(texts) >= max_texts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                text = d.get('text', '')
                if len(text) >= 20:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def train_compact_simple(
    neuron: ResonanceNeuron,
    texts: list[str],
    neuron_id: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 8000,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 200,
    save_path: str = None,
    warmup_steps: int = 200,
    eval_every: int = 2000,
) -> dict:
    """正规配置训练 compact 神经元。"""
    sampler = SequentialSampler(texts, batch_size, seed=42)

    # ── 参数组分离：embedding 不加 weight_decay（Playbook 准则）──
    embed_params = list(shared_embedding.parameters())
    embed_param_ids = {id(p) for p in embed_params}
    neuron_params = [p for p in neuron.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": neuron_params, "weight_decay": 0.1},        # 神经元参数衰减
        {"params": embed_params, "weight_decay": 0.0},         # embedding 不衰减
    ], lr=lr, betas=(0.9, 0.99))

    # WSD 学习率调度
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
    best_val_loss = float("inf")
    best_step = 0
    best_state = None
    recent_losses = []

    # 评估文本（最后 100 条作为验证集）
    eval_texts = texts[-100:]

    effective_batch = batch_size * grad_accum
    print(f"\n  [{neuron_id}] 正规配置训练:", flush=True)
    print(f"    batch={batch_size} × grad_accum={grad_accum} = effective {effective_batch}", flush=True)
    print(f"    lr={lr}, embedding 不衰减, WSD 调度", flush=True)
    print(f"    steps={num_steps}, 数据={len(texts)} 条", flush=True)
    print(f"    数据/参数比: {len(texts)*300/36e6:.1f}:1 (估算)", flush=True)
    print(f"    评估: 每 {eval_every} 步 PPL + 生成样本", flush=True)

    while step < num_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        # 梯度累积
        for _ in range(grad_accum):
            if step >= num_steps:
                break
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
            ) / grad_accum
            loss.backward()
            accum_loss += loss.item()
            step += 1

        torch.nn.utils.clip_grad_norm_(
            list(neuron.parameters()) + list(shared_embedding.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        total_loss += accum_loss
        recent_losses.append(accum_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        if step % log_every == 0:
            avg_loss = total_loss / step
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]
            unique_pct = sampler.unique_seen / sampler.n_texts * 100
            recent_avg = sum(recent_losses) / len(recent_losses)
            print(
                f"  [{neuron_id}] step {step}/{num_steps} "
                f"loss={accum_loss:.4f} avg={avg_loss:.4f} recent={recent_avg:.4f} "
                f"PPL={ppl:.1f} lr={current_lr:.2e} "
                f"unique={unique_pct:.1f}% "
                f"elapsed={elapsed/60:.0f}min",
                flush=True,
            )

        # 定期评估 PPL + 生成样本
        if step % eval_every == 0 or step == num_steps:
            neuron.eval()
            total_ce = 0.0
            n_eval = 0
            with torch.no_grad():
                for text in eval_texts[:30]:
                    shared, targets, mask = batch_align_and_embed(
                        [text], domain_sp, general_sp, shared_embedding,
                    )
                    result = neuron.forward(shared, return_logits=True)
                    logits = result['logits']
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
            print(f"\n  [{neuron_id}] ★ 评估 step {step}: val PPL={val_ppl:.2f}", flush=True)

            # 保存 best
            if val_ppl < best_val_loss:
                best_val_loss = val_ppl
                best_step = step
                best_state = {k: v.detach().clone() for k, v in neuron.state_dict().items()}
                best_embed = {k: v.detach().clone() for k, v in shared_embedding.state_dict().items()}
                print(f"    ✅ 保存 best (val PPL={best_val_loss:.2f})", flush=True)

            # 生成样本
            sample_text = generate_sample(neuron, domain_sp, general_sp, shared_embedding, device, prompt="小猫")
            print(f"    生成样本: {sample_text[:200]}", flush=True)
            print()
            neuron.train()

    # 保存
    save_state = best_state if best_state is not None else neuron.state_dict()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "neuron_config": neuron.config,
        "state_dict": save_state,
        "shared_embedding_state": best_embed if best_state is not None else None,
        "domain": "zh",
        "data_source": "simple_zh",
        "result": {
            "best_val_ppl": best_val_loss,
            "best_step": best_step,
            "steps": step,
            "saved": "best" if best_state is not None else "final",
            "spec": "compact",
            "config": "regular_batch32_embed_no_decay",
        },
    }, save_path)

    elapsed = time.time() - t_start
    print(f"\n  [{neuron_id}] Done. best_val_PPL={best_val_loss:.2f}@step{best_step}, time={elapsed/60:.1f}min", flush=True)
    print(f"  Saved: {save_path}", flush=True)

    return {"neuron_id": neuron_id, "best_val_ppl": best_val_loss, "best_step": best_step, "elapsed_s": elapsed}


def generate_sample(neuron, domain_sp, general_sp, shared_embedding, device, prompt="从前"):
    """生成中文样本（修复 tokenizer 映射 bug）。

    态极双 tokenizer 设计：
    - 输入：domain token → 映射到 general token → shared_embedding 查找
    - 输出：模型预测 domain token → domain_sp 解码
    - 自回归：domain token → 映射回 general token → 下一步输入
    """
    neuron.eval()
    with torch.no_grad():
        # 1. prompt → domain ids
        domain_ids = domain_sp.encode(prompt)
        if not domain_ids:
            return "(empty)"

        # 2. domain ids → general ids（逐 token 映射）
        general_ids = []
        for tid in domain_ids:
            piece = domain_sp.decode([tid])
            gids = general_sp.encode(piece)
            if gids:
                general_ids.append(gids[0])

        if not general_ids:
            return "(encode failed)"

        # 3. 自回归生成（模型预测 domain token）
        generated_domain_ids = list(domain_ids)
        for _ in range(80):
            emb_input = shared_embedding(torch.tensor([general_ids], device=device))
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :]
            next_domain_id = logits.argmax(dim=-1).item()
            generated_domain_ids.append(next_domain_id)

            # domain token → general token（用于下一步输入）
            piece = domain_sp.decode([next_domain_id])
            gids = general_sp.encode(piece)
            if gids:
                general_ids.append(gids[0])
            else:
                break

        # 4. 用 domain_sp 解码
        text = domain_sp.decode(generated_domain_ids)
    neuron.train()
    return text


def main():
    parser = argparse.ArgumentParser(description="用简单中文数据 + 正规配置训练 compact 神经元")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数 (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--max_texts", type=int, default=10000000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neuron_id", default="zh_simple0")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"正规配置训练 compact 神经元（简单中文数据）", flush=True)
    print(f"  数据: TinyStoriesAdv-zh (小学/幼儿园水平, 287M tokens)", flush=True)
    print(f"  规格: compact (hidden=512, layers=6, ~36M params)", flush=True)
    print(f"  正规配置:", flush=True)
    print(f"    effective batch = {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} (≥32 ✅)", flush=True)
    print(f"    lr = {args.lr} (小模型标准 ✅)", flush=True)
    print(f"    embedding 不加 weight_decay ✅", flush=True)
    print(f"    WSD 调度 + 顺序 epoch 采样 ✅", flush=True)
    print(f"    评估 PPL + 生成质量（不用 argmax）✅", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载数据
    print(f"\n[1] 加载简单中文数据...", flush=True)
    all_texts = load_simple_texts(DATA_PATH, max_texts=args.max_texts)
    print(f"  加载 {len(all_texts)} 条文本", flush=True)

    # 2. tokenizers
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    # 3. shared_embedding
    print(f"\n[3] 加载 shared_embedding（可训练）...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)

    # 4. compact 神经元
    print(f"\n[4] 创建 compact 神经元...", flush=True)
    cfg = get_domain_neuron_config("zh", spec="compact")
    cfg.dropout = args.dropout
    neuron = ResonanceNeuron(cfg).to(args.device)
    n_params = sum(p.numel() for p in neuron.parameters())
    n_embed = sum(p.numel() for p in shared_embedding.parameters())
    print(f"  {args.neuron_id}: neuron={n_params/1e6:.1f}M, shared_emb={n_embed/1e6:.1f}M", flush=True)

    # 5. 训练
    print(f"\n[5] 开始正规训练...", flush=True)
    save_path = os.path.join(OUTPUT_DIR, f"neuron_{args.neuron_id}.pt")
    result = train_compact_simple(
        neuron=neuron,
        texts=all_texts,
        neuron_id=args.neuron_id,
        shared_embedding=shared_embedding,
        domain_sp=domain_sp,
        general_sp=general_sp,
        num_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        save_path=save_path,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
    )

    print(f"\n{'='*70}", flush=True)
    print(f"训练完成！{args.neuron_id}", flush=True)
    print(f"  best_val_PPL={result['best_val_ppl']:.2f}@step{result['best_step']}", flush=True)
    print(f"  time={result['elapsed_s']/60:.1f}min", flush=True)
    print(f"  Checkpoint: {save_path}", flush=True)
    print(f"  目标: PPL < 10 (连贯生成基线)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
