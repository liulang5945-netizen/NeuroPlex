"""并行训练多个 compact 神经元（差异化数据 + shared_embedding 冻结模式）。

支持：
1. 多数据文件合并（共享核心 + 独有数据）
2. shared_embedding 模式：train（第一个神经元）/ frozen（后续神经元复用）
3. 线程限制（并行训练时不抢资源）
4. 正规配置（batch≥32, embedding不衰减, lr=1e-3, WSD调度）

Usage:
    # 第一个神经元（训练 shared_embedding）
    python -u scripts/training/train_compact_parallel.py \\
        --neuron_id zh_par0 --data_files shared_core.jsonl class_a_chinese.jsonl \\
        --shared_emb_mode train --threads 6

    # 第二个神经元（冻结 shared_embedding，并行）
    python -u scripts/training/train_compact_parallel.py \\
        --neuron_id zh_par1 --data_files shared_core.jsonl class_b_encyclopedia.jsonl \\
        --shared_emb_mode frozen --threads 6

    # 第三个神经元（冻结 shared_embedding，并行）
    python -u scripts/training/train_compact_parallel.py \\
        --neuron_id zh_par2 --data_files shared_core.jsonl class_c_story.jsonl \\
        --shared_emb_mode frozen --threads 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    load_or_create_shared_embedding,
    OUTPUT_DIR, SequentialSampler,
)

DATA_DIR = "data/simple_zh"

# 同域 compact 神经元列表（用于 side_channels 预建立）
ZH_COMPACT_NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]


def load_multi_texts(data_files: list[str], max_texts: int = 10000000) -> list[str]:
    """加载多个数据文件并合并。"""
    texts = []
    for fname in data_files:
        path = os.path.join(DATA_DIR, fname) if not os.path.isabs(fname) else fname
        if not os.path.exists(path):
            print(f"  [WARN] 文件不存在: {path}", flush=True)
            continue
        count = 0
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
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  {fname}: {count} 条", flush=True)
    return texts


class AugmentingSampler:
    """在 SequentialSampler 之上加数据增强（不增加复杂度，只增多样性）。

    增强 1：随机截断 — 每条文本随机保留 50%-100% 长度（句末截断，保完整性）
    增强 2：片段拼接 — 30% 概率把另一条短文本追加到当前文本（用换行分隔）

    不改数据复杂度（仍是 simple_zh 小学水平），只增加 epoch 间的多样性，
    防止模型逐字记忆训练集特定 token 序列。
    """

    def __init__(self, texts: list[str], batch_size: int, seed: int = 42,
                 truncate_min_ratio: float = 0.5, concat_prob: float = 0.3):
        from scripts.training.utils import SequentialSampler
        self.base = SequentialSampler(texts, batch_size, seed=seed)
        self.texts = texts
        self.n_texts = len(texts)
        self.rng = random.Random(seed + 1)
        self.truncate_min_ratio = truncate_min_ratio
        self.concat_prob = concat_prob
        # 预筛选短文本（< 200 字）用于拼接，避免长+长超长
        self._short_texts = [t for t in texts if 20 <= len(t) <= 200]
        print(f"  [AugmentingSampler] truncate_min={truncate_min_ratio}, "
              f"concat_prob={concat_prob}, short_pool={len(self._short_texts)}", flush=True)

    def _augment_one(self, text: str) -> str:
        """对单条文本做随机截断 + 拼接。"""
        # 增强 1：随机截断到 50%-100% 长度，按句末标点截断保完整
        if len(text) > 50:
            ratio = self.rng.uniform(self.truncate_min_ratio, 1.0)
            keep_chars = max(30, int(len(text) * ratio))
            sub = text[:keep_chars]
            # 找最后一个句末标点（。。！？）
            last_punct = max(
                sub.rfind('。'), sub.rfind('！'), sub.rfind('？'),
                sub.rfind('.'), sub.rfind('!'), sub.rfind('?'),
            )
            if last_punct > 30:
                text = sub[:last_punct + 1]

        # 增强 2：30% 概率拼接另一条短文本
        if (self._short_texts and
                self.rng.random() < self.concat_prob and
                len(text) < 300):
            other = self.rng.choice(self._short_texts)
            text = text + '\n' + other
        return text

    def sample_batch(self) -> list[str]:
        batch = self.base.sample_batch()
        return [self._augment_one(t) for t in batch]

    @property
    def unique_seen(self) -> int:
        return self.base.unique_seen

    @property
    def epoch(self) -> int:
        return self.base.epoch


def train_parallel(
    neuron: ResonanceNeuron,
    texts: list[str],
    neuron_id: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 16000,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 200,
    save_path: str = None,
    warmup_steps: int = 200,
    eval_every: int = 2000,
    shared_emb_mode: str = "train",
    augment: bool = True,
    truncate_min_ratio: float = 0.5,
    concat_prob: float = 0.3,
    spec: str = "compact",
) -> dict:
    """并行训练配置。"""
    if augment:
        sampler = AugmentingSampler(
            texts, batch_size, seed=42,
            truncate_min_ratio=truncate_min_ratio,
            concat_prob=concat_prob,
        )
    else:
        sampler = SequentialSampler(texts, batch_size, seed=42)

    # 参数组分离
    embed_params = list(shared_embedding.parameters())
    neuron_params = list(neuron.parameters())

    if shared_emb_mode == "frozen":
        # 冻结 shared_embedding
        for p in embed_params:
            p.requires_grad = False
        optimizer = torch.optim.AdamW([
            {"params": neuron_params, "weight_decay": 0.1},
        ], lr=lr, betas=(0.9, 0.99))
        trainable_params = neuron_params
        print(f"  shared_embedding: FROZEN（复用已有权重）", flush=True)
    else:
        # 训练 shared_embedding
        optimizer = torch.optim.AdamW([
            {"params": neuron_params, "weight_decay": 0.1},
            {"params": embed_params, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.99))
        trainable_params = neuron_params + embed_params
        print(f"  shared_embedding: TRAINABLE（第一个神经元训练）", flush=True)

    # WSD 调度
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
    if shared_emb_mode == "train":
        shared_embedding.train()

    total_loss = 0.0
    step, t_start = 0, time.time()
    best_val_loss = float("inf")
    best_step = 0
    best_state = None
    best_embed_state = None
    recent_losses = []

    eval_texts = texts[-100:]
    effective_batch = batch_size * grad_accum

    print(f"\n  [{neuron_id}] 并行训练开始:", flush=True)
    print(f"    batch={batch_size} × grad_accum={grad_accum} = effective {effective_batch}", flush=True)
    print(f"    lr={lr}, WSD调度, 数据={len(texts)} 条", flush=True)
    print(f"    steps={num_steps}, eval_every={eval_every}", flush=True)

    while step < num_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

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

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
            print(f"\n  [{neuron_id}] [EVAL] step {step}: val PPL={val_ppl:.2f}", flush=True)

            if val_ppl < best_val_loss:
                best_val_loss = val_ppl
                best_step = step
                best_state = {k: v.detach().clone() for k, v in neuron.state_dict().items()}
                if shared_emb_mode == "train":
                    best_embed_state = {k: v.detach().clone() for k, v in shared_embedding.state_dict().items()}
                print(f"    [SAVE] best (val PPL={best_val_loss:.2f})", flush=True)

            sample = generate_sample(neuron, domain_sp, general_sp, shared_embedding, device, prompt="小猫")
            print(f"    生成: {sample[:200]}", flush=True)
            print()
            neuron.train()

    save_state = best_state if best_state is not None else neuron.state_dict()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "neuron_config": neuron.config,
        "state_dict": save_state,
        "shared_embedding_state": best_embed_state if best_embed_state else None,
        "domain": "zh",
        "data_source": "simple_zh_split",
        "result": {
            "best_val_ppl": best_val_loss,
            "best_step": best_step,
            "steps": step,
            "saved": "best" if best_state is not None else "final",
            "spec": spec,
            "shared_emb_mode": shared_emb_mode,
        },
    }, save_path)

    # train 模式：自动保存 shared_embedding 到 data/shared_embedding.pt
    # 这样后续 frozen 模式的神经元可以复用（因果掩码修复后的干净 embedding）
    if shared_emb_mode == "train" and best_embed_state is not None:
        from scripts.training.utils import save_shared_embedding, SHARED_EMBEDDING_PATH
        save_shared_embedding(shared_embedding, SHARED_EMBEDDING_PATH)
        print(f"  [AUTO-SAVE] shared_embedding → {SHARED_EMBEDDING_PATH} "
              f"(供后续 frozen 模式神经元复用)", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  [{neuron_id}] Done. best_val_PPL={best_val_loss:.2f}@step{best_step}, time={elapsed/60:.1f}min", flush=True)
    print(f"  Saved: {save_path}", flush=True)
    return {"neuron_id": neuron_id, "best_val_ppl": best_val_loss, "best_step": best_step, "elapsed_s": elapsed}


def generate_sample(neuron, domain_sp, general_sp, shared_embedding, device, prompt="从前"):
    """生成中文样本（和训练输入对齐）。

    训练时输入：text → general_sp.encode → general_ids（build_position_alignment）
    生成时输入：prompt → general_sp.encode → general_ids（和训练一致）
    模型预测：domain token → domain_sp 解码
    自回归：domain token → piece → general_sp.encode → 追加所有 general ids
    """
    neuron.eval()
    with torch.no_grad():
        # 1. prompt → general_ids（用 general tokenizer，和训练一致）
        general_ids = general_sp.encode(prompt)
        if not general_ids:
            return "(empty)"

        # 2. 自回归生成（模型预测 domain token）
        generated_domain_ids = []
        for _ in range(80):
            emb_input = shared_embedding(torch.tensor([general_ids], device=device))
            result = neuron.forward(emb_input, return_logits=True)
            logits = result["logits"][:, -1, :]
            next_domain_id = logits.argmax(dim=-1).item()
            generated_domain_ids.append(next_domain_id)

            # domain token → general tokens（追加所有，和训练对齐一致）
            piece = domain_sp.decode([next_domain_id])
            gids = general_sp.encode(piece)
            if gids:
                general_ids.extend(gids)
            else:
                general_ids.append(general_sp.unk_id() if hasattr(general_sp, "unk_id") else 0)

        # 3. 用 domain_sp 解码生成的 domain tokens
        text = domain_sp.decode(generated_domain_ids)
    neuron.train()
    return text


def main():
    parser = argparse.ArgumentParser(description="并行训练 compact/standard 神经元（差异化数据）")
    parser.add_argument("--neuron_id", required=True)
    parser.add_argument("--data_files", nargs='+', required=True, help="数据文件名（在 data/simple_zh/ 下）")
    parser.add_argument("--spec", choices=["compact", "standard"], default="compact",
                        help="神经元规格：compact(36M) 或 standard(~100M)")
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--max_texts", type=int, default=10000000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=6, help="PyTorch 线程数（并行时限制）")
    parser.add_argument("--no_augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--truncate_min_ratio", type=float, default=0.5,
                        help="随机截断最小保留比例（0.5=保留 50%-100%）")
    parser.add_argument("--concat_prob", type=float, default=0.3,
                        help="片段拼接概率")
    parser.add_argument("--shared_emb_mode", choices=["train", "frozen"], default="frozen",
                        help="train=训练shared_embedding, frozen=冻结复用")
    args = parser.parse_args()

    # 限制线程数（并行训练不抢资源）
    torch.set_num_threads(args.threads)
    print(f"PyTorch 线程数: {args.threads}", flush=True)

    print("=" * 70, flush=True)
    print(f"并行训练 {args.spec} 神经元（差异化数据）", flush=True)
    print(f"  neuron_id: {args.neuron_id}", flush=True)
    print(f"  spec: {args.spec}", flush=True)
    print(f"  data_files: {args.data_files}", flush=True)
    print(f"  shared_emb_mode: {args.shared_emb_mode}", flush=True)
    print(f"  effective batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}", flush=True)
    print(f"  lr: {args.lr}, threads: {args.threads}", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载数据
    print(f"\n[1] 加载数据...", flush=True)
    all_texts = load_multi_texts(args.data_files, max_texts=args.max_texts)
    print(f"  合计: {len(all_texts)} 条", flush=True)

    # 2. tokenizers
    print(f"\n[2] tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    # 3. shared_embedding
    print(f"\n[3] shared_embedding (mode={args.shared_emb_mode})...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)

    # 4. 神经元
    print(f"\n[4] 创建 {args.spec} 神经元...", flush=True)
    cfg = get_domain_neuron_config("zh", spec=args.spec)
    cfg.dropout = args.dropout
    neuron = ResonanceNeuron(cfg).to(args.device)

    # 4.1 建立指向同域其他神经元的 side_channels（per-pair 突触投影）
    # 这样训练时 side_channels 也会通过反向传播学习如何转译 peer 的 field_vector
    # 注意：ZH_COMPACT_NEURON_IDS 中的神经元是 compact 规格，peer_cfg 固定 compact
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    for peer_id in ZH_COMPACT_NEURON_IDS:
        if peer_id == args.neuron_id:
            continue
        peer_neuron = ResonanceNeuron(peer_cfg).to(args.device)
        neuron.establish_side_channel(peer_id, peer_neuron, channel_type="excite")
        del peer_neuron
    n_side = len(neuron.excite_channels)
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  {args.neuron_id}: {n_params/1e6:.1f}M params ({n_side} excite side_channels)", flush=True)

    # 5. 训练
    print(f"\n[5] 开始训练...", flush=True)
    save_path = os.path.join(OUTPUT_DIR, f"neuron_{args.neuron_id}.pt")
    result = train_parallel(
        neuron=neuron, texts=all_texts, neuron_id=args.neuron_id,
        shared_embedding=shared_embedding, domain_sp=domain_sp, general_sp=general_sp,
        num_steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, device=args.device, log_every=args.log_every, save_path=save_path,
        warmup_steps=args.warmup_steps, eval_every=args.eval_every,
        shared_emb_mode=args.shared_emb_mode,
        augment=not args.no_augment,
        truncate_min_ratio=args.truncate_min_ratio,
        concat_prob=args.concat_prob,
        spec=args.spec,
    )

    print(f"\n{'='*70}", flush=True)
    print(f"完成！{args.neuron_id}", flush=True)
    print(f"  best_val_PPL={result['best_val_ppl']:.2f}@step{result['best_step']}", flush=True)
    print(f"  time={result['elapsed_s']/60:.1f}min", flush=True)
    print(f"  Checkpoint: {save_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
