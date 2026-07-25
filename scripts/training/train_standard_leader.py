"""训练 standard 族长神经元（111M），突破 compact argmax 天花板。

核心改进（对比 train_individual_neurons.py）：
1. 全量数据：加载全部 4.3M 文本（不分割，100% 唯一，无共享核心重复）
2. 顺序 epoch 采样：shuffle → 顺序遍历 → 重洗，保证每步看到新内容（vs 随机采样 2% 利用率）
3. 更大 batch（8）：提高单步数据覆盖
4. 更低 lr（1e-4）：standard 3x 参数，需更低学习率防发散
5. 复用冻结 shared_embedding：与 9 个 compact 跟随者兼容

Usage:
    python -u scripts/training/train_standard_leader.py --steps 8000
    python -u scripts/training/train_standard_leader.py --steps 12000 --batch_size 8
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
    load_or_create_shared_embedding, save_shared_embedding,
    OUTPUT_DIR, SHARED_EMBEDDING_PATH,
)

DATA_PATH = "data/distill/zh_texts.jsonl"


def load_all_texts(data_path: str, max_texts: int = 10000000, min_len: int = 10) -> list[str]:
    """加载全部文本（不分割，给 standard 族长独享全部数据）。"""
    print(f"  加载文本: {data_path}", flush=True)
    all_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_texts:
                break
            line = line.strip()
            if len(line) >= min_len:
                all_texts.append(line)

    total_chars = sum(len(t) for t in all_texts)
    print(f"  加载 {len(all_texts)} 条非空文本（上限 {max_texts}）", flush=True)
    print(f"  总字符: {total_chars/1e6:.1f}M, 估计 tokens: {total_chars/1.7/1e6:.0f}M", flush=True)
    print(f"  数据/参数比 (111M): {total_chars/1.7/111e6:.2f}", flush=True)
    return all_texts


class SequentialSampler:
    """顺序 epoch 采样：shuffle → 顺序遍历 → 重洗，保证每步看到新内容。

    对比随机采样（torch.randint）：
    - 随机采样：8000步×batch4=32K样本，从1.6M池中随机抽，利用率2%
    - 顺序采样：8000步×batch8=64K样本，前64K条全部唯一，利用率100%（无重复）
    """

    def __init__(self, texts: list[str], batch_size: int, seed: int = 42):
        self.texts = texts
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices = list(range(len(texts)))
        self.rng.shuffle(self.indices)
        self.cursor = 0
        self.epoch = 0
        self.n_texts = len(texts)

    def sample_batch(self) -> list[str]:
        """获取下一批，顺序遍历，epoch 结束自动重洗。"""
        if self.cursor + self.batch_size > self.n_texts:
            # Epoch 结束，重洗
            self.rng.shuffle(self.indices)
            self.cursor = 0
            self.epoch += 1

        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return [self.texts[i] for i in batch_indices]

    @property
    def unique_seen(self) -> int:
        """已看到的唯一文本数（当前 epoch 内）。"""
        return min(self.cursor, self.n_texts)


def train_standard_leader(
    neuron: ResonanceNeuron,
    texts: list[str],
    neuron_id: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 8000,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
    log_every: int = 200,
    save_path: str = None,
    weight_decay: float = 0.1,
    warmup_steps: int = 300,
    freeze_embedding: bool = True,
) -> dict:
    """训练 standard 族长（带顺序采样 + WSD 调度 + best 步保存）。

    Args:
        freeze_embedding: True=复用已冻结 shared_embedding（与 compact 兼容）
                         False=训练 shared_embedding（首次需此）
    """
    sampler = SequentialSampler(texts, batch_size, seed=42)

    # 优化器
    all_params = list(neuron.parameters())
    if not freeze_embedding:
        all_params += list(shared_embedding.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # WSD 学习率调度（standard 用更长 warmup=300）
    decay_start = max(warmup_steps + 1, int(num_steps * 0.8))
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
    if not freeze_embedding:
        shared_embedding.train()

    total_loss = 0.0
    step, t_start = 0, time.time()
    best_loss = float("inf")
    best_step = 0
    best_state = None
    recent_losses = []

    print(f"\n  [{neuron_id}] 开始训练: {num_steps} 步, batch={batch_size}, lr={lr}", flush=True)
    print(f"  预计每步 ~{0.7 * (111/36) * (batch_size/4):.1f}s, 总计 ~{num_steps * 0.7 * (111/36) * (batch_size/4) / 3600:.1f}h", flush=True)

    for _ in range(num_steps):
        batch_texts = sampler.sample_batch()

        # 数据对齐
        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding,
        )
        shared_emb = shared_emb.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        # Forward
        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]

        # Shift for next-token prediction
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

        # 滑动窗口 avg loss 追踪 best
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
                f"epoch={sampler.epoch} unique={unique_pct:.1f}% "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    avg_loss = total_loss / max(step, 1)
    ppl = math.exp(min(avg_loss, 20))
    elapsed = time.time() - t_start

    print(
        f"\n  [{neuron_id}] Done. {step} steps, "
        f"avg_loss={avg_loss:.4f}, PPL={ppl:.1f}, "
        f"best_loss={best_loss:.4f}@step{best_step}, "
        f"time={elapsed:.0f}s ({elapsed/60:.1f}min)",
        flush=True,
    )

    # 保存 best 模型
    save_state = best_state if best_state is not None else neuron.state_dict()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "neuron_config": neuron.config,
        "state_dict": save_state,
        "domain": "zh",
        "result": {
            "final_loss": avg_loss,
            "final_ppl": ppl,
            "steps": step,
            "best_loss": best_loss,
            "best_step": best_step,
            "saved": "best" if best_state is not None else "final",
            "spec": "standard",
            "role": "tribal_leader",
        },
    }, save_path)
    print(f"  Saved: {save_path} (best@step{best_step}, loss={best_loss:.4f})", flush=True)

    return {
        "neuron_id": neuron_id,
        "final_loss": avg_loss,
        "final_ppl": ppl,
        "best_loss": best_loss,
        "best_step": best_step,
        "elapsed_s": elapsed,
        "save_path": save_path,
    }


def main():
    parser = argparse.ArgumentParser(description="训练 standard 族长神经元")
    parser.add_argument("--steps", type=int, default=8000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size（更大=更多数据覆盖）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（standard 默认 1e-4）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--max_texts", type=int, default=10000000, help="最大加载文本数")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=300, help="WSD warmup（standard 用 300）")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--neuron_id", default="zh_leader0", help="族长神经元 ID")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"训练 standard 族长神经元（突破 compact argmax 天花板）", flush=True)
    print(f"  规格: standard (hidden=768, layers=10, ~111M params)", flush=True)
    print(f"  步数: {args.steps}, batch: {args.batch_size}, lr: {args.lr}", flush=True)
    print(f"  正则化: weight_decay={args.weight_decay} dropout={args.dropout} warmup={args.warmup_steps}", flush=True)
    print(f"  数据: {args.data_path} (全量加载，不分割)", flush=True)
    print(f"  采样: 顺序 epoch（无重复，100% 利用率）", flush=True)
    print(f"  shared_embedding: 冻结复用（与 9 compact 兼容）", flush=True)
    print("=" * 70, flush=True)

    # 1. 加载全部数据
    print(f"\n[1] 加载全部训练数据...", flush=True)
    all_texts = load_all_texts(args.data_path, max_texts=args.max_texts)

    # 2. 加载 tokenizers
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    print(f"  domain vocab={domain_sp.vocab_size()}, general vocab={general_sp.vocab_size()}", flush=True)

    # 3. 加载 shared_embedding（冻结复用）
    print(f"\n[3] 加载 shared_embedding（冻结复用）...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)
    shared_embedding.requires_grad_(False)
    shared_embedding.eval()
    print(f"  shared_embedding: {shared_embedding.num_embeddings} × {shared_embedding.embedding_dim} (frozen)", flush=True)

    # 4. 创建 standard 族长
    print(f"\n[4] 创建 standard 族长神经元...", flush=True)
    cfg = get_domain_neuron_config("zh", spec="standard")
    cfg.dropout = args.dropout
    # 启用突触投影：field_dim=3072 → unified_field_dim=4096（与 compact 混合时投影）
    # 但目前 9 个 compact 用 field_dim=2048，standard 用 3072
    # field_projector 会处理：Linear(3072 → unified) 和 Linear(2048 → unified)
    cfg.unified_field_dim = 4096  # 统一场空间
    neuron = ResonanceNeuron(cfg).to(args.device)
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  {args.neuron_id}: spec=standard, params={n_params/1e6:.1f}M", flush=True)
    print(f"  hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, field_dim={cfg.field_dim}", flush=True)
    print(f"  unified_field_dim={cfg.unified_field_dim} (突触投影启用)", flush=True)

    # 5. 训练
    print(f"\n[5] 开始训练...", flush=True)
    save_path = os.path.join(args.output_dir, f"neuron_{args.neuron_id}.pt")
    result = train_standard_leader(
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
        freeze_embedding=True,
    )

    # 6. 保存 shared_embedding（虽然是冻结的，但确保一致性）
    print(f"\n[6] 确认 shared_embedding 未变（冻结）...", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"训练完成！族长 {args.neuron_id}", flush=True)
    print(f"  best_loss={result['best_loss']:.4f}@step{result['best_step']}", flush=True)
    print(f"  final PPL={result['final_ppl']:.1f}", flush=True)
    print(f"  time={result['elapsed_s']/60:.1f}min", flush=True)
    print(f"  Checkpoint: {save_path}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n下一步: 运行 eval_single.py 评估族长 argmax（目标 85%+）", flush=True)


if __name__ == "__main__":
    main()
