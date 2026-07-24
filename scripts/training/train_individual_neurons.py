"""两阶段训练 —— 阶段一：逐个训练神经元（每人不同数据）。

核心改进（对比联合训练）：
1. 数据多样性：10 个神经元各有独有数据（70%）+ 共享核心（30%），不是 10 个复制品
2. 训练效率：每次只训练 1 个神经元，每步快 10 倍，每人获得 100% 梯度
3. 正则化：dropout + weight_decay + WSD 学习率调度（社区规范 SmolLM3）
4. 数据/参数比：每人 ~74K 文本 ≈ 43M tokens / 36M params ≈ 1.2（远优于联合的 0.006）
5. Best 步保存：按滑动 avg loss 保存最佳模型，非末步

Usage:
    # 逐个训练 10 个 zh compact 神经元，每人 12000 步
    python -u scripts/training/train_individual_neurons.py --n_neurons 10 --steps 12000

    # 减少步数快速验证
    python -u scripts/training/train_individual_neurons.py --n_neurons 10 --steps 4000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import random

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed

# 复用 train_neuron.py 的 tokenizer 加载函数
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    load_or_create_shared_embedding, save_shared_embedding,
    OUTPUT_DIR, SHARED_EMBEDDING_PATH,
)

DATA_PATH = "data/distill/zh_texts.jsonl"


# ── 数据分割：共享核心 + 独有数据 ──────────────────────────────────────────

def load_and_split_texts(
    data_path: str,
    n_neurons: int,
    max_texts: int = 200000,
    shared_ratio: float = 0.3,
    seed: int = 42,
) -> list[list[str]]:
    """加载文本并分割为 n_neurons 份，每份 = 共享核心 + 独有数据。

    分割策略：
    - 总数据池随机打乱
    - 30% 作为共享核心（所有神经元都训练，建立共同语言基础）
    - 70% 等分为 n_neurons 份独有数据（每人不同，保证多样性）

    Args:
        data_path: zh_texts.jsonl 路径
        n_neurons: 神经元数量
        max_texts: 最多加载多少条文本（内存控制）
        shared_ratio: 共享比例（默认 0.3）
        seed: 随机种子

    Returns:
        list of n_neurons 个 text lists，每个是该神经元的训练数据
    """
    print(f"  加载文本: {data_path}", flush=True)
    all_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_texts:
                break
            line = line.strip()
            if len(line) >= 10:  # 跳过太短的行
                all_texts.append(line)

    print(f"  加载 {len(all_texts)} 条非空文本（上限 {max_texts}）", flush=True)

    # 随机打乱
    rng = random.Random(seed)
    rng.shuffle(all_texts)

    # 分割：共享核心 + 独有数据
    n_shared = int(len(all_texts) * shared_ratio)
    shared_core = all_texts[:n_shared]
    unique_pool = all_texts[n_shared:]

    # 独有数据等分
    per_neuron_unique = len(unique_pool) // n_neurons
    subsets = []
    for i in range(n_neurons):
        start = i * per_neuron_unique
        end = start + per_neuron_unique
        unique_i = unique_pool[start:end]
        # 每个神经元 = 共享核心 + 独有数据
        neuron_texts = shared_core + unique_i
        rng.shuffle(neuron_texts)  # 再次打乱，避免共享数据集中在开头
        subsets.append(neuron_texts)

    # 统计
    total_chars = sum(len(t) for t in all_texts)
    per_neuron_chars = sum(len(t) for t in subsets[0])
    print(f"  数据分割完成:", flush=True)
    print(f"    总文本: {len(all_texts)} 条, {total_chars/1e6:.1f}M 字符", flush=True)
    print(f"    共享核心: {n_shared} 条 ({shared_ratio*100:.0f}%)", flush=True)
    print(f"    每人独有: {per_neuron_unique} 条 ({(1-shared_ratio)*100:.0f}% / {n_neurons})", flush=True)
    print(f"    每人总计: {len(subsets[0])} 条, {per_neuron_chars/1e6:.1f}M 字符", flush=True)
    print(f"    估计 tokens/人: {per_neuron_chars/1.7/1e6:.1f}M", flush=True)
    print(f"    数据/参数比 (36M): {per_neuron_chars/1.7/36e6:.2f}", flush=True)

    return subsets


# ── 单神经元训练（带正则化）──────────────────────────────────────────────────

def train_one_neuron(
    neuron: ResonanceNeuron,
    texts: list[str],
    neuron_id: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 12000,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    log_every: int = 200,
    save_path: str = None,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    freeze_embedding: bool = False,
) -> dict:
    """训练单个神经元（带正则化 + WSD 调度 + best 步保存）。

    正则化（社区规范 SmolLM3 / Chinchilla）：
    - weight_decay: AdamW 权重衰减（默认 0.1）
    - warmup_steps: WSD 调度 warmup 步数（默认 200）
    - dropout: 在 NeuronConfig 层面设置（调用前已设好）

    WSD 学习率调度：
    - Warmup（线性升温）→ Stable（稳定）→ Decay（余弦衰减到最后 10%）
    - decay 在最后 20% 步数进行
    """
    n_texts = len(texts)

    def _sample_batch() -> list[str]:
        idx = torch.randint(0, n_texts, (batch_size,))
        return [texts[int(i)] for i in idx]

    # 优化器：neuron 参数 +（可选）shared_embedding
    all_params = list(neuron.parameters())
    if not freeze_embedding:
        all_params += list(shared_embedding.parameters())

    optimizer = torch.optim.AdamW(
        all_params, lr=lr, weight_decay=weight_decay,
    )

    # WSD 学习率调度
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

    for _ in range(num_steps):
        batch_texts = _sample_batch()

        # 数据对齐：text → shared_emb + domain targets
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
            print(
                f"  [{neuron_id}] step {step}/{num_steps} "
                f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                f"PPL={ppl:.1f} lr={current_lr:.2e} "
                f"best={best_loss:.4f}@{best_step} "
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


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="两阶段训练 —— 阶段一：逐个训练神经元")
    parser.add_argument("--n_neurons", type=int, default=10,
                        help="神经元数量（默认 10）")
    parser.add_argument("--steps", type=int, default=12000,
                        help="每个神经元训练步数（默认 12000）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--spec", default="compact",
                        help="神经元规格 (compact/standard/expert)")
    parser.add_argument("--max_texts", type=int, default=200000,
                        help="最多加载多少条文本（内存控制）")
    parser.add_argument("--shared_ratio", type=float, default=0.3,
                        help="共享核心比例（默认 0.3 = 30%）")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW 权重衰减（默认 0.1）")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="WSD 调度 warmup 步数（默认 200）")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout 率（默认 0.1）")
    parser.add_argument("--freeze_after_first", action="store_true",
                        help="第一个神经元训练 embedding，之后冻结复用（节省 60% 训练时间）")
    parser.add_argument("--data_path", default=DATA_PATH,
                        help="训练数据路径")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="checkpoint 保存目录")
    parser.add_argument("--skip_neurons", type=int, default=0,
                        help="跳过前 N 个神经元（断点续训用）")
    parser.add_argument("--resume", action="store_true",
                        help="从已有ckpt加载best状态继续训练（断点续训）")
    parser.add_argument("--resume_lr", type=float, default=1e-4,
                        help="续训学习率（默认1e-4，比初始3e-4低）")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"两阶段训练 —— 阶段一：逐个训练 {args.n_neurons} 个神经元", flush=True)
    print(f"  规格: {args.spec}, 步数/人: {args.steps}", flush=True)
    print(f"  正则化: weight_decay={args.weight_decay} dropout={args.dropout} "
          f"warmup={args.warmup_steps}", flush=True)
    print(f"  数据: {args.data_path}, max_texts={args.max_texts}", flush=True)
    print(f"  共享比例: {args.shared_ratio*100:.0f}% 共享 + "
          f"{(1-args.shared_ratio)*100:.0f}% 独有", flush=True)
    print(f"  freeze_after_first: {args.freeze_after_first}", flush=True)
    print("=" * 70, flush=True)

    # ── 1. 加载并分割数据 ──
    print(f"\n[1] 加载并分割训练数据...", flush=True)
    subsets = load_and_split_texts(
        args.data_path,
        n_neurons=args.n_neurons,
        max_texts=args.max_texts,
        shared_ratio=args.shared_ratio,
    )

    # ── 2. 加载 tokenizers ──
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    print(f"  domain vocab={domain_sp.vocab_size()}, general vocab={general_sp.vocab_size()}", flush=True)

    # ── 3. 创建/加载 shared_embedding ──
    print(f"\n[3] 加载 shared_embedding...", flush=True)
    shared_embedding = load_or_create_shared_embedding(args.device)
    print(f"  shared_embedding: {shared_embedding.num_embeddings} × {shared_embedding.embedding_dim}", flush=True)

    # ── 4. 逐个训练神经元 ──
    print(f"\n[4] 开始逐个训练 {args.n_neurons} 个神经元...", flush=True)
    print(f"  预计每步 ~0.7s, 每人 {args.steps} 步 ≈ {args.steps * 0.7 / 60:.0f} 分钟", flush=True)
    print(f"  总预计: {args.n_neurons * args.steps * 0.7 / 3600:.1f} 小时", flush=True)
    print()

    all_results = []
    total_start = time.time()
    embedding_frozen = False  # 是否已冻结 embedding

    for i in range(args.n_neurons):
        if i < args.skip_neurons:
            print(f"\n--- 跳过神经元 {i}（断点续训）---", flush=True)
            continue

        neuron_id = f"zh_j{i}"
        print(f"\n{'='*60}", flush=True)
        print(f"[{neuron_id}] 神经元 {i+1}/{args.n_neurons}", flush=True)
        print(f"  训练数据: {len(subsets[i])} 条文本", flush=True)

        # freeze_after_first 逻辑：第一个神经元训练 embedding，之后冻结
        if args.freeze_after_first and i > 0 and not embedding_frozen:
            shared_embedding.requires_grad_(False)
            shared_embedding.eval()
            embedding_frozen = True
            print(f"  ⚡ shared_embedding 已冻结（复用神经元 0 的训练结果）", flush=True)

        # 创建神经元
        cfg = get_domain_neuron_config("zh", spec=args.spec)
        cfg.dropout = args.dropout  # 正则化
        neuron = ResonanceNeuron(cfg).to(args.device)
        n_params = sum(p.numel() for p in neuron.parameters())
        emb_status = "冻结" if embedding_frozen else "可训练"
        print(f"  参数: {n_params/1e6:.1f}M, dropout={cfg.dropout}, embedding={emb_status}", flush=True)

        # 断点续训：从已有 ckpt 加载 best 状态
        resume_ckpts_path = os.path.join(args.output_dir, f"neuron_zh_j{i}.pt")
        if args.resume and os.path.exists(resume_ckpts_path):
            old_ckpt = torch.load(resume_ckpts_path, map_location=args.device, weights_only=False)
            neuron.load_state_dict(old_ckpt["state_dict"], strict=False)
            old_best = old_ckpt.get("result", {}).get("best_loss", "?")
            old_step = old_ckpt.get("result", {}).get("best_step", "?")
            print(f"  📂 续训: 加载已有权重 (best_loss={old_best}@step{old_step})", flush=True)

        # 续训时用更低 lr
        current_lr = args.resume_lr if (args.resume and os.path.exists(resume_ckpts_path)) else args.lr
        if current_lr != args.lr:
            print(f"  续训 lr={current_lr} (低于初始 {args.lr})", flush=True)

        # 训练
        save_path = os.path.join(args.output_dir, f"neuron_zh_j{i}.pt")
        result = train_one_neuron(
            neuron=neuron,
            texts=subsets[i],
            neuron_id=neuron_id,
            shared_embedding=shared_embedding,
            domain_sp=domain_sp,
            general_sp=general_sp,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=current_lr,
            device=args.device,
            log_every=args.log_every,
            save_path=save_path,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            freeze_embedding=embedding_frozen,
        )
        all_results.append(result)

        # 第一个神经元训练后保存 shared_embedding
        if args.freeze_after_first and i == 0:
            save_shared_embedding(shared_embedding)
            print(f"  ⚡ shared_embedding 已保存（后续神经元复用）", flush=True)

        elapsed_so_far = time.time() - total_start
        remaining = (args.n_neurons - i - 1) * (elapsed_so_far / max(i + 1 - args.skip_neurons, 1))
        print(f"\n  已完成 {i+1}/{args.n_neurons}, "
              f"已用 {elapsed_so_far/3600:.1f}h, "
              f"预计剩余 {remaining/3600:.1f}h", flush=True)

    # ── 5. 保存 shared_embedding（如果未冻结）──
    if not embedding_frozen:
        print(f"\n[5] 保存 shared_embedding...", flush=True)
        save_shared_embedding(shared_embedding)

    # ── 6. 汇总 ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"训练完成！总耗时: {total_elapsed/3600:.1f} 小时", flush=True)
    print(f"{'neuron':<12} {'loss':<10} {'PPL':<10} {'best_loss':<12} {'best_step':<10} {'time_min':<10}", flush=True)
    print("-" * 65, flush=True)
    for r in all_results:
        print(f"{r['neuron_id']:<12} {r['final_loss']:<10.4f} {r['final_ppl']:<10.1f} "
              f"{r['best_loss']:<12.4f} {r['best_step']:<10} {r['elapsed_s']/60:<10.1f}", flush=True)
    print(f"\nCheckpoints: {args.output_dir}/neuron_zh_j*.pt", flush=True)

    # 检查是否所有神经元都达标（PPL < 50）
    qualified = sum(1 for r in all_results if r["final_ppl"] < 50)
    print(f"\n达标神经元 (PPL<50): {qualified}/{len(all_results)}", flush=True)
    if qualified == len(all_results):
        print("✅ 所有神经元达标，可进入阶段二（协作训练）", flush=True)
    else:
        print("⚠️ 部分神经元未达标，可能需要增加步数或检查数据", flush=True)


if __name__ == "__main__":
    main()
