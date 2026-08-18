"""TinyStories 验证实验：纯 Transformer baseline。

基于 nanoGPT 设计，用项目现有的 TransformerBlock/RMSNorm 组件。
目标：验证训练 pipeline 是否正确（能否生成连贯英文故事）。

如果这个 baseline 能生成连贯文本 → pipeline 正确，问题在数据/架构
如果 baseline 也乱码 → pipeline 有 bug

配置：~10M 参数, GPT-2 BPE (50257), batch=32, lr=1e-3
符合 AI_TRAINING_PLAYBOOK.md 准则。
"""
from __future__ import annotations

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.layers import TransformerBlock, RMSNorm

# ── 配置 ──
class Config:
    vocab_size = 50257       # GPT-2 BPE
    hidden_size = 192        # CPU 友好 (~10M 参数)
    num_layers = 4
    num_heads = 4
    num_kv_heads = 4         # MHA (不用 GQA，简化)
    intermediate_size = 768  # SwiGLU
    block_size = 128         # 序列长度 (CPU 友好)
    rms_norm_eps = 1e-5
    dropout = 0.1
    # 训练
    batch_size = 12          # nanoGPT CPU 配置
    lr = 1e-3                # 小模型用高 lr (Playbook)
    max_iters = 3000
    warmup_iters = 100
    eval_interval = 500
    eval_iters = 30
    save_path = "data/tinystories/baseline_model.pt"


class PureTransformerLM(nn.Module):
    """纯 Transformer 语言模型（nanoGPT 风格）。

    不含 field_write/read/projector 等额外组件。
    Tied embeddings（输入输出共享）节省参数。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                intermediate_size=cfg.intermediate_size,
                rms_norm_eps=cfg.rms_norm_eps,
                bias=False,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])
        self.norm_f = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Tied embedding: lm_head 权重 = tok_emb 权重
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

        # 初始化
        self.apply(self._init_weights)
        # 对残差层缩放 (GPT-2 风格)
        for pn, p in self.named_parameters():
            if pn.endswith("attention.out_proj.weight") or pn.endswith("feed_forward.w2.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.hidden_size ** -0.5 / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * float('-inf')

        for block in self.blocks:
            x, _, _ = block(x, mask=mask)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx


def load_data():
    """加载二进制 token 数据。"""
    train_data = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/tinystories/val.bin", dtype=np.uint16, mode="r")
    print(f"Train: {len(train_data)} tokens ({len(train_data)/1e6:.1f}M)")
    print(f"Val:   {len(val_data)} tokens ({len(val_data)/1e6:.1f}M)")
    return train_data, val_data


def get_batch(data, cfg: Config, device='cpu'):
    """随机采样一个 batch。"""
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: Config):
    """评估 train/val loss。"""
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def generate_sample(model, cfg: Config, enc, prompt="Once upon a time"):
    """生成样本文本。"""
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    text = enc.decode(out[0].tolist())
    return text


def main():
    cfg = Config()
    device = 'cpu'
    enc = tiktoken.get_encoding("gpt2")

    print("=" * 60)
    print("TinyStories 验证实验：纯 Transformer baseline")
    print("=" * 60)
    print(f"模型: {cfg.num_layers}层, {cfg.num_heads}头, hidden={cfg.hidden_size}")
    print(f"block_size={cfg.block_size}, batch={cfg.batch_size}, lr={cfg.lr}")
    print(f"max_iters={cfg.max_iters}, warmup={cfg.warmup_iters}")

    # 参数量
    model = PureTransformerLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.1f}M")

    # 数据
    print(f"\n[1] 加载数据...")
    train_data, val_data = load_data()
    data_param_ratio = len(train_data) / n_params
    print(f"数据/参数比: {data_param_ratio:.1f} (Chinchilla 最优 20:1)")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        weight_decay=0.1,
    )

    # 学习率调度 (cosine with warmup)
    def get_lr(it):
        if it < cfg.warmup_iters:
            return cfg.lr * it / cfg.warmup_iters
        if it > cfg.max_iters:
            return cfg.lr * 0.1
        decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.lr * 0.1 + 0.9 * cfg.lr * coeff

    # 训练
    print(f"\n[2] 开始训练...")
    best_val_loss = float('inf')
    t0 = time.time()

    for it in range(cfg.max_iters):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch(train_data, cfg, device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {it:5d}/{cfg.max_iters} loss={loss.item():.4f} lr={lr:.2e} "
                  f"PPL={math.exp(loss.item()):.1f} elapsed={elapsed:.0f}s")

        if (it + 1) % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"\n  ── 评估 step {it+1} ──")
            print(f"  train loss={losses['train']:.4f} PPL={math.exp(losses['train']):.1f}")
            print(f"  val   loss={losses['val']:.4f} PPL={math.exp(losses['val']):.1f}")

            # 生成样本
            sample = generate_sample(model, cfg, enc, "Once upon a time")
            print(f"  生成样本: {sample[:300]}...")
            print()

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state': model.state_dict(),
                    'config': cfg.__dict__,
                    'val_loss': best_val_loss,
                }, cfg.save_path)
                print(f"  ✅ 保存 best model (val_loss={best_val_loss:.4f})")

    # 最终生成
    print(f"\n[3] 最终生成样本:")
    print("=" * 60)
    for prompt in ["Once upon a time", "The little bear", "In a forest"]:
        sample = generate_sample(model, cfg, enc, prompt)
        print(f"\n提示: {prompt}")
        print(f"生成: {sample}")
        print("-" * 60)

    elapsed = time.time() - t0
    print(f"\n训练完成！总时间: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Best val loss: {best_val_loss:.4f} (PPL={math.exp(best_val_loss):.1f})")
    print(f"模型保存: {cfg.save_path}")


if __name__ == "__main__":
    main()
