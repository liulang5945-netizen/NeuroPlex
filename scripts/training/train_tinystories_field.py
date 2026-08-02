"""TinyStories 实验 B：field-augmented Transformer 消融对比。

基于 baseline (train_tinystories.py) 添加 field 组件：
- field_write: hidden → field_dim（L2 归一化）
- field_read_layers: per-layer, field_dim → hidden（门控残差）
- 两轮前向：round 1 独立，round 2 field conditioning

目的：严格消融验证 field 组件是否有用。
- 控制变量：同数据 + 同 tokenizer(GPT-2 BPE) + 同规模 + 同超参
- 唯一差异：是否有 field 组件

预期：
- 若 field-augmented PPL ≤ baseline(16.6) → field 组件有用
- 若 field-augmented PPL > baseline → field 组件在单神经元场景有害（预期，field 为协作设计）

符合 AI_TRAINING_PLAYBOOK.md 准则 0.3 "消融一切"。
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

from taiji.layers import TransformerBlock, RMSNorm


# ── 配置（和 baseline 完全一致，只多了 field_dim）──
class Config:
    vocab_size = 50257       # GPT-2 BPE（和 baseline 一致）
    hidden_size = 192        # CPU 友好 (~10M 参数，和 baseline 一致)
    num_layers = 4
    num_heads = 4
    num_kv_heads = 4         # MHA（和 baseline 一致）
    intermediate_size = 768  # SwiGLU（和 baseline 一致）
    block_size = 128         # 序列长度（和 baseline 一致）
    rms_norm_eps = 1e-5
    dropout = 0.1
    # field 组件配置（新增）
    field_dim = 192          # 和 hidden_size 一致（简化，避免额外参数膨胀）
    field_rounds = 2         # 两轮前向：round 1 独立，round 2 field conditioning
    # 训练（和 baseline 一致）
    batch_size = 12
    lr = 1e-3
    max_iters = 3000
    warmup_iters = 100
    eval_interval = 500
    eval_iters = 30
    save_path = "data/tinystories/field_model.pt"


class FieldAugmentedLM(nn.Module):
    """带 field 组件的 Transformer 语言模型。

    和 baseline 的 PureTransformerLM 唯一区别：添加了 field 组件。
    - field_write: 最后 hidden → field_dim（L2 归一化）
    - field_read_layers: per-layer, field_dim → hidden（门控残差）
    - 两轮前向：round 1 独立 forward + write field；round 2 用 field conditioning
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # ── 和 baseline 完全一致的部分 ──
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
        # Tied embedding（和 baseline 一致）
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

        # ── 新增：field 组件（模拟 ResonanceNeuron）──
        # field_write: 把 hidden 池化后写到 field 空间
        self.field_write = nn.Linear(cfg.hidden_size, cfg.field_dim, bias=False)
        # field_read_layers: per-layer, field → hidden conditioning
        self.field_read_layers = nn.ModuleList([
            nn.Linear(cfg.field_dim, cfg.hidden_size, bias=False)
            for _ in range(cfg.num_layers)
        ])
        # 门控（v2 风格：per-position gated read）
        self.field_read_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

        # 初始化（和 baseline 一致）
        self.apply(self._init_weights)
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

    def _run_blocks(self, x, mask, field_state=None, round_num=1):
        """运行 transformer blocks，可选 field conditioning。"""
        for i, block in enumerate(self.blocks):
            # 标准 transformer block
            h_normed = block.attention_norm(x)
            attn_out, _, _ = block.attention(h_normed, mask=mask)
            x = x + attn_out
            x = x + block.feed_forward(block.ffn_norm(x))

            # Field conditioning（round 2+ 才启用）
            if field_state is not None and round_num > 1:
                conditioning = self.field_read_layers[i](field_state)  # [B, hidden] or [1, hidden]
                if conditioning.dim() == 1:
                    conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
                else:
                    conditioning = conditioning.unsqueeze(1)  # [B, 1, hidden]
                # per-position gated read（v2 风格）
                gate = torch.sigmoid(self.field_read_gate(x))  # [B, L, 1]
                x = x + gate * conditioning
        return x

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * float('-inf')

        # ── Round 1: 独立 forward + write field ──
        h1 = self._run_blocks(x, mask, field_state=None, round_num=1)
        # attention-pooled field write（v2 风格，简化版：用最后一个 token）
        hidden_last = h1[:, -1, :]  # [B, hidden]
        v_raw = self.field_write(hidden_last)  # [B, field_dim]
        field_state = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)  # L2 归一化

        # ── Round 2: 重新 forward + field conditioning ──
        h2 = self._run_blocks(x, mask, field_state=field_state, round_num=2)

        x = self.norm_f(h2)
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
    """加载二进制 token 数据（和 baseline 共用）。"""
    train_data = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/tinystories/val.bin", dtype=np.uint16, mode="r")
    print(f"Train: {len(train_data)} tokens ({len(train_data)/1e6:.1f}M)")
    print(f"Val:   {len(val_data)} tokens ({len(val_data)/1e6:.1f}M)")
    return train_data, val_data


def get_batch(data, cfg: Config, device='cpu'):
    """随机采样一个 batch（和 baseline 一致）。"""
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: Config):
    """评估 train/val loss（和 baseline 一致）。"""
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
    print("TinyStories 实验 B：field-augmented Transformer 消融")
    print("=" * 60)
    print(f"模型: {cfg.num_layers}层, {cfg.num_heads}头, hidden={cfg.hidden_size}")
    print(f"field_dim={cfg.field_dim}, field_rounds={cfg.field_rounds}")
    print(f"block_size={cfg.block_size}, batch={cfg.batch_size}, lr={cfg.lr}")
    print(f"max_iters={cfg.max_iters}, warmup={cfg.warmup_iters}")

    # 参数量
    model = FieldAugmentedLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_field_params = sum(p.numel() for n, p in model.named_parameters()
                         if 'field_' in n)
    print(f"总参数量: {n_params/1e6:.2f}M")
    print(f"  field 组件参数: {n_field_params/1e6:.2f}M ({n_field_params/n_params*100:.1f}%)")
    print(f"  baseline 参考: ~12.0M")

    # 数据
    print(f"\n[1] 加载数据...")
    train_data, val_data = load_data()
    data_param_ratio = len(train_data) / n_params
    print(f"数据/参数比: {data_param_ratio:.1f} (Chinchilla 最优 20:1)")

    # 优化器（和 baseline 一致）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        weight_decay=0.1,
    )

    # 学习率调度（和 baseline 一致：cosine with warmup）
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
            print(f"  baseline 参考: val PPL=16.6 (step 3000)")

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
                    'n_params': n_params,
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
    print(f"\n{'='*60}")
    print(f"消融对比结果：")
    print(f"  baseline (无 field):        PPL=16.6, 20.9min, ~12.0M 参数")
    print(f"  field-augmented (本实验):   PPL={math.exp(best_val_loss):.1f}, {elapsed/60:.1f}min, {n_params/1e6:.2f}M 参数")
    if math.exp(best_val_loss) <= 16.6:
        print(f"  → ✅ field 组件有用 (PPL ≤ baseline)")
    else:
        print(f"  → ⚠️ field 组件在单神经元场景有害 (PPL > baseline)")
        print(f"     （预期结果：field 为多神经元协作设计，单神经元用无意义）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
