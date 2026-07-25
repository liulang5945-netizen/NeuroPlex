"""TinyStories 实验 C：多神经元协作验证（态极核心假设）。

设计：3 × 4M 小神经元协作 vs 1 × 12M baseline（同总参数量、同数据、同超参）
验证：多神经元协作能否涌现出超过单大神经元的智能？

协作机制（基于实验 B 已验证的 field 机制）：
  Round 1: 每个神经元独立 forward（无 field conditioning）
  Field 交换: 聚合所有神经元的 field（平均）→ 共享场
  Round 2: 每个神经元用共享 field 做 conditioning，重新 forward
  Logits 融合: 平均所有神经元的 logits

参数量对比：
  baseline (1×12M):        12.0M 参数, PPL=16.6
  field-augmented (1×12M): 12.22M 参数, PPL=14.3
  本实验 (3×~3.3M + shared): ~12M 参数, PPL=?

预期：
  若 PPL < 14.3 → 协作涌现确认，态极方向正确
  若 PPL ≥ 14.3 → 协作无效，需重新评估
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


# ── 配置 ──
class Config:
    # Tokenizer & 数据（和 baseline 一致）
    vocab_size = 50257       # GPT-2 BPE
    block_size = 128
    # 单神经元规格（3 个小神经元，总参数量匹配 baseline ~12M）
    num_neurons = 3
    hidden_size = 128        # 每个神经元 hidden（vs baseline 192）
    num_layers = 4           # 每个神经元层数（和 baseline 一致）
    num_heads = 4
    num_kv_heads = 4
    intermediate_size = 512  # SwiGLU（缩小匹配 hidden）
    rms_norm_eps = 1e-5
    dropout = 0.1
    # field 组件（和实验 B 同设计）
    field_dim = 128          # 和 hidden_size 一致
    # 训练（和 baseline 一致）
    batch_size = 12
    lr = 1e-3
    max_iters = 3000
    warmup_iters = 100
    eval_interval = 500
    eval_iters = 30
    save_path = "data/tinystories/collab_model.pt"


class SmallNeuron(nn.Module):
    """单个小神经元（含 field 组件）。

    基于实验 B 的 FieldAugmentedLM，但：
    - 不含 embedding（共享外部 embedding）
    - 不含 lm_head（共享外部 lm_head）
    - 只含 transformer body + field 组件
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Transformer body
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

        # Field 组件（和实验 B 同设计）
        self.field_write = nn.Linear(cfg.hidden_size, cfg.field_dim, bias=False)
        self.field_read_layers = nn.ModuleList([
            nn.Linear(cfg.field_dim, cfg.hidden_size, bias=False)
            for _ in range(cfg.num_layers)
        ])
        self.field_read_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _run_blocks(self, x, mask, field_state=None, round_num=1):
        """运行 transformer blocks，可选 field conditioning。"""
        for i, block in enumerate(self.blocks):
            h_normed = block.attention_norm(x)
            attn_out, _ = block.attention(h_normed, mask=mask)
            x = x + attn_out
            x = x + block.feed_forward(block.ffn_norm(x))

            # Field conditioning（round 2+ 才启用）
            if field_state is not None and round_num > 1:
                conditioning = self.field_read_layers[i](field_state)  # [B, hidden]
                if conditioning.dim() == 1:
                    conditioning = conditioning.unsqueeze(0).unsqueeze(0)
                else:
                    conditioning = conditioning.unsqueeze(1)  # [B, 1, hidden]
                gate = torch.sigmoid(self.field_read_gate(x))  # [B, L, 1]
                x = x + gate * conditioning
        return x

    def forward(self, x_emb, mask, field_state=None, round_num=1):
        """前向传播。

        Args:
            x_emb: [B, L, hidden] embedding 输入
            mask: 因果掩码
            field_state: [B, field_dim] 或 None
            round_num: 1=独立, 2=field conditioning

        Returns:
            hidden: [B, L, hidden] 最终 hidden
            field_vector: [B, field_dim] 本神经元写入的 field
        """
        h = self._run_blocks(x_emb, mask, field_state=field_state, round_num=round_num)
        h = self.norm_f(h)
        # Field write: 用最后一个 token 的 hidden
        hidden_last = h[:, -1, :]  # [B, hidden]
        v_raw = self.field_write(hidden_last)  # [B, field_dim]
        field_vector = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)  # L2 归一化
        return h, field_vector


class CollaborativeBrain(nn.Module):
    """多神经元协作大脑。

    3 个小神经元通过共享 field 协作：
    - Round 1: 各自独立 forward
    - Field 聚合: 平均所有神经元的 field_vector
    - Round 2: 各自用聚合 field 做 conditioning，重新 forward
    - Logits 融合: 平均所有神经元的 logits
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 共享 embedding + lm_head（tied）
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

        # N 个小神经元
        self.neurons = nn.ModuleList([
            SmallNeuron(cfg) for _ in range(cfg.num_neurons)
        ])

        self.apply(self._init_weights)
        # 缩放残差层初始化（和 baseline/实验B 一致）
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
        x_emb = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * float('-inf')

        # ── Round 1: 各自独立 forward ──
        field_vectors = []
        for neuron in self.neurons:
            _, fv = neuron(x_emb, mask, field_state=None, round_num=1)
            field_vectors.append(fv)

        # ── Field 聚合: 平均所有神经元的 field ──
        shared_field = torch.stack(field_vectors, dim=0).mean(dim=0)  # [B, field_dim]

        # ── Round 2: 各自用共享 field 做 conditioning ──
        all_logits = []
        for neuron in self.neurons:
            h, _ = neuron(x_emb, mask, field_state=shared_field, round_num=2)
            logits = self.lm_head(h)  # [B, L, vocab]
            all_logits.append(logits)

        # ── Logits 融合: 平均 ──
        logits = torch.stack(all_logits, dim=0).mean(dim=0)  # [B, L, vocab]

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
    train_data = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/tinystories/val.bin", dtype=np.uint16, mode="r")
    print(f"Train: {len(train_data)} tokens ({len(train_data)/1e6:.1f}M)")
    print(f"Val:   {len(val_data)} tokens ({len(val_data)/1e6:.1f}M)")
    return train_data, val_data


def get_batch(data, cfg: Config, device='cpu'):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: Config):
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
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    text = enc.decode(out[0].tolist())
    return text


def main():
    cfg = Config()
    device = 'cpu'
    enc = tiktoken.get_encoding("gpt2")

    print("=" * 60)
    print("TinyStories 实验 C：多神经元协作验证（态极核心假设）")
    print("=" * 60)
    print(f"配置: {cfg.num_neurons} 个小神经元, 每个 {cfg.num_layers}层 {cfg.num_heads}头 hidden={cfg.hidden_size}")
    print(f"field_dim={cfg.field_dim}, block_size={cfg.block_size}, batch={cfg.batch_size}")
    print(f"协作机制: round1独立→field聚合→round2 conditioning→logits平均")
    print(f"max_iters={cfg.max_iters}, lr={cfg.lr}")

    model = CollaborativeBrain(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_shared = sum(p.numel() for n, p in model.named_parameters() if 'neurons' not in n)
    n_neurons = sum(p.numel() for n, p in model.named_parameters() if 'neurons' in n)
    n_per_neuron = n_neurons / cfg.num_neurons
    print(f"\n参数量分析:")
    print(f"  总参数量:   {n_params/1e6:.2f}M")
    print(f"  共享部分:   {n_shared/1e6:.2f}M (embedding + lm_head tied)")
    print(f"  神经元部分: {n_neurons/1e6:.2f}M ({cfg.num_neurons} × {n_per_neuron/1e6:.2f}M)")
    print(f"  baseline 参考: ~12.0M (1×12M)")
    print(f"  field-augmented 参考: ~12.22M (1×12M)")

    print(f"\n[1] 加载数据...")
    train_data, val_data = load_data()
    data_param_ratio = len(train_data) / n_params
    print(f"数据/参数比: {data_param_ratio:.1f} (Chinchilla 最优 20:1)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        weight_decay=0.1,
    )

    def get_lr(it):
        if it < cfg.warmup_iters:
            return cfg.lr * it / cfg.warmup_iters
        if it > cfg.max_iters:
            return cfg.lr * 0.1
        decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.lr * 0.1 + 0.9 * cfg.lr * coeff

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
            print(f"  baseline 参考:     val PPL=16.6")
            print(f"  field-augmented:   val PPL=14.3")

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
    print(f"态极核心假设验证结果：")
    print(f"  baseline (1×12M):         PPL=16.6, 20.9min")
    print(f"  field-augmented (1×12M):  PPL=14.3, 25.6min")
    print(f"  collab (3×~4M, 本实验):   PPL={math.exp(best_val_loss):.1f}, {elapsed/60:.1f}min, {n_params/1e6:.2f}M 参数")
    collab_ppl = math.exp(best_val_loss)
    if collab_ppl < 14.3:
        print(f"  → ✅✅ 协作涌现确认！多神经元协作超过单大神经元 (PPL {collab_ppl:.1f} < 14.3)")
        print(f"     态极架构核心假设成立：多神经元协作能涌现出超过单大神经元的智能")
    elif collab_ppl < 16.6:
        print(f"  → ⚠️ 协作优于纯baseline但不如field-augmented (PPL {collab_ppl:.1f} 在 14.3~16.6 之间)")
        print(f"     协作有一定效果但未超过单神经元field版本，需优化协作机制")
    else:
        print(f"  → ❌ 协作无效 (PPL {collab_ppl:.1f} ≥ 16.6)")
        print(f"     多神经元协作未涌现，需重新评估架构或协作机制")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
