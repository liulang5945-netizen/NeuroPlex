"""SharedContextEncoder + SelfEvolver — 自主进化 embedding/encoder（P6-6 v2）.

设计哲学（响应"教师得来的不能成为永久原罪"质疑）：
  教师只在启动期提供初始化种子，之后 embedding + encoder 通过三机制自主进化：
  1. Hebbian: 共激活的 token embedding 互相吸引（生物学启发）
  2. Contrastive: 同 domain 输入拉近，跨 domain 推远（路由当弱监督）
  3. MLM: 随机 mask token 预测（自监督，NLP 已验证的最强方案）

  三机制组合让 embedding/encoder 逐步摆脱教师初始化影响，
  当态极扩展到更大规模时能自主扩展（vocab/dim 不被教师限制）。

模块组成：
  - SharedContextEncoder: 共享 transformer encoder（可训练，非 per-neuron）
  - HebbianUpdater: token 共激活统计 + embedding 更新
  - ContrastiveLoss: domain 内/间对比损失
  - MLMLoss: mask language modeling 损失
  - SelfEvolver: 三机制组合训练器（sleep cycle 或独立训练时调用）

Usage:
    # 启动期：从教师初始化（一次性）
    encoder = SharedContextEncoder.build_from_teacher(teacher, ...)

    # 自训期：sleep cycle 中调用
    evolver = SelfEvolver(encoder, hebbian_updater, router_weak_supervision)
    for batch in sleep_data:
        loss = evolver.training_step(batch)
        loss.backward()

    # 成熟期：教师初始化影响已被自组织覆盖
"""
from __future__ import annotations

import os
import math
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.layers import TransformerBlock, RMSNorm


class SharedContextEncoder(nn.Module):
    """共享上下文 encoder — 把 token embedding 处理成上下文感知的 hidden state.

    与 StandaloneEmbedding 的关系：
    - StandaloneEmbedding: token_id -> [embed_dim] embedding lookup（无上下文）
    - SharedContextEncoder: [B, L, embed_dim] -> [B, L, hidden_dim] 上下文感知

    与教师的关系：
    - 启动期：可从教师前几层 distill 初始化
    - 自训期：通过 SelfEvolver 三机制自主更新
    - 成熟期：完全自主，教师可弃

    共享性：
    - 这是共享模块（不是 per-neuron），符合"小 neuron 协同"哲学
    - 类比人脑：小脑/丘脑作为共享预处理层
    """

    def __init__(
        self,
        vocab_size: int = 256000,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        intermediate_size: int = 1408,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        n_domains: int = 0,  # P6-9 fix: domain classifier head（>0 启用）
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_domains = n_domains

        # Token embedding（可训练，启动期从教师 SVD 初始化）
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # embed_dim -> hidden_dim 投影（若不同）
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Transformer encoder 层
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)

        # 位置编码（简单的可学习位置嵌入，避免 RoPE 复杂度）
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # MLM head（用于自监督训练，与 embedding 权重绑定 tied）
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # 权重绑定：lm_head.weight = embedding.weight (类似 GPT/BERT)
        # 但维度可能不匹配（hidden_dim vs embed_dim），只在相等时绑定
        if hidden_dim == embed_dim:
            self.lm_head.weight = self.embedding.weight

        # P6-9 fix: domain classifier head（supervised domain routing signal）
        # 让 encoder 显式学习 domain 区分，避免 representation collapse
        # 用 mean pool 后的 hidden state -> n_domains logits
        if n_domains > 0:
            self.domain_classifier = nn.Linear(hidden_dim, n_domains)
        else:
            self.domain_classifier = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = True,
        return_domain_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """前向传播：input_ids -> hidden states (+ 可选 logits).

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] 可选，1=有效 token，0=padding。
                P6-9 fix: 用于 mean pool 排除 padding，避免短 prompt 被 padding 淹没。
            return_hidden: 是否返回 hidden states（用于路由/encoder 输出）
            return_domain_logits: 是否返回 domain 分类 logits（P6-9）

        Returns:
            (logits, hidden, domain_logits)
            - logits: [B, L, vocab_size] MLM 预测（用于自监督训练）
            - hidden: [B, L, hidden_dim] 上下文感知的 hidden states（用于路由）
            - domain_logits: [B, n_domains] domain 分类 logits（P6-9，可选）
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Token embedding + position embedding
        emb = self.embedding(input_ids)  # [B, L, embed_dim]
        emb = self.input_proj(emb)  # [B, L, hidden_dim]
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        emb = emb + self.pos_embedding(positions)
        emb = self.dropout(emb)

        # Transformer layers
        h = emb
        for layer in self.layers:
            h, _ = layer(h)  # TransformerBlock 返回 (x, kv_cache)
        h = self.final_norm(h)

        # MLM head
        logits = self.lm_head(h)

        # P6-9: domain classifier（attention-masked mean pool 后过 classifier）
        # 关键修复：用 attention_mask 排除 padding tokens，否则短 prompt
        # （如 "你好"=2 tokens padded 到 64）的 mean pool 被 padding 主导
        domain_logits = None
        if return_domain_logits and self.domain_classifier is not None:
            if attention_mask is not None:
                # masked mean pool: 只对有效 tokens 取平均
                mask = attention_mask.to(h.dtype).unsqueeze(-1)  # [B, L, 1]
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, hidden_dim]
            else:
                pooled = h.mean(dim=1)  # [B, hidden_dim]
            domain_logits = self.domain_classifier(pooled)  # [B, n_domains]

        if return_hidden:
            return logits, h, domain_logits
        return logits, None, domain_logits

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """便捷方法：只返回 hidden states（用于路由）."""
        _, hidden, _ = self.forward(input_ids, attention_mask=attention_mask, return_hidden=True)
        return hidden

    def classify_domain(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """P6-9: 直接返回 domain logits（用于 supervised routing）."""
        _, _, domain_logits = self.forward(
            input_ids, attention_mask=attention_mask,
            return_hidden=False, return_domain_logits=True,
        )
        return domain_logits  # [B, n_domains]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "n_domains": self.n_domains,  # P6-9
            },
            "state_dict": self.state_dict(),
        }, path)
        print(f"[SharedContextEncoder] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SharedContextEncoder":
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        m = cls(**cfg)
        m.load_state_dict(data["state_dict"])
        m.to(device)
        m.eval()
        return m

    @classmethod
    def build_from_standalone_embedding(
        cls,
        standalone_embedding,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        intermediate_size: int = 1408,
        n_domains: int = 0,  # P6-9
    ) -> "SharedContextEncoder":
        """从已有 StandaloneEmbedding 构建（复用其 embedding 权重作为初始化）.

        Args:
            standalone_embedding: StandaloneEmbedding 实例（已从教师 SVD 初始化）
            hidden_dim: encoder hidden dim
            num_layers: encoder 层数（推荐 2-4）
            n_domains: P6-9 domain classifier head 的类别数（0=不启用）
            ...
        """
        m = cls(
            vocab_size=standalone_embedding.vocab_size,
            embed_dim=standalone_embedding.embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            n_domains=n_domains,
        )
        # 复用 embedding 权重
        m.embedding.weight.data = standalone_embedding.embedding.weight.data.clone()
        print(f"[SharedContextEncoder] Built from StandaloneEmbedding "
              f"(vocab={m.vocab_size}, embed_dim={m.embed_dim}, layers={num_layers}, "
              f"n_domains={n_domains})")
        return m


class HebbianUpdater:
    """Hebbian token 共激活更新器.

    生物学启发：
    - 经常在同一 context 中出现的 token，其 embedding 应该互相"靠近"
    - 类似 Hebbian learning: "cells that fire together, wire together"

    实现：
    - 维护 token 共现矩阵（稀疏，top-K co-occurrence）
    - 定期用共现统计更新 embedding：拉近共现 token，推远随机 token

    注意：这是离线/异步更新，不在 forward path 中。
    """

    def __init__(
        self,
        vocab_size: int,
        cooc_top_k: int = 50,
        lr: float = 0.01,
        decay: float = 0.999,  # EMA decay for co-occurrence counts
        neg_ratio: float = 0.5,  # P6-8 fix: 负采样推远强度（相对正样本拉近的倍数）
        n_negatives: int = 10,   # P6-8 fix: 每个 token 推远的负样本数
    ):
        self.vocab_size = vocab_size
        self.cooc_top_k = cooc_top_k
        self.lr = lr
        self.decay = decay
        self.neg_ratio = neg_ratio
        self.n_negatives = n_negatives
        # 共现统计：{token_id: {peer_token_id: count}}
        # 用 dict-of-dict 节省内存（稀疏）
        self._cooc: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # 已观察到的 token 集合（用于负采样）
        self._observed_tokens: set = set()

    def update_cooc(self, input_ids: torch.Tensor) -> None:
        """从一个 batch 的输入更新共现统计.

        Args:
            input_ids: [B, L] token IDs
        """
        ids = input_ids.tolist()
        for seq in ids:
            # 在一个 sequence 内，相邻 window_size 的 token 算共现
            L = len(seq)
            window = 8  # 上下文窗口
            for i, t in enumerate(seq):
                if t >= self.vocab_size:
                    continue
                # P6-8 fix: 记录已观察到的 token（用于负采样）
                self._observed_tokens.add(t)
                lo = max(0, i - window)
                hi = min(L, i + window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    peer = seq[j]
                    if peer >= self.vocab_size:
                        continue
                    # EMA 更新
                    old = self._cooc[t][peer]
                    self._cooc[t][peer] = old * self.decay + (1 - self.decay)

    def apply_hebbian_update(
        self,
        embedding_layer: nn.Embedding,
        batch_size: int = 256,
    ) -> float:
        """应用 Hebbian 更新到 embedding 层.

        P6-8 fix: 实现"正样本拉近 + 负样本推远"双向机制（原代码只有拉近，
        导致所有 token embedding 都被拉向共现均值，加剧表征坍塌）。

        对每个 token t：
        - 正样本：top-K 共现 peer，拉近 embedding[t] -> peer_mean
        - 负样本：随机采样 n_negatives 个未共现 token，推远 embedding[t] <- neg_mean
        - 总更新：delta = lr * (peer_mean - v_t) - lr * neg_ratio * (neg_mean - v_t)

        Args:
            embedding_layer: nn.Embedding 实例
            batch_size: 每次更新的 token 数（避免全量更新 OOM）

        Returns:
            平均更新幅度（诊断用）
        """
        if not self._cooc:
            return 0.0

        weight = embedding_layer.weight.data  # [vocab, dim]
        all_tokens = list(self._cooc.keys())
        random.shuffle(all_tokens)
        # 负采样池：已观察到但与当前 token 不共现的 token
        observed_list = list(self._observed_tokens)
        total_update = 0.0
        n_updated = 0

        for t in all_tokens[:batch_size]:
            peers = self._cooc[t]
            if not peers:
                continue
            # top-K 共现 peer（正样本）
            top_peers = sorted(peers.items(), key=lambda x: -x[1])[:self.cooc_top_k]
            peer_ids = [p for p, _ in top_peers]
            if not peer_ids:
                continue

            peer_tensor = weight[peer_ids]  # [K, dim]
            peer_mean = peer_tensor.mean(dim=0)  # [dim]
            v_t = weight[t]  # [dim]

            # Hebbian pull: 拉近到 peer 均值
            pull_delta = self.lr * (peer_mean - v_t)

            # P6-8 fix: 负采样推远
            # 从 observed_list 中随机采样 n_negatives 个不与 t 共现的 token
            peer_set = set(peer_ids)
            neg_candidates = [tok for tok in observed_list if tok != t and tok not in peer_set]
            if len(neg_candidates) >= self.n_negatives:
                neg_samples = random.sample(neg_candidates, self.n_negatives)
                neg_tensor = weight[neg_samples]  # [n_neg, dim]
                neg_mean = neg_tensor.mean(dim=0)  # [dim]
                # 推远：让 v_t 远离 neg_mean
                push_delta = -self.lr * self.neg_ratio * (neg_mean - v_t)
            else:
                push_delta = torch.zeros_like(v_t)

            delta = pull_delta + push_delta
            weight[t] = v_t + delta
            total_update += delta.norm().item()
            n_updated += 1

        return total_update / max(n_updated, 1)

    def get_stats(self) -> dict:
        return {
            "n_tracked_tokens": len(self._cooc),
            "avg_peers_per_token": (
                sum(len(p) for p in self._cooc.values()) / max(len(self._cooc), 1)
            ),
        }


class ContrastiveLoss(nn.Module):
    """Domain-aware contrastive loss — 用路由结果当弱监督.

    思路：
    - 同 domain 的输入，其 hidden state 应该拉近
    - 跨 domain 的输入，其 hidden state 应该推远
    - "domain" 由 ThalamicRouter 的路由结果决定（弱监督，不需要人工标注）

    P6-8 fix v2: margin loss + uniformity loss，双重防坍塌.
    - margin loss: 显式要求 sim_pos > margin_pos 且 sim_neg < margin_neg
      坍塌时 sim_neg = 1 > margin_neg，neg_loss 仍然大
    - uniformity loss: 约束 hidden state 在超球面均匀分布
      坍塌时所有点聚在一起，uniformity loss 大
      参考 "Understanding Contrastive Representation Learning through
      Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020)

    L = margin_loss + lambda_uniform * uniformity_loss
    """

    def __init__(
        self,
        margin_pos: float = 0.8,  # 同 domain 期望 sim > 0.8
        margin_neg: float = 0.3,  # 跨 domain 期望 sim < 0.3
        lambda_uniform: float = 1.0,  # P6-8 fix v2: uniformity loss 权重
        uniformity_t: float = 2.0,    # uniformity loss 温度参数
        temperature: float = 0.1,  # 保留兼容性（margin loss 不用）
    ):
        super().__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.lambda_uniform = lambda_uniform
        self.uniformity_t = uniformity_t
        self.temperature = temperature  # 兼容旧参数

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, hidden_dim]
        domain_labels: torch.Tensor,  # [B] domain ID per sample
    ) -> torch.Tensor:
        """计算 margin + uniformity contrastive loss.

        Args:
            hidden_states: [B, hidden_dim] 句级 hidden state（mean pool 后）
            domain_labels: [B] 每个样本的 domain ID

        Returns:
            scalar loss
        """
        B = hidden_states.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        # L2 normalize
        z = F.normalize(hidden_states, dim=-1)  # [B, hidden_dim]

        # cosine 相似度矩阵 [B, B]
        sim = z @ z.T  # [B, B] in [-1, 1]

        # 排除对角线
        eye = torch.eye(B, device=hidden_states.device)
        sim_offdiag = sim * (1 - eye)  # 自己跟自己的 sim 设为 0

        # 正样本 mask: 同 domain 的其他样本
        labels = domain_labels.unsqueeze(0)  # [1, B]
        pos_mask = (labels == labels.T).float() * (1 - eye)  # [B, B]
        # 负样本 mask: 跨 domain
        neg_mask = (labels != labels.T).float()  # [B, B]

        # Margin loss
        pos_loss = (self.margin_pos - sim_offdiag).clamp(min=0) * pos_mask
        neg_loss = (sim_offdiag - self.margin_neg).clamp(min=0) * neg_mask

        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)
        margin_loss = pos_loss.sum() / n_pos + neg_loss.sum() / n_neg

        # P6-8 fix v2: Uniformity loss — 防止所有点坍塌到一点
        # L_uniform = log(E[exp(-t * ||x - y||^2)])
        # 坍塌时 ||x-y||=0, exp(0)=1, L=log(1)=0
        # 均匀时 ||x-y|| 大, exp 小, L 很负
        # 最小化 L_uniform → 推动点均匀分布
        # 用 pdist 计算 batch 内所有对的 ||x-y||^2
        if B >= 2:
            # pdist 返回上三角对的距离
            dist_sq = torch.pdist(z, p=2).pow(2)  # [B*(B-1)/2]
            if dist_sq.numel() > 0:
                uniformity = torch.log(
                    torch.exp(-self.uniformity_t * dist_sq).mean() + 1e-8
                )
            else:
                uniformity = torch.tensor(0.0, device=hidden_states.device)
        else:
            uniformity = torch.tensor(0.0, device=hidden_states.device)

        loss = margin_loss + self.lambda_uniform * uniformity
        return loss


class MLMLoss(nn.Module):
    """Masked Language Modeling loss — 自监督训练.

    随机 mask 15% 的 token，让 encoder 预测被 mask 的 token.
    完全自监督，不需要教师.
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 4,  # 假设 [MASK] token id=4，根据实际 tokenizer 调整
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def forward(
        self,
        encoder: SharedContextEncoder,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """计算 MLM loss.

        Args:
            encoder: SharedContextEncoder 实例
            input_ids: [B, L] 原始 token IDs

        Returns:
            scalar loss
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 随机选择 mask 位置
        prob_matrix = torch.full(input_ids.shape, self.mask_prob, device=device)
        masked_positions = torch.bernoulli(prob_matrix).bool()  # [B, L]

        # 若没有任何 mask 位置，返回 0
        if not masked_positions.any():
            return torch.tensor(0.0, device=device)

        # 构造 masked input
        masked_input = input_ids.clone()
        masked_input[masked_positions] = self.mask_token_id

        # Forward
        # P6-9: forward 现在返回 (logits, hidden, domain_logits) 3-tuple
        logits, _, _ = encoder(masked_input, return_hidden=False)  # [B, L, vocab]

        # 只计算被 mask 位置的 loss
        masked_logits = logits[masked_positions]  # [N_masked, vocab]
        masked_labels = input_ids[masked_positions]  # [N_masked]

        loss = F.cross_entropy(masked_logits, masked_labels)
        return loss


class SelfEvolver:
    """三机制组合训练器 — 让 embedding/encoder 自主进化.

    三机制权重：
    - Hebbian: 拉近共现 token（embedding 层更新，离线）
    - Contrastive: 同 domain 拉近，跨 domain 推远（hidden 层梯度）
    - MLM: mask 预测（全栈梯度，最强自监督）

    Usage:
        evolver = SelfEvolver(encoder, hebbian, contrastive, mlm)
        for batch in sleep_data:
            loss = evolver.training_step(batch)
            loss.backward()
            optimizer.step()
        evolver.apply_hebbian_to_embedding()
    """

    def __init__(
        self,
        encoder: SharedContextEncoder,
        hebbian_updater: HebbianUpdater,
        contrastive_loss: ContrastiveLoss,
        mlm_loss: MLMLoss,
        weights: Tuple[float, float, float] = (1.0, 0.5, 1.0),
        w_domain: float = 2.0,  # P6-9: domain classification loss weight
    ):
        """
        Args:
            encoder: SharedContextEncoder 实例
            hebbian_updater: Hebbian token 共激活更新器
            contrastive_loss: domain 对比损失
            mlm_loss: MLM 自监督损失
            weights: (hebbian, contrastive, mlm) 三机制权重
            w_domain: P6-9 domain classification loss 权重（supervised signal，
                     直接优化 domain 区分，避免 representation collapse）
        """
        self.encoder = encoder
        self.hebbian = hebbian_updater
        self.contrastive = contrastive_loss
        self.mlm = mlm_loss
        self.w_hebbian, self.w_contrastive, self.w_mlm = weights
        self.w_domain = w_domain  # P6-9

        # 训练统计
        self.last_loss = {
            "total": 0.0, "mlm": 0.0, "contrastive": 0.0,
            "hebbian": 0.0, "domain": 0.0,  # P6-9
        }

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        domain_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """一个训练 step：计算组合 loss（mlm + contrastive + domain classification）.

        Args:
            batch: {"input_ids": [B, L]}
            domain_labels: [B] 可选，若提供则计算 contrastive loss + domain classification loss

        Returns:
            total_loss (scalar tensor, 可 backward)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")  # P6-9 fix

        # 1. 更新 Hebbian 共现统计（无梯度，只在有效 tokens 上统计）
        with torch.no_grad():
            self.hebbian.update_cooc(input_ids)

        # 2. MLM loss（自监督）
        mlm_loss = self.mlm(self.encoder, input_ids)

        # 3. Contrastive loss + Domain classification loss
        # P6-9: 共享一次 forward，同时算 contrastive 和 domain classifier
        # P6-9 fix: 传 attention_mask 让 mean pool 排除 padding
        if domain_labels is not None:
            # 一次有梯度 forward，同时返回 hidden 和 domain_logits
            _, hidden, domain_logits = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                return_hidden=True,
                return_domain_logits=(self.encoder.domain_classifier is not None),
            )
            # P6-9 fix: contrastive loss 也用 attention-masked mean pool
            if attention_mask is not None:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # [B, L, 1]
                sent_hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                sent_hidden = hidden.mean(dim=1)  # [B, hidden_dim]
            contrastive_loss = self.contrastive(sent_hidden, domain_labels)

            # P6-9: domain classification loss（supervised）
            if domain_logits is not None:
                domain_loss = F.cross_entropy(domain_logits, domain_labels)
            else:
                domain_loss = torch.tensor(0.0, device=input_ids.device)
        else:
            contrastive_loss = torch.tensor(0.0, device=input_ids.device)
            domain_loss = torch.tensor(0.0, device=input_ids.device)

        # 4. 组合 loss
        total = (self.w_mlm * mlm_loss
                 + self.w_contrastive * contrastive_loss
                 + self.w_domain * domain_loss)  # P6-9

        self.last_loss = {
            "total": float(total.item()),
            "mlm": float(mlm_loss.item()),
            "contrastive": float(contrastive_loss.item()),
            "hebbian": 0.0,
            "domain": float(domain_loss.item()),  # P6-9
        }

        return total

    def apply_hebbian_to_embedding(self) -> float:
        """应用 Hebbian 更新到 encoder 的 embedding 层（离线/异步）."""
        update = self.hebbian.apply_hebbian_update(self.encoder.embedding)
        self.last_loss["hebbian"] = update
        return update

    def get_loss_summary(self) -> dict:
        return dict(self.last_loss)
