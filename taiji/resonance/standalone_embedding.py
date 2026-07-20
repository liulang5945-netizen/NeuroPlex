"""Standalone embedding table — 脱离教师模型独立推理 (P6-1).

人脑对照：
  婴儿学习时需要"教师"（父母/环境）提供信号
  成年后大脑独立工作，教师的影响已"内化"为神经元连接

设计动机：
  原 SharedEmbedProj 在推理时仍需 forward 1.5B 教师模型算 hidden state，
  这违背了"小 neuron 协同比肩大模型"的设计哲学。
  P6-1: 用独立 nn.Embedding 替代教师 hidden state 路径。

初始化策略：
  蒸馏完成后，用教师 embedding 矩阵的 SVD 初始化这个独立 embedding 表。
  之后可以：
    - 冻结（保持与蒸馏时一致）
    - 微调（继续训练让 embedding 适配 neuron 集合）

使用方式：
  # 1. 蒸馏后构建（一次性）
  embed_table = StandaloneEmbedding.build_from_teacher(teacher_model, target_dim=512)
  embed_table.save("data/distill/standalone_embedding.pt")

  # 2. 推理时直接用（无需教师）
  cortex.set_standalone_embedding(embed_table)  # 替代 set_teacher_pipeline

文件：
  data/distill/standalone_embedding.pt: {weight: [vocab, target_dim]}
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional


class StandaloneEmbedding(nn.Module):
    """独立 embedding 表，推理时直接 lookup，无需教师 forward。

    与 SharedEmbedProj 的关键区别：
    - SharedEmbedProj: input_ids -> teacher.forward -> hidden [B,L,2048] -> proj -> [B,L,512]
      推理时需要跑 1.5B 教师模型
    - StandaloneEmbedding: input_ids -> embedding lookup -> [B,L,512]
      推理时只需 O(1) lookup，零教师依赖

    代价：
      失去教师 hidden state 的"上下文感知"能力（教师 forward 会用 transformer
      处理上下文，embedding lookup 是纯 token-level）
      但蒸馏时 neuron 学到的 transformer 层会自己处理上下文
    """

    def __init__(self, vocab_size: int = 256000, embed_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 初始化为正交（默认 nn.Embedding 是 N(0,1)，太大会让后续层不稳定）
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """直接 lookup，与 nn.Embedding 接口一致。

        Args:
            input_ids: [B, L] long tensor

        Returns:
            [B, L, embed_dim] embedding
        """
        return self.embedding(input_ids)

    def save(self, path: str) -> None:
        """保存 embedding 表到磁盘。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "weight": self.embedding.weight.data.cpu(),
        }, path)
        print(f"[StandaloneEmbedding] Saved to {path} "
              f"(vocab={self.vocab_size}, dim={self.embed_dim})")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StandaloneEmbedding":
        """从磁盘加载。"""
        data = torch.load(path, map_location=device, weights_only=False)
        m = cls(vocab_size=data["vocab_size"], embed_dim=data["embed_dim"])
        m.embedding.weight.data = data["weight"].to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        print(f"[StandaloneEmbedding] Loaded from {path} "
              f"(vocab={m.vocab_size}, dim={m.embed_dim})")
        return m

    @classmethod
    def build_from_teacher(
        cls,
        teacher_model,
        target_dim: int = 512,
        method: str = "svd",
    ) -> "StandaloneEmbedding":
        """从教师模型构建独立 embedding 表（蒸馏后一次性调用）。

        Args:
            teacher_model: 教师模型（含 backbone.embedding）
            target_dim: 目标维度（与 neuron.config.base_embed_dim 一致）
            method: 初始化方法
                "svd": 用教师 embedding 矩阵的 SVD 降维（推荐，保留语义结构）
                "random": 随机正交初始化（基线对比用）

        Returns:
            StandaloneEmbedding 实例（已初始化，未训练）
        """
        # 获取教师 embedding 矩阵
        if hasattr(teacher_model, 'backbone'):
            teacher_emb = teacher_model.backbone.embedding
        elif hasattr(teacher_model, 'embedding'):
            teacher_emb = teacher_model.embedding
        else:
            raise AttributeError("Cannot find teacher embedding")

        teacher_weight = teacher_emb.weight.data  # [vocab, teacher_dim]
        vocab_size, teacher_dim = teacher_weight.shape
        print(f"[StandaloneEmbedding] Building from teacher: "
              f"vocab={vocab_size}, teacher_dim={teacher_dim}, target_dim={target_dim}")

        m = cls(vocab_size=vocab_size, embed_dim=target_dim)

        if method == "svd":
            # SVD 降维：teacher_weight [vocab, 2048] -> [vocab, 512]
            # 用左奇异向量 × 奇异值保留主要语义信息
            U, S, Vh = torch.linalg.svd(teacher_weight, full_matrices=False)
            # 取前 target_dim 个分量
            target_weight = U[:, :target_dim] * S[:target_dim].unsqueeze(0)
            # 归一化到合理 scale（与 neuron 期望的 std≈0.02 一致）
            current_std = target_weight.std()
            if current_std > 1e-6:
                target_weight = target_weight * (0.02 / current_std)
            m.embedding.weight.data = target_weight.contiguous()
            print(f"  SVD init: kept {target_dim}/{teacher_dim} dims, "
                  f"explained_var={S[:target_dim].pow(2).sum() / S.pow(2).sum():.4f}")
        else:
            # 随机初始化（默认 normal std=0.02，已在 __init__ 完成）
            print(f"  Random init (std=0.02)")

        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    @classmethod
    def build_from_shared_proj(
        cls,
        teacher_model,
        shared_proj_path: str,
        target_dim: int = 512,
    ) -> "StandaloneEmbedding":
        """从已有 SharedEmbedProj 构建独立 embedding 表。

        策略：用教师 embedding 矩阵 [vocab, 2048] 经过 SharedEmbedProj 投影到
        [vocab, 512]，作为独立 embedding 表的初始化。
        这样独立 embedding 表在初始化时与原 SharedEmbedProj 路径等价。

        Args:
            teacher_model: 教师模型
            shared_proj_path: SharedEmbedProj 文件路径
            target_dim: 目标维度

        Returns:
            StandaloneEmbedding 实例
        """
        from taiji.resonance.shared_embed import SharedEmbedProj

        # 获取教师 embedding 矩阵
        if hasattr(teacher_model, 'backbone'):
            teacher_emb = teacher_model.backbone.embedding
        elif hasattr(teacher_model, 'embedding'):
            teacher_emb = teacher_model.embedding
        else:
            raise AttributeError("Cannot find teacher embedding")

        teacher_weight = teacher_emb.weight.data  # [vocab, 2048]
        vocab_size, teacher_dim = teacher_weight.shape
        print(f"[StandaloneEmbedding] Building from SharedEmbedProj: "
              f"vocab={vocab_size}, teacher_dim={teacher_dim}, target_dim={target_dim}")

        # 加载 SharedEmbedProj
        proj = SharedEmbedProj.load(shared_proj_path, src_dim=teacher_dim, target_dim=target_dim)

        # 对每个 token 做 projection: [vocab, 2048] @ [2048, 512] -> [vocab, 512]
        with torch.no_grad():
            target_weight = proj(teacher_weight)  # [vocab, 512]

        m = cls(vocab_size=vocab_size, embed_dim=target_dim)
        m.embedding.weight.data = target_weight.contiguous()
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        print(f"  Initialized from SharedEmbedProj (equivalent to teacher path)")
        return m
