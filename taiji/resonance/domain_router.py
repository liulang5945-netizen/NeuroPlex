"""DomainRouter - 基于 field_vector 与 domain anchor 的相似度路由。

核心思想：
  每个 neuron 在自己擅长的领域会写出独特的 field_vector，
  在不擅长的领域 field_vector 方向会偏离本域典型方向。

  1. 蒸馏后，用本域数据计算 neuron 的 "domain anchor" = 平均 field_vector
  2. 推理时，每个 neuron 对输入生成 field_vector
  3. 计算当前 field_vector 与各 neuron 的 domain anchor 的余弦相似度
  4. 相似度越高 -> 该 neuron 越匹配本域 -> 给更高权重

这取代了原 ensemble 的 entropy-based weighting：
  原方案问题：neuron 对任何输入都给出自信答案（logits entropy 低），
              但自信 ≠ 正确，导致错误 neuron 抢占权重。
  新方案：用 field_vector 的方向匹配度判断 "这个输入是否属于本域"，
         不依赖 logits 自信度，避免被"错误但自信"的 neuron 误导。

文件：
  - data/distill/domain_anchors.pt: {neuron_id: [field_dim] tensor}
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class DomainRouter:
    """基于 field_vector × domain anchor 相似度的路由器。

    Usage:
        # 1. 蒸馏后计算 anchors
        router = DomainRouter(field_dim=4096)
        router.compute_anchors(neurons, datasets, shared_embedding_fn, device)

        # 2. 推理时查询
        weights = router.route(field_vectors)  # {nid: float}
    """

    def __init__(self, field_dim: int, temperature: float = 0.1):
        """
        Args:
            field_dim: field_vector 维度（与 neuron.config.field_dim 一致）
            temperature: softmax 温度，越低路由越 sharp（hard routing）
                         默认 0.1 让最匹配的 neuron 拿到 ~90% 权重
        """
        self.field_dim = field_dim
        self.temperature = temperature
        # anchors[nid] = [field_dim] 归一化向量
        self.anchors: Dict[str, torch.Tensor] = {}

    def compute_anchors(
        self,
        neurons: Dict,  # Dict[str, ResonanceNeuron]
        domain_data: Dict[str, torch.Tensor],  # {nid: [N, L] token IDs}
        shared_embedding_fn,  # callable(input_ids) -> [B, L, base_embed_dim]
        device: str = "cpu",
        batch_size: int = 4,
        max_samples: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """计算每个 neuron 的 domain anchor。

        对每个 neuron：
          1. 取本域数据前 max_samples 条
          2. 跑 forward 获取 field_vector（round 1，无场状态）
          3. 平均所有样本的 field_vector -> anchor
          4. L2 归一化

        Args:
            neurons: {nid: ResonanceNeuron}
            domain_data: {nid: [N, L] tensor}（应该与 neuron 的 domain 匹配）
            shared_embedding_fn: 输入 input_ids 返回 shared_emb 的函数
            device: 计算设备
            batch_size: 批大小
            max_samples: 每个域最多用多少样本计算 anchor

        Returns:
            {nid: [field_dim] normalized tensor}
        """
        self.anchors = {}
        print(f"[DomainRouter] Computing domain anchors (field_dim={self.field_dim})...")

        for nid, neuron in neurons.items():
            if nid not in domain_data:
                print(f"  [{nid}] no domain data, skip")
                continue

            data = domain_data[nid]
            if data.shape[0] == 0:
                continue
            n = min(data.shape[0], max_samples)
            data = data[:n].to(device)

            anchor_sum = torch.zeros(self.field_dim, device=device)
            count = 0

            neuron.eval()
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = data[i:i + batch_size]
                    shared_emb = shared_embedding_fn(batch)
                    result = neuron.forward(shared_emb, return_logits=False)
                    field_vec = result["field_vector"]  # [B, field_dim]
                    # 平均 batch 内的 field_vector
                    anchor_sum += field_vec.sum(dim=0)
                    count += field_vec.shape[0]

            if count > 0:
                anchor = anchor_sum / count
                anchor = anchor / (anchor.norm() + 1e-8)
                self.anchors[nid] = anchor.cpu()
                print(f"  [{nid}] anchor computed from {count} samples, norm={anchor.norm().item():.4f}")

        return self.anchors

    def route(
        self,
        field_vectors: Dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, float]:
        """根据当前 field_vector 与 anchor 的相似度计算路由权重。

        Args:
            field_vectors: {nid: [B, field_dim] 或 [field_dim]} 当前 forward 产生的 field_vector
            batch_idx: 如果 field_vector 是 batched，取哪个 sample

        Returns:
            {nid: float} 归一化权重（和为 1）
        """
        if not self.anchors:
            # 无 anchor fallback：均匀权重
            n = len(field_vectors)
            return {nid: 1.0 / n for nid in field_vectors} if n > 0 else {}

        # 计算每个 neuron 的相似度
        sims = {}
        for nid, fv in field_vectors.items():
            if nid not in self.anchors:
                sims[nid] = 0.0
                continue
            # fv: [B, field_dim] 或 [field_dim]
            if fv.dim() == 2:
                fv_sample = fv[batch_idx]  # [field_dim]
            else:
                fv_sample = fv  # [field_dim]
            anchor = self.anchors[nid].to(fv_sample.device)
            # field_vector 已归一化，anchor 也已归一化，cosine = dot
            sim = torch.dot(fv_sample, anchor).item()
            sims[nid] = sim

        # 用相似度做 softmax 路由
        # temperature 越低 -> 越接近 hard routing
        nids = list(sims.keys())
        sim_tensor = torch.tensor([sims[nid] for nid in nids], dtype=torch.float32)
        # 用 temperature 缩放后 softmax
        weights_tensor = F.softmax(sim_tensor / self.temperature, dim=0)

        return {nid: float(weights_tensor[i].item()) for i, nid in enumerate(nids)}

    def route_per_position(
        self,
        field_vectors_by_position: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-position routing 版本（用于 ensemble 的 [B, L] 加权）。

        Args:
            field_vectors_by_position: {nid: [B, L, field_dim]} 或 {nid: [B, field_dim]}
                                       注意：当前 neuron.forward 只产生 [B, field_dim]，
                                       所以这个方法主要用于 [B] 维度的 batched routing。

        Returns:
            weights: [N, B] tensor（N=neuron 数, B=batch）
        """
        if not self.anchors:
            n = len(field_vectors_by_position)
            # fallback: 均匀
            first = next(iter(field_vectors_by_position.values()))
            B = first.shape[0] if first.dim() >= 2 else 1
            return torch.full((n, B), 1.0 / n)

        nids = list(field_vectors_by_position.keys())
        first = next(iter(field_vectors_by_position.values()))
        if first.dim() == 1:
            first = first.unsqueeze(0)  # [1, field_dim]
        B = first.shape[0]

        # 计算每个 neuron × 每个 batch sample 的相似度
        sim_matrix = torch.zeros(len(nids), B)
        for i, nid in enumerate(nids):
            fv = field_vectors_by_position[nid]
            if fv.dim() == 1:
                fv = fv.unsqueeze(0)
            anchor = self.anchors[nid].to(fv.device) if nid in self.anchors else None
            if anchor is None:
                continue
            # fv: [B, field_dim], anchor: [field_dim]
            sims = (fv * anchor.unsqueeze(0)).sum(dim=-1)  # [B]
            sim_matrix[i] = sims

        # softmax over neurons（dim=0）
        weights = F.softmax(sim_matrix / self.temperature, dim=0)  # [N, B]
        return weights

    def save(self, path: str) -> None:
        """保存 anchors 到磁盘。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "field_dim": self.field_dim,
            "temperature": self.temperature,
            "anchors": self.anchors,
        }, path)
        print(f"[DomainRouter] Saved {len(self.anchors)} anchors to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DomainRouter":
        """从磁盘加载 anchors。"""
        data = torch.load(path, map_location=device, weights_only=False)
        router = cls(field_dim=data["field_dim"], temperature=data["temperature"])
        router.anchors = {nid: v.to(device) for nid, v in data["anchors"].items()}
        print(f"[DomainRouter] Loaded {len(router.anchors)} anchors from {path}")
        return router

    def get_anchor(self, nid: str) -> Optional[torch.Tensor]:
        return self.anchors.get(nid)

    def similarity(self, nid: str, field_vector: torch.Tensor) -> float:
        """计算单个 neuron 的 field_vector 与其 anchor 的相似度。"""
        if nid not in self.anchors:
            return 0.0
        anchor = self.anchors[nid].to(field_vector.device)
        if field_vector.dim() == 2:
            field_vector = field_vector.mean(dim=0)
        return torch.dot(field_vector, anchor).item()
