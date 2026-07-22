"""NeuronGeometry — RSGN-inspired geometric position space for neurons.

RSGN (arXiv 2601.18064) 核心思想：
  Neurons have coordinates in a geometric space. Connections are not all-to-all
  but decay with distance — closer neurons interact more, O(N·k) instead of O(N²).

态极融合方式（非替换）：
  - 每个 neuron 在 8D 几何空间有坐标
  - 同域 neuron 初始聚集在近邻区域（半径 0.2）
  - 跨域 neuron 均匀分布在圆周上
  - distance_gate = exp(-dist² / (2σ²))，σ=0.5
  - 作为 CoactivationTracker 的先验权重：近距离 neuron 共激活权重更高

Usage:
    geo = NeuronGeometry(embedding_dim=8)
    geo.assign_domain_positions({"zh": ["zh_1", "zh_2"], "en": ["en_1"]})
    gate = geo.distance_gate("zh_1", "zh_2")  # → ~0.92 (同域近邻)
    gate = geo.distance_gate("zh_1", "en_1")  # → ~0.04 (跨域远邻)
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch


class NeuronGeometry:
    """RSGN-inspired geometric position space for neurons.

    Attributes:
        positions: {nid: 8D tensor} neuron coordinates
        sigma: σ for gaussian decay gate = exp(-dist²/(2σ²))
        embedding_dim: dimension of geometric space (default 8)
    """

    def __init__(self, embedding_dim: int = 8, sigma: float = 0.5):
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        self.positions: Dict[str, torch.Tensor] = {}

    def assign_position(self, nid: str, position: torch.Tensor) -> None:
        """手动设置 neuron 坐标。"""
        self.positions[nid] = position.detach()

    def assign_domain_positions(
        self,
        domain_to_nids: Dict[str, List[str]],
        intra_domain_radius: float = 0.2,
        inter_domain_radius: float = 1.0,
        seed: int = 42,
    ) -> None:
        """按域分配初始坐标。

        同域 neuron 聚集在球半径 intra_domain_radius 内，
        跨域中心均匀分布在半径 inter_domain_radius 的球面上。

        Args:
            domain_to_nids: {domain: [nid, ...]}
            intra_domain_radius: 同域聚集半径（默认 0.2）
            inter_domain_radius: 跨域中心分布半径（默认 1.0）
            seed: 随机种子
        """
        rng = random.Random(seed)
        domains = list(domain_to_nids.keys())
        n_domains = len(domains)

        if n_domains == 0:
            return

        # 为每个域生成一个中心点（均匀分布在球面上）
        domain_centers: Dict[str, torch.Tensor] = {}
        for i, domain in enumerate(domains):
            if n_domains == 1:
                # 单域：中心在原点
                center = torch.zeros(self.embedding_dim)
            else:
                # 多个域：均匀分布在球面上（用 Fibonacci sphere 近似）
                phi = math.acos(1 - 2 * (i + 0.5) / n_domains)
                theta = math.pi * (1 + 5 ** 0.5) * i
                center = torch.zeros(self.embedding_dim)
                center[0] = inter_domain_radius * math.sin(phi) * math.cos(theta)
                center[1] = inter_domain_radius * math.sin(phi) * math.sin(theta)
                center[2] = inter_domain_radius * math.cos(phi)
                # 剩余维度用小随机数填充（避免退化到低维子空间）
                for d in range(3, self.embedding_dim):
                    center[d] = (rng.random() - 0.5) * 0.1 * inter_domain_radius
            domain_centers[domain] = center

        # 为每个 neuron 分配坐标（域中心 + 小偏移）
        for domain, nids in domain_to_nids.items():
            center = domain_centers[domain]
            for nid in nids:
                # 同域 neuron 在 center 附近随机偏移
                offset = torch.randn(self.embedding_dim) * (intra_domain_radius / 3)
                pos = center + offset
                # 确保偏移后仍在球内（clip 到半径）
                if pos.norm().item() > intra_domain_radius + 0.01:
                    pos = pos / pos.norm() * intra_domain_radius * 0.9
                if center.norm().item() > 0.01:
                    # 以 center 为中心，限制偏移半径
                    delta = pos - center
                    if delta.norm().item() > intra_domain_radius:
                        delta = delta / delta.norm() * intra_domain_radius * 0.9
                        pos = center + delta
                self.positions[nid] = pos

    def distance(self, nid_a: str, nid_b: str) -> float:
        """计算两个 neuron 之间的欧氏距离。"""
        if nid_a not in self.positions or nid_b not in self.positions:
            return 1.0  # 未注册的 neuron 默认距离 = 1（非近邻）
        return (self.positions[nid_a] - self.positions[nid_b]).norm().item()

    def distance_gate(self, nid_a: str, nid_b: str) -> float:
        """Gaussian 距离衰减门控：exp(-dist² / (2σ²))。

        - 同域近邻：gate ≈ 1.0（强共激活先验）
        - 跨域远邻：gate ≈ 0.0（弱共激活先验，但不为零）
        - 未注册 neuron：gate = 0.01（最小先验）
        """
        if nid_a not in self.positions or nid_b not in self.positions:
            return 0.01
        dist = self.distance(nid_a, nid_b)
        return math.exp(-(dist ** 2) / (2 * self.sigma ** 2))

    def batch_distance_gates(
        self, nids: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """批量计算所有 pair 的距离门控。"""
        gates = {}
        for i, ni in enumerate(nids):
            for j, nj in enumerate(nids[i + 1:], i + 1):
                gates[(ni, nj)] = self.distance_gate(ni, nj)
        return gates

    def get_position(self, nid: str) -> Optional[torch.Tensor]:
        return self.positions.get(nid)

    def list_positions(self) -> Dict[str, torch.Tensor]:
        return dict(self.positions)

    def get_state_dict(self) -> dict:
        """持久化：positions 转为可序列化格式。"""
        return {
            "positions": {nid: pos.tolist() for nid, pos in self.positions.items()},
            "embedding_dim": self.embedding_dim,
            "sigma": self.sigma,
        }

    def load_state_dict(self, state: dict) -> None:
        """从持久化恢复 positions。"""
        self.embedding_dim = state.get("embedding_dim", 8)
        self.sigma = state.get("sigma", 0.5)
        self.positions = {
            nid: torch.tensor(pos_list, dtype=torch.float32)
            for nid, pos_list in state["positions"].items()
        }