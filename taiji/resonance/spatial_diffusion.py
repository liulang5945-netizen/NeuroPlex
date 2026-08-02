"""C7: 空间场扩散动力学 — 图拉普拉斯扩散。

将"场是单一 D 维向量"升级为"空间场 + 扩散动力学"：
每个神经元有自己的位置（NeuronGeometry），信号在神经元之间扩散。

扩散方程（线性、可微）：
    V' = normalize(V - alpha * L @ V)

其中：
- V: [N, B, D] per-neuron field vectors（L2-normalized）
- L: [N, N] 对称归一化图拉普拉斯 = I - D^(-1/2) W D^(-1/2)
- W[i,j] = exp(-dist(i,j)^2 / (2*sigma^2))（高斯衰减权重，复用 NeuronGeometry）
- alpha: 扩散强度（可由调质驱动：高多巴胺→强扩散=信号传播远）

等价形式：V' = normalize((1-alpha)*V + alpha*W_norm@V)
其中 W_norm = D^(-1/2) W D^(-1/2)（归一化权重矩阵）
- alpha=0: V'=V（退化）
- alpha=1: V'=normalize(邻居加权平均)（完全平滑）
- alpha∈(0,1): V 和邻居平均的凸组合

特性：
- 线性可微：L 是常数矩阵，扩散是矩阵乘法，全程可微，进入梯度流
- 向后兼容：alpha=0 时 V'=V，完全退化
- 空间结构：近邻神经元信号互相扩散，远邻不扩散
- 子集支持：active_ids 可能是 neuron_ids 子集，自动构建子图拉普拉斯

生物学启发：
- 人脑皮层信号在局部皮层柱间扩散传播
- 近邻神经元强耦合，远距离弱耦合
- 扩散使"局部共识"传播为"全局共识"
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


class SpatialDiffuser:
    """图拉普拉斯空间扩散器。

    在 per-neuron field vectors 上施加扩散动力学，
    使信号在神经元空间中传播。

    Usage:
        diffuser = SpatialDiffuser(["n0", "n1", "n2"], geometry, alpha=0.1)
        # per_neuron_vectors: [N, B, D]
        diffused = diffuser.diffuse(per_neuron_vectors)
    """

    def __init__(
        self,
        neuron_ids: List[str],
        geometry,  # NeuronGeometry
        alpha: float = 0.1,
        sigma: Optional[float] = None,
    ):
        self.neuron_ids = list(neuron_ids)
        self.alpha = alpha
        self.sigma = sigma if sigma is not None else getattr(geometry, "sigma", 0.5)
        # 构建图拉普拉斯 L_sym [N, N]（常数矩阵，不参与训练）
        self.graph_laplacian: torch.Tensor = self._build_graph_laplacian(geometry)

    def _build_graph_laplacian(self, geometry) -> torch.Tensor:
        """构建对称归一化图拉普拉斯 L = I - D^(-1/2) W D^(-1/2)。

        W[i,j] = exp(-dist(i,j)^2 / (2*sigma^2))（高斯衰减，复用 geometry.distance_gate）
        D[i,i] = sum_j W[i,j]（度矩阵）
        """
        N = len(self.neuron_ids)
        if N == 0:
            return torch.zeros(0, 0)

        # 构建权重矩阵 W [N, N]
        W = torch.zeros(N, N)
        for i, ni in enumerate(self.neuron_ids):
            for j, nj in enumerate(self.neuron_ids):
                if i == j:
                    continue
                W[i, j] = geometry.distance_gate(ni, nj)

        # 度矩阵 D（对角向量）
        D = W.sum(dim=1)  # [N]

        # D^(-1/2)（对角矩阵，避免除零）
        D_inv_sqrt = torch.zeros(N)
        D_inv_sqrt[D > 1e-8] = 1.0 / D[D > 1e-8].sqrt()

        # L_sym = I - D^(-1/2) W D^(-1/2)
        # 用向量运算：D^(-1/2) W D^(-1/2) = (D_inv_sqrt[:,None] * W) * D_inv_sqrt[None,:]
        W_norm = D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
        I = torch.eye(N)
        L = I - W_norm

        return L

    def _get_sub_laplacian(self, active_ids: List[str]) -> torch.Tensor:
        """获取 active_ids 对应的子图拉普拉斯。

        若 active_ids 与 self.neuron_ids 完全一致，直接返回完整 L；
        否则取子矩阵。
        """
        if active_ids == self.neuron_ids:
            return self.graph_laplacian

        idx_map = {nid: i for i, nid in enumerate(self.neuron_ids)}
        active_idx = [idx_map[nid] for nid in active_ids if nid in idx_map]
        if len(active_idx) < 2:
            # 少于 2 个节点，无需扩散
            return None
        return self.graph_laplacian[active_idx][:, active_idx]

    def diffuse(
        self,
        per_neuron_vectors: torch.Tensor,
        active_ids: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """对 per-neuron vectors 做图拉普拉斯扩散。

        V' = normalize(V - alpha * L @ V)
        等价于 V' = normalize((1-alpha)*V + alpha*W_norm@V)

        Args:
            per_neuron_vectors: [N, B, D] 或 [N, D]，L2-normalized
            active_ids: 本次 active 的 neuron IDs（None 时用 self.neuron_ids）
            alpha: 扩散强度（None 时用 self.alpha）

        Returns:
            扩散后的 vectors，同形状，重新归一化到单位向量
        """
        a = alpha if alpha is not None else self.alpha
        if a == 0:
            return per_neuron_vectors

        ids = active_ids if active_ids is not None else self.neuron_ids
        L = self._get_sub_laplacian(ids)
        if L is None:
            return per_neuron_vectors  # 少于 2 个节点，无需扩散

        # 确保 L 与 V 在同一 device
        L = L.to(per_neuron_vectors.device)

        if per_neuron_vectors.dim() == 2:  # [N, D]
            # L @ V: [N, N] @ [N, D] -> [N, D]
            diffused = per_neuron_vectors - a * (L @ per_neuron_vectors)
        elif per_neuron_vectors.dim() == 3:  # [N, B, D]
            # L @ V: einsum('ij,jbd->ibd')
            diffused = per_neuron_vectors - a * torch.einsum(
                "ij,jbd->ibd", L, per_neuron_vectors
            )
        else:
            raise ValueError(
                f"per_neuron_vectors dim must be 2 or 3, got {per_neuron_vectors.dim()}"
            )

        # 重新归一化（保持单位向量语义，扩散只改变方向）
        return F.normalize(diffused, dim=-1)

    def rebuild(self, neuron_ids: List[str], geometry) -> None:
        """重建图拉普拉斯（geometry 更新后调用）。"""
        self.neuron_ids = list(neuron_ids)
        self.graph_laplacian = self._build_graph_laplacian(geometry)
