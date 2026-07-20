"""认知增强机制（人脑启发）。

包含三个 Phase 4 机制：
1. 功能柱 (Cortical Column) 原生实现：部落作为一等公民
2. 注意力增益 (Attentional Modulation)：自上而下注意力增强相关神经元
3. 阈值可塑性 (Threshold Plasticity)：per-neuron 发放阈值自适应

人脑参考：
- 皮层功能柱：~100 微米直径，~100 神经元处理同类输入
- 注意力增益：皮层-丘脑回路增强特定区域
- 阈值可塑性：homeostatic scaling，防止过度兴奋
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("Taiji.Cognitive")


class CorticalColumn:
    """功能柱（人脑启发：皮层功能柱原生实现）。

    人脑皮层功能柱：
    - ~100 微米直径，~100 神经元
    - 内部强连接（兴奋性+抑制性）
    - 处理同类输入
    - 柱间通过横向连接形成更大尺度组织

    态极实现：
    - TribeSuperNeuron 作为一等公民（不再只是事后压缩）
    - 新神经元直接创建到某个部落（基于 domain）
    - 部落内部 dense side_channels，部落间 sparse
    - 部落可分裂（规模过大）或合并（功能重叠）

    与现有 TribeSuperNeuron 的区别：
    - TribeSuperNeuron 是"压缩器"，把 N 个成员压成 1 个向量
    - CorticalColumn 是"组织单元"，管理成员的创建、连接、生命周期
    """

    def __init__(
        self,
        column_id: str,
        domain: str,
        max_members: int = 100,
        split_threshold: int = 150,
        merge_similarity: float = 0.9,
    ):
        self.column_id = column_id
        self.domain = domain
        self.max_members = max_members
        self.split_threshold = split_threshold  # 超过此规模考虑分裂
        self.merge_similarity = merge_similarity  # 相似度高于此值考虑合并

        self.member_ids: List[str] = []
        self._internal_connections: Dict[tuple, float] = {}  # (id_a, id_b) -> strength

    def add_member(self, neuron_id: str) -> None:
        """添加成员到功能柱。"""
        if neuron_id not in self.member_ids:
            self.member_ids.append(neuron_id)
            # 与所有现有成员建立内部连接（初始强度=0.5）
            for other_id in self.member_ids[:-1]:
                self._internal_connections[(neuron_id, other_id)] = 0.5
                self._internal_connections[(other_id, neuron_id)] = 0.5

    def remove_member(self, neuron_id: str) -> None:
        """移除成员。"""
        if neuron_id in self.member_ids:
            self.member_ids.remove(neuron_id)
            # 清理相关连接
            to_remove = [k for k in self._internal_connections if neuron_id in k]
            for key in to_remove:
                del self._internal_connections[key]

    def should_split(self) -> bool:
        """是否应该分裂（规模过大）。"""
        return len(self.member_ids) > self.split_threshold

    def should_merge_with(self, other: "CorticalColumn", similarity: float) -> bool:
        """是否应该与另一个功能柱合并。"""
        return (
            self.domain == other.domain
            and similarity > self.merge_similarity
        )

    def get_internal_connection_strength(self, id_a: str, id_b: str) -> float:
        """获取两个成员间的内部连接强度。"""
        return self._internal_connections.get((id_a, id_b), 0.0)

    def strengthen_internal(self, id_a: str, id_b: str, delta: float = 0.1) -> None:
        """强化内部连接（Hebbian）。"""
        key = (id_a, id_b)
        current = self._internal_connections.get(key, 0.0)
        self._internal_connections[key] = min(1.0, current + delta)

    def get_stats(self) -> dict:
        return {
            "column_id": self.column_id,
            "domain": self.domain,
            "n_members": len(self.member_ids),
            "n_internal_connections": len(self._internal_connections) // 2,
            "should_split": self.should_split(),
        }


class ColumnRegistry:
    """功能柱注册表：管理所有功能柱。"""

    def __init__(self):
        self._columns: Dict[str, CorticalColumn] = {}
        # neuron_id -> column_id 反向映射
        self._neuron_to_column: Dict[str, str] = {}

    def create_column(self, column_id: str, domain: str) -> CorticalColumn:
        """创建新功能柱。"""
        column = CorticalColumn(column_id, domain)
        self._columns[column_id] = column
        return column

    def assign_neuron(self, neuron_id: str, column_id: str) -> None:
        """将神经元分配到功能柱。"""
        if column_id not in self._columns:
            raise ValueError(f"功能柱 {column_id} 不存在")
        # 从原功能柱移除
        old_column = self._neuron_to_column.get(neuron_id)
        if old_column and old_column in self._columns:
            self._columns[old_column].remove_member(neuron_id)
        # 添加到新功能柱
        self._columns[column_id].add_member(neuron_id)
        self._neuron_to_column[neuron_id] = column_id

    def get_column(self, neuron_id: str) -> Optional[CorticalColumn]:
        """获取神经元所属的功能柱。"""
        column_id = self._neuron_to_column.get(neuron_id)
        if column_id:
            return self._columns.get(column_id)
        return None

    def auto_assign_by_domain(self, neuron_id: str, domain: str) -> str:
        """根据 domain 自动分配到功能柱（无则创建）。

        Returns:
            分配到的 column_id
        """
        # 找同 domain 的现有功能柱
        for col_id, column in self._columns.items():
            if column.domain == domain and not column.should_split():
                self.assign_neuron(neuron_id, col_id)
                return col_id

        # 没有合适的，创建新的
        col_id = f"col_{domain}_{len(self._columns)}"
        self.create_column(col_id, domain)
        self.assign_neuron(neuron_id, col_id)
        return col_id

    def get_all_columns(self) -> List[CorticalColumn]:
        return list(self._columns.values())

    def get_stats(self) -> dict:
        return {
            "total_columns": len(self._columns),
            "total_neurons_assigned": len(self._neuron_to_column),
            "columns": {col_id: col.get_stats() for col_id, col in self._columns.items()},
        }


class AttentionBeam:
    """注意力增益（人脑启发：自上而下注意力增强）。

    人脑注意力机制：
    - 自上而下信号（前额叶→感觉皮层）
    - 增强特定区域神经元响应
    - 抑制其他区域

    态极实现：
    - attention_beam 向量来自用户 query embedding 或任务上下文
    - 场评分时与 attention_beam 对齐的神经元获得 boost
    - lm_head 加权时注意力之外的神经常被抑制
    """

    def __init__(
        self,
        field_dim: int = 4096,
        boost_factor: float = 2.0,
        suppress_factor: float = 0.5,
        alignment_threshold: float = 0.3,
    ):
        self.field_dim = field_dim
        self.boost_factor = boost_factor
        self.suppress_factor = suppress_factor
        self.alignment_threshold = alignment_threshold

        # 当前注意力向量（[D] 维，L2 归一化）
        self._beam: Optional[torch.Tensor] = None

    def set_beam(self, context_vector: torch.Tensor) -> None:
        """设置注意力向量（来自任务上下文或用户 query）。

        Args:
            context_vector: [D] 或 [B, D] 上下文向量
        """
        if context_vector.dim() == 1:
            context_vector = context_vector.unsqueeze(0)
        # L2 归一化
        self._beam = context_vector / (
            context_vector.norm(dim=-1, keepdim=True) + 1e-8
        )

    def clear_beam(self) -> None:
        """清除注意力（恢复民主模式）。"""
        self._beam = None

    def get_score_boost(
        self,
        neuron_field_vector: torch.Tensor,
    ) -> float:
        """获取神经元的注意力分数 boost。

        与 attention_beam 对齐的神经元获得 boost_factor 倍分数。
        不对齐的被 suppress_factor 倍抑制。

        Returns:
            分数倍数（1.0 = 中性，>1 = boost，<1 = suppress）
        """
        if self._beam is None:
            return 1.0

        vec = neuron_field_vector
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        # per-sample cosine
        sim = ((vec * self._beam).sum(dim=-1) /
               (vec.norm(dim=-1) * self._beam.norm(dim=-1) + 1e-8)).mean().item()

        if sim > self.alignment_threshold:
            # 对齐：boost
            # 线性插值：threshold → 1.0, sim=1.0 → boost_factor
            t = (sim - self.alignment_threshold) / (1.0 - self.alignment_threshold + 1e-8)
            return 1.0 + (self.boost_factor - 1.0) * t
        else:
            # 不对齐：suppress
            t = sim / self.alignment_threshold
            return self.suppress_factor + (1.0 - self.suppress_factor) * t

    def get_lm_head_weight_modifier(
        self,
        neuron_field_vector: torch.Tensor,
    ) -> float:
        """获取 lm_head 权重的注意力修饰。

        Returns:
            权重倍数（与 score_boost 类似但可能更激进）
        """
        return self.get_score_boost(neuron_field_vector)

    @property
    def is_active(self) -> bool:
        return self._beam is not None

    def get_stats(self) -> dict:
        return {
            "active": self.is_active,
            "boost_factor": self.boost_factor,
            "suppress_factor": self.suppress_factor,
            "alignment_threshold": self.alignment_threshold,
        }


class ThresholdPlasticity:
    """阈值可塑性（人脑启发：homeostatic scaling）。

    人脑神经元发放阈值根据近期活动历史自适应调整：
    - 频繁发放 → threshold 上升（更难再激活，防止过度兴奋）
    - 长期沉默 → threshold 下降（更容易被唤醒）

    态极实现：
    - 每个神经元维护 firing_threshold buffer
    - 基于 CoactivationTracker 的 activation_count 调整
    - 目标：让所有神经元都有公平的激活机会
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        max_threshold: float = 0.8,
        min_threshold: float = 0.1,
        adaptation_rate: float = 0.01,
        target_activation_rate: float = 0.3,  # 目标激活率
    ):
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.adaptation_rate = adaptation_rate
        self.target_activation_rate = target_activation_rate

        # neuron_id -> 当前阈值
        self._thresholds: Dict[str, float] = {}

    def get_threshold(self, neuron_id: str) -> float:
        """获取神经元的当前发放阈值。"""
        return self._thresholds.get(neuron_id, self.base_threshold)

    def adapt(
        self,
        neuron_id: str,
        activation_count: int,
        total_rounds: int,
    ) -> float:
        """根据激活历史调整神经元的发放阈值。

        Args:
            neuron_id: 神经元 ID
            activation_count: 总激活次数
            total_rounds: 总轮次

        Returns:
            新的阈值
        """
        if total_rounds == 0:
            return self.get_threshold(neuron_id)

        actual_rate = activation_count / total_rounds
        current = self.get_threshold(neuron_id)

        # 误差信号：实际激活率 - 目标激活率
        # 正误差 = 激活过多 → 提高阈值
        # 负误差 = 激活过少 → 降低阈值
        error = actual_rate - self.target_activation_rate

        # 比例控制
        new_threshold = current + self.adaptation_rate * error * 10  # ×10 放大自适应速度

        # 限幅
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        self._thresholds[neuron_id] = new_threshold
        return new_threshold

    def adapt_all(
        self,
        activation_counts: Dict[str, int],
        total_rounds: int,
    ) -> Dict[str, float]:
        """批量调整所有神经元的阈值。

        Args:
            activation_counts: {neuron_id: activation_count}
            total_rounds: 总轮次

        Returns:
            {neuron_id: new_threshold}
        """
        new_thresholds = {}
        for nid, count in activation_counts.items():
            new_thresholds[nid] = self.adapt(nid, count, total_rounds)
        return new_thresholds

    def reset(self, neuron_id: str) -> None:
        """重置某神经元的阈值到基础值。"""
        self._thresholds.pop(neuron_id, None)

    def get_stats(self) -> dict:
        if not self._thresholds:
            return {"n_neurons_tracked": 0}
        return {
            "n_neurons_tracked": len(self._thresholds),
            "mean_threshold": sum(self._thresholds.values()) / len(self._thresholds),
            "min_threshold": min(self._thresholds.values()),
            "max_threshold": max(self._thresholds.values()),
        }
