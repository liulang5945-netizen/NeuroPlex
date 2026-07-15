"""
部落压缩质量量化 (Tribal Compression Quality Metrics)
=====================================================

共振场三层架构中，部落对外表现为一个超级神经元。N 个成员的内部子场动态
被压缩为一个 4096 维单位向量写入上级场。

本模块量化两件事：
1. 信号质量因子 Q = α·β·γ — 压缩后的输出有多可信
2. 压缩损失 — 单向量能否代表所有成员的输出（决定部落是否该解散）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("Taiji.Tribal")


# ── 纯计算类：不继承 nn.Module，无训练参数 ──


class TribalMetrics:
    """
    部落压缩质量的三个内部指标 + 信号质量因子 + 压缩损失。

    使用方法：
        metrics = TribalMetrics()

        for round_num in range(max_rounds):
            member_writes = tribe.run_round(input_data, sub_field_state)
            sub_field_state = tribe.sub_field.state
            metrics.record_round(member_writes, sub_field_state)

        Q = metrics.quality_factor()
        loss = metrics.compression_loss()
    """

    def __init__(self):
        self._member_writes_history: list[list[torch.Tensor]] = []
        self._sub_field_history: list[torch.Tensor] = []

    def record_round(
        self,
        member_writes: list[torch.Tensor],
        sub_field_state: torch.Tensor,
    ) -> None:
        """
        记录一轮共振后的部落状态。

        Args:
            member_writes: 每个成员本轮最终写入的场向量（已归一化）
            sub_field_state: 子场聚合后的状态向量
        """
        self._member_writes_history.append([v.detach().clone() for v in member_writes])
        self._sub_field_history.append(sub_field_state.detach().clone())

    @property
    def latest_writes(self) -> list[torch.Tensor]:
        """最后一轮各成员的写入向量。"""
        if not self._member_writes_history:
            return []
        return self._member_writes_history[-1]

    @property
    def field_history(self) -> list[torch.Tensor]:
        """子场状态的完整历史。"""
        return self._sub_field_history

    # ── 指标 1：内部相干度 α ──

    @property
    def coherence(self) -> float:
        """
        部落内所有成员最终轮写入向量的 pairwise cosine 均值。

        α ∈ [0, 1]
        高 → "大家在说同一件事"，输出方向可信。
        低 → "内部分歧"，压缩后的方向不可靠。
        """
        writes = self.latest_writes
        if len(writes) < 2:
            return 1.0

        pairwise_cos = []
        for vi, vj in combinations(writes, 2):
            pairwise_cos.append(float(torch.dot(vi, vj)))
        return sum(pairwise_cos) / len(pairwise_cos)

    # ── 指标 2：收敛速度 β ──

    @property
    def stability(self) -> float:
        """
        子场状态在共振轮次间的指数衰减加权稳定度。

        β ∈ [0, 1]
        高 → 子场快速收敛，部落对这个输入理解透彻。
        低 → 持续震荡，部落内部在 struggled。
        """
        history = self._sub_field_history
        if len(history) < 2:
            return 1.0

        deltas = []
        for t in range(1, len(history)):
            cos_sim = float(
                torch.dot(history[t], history[t - 1])
                / (history[t].norm() * history[t - 1].norm() + 1e-8)
            )
            deltas.append(1.0 - cos_sim)

        # 指数衰减加权：最近的变化最重要
        weights = [0.7 ** i for i in range(len(deltas))]
        weighted_delta = sum(
            w * d for w, d in zip(reversed(weights), reversed(deltas))
        )
        return 1.0 - weighted_delta

    # ── 指标 3：方向散布度 γ ──

    @property
    def spread(self) -> float:
        """
        成员方向在质心周围的空间集中度。

        使用成员写入向量到质心的平均欧氏距离（而非投影方差）。
        投影方差在余弦接近时区分度弱（成员分别位于两个正交簇时，
        投影到均值的方差可能仍接近 0），欧氏距离不受此限制。

        γ ∈ [0, 1]
        高 → 成员紧密聚集，压缩损失可忽略。
        低 → 方向四散（可能是多簇混合），单向量压缩无法概括。
        """
        writes = self.latest_writes
        if len(writes) < 2:
            return 1.0

        # 质心方向（L2 归一化）
        stacked = torch.stack(writes, dim=0)  # [N, D]
        raw_centroid = stacked.mean(dim=0)
        centroid = raw_centroid / (raw_centroid.norm() + 1e-8)

        # 每个成员到质心的欧氏距离（已归一化向量间，最大距离 = √2）
        distances = [(v - centroid).norm().item() for v in writes]
        mean_dist = sum(distances) / len(distances)

        # mean_dist ∈ [0, √2] → γ ∈ [1/(1+√2), 1] ≈ [0.41, 1]
        # 紧密聚集：mean_dist ≈ 0.1 → γ ≈ 0.91
        # 双正交簇：mean_dist ≈ 0.9 → γ ≈ 0.53
        return 1.0 / (1.0 + mean_dist)

    # ── 合成：信号质量因子 Q ──

    def quality_factor(self) -> float:
        """
        信号质量因子 Q = α · β · γ。

        Q ∈ [0, 1]
        任何一项低 → 整体可信度就低。

        Q ≈ 0.8: 方向清晰、内部一致 → 上级场可重度依赖。
        Q ≈ 0.3: 有分歧 → 上级场应谨慎。
        Q ≈ 0.05: 不可信 → 上级场应忽略。
        """
        return self.coherence * self.stability * self.spread

    # ── 压缩损失 ──

    def compression_loss(self) -> float:
        """
        用最终子场方向 v_tribe 解释各成员写入的残差。

        ∈ [0, √2]（近似）
        0 = 部落输出完美代表所有人。
        > 0.5 = 建议触发解散（成员差异太大，强行压缩降低信息效率）。

        这是可逆部落机制（文档 18.5）的解散触发条件之一。
        """
        writes = self.latest_writes
        if len(writes) < 2:
            return 0.0

        # v_tribe = 子场最终状态的归一化方向
        final_field = self._sub_field_history[-1] if self._sub_field_history else writes[0]
        v_tribe = final_field / (final_field.norm() + 1e-8)

        residuals = []
        for v in writes:
            proj = (v @ v_tribe) * v_tribe
            residual = (v - proj).norm().item()
            residuals.append(residual)

        return sum(residuals) / len(residuals)

    # ── 完整摘要 ──

    def summary(self) -> dict[str, float]:
        """返回所有指标的一次性汇总。"""
        return {
            "coherence_alpha": round(self.coherence, 4),
            "stability_beta": round(self.stability, 4),
            "spread_gamma": round(self.spread, 4),
            "quality_factor_Q": round(self.quality_factor(), 4),
            "compression_loss": round(self.compression_loss(), 4),
        }

    def reset(self) -> None:
        """清空历史，准备下一次推理。"""
        self._member_writes_history.clear()
        self._sub_field_history.clear()


# ── 超级神经元：部落对外接口 ──


class TribeSuperNeuron(nn.Module):
    """
    部落超级神经元 — 在上级场中表现为一个普通神经元。

    封装了：
    - 内部子场引用
    - 成员列表
    - 质量因子的自动计算与注入

    设计原则：
    - field_write() 永远返回单位向量（不破坏归一化）
    - 质量因子通过 compute_resonance() 影响上级场的共振度计算
    - 对上级场完全透明——上级场不需要知道这是一个部落
    """

    # 部落压缩损失的解散阈值
    DISSOLVE_THRESHOLD: float = 0.5

    def __init__(
        self,
        tribe_id: int,
        members: list[Any],          # list[ResonanceNeuron]
        sub_field: Any,              # ResonanceField（部落内部子场）
        parent_field: Any = None,    # 上级 ResonanceField
    ):
        """
        Args:
            tribe_id: 部落在上级场中的唯一标识
            members: 部落成员（ResonanceNeuron 实例）
            sub_field: 部落内部子场（ResonanceField 实例）
            parent_field: 上级共振场。如果为 None，D 从 sub_field.D 推断。
                D 从上级场继承，而非硬编码——上级场扩张时部落跟随。
        """
        super().__init__()
        self.tribe_id = tribe_id
        self.members = members
        self.sub_field = sub_field
        self.parent_field = parent_field

        # D 继承：上级场维度 > sub_field.D > 4096 回退
        if parent_field is not None and hasattr(parent_field, "D"):
            parent_D = parent_field.D
        elif hasattr(sub_field, "D"):
            parent_D = sub_field.D
        else:
            parent_D = 4096

        sub_D = getattr(sub_field, "D", parent_D)

        # 部落对外场投影（与普通神经元的 field_write 等价）
        self.field_write_proj = nn.Linear(sub_D, parent_D, bias=False)

        # 部落指纹（与普通神经元的 fingerprint 等价）
        self.register_buffer("fingerprint", torch.zeros(parent_D))

        # 部落在上级场中的评估状态
        self.register_buffer("L_score_fast", torch.tensor(0.0))
        self.register_buffer("L_score_slow", torch.tensor(0.0))
        self.register_buffer("days_since_creation", torch.tensor(0.0))

        self._last_v_tribe: Optional[torch.Tensor] = None
        self._last_Q: float = 1.0

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def last_field_vector(self) -> Optional[torch.Tensor]:
        """最后一次推理的部落写入向量（单位向量）。"""
        return self._last_v_tribe

    @property
    def last_quality(self) -> float:
        """最后一次推理的信号质量因子。"""
        return self._last_Q

    def forward_tribe(
        self,
        input_ids: torch.Tensor,
        max_rounds: int = 5,
    ) -> dict:
        """
        执行部落内部完整共振循环，返回对外输出。

        Args:
            input_ids: 输入 token IDs [B, L]
            max_rounds: 最大内部共振轮数

        Returns:
            dict with:
                - v_tribe: 部落对外写入向量（单位长度）
                - quality_factor: Q = α·β·γ
                - metrics_summary: 所有指标的 dict
                - should_dissolve: 是否建议解散
        """
        self.metrics.reset()
        self.sub_field.reset() if hasattr(self.sub_field, "reset") else None

        # ── 第 1 轮：所有成员独立前向 ──
        member_writes = []
        for neuron in self.members:
            v = neuron.forward(input_ids, field_state=None, round_num=1)["field_vector"]
            member_writes.append(v)
            if hasattr(self.sub_field, "write"):
                self.sub_field.write(neuron.neuron_id, v)

        sub_field_state = (
            self.sub_field.state
            if hasattr(self.sub_field, "state")
            else sum(member_writes) / len(member_writes)
        )
        self.metrics.record_round(member_writes, sub_field_state)

        # ── 第 2-N 轮：读子场 → 条件化 → 重新写入 ──
        for r in range(2, max_rounds + 1):
            member_writes = []
            for neuron in self.members:
                v = neuron.forward(
                    input_ids,
                    field_state=sub_field_state,
                    round_num=r,
                )["field_vector"]
                member_writes.append(v)
                if hasattr(self.sub_field, "update"):
                    self.sub_field.update(neuron.neuron_id, v)

            sub_field_state = (
                self.sub_field.state
                if hasattr(self.sub_field, "state")
                else sum(member_writes) / len(member_writes)
            )
            self.metrics.record_round(member_writes, sub_field_state)

        # ── 合成部落对外输出 ──
        v_tribe_raw = self.field_write_proj(sub_field_state)
        self._last_v_tribe = v_tribe_raw / (v_tribe_raw.norm(dim=-1, keepdim=True) + 1e-8)
        self._last_Q = self.metrics.quality_factor()

        compression = self.metrics.compression_loss()
        should_dissolve = compression > self.DISSOLVE_THRESHOLD

        if should_dissolve:
            logger.warning(
                "部落 %s 压缩损失 %.3f > %.2f，建议解散",
                self.tribe_id,
                compression,
                self.DISSOLVE_THRESHOLD,
            )

        return {
            "v_tribe": self._last_v_tribe,
            "quality_factor": self._last_Q,
            "metrics_summary": self.metrics.summary(),
            "should_dissolve": should_dissolve,
        }

    def compute_resonance(self, parent_field_state: torch.Tensor) -> float:
        """
        在上级场中计算共振度（质量因子调权）。

        与普通神经元的关键区别：
            score_effective = cosine(v_tribe, F_parent) * Q

        普通神经元 Q=1.0，行为不变。

        Args:
            parent_field_state: 上级场当前状态 [D]

        Returns:
            质量因子调权后的有效共振度 ∈ [-1, 1]
        """
        if self._last_v_tribe is None:
            return 0.0

        v = self._last_v_tribe.flatten()
        F = parent_field_state.flatten()
        raw_score = float(
            torch.dot(v, F) / (v.norm() * F.norm() + 1e-8)
        )
        return raw_score * self._last_Q

    def freeze_fingerprint(self) -> None:
        """固化部落指纹——field_write 投影的归一化均值方向。"""
        with torch.no_grad():
            fp = self.field_write_proj.weight.mean(dim=0)
            self.fingerprint.copy_(fp / (fp.norm() + 1e-8))

    def get_status(self) -> dict:
        """获取部落状态摘要。"""
        return {
            "tribe_id": self.tribe_id,
            "n_members": self.n_members,
            "L_score_fast": round(self.L_score_fast.item(), 4),
            "L_score_slow": round(self.L_score_slow.item(), 4),
            "days_since_creation": round(self.days_since_creation.item(), 1),
            "last_quality": round(self._last_Q, 4),
            "should_dissolve": (
                self.metrics.compression_loss() > self.DISSOLVE_THRESHOLD
                if self._last_v_tribe is not None
                else False
            ),
        }


# ── 部落评估辅助 ──


@dataclass
class CoactivationTracker:
    """
    跨神经元共激活矩阵追踪。

    用于检测应该主动部落化的神经元组。
    """

    ema_alpha: float = 0.1
    threshold: float = 0.6
    _matrix: dict[tuple[int, int], float] = field(default_factory=dict)

    def update(self, active_neuron_ids: list[int]) -> None:
        """每轮共振后更新共激活频率。"""
        for i, j in combinations(sorted(active_neuron_ids), 2):
            key = (i, j)
            prev = self._matrix.get(key, 0.0)
            self._matrix[key] = self.ema_alpha * 1.0 + (1 - self.ema_alpha) * prev

    def get_dense_groups(self, min_size: int = 8) -> list[list[int]]:
        """
        找出共激活密度超过阈值的神经元组。

        简单实现：对超过阈值的边做连通分量分析。
        """
        dense_edges = [
            (i, j)
            for (i, j), freq in self._matrix.items()
            if freq > self.threshold
        ]

        # 连通分量（Union-Find）
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in dense_edges:
            union(i, j)

        # 按根节点分组
        groups: dict[int, list[int]] = {}
        for node in parent:
            root = find(node)
            groups.setdefault(root, []).append(node)

        return [sorted(g) for g in groups.values() if len(g) >= min_size]
