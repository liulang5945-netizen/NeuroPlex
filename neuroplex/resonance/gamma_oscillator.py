"""GammaOscillator — 40Hz Gamma 同步绑定（P6-3）。

人脑参考：
  皮层 Gamma 振荡（30-80Hz，典型 40Hz）是 feature binding 的核心机制。
  不同脑区的神经元群体通过同步振荡绑定成"知觉单元"。
  - 同 phase 的 neuron 放电相互增强（绑定）
  - 异 phase 的 neuron 放电相互干扰（解绑）

态极实现：
  每个 neuron 有一个 phase ∈ [0, 2π)，全局时钟 t 每轮共振推进 ω*dt。
  coherence(neuron) = cos(phase_neuron - phase_global)
  - coherence=+1: 同步，写入增强
  - coherence=-1: 反相，写入衰减

  phase 分配策略：
  - 同 domain 的 neuron 同 phase（绑定成知觉单元）
  - 不同 domain 的 neuron 不同 phase（解绑）
  - 路由器选出的 top-K neuron 天然同 domain，写入时互相增强

接入方式：
  ResonanceField.write() / update() 时，若注册了 gamma_oscillator，
  写入向量乘以 gate_factor = 0.5 + 0.5 * coherence（∈ [0, 1]）。
  这是可选的，不影响现有调用。

Usage:
    osc = GammaOscillator()
    osc.assign_phase("zh_1", phase=0.0)
    osc.assign_phase("en_1", phase=math.pi)  # 反相
    osc.tick()  # 每轮共振推进
    gate = osc.gate_factor("zh_1")  # 调制写入强度
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional
import torch


class GammaOscillator:
    """40Hz Gamma 同步振荡器 — feature binding 机制。

    Attributes:
        phases: {neuron_id: phase ∈ [0, 2π)}
        global_phase: 全局相位（每轮 tick 推进 omega）
        omega: 每轮相位推进（默认 π/4，8 轮走一圈）
    """

    def __init__(
        self,
        omega: float = math.pi / 4,
        min_gate: float = 0.2,
        max_gate: float = 1.0,
        binding_scale: float = 0.3,
    ):
        """
        Args:
            omega: 每轮 tick 的相位推进（默认 π/4，8 轮一个周期）
            min_gate: 最小门控因子（反相时也有最小写入，避免完全静默）
            max_gate: 最大门控因子（同相时增强上限）
            binding_scale: 相位绑定强度（C23：共振分 × (1 + binding_scale·binding)）。
                           >0 时相位同步度直接驱动共振强度（同相群体增强/异相解绑）；
                           0 = 关闭绑定（仅保留写入门控，兼容旧行为）。
        """
        self.phases: Dict[str, float] = {}
        self.global_phase: float = 0.0
        self.omega = omega
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.binding_scale = binding_scale

    def assign_phase(self, neuron_id: str, phase: float) -> None:
        """为 neuron 分配 phase（弧度）。"""
        self.phases[neuron_id] = phase % (2 * math.pi)

    def assign_phase_by_domain(
        self,
        domain_to_nids: Dict[str, list],
        phase_offset_per_domain: float = math.pi / 3,
    ) -> None:
        """按 domain 批量分配 phase：同 domain 同 phase，跨 domain 等距分布。

        Args:
            domain_to_nids: {domain_name: [neuron_id, ...]}
            phase_offset_per_domain: 跨 domain 的 phase 步长（默认 π/3，6 个域均匀分布）
        """
        for i, (domain, nids) in enumerate(domain_to_nids.items()):
            phase = i * phase_offset_per_domain
            for nid in nids:
                self.assign_phase(nid, phase)

    def tick(self, dt: float = 1.0) -> float:
        """推进全局相位。

        Args:
            dt: 时间步长（默认 1.0，表示一轮共振）

        Returns:
            推进后的全局 phase
        """
        self.global_phase = (self.global_phase + self.omega * dt) % (2 * math.pi)
        return self.global_phase

    def kuramoto_step(
        self,
        coupling_strength: float = 0.05,
        active_ids: Optional[list] = None,
        coactivation: Optional[Any] = None,
    ) -> None:
        """KoPE-inspired Kuramoto 相位耦合。

        神经元通过共振耦合相互牵引相位：
        - 共激活强的 neuron 对相互牵引相位同步（绑定成知觉单元）
        - 无共激活的 neuron 对相位独立（解绑）

        dθ_i/dt = ω + K/N * Σ_j sin(θ_j - θ_i) * coactivation(i,j)

        Args:
            coupling_strength: K, 耦合强度（默认 0.05，温和牵引）
            active_ids: 本轮激活的 neuron ID 列表（None=全部）
            coactivation: CoactivationTracker（提供 pair 强度）
        """
        if not self.phases:
            return
        ids = active_ids if active_ids is not None else list(self.phases.keys())
        if len(ids) < 2:
            return

        # 计算每个 neuron 的相位增量
        delta = {nid: 0.0 for nid in ids}
        for i, nid_i in enumerate(ids):
            if nid_i not in self.phases:
                continue
            for nid_j in ids[i + 1:]:
                if nid_j not in self.phases:
                    continue
                # 耦合强度由共激活强度调制
                k = coupling_strength
                if coactivation is not None and hasattr(coactivation, "get_coactivation"):
                    ca = coactivation.get_coactivation(nid_i, nid_j)
                    k *= max(ca, 0.01)  # 无共激活时保留最小耦合
                phase_diff = self.phases[nid_j] - self.phases[nid_i]
                # sin(θ_j - θ_i) 对称：i 受 j 牵引为正，j 受 i 牵引为负
                delta[nid_i] += k * math.sin(phase_diff)
                delta[nid_j] -= k * math.sin(phase_diff)

        # 应用增量
        for nid in ids:
            if nid in self.phases:
                self.phases[nid] = (self.phases[nid] + delta[nid]) % (2 * math.pi)

    def coherence(self, neuron_id: str) -> float:
        """计算 neuron 与全局 phase 的对齐度 ∈ [-1, 1]。

        - +1: 完全同步
        - 0: 正交
        - -1: 完全反相
        """
        if neuron_id not in self.phases:
            return 1.0  # 未注册的 neuron 默认同步（不施加调制）
        return math.cos(self.phases[neuron_id] - self.global_phase)

    def gate_factor(self, neuron_id: str) -> float:
        """门控因子 ∈ [min_gate, max_gate]，用于调制写入强度。

        coherence=+1 → max_gate（同相增强）
        coherence=-1 → min_gate（反相衰减但不完全静默）
        """
        c = self.coherence(neuron_id)
        # 线性映射 [-1, 1] -> [min_gate, max_gate]
        return self.min_gate + (self.max_gate - self.min_gate) * (c + 1.0) / 2.0

    def batch_gate_factors(self, neuron_ids: list) -> torch.Tensor:
        """批量获取门控因子（用于 ensemble forward 时一次计算）。"""
        return torch.tensor(
            [self.gate_factor(nid) for nid in neuron_ids],
            dtype=torch.float32,
        )

    def pairwise_binding(
        self,
        active_ids: Optional[list] = None,
        coactivation: Optional[Any] = None,
    ) -> Dict[str, float]:
        """计算每个 neuron 与其他激活 neuron 的平均相位同步度（pairwise 关系）。

        C23（2026-08-08）：相位同步本体化——相位不再是"对全局相位的标量门控"，
        而是 neuron 之间的**关系度量**：同相 neuron 群体绑结成知觉单元
        （binding → +1），异相解绑（binding → -1）。

            binding_i = mean_{j≠i} [ cos(θ_i - θ_j) ]      ∈ [-1, 1]

        可选共激活调制（与 Kuramoto 耦合一致：k *= max(ca, 0.01)）——
        共激活强的 neuron 对相位相互牵引（逐渐同相），binding 随共振演化增强，
        形成"共激活 → 相位同步 → 绑结"的动态闭环。

        Args:
            active_ids: 本轮激活的 neuron ID 列表（None=全部）
            coactivation: CoactivationTracker（提供 pair 强度）

        Returns:
            {nid: binding ∈ [-1, 1]}（不足 2 个时全部 0.0）
        """
        if not self.phases:
            return {}
        ids = active_ids if active_ids is not None else list(self.phases.keys())
        ids = [nid for nid in ids if nid in self.phases]
        if len(ids) < 2:
            return {nid: 0.0 for nid in ids}
        result: Dict[str, float] = {}
        for nid_i in ids:
            theta_i = self.phases[nid_i]
            total = 0.0
            for nid_j in ids:
                if nid_i == nid_j:
                    continue
                s = math.cos(theta_i - self.phases[nid_j])
                if coactivation is not None and hasattr(coactivation, "get_coactivation"):
                    s *= max(coactivation.get_coactivation(nid_i, nid_j), 0.01)
                total += s
            result[nid_i] = total / (len(ids) - 1)
        return result

    def reset(self) -> None:
        """重置全局 phase 到 0（新输入开始时调用）。"""
        self.global_phase = 0.0

    def get_phase(self, neuron_id: str) -> Optional[float]:
        return self.phases.get(neuron_id)

    def list_phases(self) -> Dict[str, float]:
        return dict(self.phases)


def apply_gamma_gate(
    field,
    gamma_oscillator: Optional[GammaOscillator],
) -> None:
    """把 GammaOscillator 注入到 ResonanceField（不破坏现有接口）。

    注入后，field.write() 和 field.update() 会自动用 gamma gate 调制写入强度。
    传入 None 可移除注入。

    Args:
        field: ResonanceField 实例
        gamma_oscillator: GammaOscillator 实例或 None
    """
    field._gamma_oscillator = gamma_oscillator
    if gamma_oscillator is not None:
        # monkey-patch write/update（保留原方法）
        if not hasattr(field, '_original_write'):
            field._original_write = field.write
            field._original_update = field.update

            def gated_write(neuron_id, vector, scale=1.0):
                osc = field._gamma_oscillator
                if osc is not None and neuron_id in osc.phases:
                    gate = osc.gate_factor(neuron_id)
                    vector = vector * gate
                return field._original_write(neuron_id, vector, scale=scale)

            def gated_update(neuron_id, vector, scale=1.0):
                osc = field._gamma_oscillator
                if osc is not None and neuron_id in osc.phases:
                    gate = osc.gate_factor(neuron_id)
                    vector = vector * gate
                return field._original_update(neuron_id, vector, scale=scale)

            field.write = gated_write
            field.update = gated_update
