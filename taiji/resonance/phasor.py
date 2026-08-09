"""PhasorDynamics — 可微相位动力学（C23-C 增量：相位同步本体化最终形态）。

设计本意：振荡相位同步是态极共振的本体机制——"谁与谁同相"决定绑结
（feature binding）。标量 GammaOscillator 的相位是离散标量（cos 差无梯度），
只能做启发式调制；PhasorDynamics 用 2D 相位向量 + 可微 Kuramoto 演化，
让相位成为端到端可学机制。

核心设计：
- 相位 = 2D 单位向量 p_i = (cosθ_i, sinθ_i)
  binding_i = mean_j (p_i·p_j) = mean_j cos(θ_i-θ_j)   （可微点积）
- Kuramoto 牵引 sin(θ_j-θ_i) = det([p_i, p_j])          （可微叉积）
- 双驱动相位动力学：
  · 前向物理：Kuramoto 演化（自然频率 ω + 耦合 K + 共激活调制），in-place 推进状态
  · 反向任务：phasors / ω / K 均为 nn.Parameter，loss 梯度直接调整相位
  → "同相/异相"由任务学出，而非先验同域同相

演化（Kuramoto ODE 离散化）：
  Δθ_i = ω_i·dt + (K/N)·Σ_j det([p_i,p_j])·c_ij
  p_i ← normalize( R(Δθ_i)·p_i )

与标量 GammaOscillator 的接口兼容（assign_phase_by_domain / tick /
kuramoto_step / gate_factor / batch_gate_factors / pairwise_binding），
额外提供可微版 binding_tensor（forward_train 用）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhasorDynamics(nn.Module):
    """可微相位动力学（C23-C）。

    Attributes:
        phasors: Parameter [N, 2]，2D 单位相位向量（任务梯度直接可调）
        omega: Parameter [N]，自然频率（可学：任务决定"谁该同相/异相"的频差）
        coupling_k: Parameter 标量，全局 Kuramoto 耦合强度
        binding_scale: 绑定强度 β（scores/场写入 × (1 + β·binding)）
    """

    def __init__(
        self,
        omega_init: float = math.pi / 4,
        min_gate: float = 0.2,
        max_gate: float = 1.0,
        binding_scale: float = 0.3,
        coupling_init: float = 0.05,
        dt: float = 0.2,
    ):
        """
        Args:
            omega_init: 自然频率初始值（每轮相位推进，π/4 ≈ 8 轮一圈）
            min_gate: 最小门控因子（兼容标量接口）
            max_gate: 最大门控因子
            binding_scale: 绑定强度 β（0 = 关闭绑定）
            coupling_init: Kuramoto 耦合强度初始值（可学）
            dt: Kuramoto 演化时间步长
        """
        super().__init__()
        self.differentiable = True
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.binding_scale = binding_scale
        self.dt = dt
        self.omega_init = omega_init
        self._id_to_idx: Dict[str, int] = {}
        self.register_parameter(
            "coupling_k", nn.Parameter(torch.tensor(float(coupling_init)))
        )
        # phasors/omega 在 register_neurons 时注册为 Parameter（None 不会被 nn.Module
        # 注册，避免空 buffer 与后续 Parameter 同名冲突）
        self.phasors: Optional[nn.Parameter] = None
        self.omega: Optional[nn.Parameter] = None
        self.global_phase: float = 0.0

    # ── 注册 / 相位分配（兼容标量接口）──

    def register_neurons(
        self,
        ids: List[str],
        phases: Optional[List[float]] = None,
    ) -> None:
        """一次性注册所有 neuron，构建 phasors/omega 参数。

        Args:
            ids: neuron ID 列表（顺序固定，之后所有调用按 ID 索引）
            phases: 初始相位（弧度）；None = 全部 0
        """
        n = len(ids)
        if n == 0:
            return
        self._id_to_idx = {nid: i for i, nid in enumerate(ids)}
        if phases is None:
            phases = [0.0] * n
        pv = torch.tensor(
            [[math.cos(ph), math.sin(ph)] for ph in phases], dtype=torch.float32
        )
        # 若已存在普通属性（None 占位或旧参数），先删除再注册 Parameter
        for _name in ("phasors", "omega"):
            if hasattr(self, _name):
                delattr(self, _name)
        self.register_parameter("phasors", nn.Parameter(pv))
        self.register_parameter(
            "omega", nn.Parameter(torch.full((n,), float(self.omega_init)))
        )

    def assign_phase(self, neuron_id: str, phase: float) -> None:
        """为单个 neuron 分配相位（兼容标量接口；首次调用需已 register 或自动追加）。

        若 phasors 尚未注册，累积到暂存；register_neurons 后统一构建。
        """
        if self.phasors is not None and self.phasors.numel() > 0:
            # 已注册：更新已有相位（单位向量重建）
            if neuron_id in self._id_to_idx:
                idx = self._id_to_idx[neuron_id]
                with torch.no_grad():
                    self.phasors[idx, 0] = math.cos(phase)
                    self.phasors[idx, 1] = math.sin(phase)
        else:
            # 未注册：暂存（register_neurons 时消费）
            if not hasattr(self, "_pending_phases"):
                self._pending_phases: Dict[str, float] = {}
            self._pending_phases[neuron_id] = phase

    def assign_phase_by_domain(
        self,
        domain_to_nids: Dict[str, list],
        phase_offset_per_domain: float = math.pi / 3,
    ) -> None:
        """按 domain 批量分配相位（兼容标量接口）：同 domain 同相、跨 domain 等距。

        收集全部 (nid, phase) 后统一 register_neurons。
        """
        ids: List[str] = []
        phases: List[float] = []
        pending = getattr(self, "_pending_phases", {})
        for i, (domain, nids) in enumerate(domain_to_nids.items()):
            base = i * phase_offset_per_domain
            for nid in nids:
                ids.append(nid)
                phases.append(pending.get(nid, base))
        self.register_neurons(ids, phases)

    # ── 可微绑定（forward_train 用）──

    def binding_tensor(
        self,
        active_ids: Optional[List[str]] = None,
        coactivation: Optional[Any] = None,
        phasors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """可微平均相位绑定：[N] 张量，梯度可达 phasors/omega/coupling_k。

        binding_i = mean_{j≠i} (p_i·p_j) × 共激活调制   ∈ [-1, 1]

        Args:
            active_ids: 本轮激活 neuron ID（None = 全部）
            coactivation: CoactivationTracker（pair 强度调制，与 Kuramoto 一致）
            phasors: 外部相位张量 [N,2]（顺序 = active_ids，可微演化输出用）；
                     None = 用 self.phasors

        Returns:
            [N] 张量（按 active_ids 顺序）
        """
        ids = active_ids if active_ids is not None else list(self._id_to_idx.keys())
        idxs = [self._id_to_idx[nid] for nid in ids if nid in self._id_to_idx]
        N = len(idxs)
        if N == 0 or (self.phasors is None and phasors is None):
            return torch.zeros(len(ids))
        p = phasors if phasors is not None else self.phasors[idxs]  # [N, 2]
        if p.shape[0] != N:
            # 外部相位顺序/数量不匹配 → 回退自身相位
            p = self.phasors[idxs]
        sim = p @ p.t()                 # [N, N] cos(θ_i-θ_j)，可微
        if N >= 2 and coactivation is not None:
            c = torch.ones(N, N, device=sim.device)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        c[i, j] = max(
                            coactivation.get_coactivation(ids[i], ids[j]), 0.01
                        )
            sim = sim * c
        b = (sim.sum(dim=1) - torch.diag(sim)) / max(N - 1, 1)  # [N] 平均绑定
        return b

    # ── Kuramoto 演化（可微；in-place 状态推进）──

    def evolve(
        self,
        active_ids: Optional[List[str]] = None,
        coactivation: Optional[Any] = None,
        dt: Optional[float] = None,
        coupling_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """可微 Kuramoto 演化：返回归一化后的新相位 [N,2]（梯度到 ω/K/phasors）。

        与 kuramoto_step 的区别：**不 in-place 更新状态**，演化计算全可微——
        ω/K 的梯度路径经此打通（任务 loss → binding → new_p → dtheta → ω/K）。
        状态推进由调用方负责（kuramoto_step 内部 detach 推进；forward_train
        用本方法的输出直接算绑定，梯度流完整）。

        Δθ_i = ω_i·dt + (K/N)·Σ_j det([p_i,p_j])·c_ij

        Args:
            active_ids: 本轮激活 neuron ID（None = 全部）
            coactivation: CoactivationTracker（pair 强度调制）
            dt: 时间步长（None = self.dt）
            coupling_strength: 覆盖耦合强度（None = self.coupling_k 参数）

        Returns:
            [N,2] 归一化新相位（顺序 = active_ids），可微
        """
        ids = active_ids if active_ids is not None else list(self._id_to_idx.keys())
        idxs = [self._id_to_idx[nid] for nid in ids if nid in self._id_to_idx]
        N = len(idxs)
        if N == 0 or self.phasors is None or self.phasors.numel() == 0:
            return torch.zeros(len(ids), 2)
        p = self.phasors[idxs]          # Parameter [N,2]，可微
        dets = torch.zeros(N, N, device=p.device)
        for i in range(N):
            pi = p[i]
            for j in range(N):
                if i == j:
                    continue
                d = pi[0] * p[j, 1] - pi[1] * p[j, 0]
                if coactivation is not None:
                    d = d * max(coactivation.get_coactivation(ids[i], ids[j]), 0.01)
                dets[i, j] = d
        K = self.coupling_k if coupling_strength is None else torch.tensor(
            float(coupling_strength), device=p.device
        )
        step = dt if dt is not None else self.dt
        dtheta = self.omega[idxs] * step + (K / N) * dets.sum(dim=1)  # [N] 可微
        cos_d, sin_d = torch.cos(dtheta), torch.sin(dtheta)
        new_x = p[:, 0] * cos_d - p[:, 1] * sin_d
        new_y = p[:, 0] * sin_d + p[:, 1] * cos_d
        new_p = F.normalize(torch.stack([new_x, new_y], dim=1), dim=-1)
        return new_p  # [N,2]

    def kuramoto_step(
        self,
        coupling_strength: Optional[float] = None,
        active_ids: Optional[List[str]] = None,
        coactivation: Optional[Any] = None,
        dt: Optional[float] = None,
    ) -> None:
        """可微 Kuramoto 相位耦合（状态推进，no_grad）。

        内部调用 evolve()（可微计算）+ detach 状态推进：
        状态是动力学轨迹（buffer 语义），不参与计算图；梯度经 forward_train 的
        evolve 输出（最后一轮）流向 ω/K。

        注：完全对齐/反相是绑定驻点（det=0 时耦合无牵引），物理正确。
        """
        ids = active_ids if active_ids is not None else list(self._id_to_idx.keys())
        idxs = [self._id_to_idx[nid] for nid in ids if nid in self._id_to_idx]
        N = len(idxs)
        if N < 2 or self.phasors is None or self.phasors.numel() == 0:
            return
        new_p = self.evolve(
            active_ids=ids, coactivation=coactivation,
            dt=dt, coupling_strength=coupling_strength,
        )
        with torch.no_grad():
            self.phasors[idxs] = new_p
        self.global_phase = (
            self.global_phase + float(self.omega[idxs].mean().item()) * (dt if dt is not None else self.dt)
        ) % (2 * math.pi)

    def tick(self, dt: float = 1.0) -> float:
        """推进全局相位（兼容标量接口）。"""
        self.global_phase = (self.global_phase + self.omega_init * dt) % (2 * math.pi)
        return self.global_phase

    def task_gradient_step(self, lr: float = 0.1) -> None:
        """任务梯度驱动相位演化（黎曼梯度下降：切向投影 + 单位归一）。

        相位是单位向量（流形约束），普通 SGD 的径向梯度分量会被归一化抹掉
        （完全对齐时梯度纯径向 → SGD 无效）。正确更新 = 只保留切向旋转：

            tangent = g − (g·p)·p        （去掉径向分量）
            p ← normalize(p − lr·tangent)

        与 Kuramoto 物理牵引并存 → 双驱动相位动力学：
        forward 内 Kuramoto 前向推进（物理），backward 后此方法（任务信号）。

        Args:
            lr: 学习率（相位演化步长）
        """
        if self.phasors is None or self.phasors.grad is None:
            return
        g = self.phasors.grad
        radial = (g * self.phasors).sum(dim=1, keepdim=True) * self.phasors
        tangent = g - radial
        with torch.no_grad():
            self.phasors.sub_(lr * tangent)
            self.phasors.data = (
                self.phasors.data
                / self.phasors.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            )

    # ── 兼容标量接口（门控 / dict binding）──

    def phase_of(self, neuron_id: str) -> float:
        idx = self._id_to_idx.get(neuron_id)
        if idx is None or self.phasors is None or self.phasors.numel() == 0:
            return 0.0
        return math.atan2(
            float(self.phasors[idx, 1].item()), float(self.phasors[idx, 0].item())
        )

    def coherence(self, neuron_id: str) -> float:
        return math.cos(self.phase_of(neuron_id) - self.global_phase)

    def gate_factor(self, neuron_id: str) -> float:
        c = self.coherence(neuron_id)
        return self.min_gate + (self.max_gate - self.min_gate) * (c + 1.0) / 2.0

    def batch_gate_factors(self, neuron_ids: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.gate_factor(nid) for nid in neuron_ids], dtype=torch.float32
        )

    def pairwise_binding(
        self,
        active_ids: Optional[List[str]] = None,
        coactivation: Optional[Any] = None,
    ) -> Dict[str, float]:
        """dict 版绑定（兼容 ensemble 推理标量路径；可微路径用 binding_tensor）。"""
        b = self.binding_tensor(active_ids, coactivation)
        ids = active_ids if active_ids is not None else list(self._id_to_idx.keys())
        return {nid: float(b[i].detach()) for i, nid in enumerate(ids) if i < len(b)}

    def list_phases(self) -> Dict[str, float]:
        return {nid: self.phase_of(nid) for nid in self._id_to_idx}

    @property
    def phases(self) -> Dict[str, float]:
        """兼容标量 GammaOscillator 的 `phases` dict（{nid: 弧度}）。

        apply_gamma_gate / cortex.set_gamma_oscillator / loader 日志都读取
        `osc.phases`（成员判断 `nid in osc.phases`）。PhasorDynamics 的相位
        状态在 phasors Parameter 里，这里动态推导为 dict 保持接口兼容。
        """
        return {nid: self.phase_of(nid) for nid in self._id_to_idx}

    def get_phase(self, neuron_id: str) -> Optional[float]:
        """兼容标量接口：返回相位弧度（None = 未注册）。"""
        if neuron_id not in self._id_to_idx:
            return None
        return self.phase_of(neuron_id)

    def reset(self) -> None:
        self.global_phase = 0.0
