"""ContinuousResonance — C25-E 连续时间共振（相位同步驱动的连续动力学）。

替代离散共振轮次（round1 全量 + 不应期硬门 + max_rounds 限制）：
- 时间步进 T 微步（dt），每步相位 Kuramoto 演化（复用 PhasorDynamics.evolve）
- 激活强度 a_i(t) = σ(β·(binding_i(t) - b0)) 连续驱动"谁参与、权重多少"——
  与当前共振主体同相 → 强激活持续参与；异相 → 弱激活退场。
  **替代不应期硬门的信息轮替**：轮替由相位关系连续决定，无硬跳变。
- 场随积分单调演化：F(t+dt) = F(t) + dt·Σ a_i(t)·project(v_i(t))
- 融合权重 w_i = Σ_t dt·a_i(t)·confidence_i（时间平均激活）
- 收敛：相位绑定分布稳定（锁定）或 T 步满

安全性边界（C23 同款）：executive 判定（judge NLL 主信号，C20v2）不消费
本模块输出——连续共振只作用于协作融合路径；t=0 独立前向采集判定信号，
与离散 round1 语义一致。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousResonance(nn.Module):
    """连续时间共振核心（纯数学：激活 / 权重累积 / 收敛判据）。

    ensemble.continuous_forward 负责编排（forward neuron、写场、logits），
    本模块提供连续动力学的可测单元。
    """

    def __init__(
        self,
        steps: int = 8,
        dt: float = 1 / 8,
        act_temp: float = 1.5,
        act_offset: float = 0.0,
        min_activ: float = 0.05,
        conv_tol: float = 0.02,
        min_steps: int = 2,
        # 缺口 R：多频段振荡——theta-gamma 嵌套（人脑：慢 theta 相位调制 gamma 振幅包络）
        theta_omega: float = 0.0,   # theta 频率（rad/单位时间；0 = 不启用，向后兼容）
        theta_amp: float = 0.2,     # 包络调制幅度（theta_omega=0 时无效果）
        theta_init: float = 0.0,    # 初始 theta 相位
    ):
        """
        Args:
            steps: 最大时间步数 T（替代 max_rounds 的连续版）
            dt: 时间步长（积分精度；Σdt = 1）
            act_temp: 激活温度 β——binding 差放大倍数（越高越接近硬门，越低越平滑）
            act_offset: 激活偏置 b0（binding=b0 时 a=0.5）
            min_activ: 参与门槛（activ < 该值的 neuron 本步退场，软过滤替代硬不应期）
            conv_tol: 收敛容差（绑定 std 相邻步变化 < 该值视为锁定）
            min_steps: 收敛检查最早步数（防止相位演化尚未推进时的单步假收敛）
            theta_omega: theta 慢振荡频率（0 = 不启用嵌套，与旧行为完全一致）
            theta_amp: theta 包络调制幅度（gamma 激活振幅被 theta 相位调制）
            theta_init: theta 初始相位
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.act_temp = act_temp
        self.act_offset = act_offset
        self.min_activ = min_activ
        self.conv_tol = conv_tol
        self.min_steps = min_steps
        # R12（REMEDIATION_PLAN 2026-08-14）：显式实验开关——theta_omega>0 启用
        # 嵌套，生产默认关闭；环境变量 TAIJI_THETA_NESTING=1 免改码 A/B
        # （theta 频率 0.5 rad/单位时间，幅度 0.2，verify_c26_theta_gamma 标定值）。
        # 无 env 时行为与旧版完全一致（theta_omega=0 → 包络恒 1、调制恒等）。
        if theta_omega == 0.0 and os.environ.get("TAIJI_THETA_NESTING", "0") not in ("", "0", "false", "False"):
            theta_omega = 0.5
            theta_amp = theta_amp if theta_amp != 0.0 else 0.2
        self.theta_omega = theta_omega
        self.theta_amp = theta_amp
        self.theta_init = theta_init
        # C26 增量五（2026-08-14）：记忆驱动的跨频耦合——记忆检索对齐 theta
        # 峰值相位（记忆注意窗），无记忆时按嵌套开关行为（theta_omega=0 → 恒等）。
        self._memory_entrained = False
        # C27 增量二（2026-08-14）：相位归属记忆（KoPE）——记忆注入时对齐
        # 其沉淀相位（不同记忆不同相位唤醒），默认 0 = 峰值对齐（增量五零回归）。
        self._entrain_phase = 0.0

    # ── 激活（连续替代不应期硬门）──

    def activation(self, binding: torch.Tensor) -> torch.Tensor:
        """激活强度：σ(act_temp·(binding - act_offset)) ∈ (0, 1)。

        绑定 = cos 差 ∈ [-1,1]；同相群体 binding→+1 → a→1（强参与）；
        异相 → -1 → a→0（退场）。连续、可微、无硬跳变。
        """
        return torch.sigmoid(self.act_temp * (binding - self.act_offset))

    def activations_from_phasors(
        self,
        phasor,
        ids: List[str],
        coactivation=None,
        phasors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """由相位直接算激活（一步到位，供主循环用）。"""
        binding = phasor.binding_tensor(ids, coactivation, phasors=phasors)
        return self.activation(binding)

    # ── 权重累积（时间平均激活）──

    def weights_accum(
        self,
        weights: torch.Tensor,
        activ: torch.Tensor,
        confidence: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """w += dt·a_i·conf_i（场/融合权重的时间积分）。"""
        return weights + dt * activ * confidence

    # ── 收敛判据（相位绑定锁定）──

    def converged(
        self,
        binding_history: List[torch.Tensor],
        tol: Optional[float] = None,
    ) -> bool:
        """绑定分布稳定 = 锁定（方差相邻步变化 < tol）。

        相位锁定 → 绑定不变 → 激活不变 → 场/权重稳态——可提前停止
        （连续时间版的"自适应停止"）。
        """
        tol = tol if tol is not None else self.conv_tol
        if len(binding_history) < 2:
            return False
        b_prev, b_cur = binding_history[-2], binding_history[-1]
        if b_prev.numel() != b_cur.numel() or b_prev.numel() == 0:
            return False
        delta = (b_cur.std() - b_prev.std()).abs()
        return bool(delta.item() < tol)

    # ── 锁定度（诊断）──

    def lock_degree(self, binding: torch.Tensor) -> float:
        """当前绑定锁定度：均值高 + 方差低 = 强锁定（共识强）。"""
        if binding.numel() == 0:
            return 0.0
        m = float(binding.mean().item())
        s = float(binding.std().item())
        # 全同相 = 1，全异相 = -1；锁定 = |绑定均值| 高
        return m

    # ── 缺口 R：多频段振荡——theta-gamma 嵌套（2026-08-11）──
    # 人脑机制：慢 theta 振荡（4-8Hz）相位调制快 gamma 振荡（30-100Hz）振幅包络
    # （Lisman theta-gamma 嵌套编码）——theta 相位决定 gamma 活动窗口（长距绑定/刷新）。
    # 默认 theta_omega=0（不启用），与 C25-E 旧行为逐元素一致（零回归）。

    def theta_phase_at(self, t: float) -> float:
        """t 时刻的 theta 慢振荡相位（rad）。

        记忆 entrain 后相位对齐目标相位（默认 0 → 包络峰值）：记忆注意窗
        期间 gamma 绑定持续增强（跨频耦合：记忆这一慢变量驱动 theta 相位）。
        C27 增量二（KoPE）：对齐目标 = 记忆沉淀时的加权均值相角——相位归属
        记忆（不同记忆不同相位唤醒，θ 相位序列编码）。
        """
        if self._memory_entrained:
            return self._entrain_phase
        return self.theta_init + self.theta_omega * t

    def theta_envelope(self, t: float) -> float:
        """theta 包络：1 + theta_amp·cos(theta_phase(t)) ∈ [1-A, 1+A]。

        - 无嵌套（theta_omega=0 且无记忆）：恒 1.0（零回归）
        - 记忆 entrain：峰值 1+amp（记忆注意窗）
        - 显式嵌套：按相位振荡
        """
        if self.theta_omega == 0 and not self._memory_entrained:
            return 1.0
        return 1.0 + self.theta_amp * math.cos(self.theta_phase_at(t))

    def theta_modulate(self, activ: torch.Tensor, t: float) -> torch.Tensor:
        """theta-gamma 嵌套：gamma 激活振幅 × theta 慢振荡包络（调幅）。"""
        return activ * self.theta_envelope(t)

    def entrain_memory(self, target_phase: float = 0.0) -> None:
        """记忆检索 → theta 相位对齐目标相位（跨频耦合 + 相位归属）。

        C26 增量五：记忆注入生成时调用，theta 相位对齐包络峰值，gamma 绑定
        在记忆条件化期间增强——"记忆注意窗"（相关回路同步激活）。
        C27 增量二（KoPE）：target_phase 为记忆沉淀时的相位——相位归属记忆，
        不同记忆在不同相位被唤醒（人脑 θ 相位序列编码记忆）。默认 0.0 =
        峰值对齐（增量五行为，零回归）。无记忆的 forward 不受影响。
        """
        self._memory_entrained = True
        self._entrain_phase = float(target_phase)

    def reset_entrain(self) -> None:
        """单次 forward 结束后清除记忆 entrain 状态（下一次 forward 干净）。"""
        self._memory_entrained = False
        self._entrain_phase = 0.0

    # ── 一步相位演化 + 激活（主循环原语）──

    def step(
        self,
        phasor,
        ids: List[str],
        coactivation=None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推进相位（可微 Kuramoto）+ 返回本步激活。

        Returns:
            (new_phasors [N,2], activ [N])
        """
        new_p = phasor.evolve(ids, coactivation, dt=dt)
        binding = phasor.binding_tensor(ids, coactivation, phasors=new_p)
        return new_p, self.activation(binding)
