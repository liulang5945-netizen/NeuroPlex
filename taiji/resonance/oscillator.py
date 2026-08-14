"""BioOSS 振荡型节点（o 型）— C27 增量三/四（2026-08-14）。

背景：态极 neuron 已有人脑启发分化（excitatory/inhibitory 亚型 + is_inhibitory），
但"抑制/兴奋"是单维标记。BioOSS p/o 双模型把角色分工正式化：
- **p（projection，投射型）**：内容 neuron（dialogue/general/域），承担场内容
  投影与 lm_head 生成——现有全部 neuron 都是 p 型。
- **o（oscillation，振荡型）**：节奏 neuron（本模块 OscillatorNode）——不承担
  内容生成（无 lm_head），只做节奏动力学：
  1. **相位推进**：θ(t+dt) = θ(t) + ω·dt（theta 慢 / gamma 快双层）
  2. **p 型相位牵引**：作为 Kuramoto 外部牵引力（sin(θ_osc-θ_i)·coupling 加入
     内容 neuron 的 dtheta）——"o 型驱动 p 型锁相"（人脑：抑制性中间神经元
     节律调控兴奋性投射神经元）
  3. **GABA 式节奏门控**：按振荡相位周期性 write_inhibit 共振场（半周期窗口
     衰减，方向 = gaba_vec）——时间门控而非内容污染。

C27 增量四（2026-08-14）：o 型从固定节奏源 → **可学习节奏控制器**：
- omega / coupling / gaba_amp 升级为 nn.Parameter（进训练梯度流）
- 新增可微相位路径（theta_tensor / phase_unit_tensor / gaba_gate_tensor）——
  forward_train continuous 分支用张量相位做牵引输入与门控，梯度经
  phase_loss（ω/coupling）+ osc_rhythm_loss（ω/gaba_amp）反传
- 推理路径（step/unit/gaba_gate）保持纯 float 状态推进，零回归

轻量合成节点：装配时动态创建（无需训练 ckpt），纯动力学 + 可学习节奏参数。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OscillatorNode(nn.Module):
    """BioOSS 振荡型节点（o 型）：相位节奏源 + GABA 式时间门控 + 可学习节奏参数。

    Args:
        nid: 节点 ID（如 "osc_theta_0" / "osc_gamma_0"）
        omega: 自然频率（rad/步；theta≈0.5 慢、gamma≈π/4 快）——可学习
        coupling: 对 p 型 neuron 的 Kuramoto 牵引强度——可学习
        gaba_amp: GABA 节奏门控幅度（write_inhibit 窗口深度，0 = 关闭门控）——可学习
        dim: 共振场维度（gaba_vec 门控方向，固定 buffer）
        phase: 初始相位（rad，float 状态）
    """

    def __init__(self, nid: str, omega: float, coupling: float,
                 gaba_amp: float, dim: int, phase: float = 0.0):
        super().__init__()
        self.nid = nid
        self.dim = int(dim)
        self._phase = float(phase)
        # C27 增量四：节奏参数可学习（训练梯度流；推理只读 .item() 零影响）
        self.register_parameter("omega", nn.Parameter(torch.tensor(float(omega))))
        self.register_parameter("coupling", nn.Parameter(torch.tensor(float(coupling))))
        self.register_parameter("gaba_amp", nn.Parameter(torch.tensor(float(gaba_amp))))
        # GABA 门控方向：随机归一化场向量（L2，维度偏好由初始化给定，固定 buffer）
        self.register_buffer("gaba_vec", F.normalize(torch.randn(dim), dim=-1))

    # ── float 状态（推理路径，零回归）──

    @property
    def phase(self) -> float:
        """当前振荡相位（rad）。推理推进；训练用 theta_tensor 可微路径。"""
        return self._phase

    def step(self, dt: float = 1.0) -> None:
        """推进振荡相位：θ += ω·dt（mod 2π）。ω 读取 .item()（状态推进无梯度）。"""
        self._phase = (self._phase + float(self.omega.item()) * float(dt)) % (2.0 * math.pi)

    def unit(self, device=None, dtype=None) -> torch.Tensor:
        """当前相位单位向量 [cos θ, sin θ]（Kuramoto 牵引输入，float 状态）。"""
        t = torch.tensor(
            [math.cos(self._phase), math.sin(self._phase)],
            dtype=dtype if dtype is not None else torch.float32,
            device=device,
        )
        return t

    def gaba_gate(self) -> float:
        """GABA 半周期窗口强度：max(0, cos θ) ∈ [0, 1]。

        相位在峰值窗口内施加抑制，另一半周期无门控（人脑 GABA 中间神经元
        在特定 θ 相位抑制投射神经元）。
        """
        return max(0.0, math.cos(self._phase))

    def reset(self) -> None:
        """相位复位（单次 forward 开始时可调用，保证节奏起点一致）。"""
        self._phase = 0.0

    # ── 可微路径（C27 增量四：训练侧节奏控制器）──
    # forward_train continuous 分支用这些 API 构建张量相位：
    # θ(t) = θ0 + ω·t（θ0 从 float 状态读，ω Parameter 可微）——
    # 梯度经牵引（→new_p→bvec→phase_loss）与门控（→field_state→scores）
    # 与 osc_rhythm_loss（w=gaba_amp·gate 对齐锁相度）反传。

    def theta_tensor(self, t: float, device=None, dtype=None) -> torch.Tensor:
        """可微相位 θ(t) = θ0 + ω·t（ω 可微，θ0 状态 detach）。"""
        theta0 = torch.tensor(
            self._phase, dtype=dtype if dtype is not None else torch.float32,
            device=device,
        )
        return theta0 + self.omega * float(t)

    def phase_unit_tensor(self, t: float, device=None, dtype=None) -> torch.Tensor:
        """可微相位单位向量 [cos θ, sin θ]（Kuramoto 牵引输入）。"""
        theta = self.theta_tensor(t, device, dtype)
        return torch.stack([torch.cos(theta), torch.sin(theta)])

    def gaba_gate_tensor(self, t: float, device=None, dtype=None) -> torch.Tensor:
        """可微 GABA 门控：max(0, cos θ(t))（θ 经 ω 可微）。"""
        return torch.clamp(torch.cos(self.theta_tensor(t, device, dtype)), min=0.0)


def make_default_oscillators(dim: int) -> List[OscillatorNode]:
    """装配默认双层振荡节点：theta 慢（节奏窗）+ gamma 快（同频锁相）。

    Args:
        dim: 共振场维度（gaba_vec 方向）

    Returns:
        [OscillatorNode(theta), OscillatorNode(gamma)]
    """
    return [
        OscillatorNode(nid="osc_theta_0", omega=0.5, coupling=0.4,
                       gaba_amp=0.08, dim=dim),
        OscillatorNode(nid="osc_gamma_0", omega=math.pi / 4, coupling=0.3,
                       gaba_amp=0.04, dim=dim),
    ]
