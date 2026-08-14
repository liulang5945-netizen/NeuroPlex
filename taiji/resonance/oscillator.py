"""BioOSS 振荡型节点（o 型）— C27 增量三（2026-08-14）。

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

轻量合成节点：装配时动态创建（无需训练 ckpt），纯动力学。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


class OscillatorNode:
    """BioOSS 振荡型节点（o 型）：相位节奏源 + GABA 式时间门控。

    Args:
        nid: 节点 ID（如 "osc_theta_0" / "osc_gamma_0"）
        omega: 自然频率（rad/步；theta≈0.5 慢、gamma≈π/4 快）
        coupling: 对 p 型 neuron 的 Kuramoto 牵引强度
        gaba_amp: GABA 节奏门控幅度（write_inhibit 窗口深度，0 = 关闭门控）
        dim: 共振场维度（gaba_vec 门控方向）
        phase: 初始相位（rad）
    """

    def __init__(self, nid: str, omega: float, coupling: float,
                 gaba_amp: float, dim: int, phase: float = 0.0):
        self.nid = nid
        self.omega = float(omega)
        self.coupling = float(coupling)
        self.gaba_amp = float(gaba_amp)
        self.phase = float(phase)
        # GABA 门控方向：随机归一化场向量（L2，维度偏好由初始化给定）
        self.gaba_vec = F.normalize(torch.randn(dim), dim=-1)

    def step(self, dt: float = 1.0) -> None:
        """推进振荡相位：θ += ω·dt（mod 2π）。"""
        self.phase = (self.phase + self.omega * float(dt)) % (2.0 * math.pi)

    def unit(self, device=None, dtype=None) -> torch.Tensor:
        """当前相位单位向量 [cos θ, sin θ]（Kuramoto 牵引输入）。"""
        t = torch.tensor(
            [math.cos(self.phase), math.sin(self.phase)],
            dtype=dtype if dtype is not None else torch.float32,
            device=device,
        )
        return t

    def gaba_gate(self) -> float:
        """GABA 半周期窗口强度：max(0, cos θ) ∈ [0, 1]。

        相位在峰值窗口内施加抑制，另一半周期无门控（人脑 GABA 中间神经元
        在特定 θ 相位抑制投射神经元）。
        """
        return max(0.0, math.cos(self.phase))

    def reset(self) -> None:
        """相位复位（单次 forward 开始时可调用，保证节奏起点一致）。"""
        self.phase = 0.0


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
