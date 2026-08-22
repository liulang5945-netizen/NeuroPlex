"""Taiji 架构级实现 —— 全栈内化群体机制(替代 NeuroPlex 群体栈)

设计动机(详见 plans/active/TAIJI_ARCHITECTURE_DESIGN.md):
    旧架构(外挂式):
        ResonanceField ← ResonanceEnsemble ← ResonanceNeuron ← TransformerBlock
        群体协调(ensemble)、场(field)、相位(continuous)、E/I(topology)、
        生命周期(lifecycle/stdp/coaction)全部是外挂层,通过参数传递协调。

    Taiji 架构级(全栈内化):
        TaijiArchitecture
          └─ TaijiPopulationLayer × N  ← 每层 = 群体计算单元
               ├─ P × TaijiBlock (并行,yang+yin 双流,field-native)
               ├─ 层内场状态(field_state first-class)
               ├─ 相位绑定(_PhasorOscillator 可微 Kuramoto + theta-gamma 嵌套)
               ├─ E/I 路由(算子内原生,excite→yang / inhibit→yin)
               ├─ per-sample sparse routing(hard top-K + STE)
               ├─ 不应期调度(refractory counter)
               └─ STDP 局部学习(field 时序差,算子内原生)

    群体性的新定义:
        旧:独立 neuron 通过 ensemble 协调
        新:层内多神经元原生并行 + 跨层场传导
        "群体是能力单位"的精神保留,表达方式从外挂调度下沉到架构层原生。

四项扩展(本版本):
    1. 调质平面:接入 NeuromodulatorState(DA/5-HT/NE/ACh),驱动 temp_gain/ffn_gain
    2. 生命周期平面:STDP tracker + coaction tracker + maturity tracker
    3. Phasor 模块:_PhasorOscillator(Kuramoto 演化 + theta-gamma 嵌套)替代固定相位
    4. hard top-K routing:STE 让 top-K 选中可微,反向梯度穿透硬选择

接口契约:
    forward(input_ids) -> logits
    完全独立的架构,不依赖 ResonanceNeuron/Ensemble/Field。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.taiji import TaijiBlock
from neuroplex.layers import RMSNorm


# ─────────────────────────────────────────────────────────────────────
# 第 1 项:调质平面(NeuromodulatorState 内化版)
# ─────────────────────────────────────────────────────────────────────

class _NeuromodulatorState(nn.Module):
    """神经调质状态(吸收 neuro_modulation.py 的 NeuromodulatorState)。

    四种调质:
    - dopamine (DA): 奖励信号 → ffn_gain(高 DA→FFN 强化)
    - serotonin (5-HT): 满足感 → refractory_multiplier(高 5-HT→长不应期)
    - norepinephrine (NE): 警觉 → temp_gain(高 NE→attention 聚焦)
    - acetylcholine (ACh): 新颖性 → focus_gain(高 ACh→注意聚焦)

    调质是架构级状态,所有 TaijiPopulationLayer 共享。外部信号驱动目标值,
    EMA 缓慢趋近(避免突变)。
    """

    def __init__(
        self,
        dopamine: float = 0.5,
        serotonin: float = 0.5,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
        ema_alpha: float = 0.1,
    ):
        super().__init__()
        # 当前值(EMA 趋近目标)
        self.register_buffer("dopamine", torch.tensor(float(dopamine)))
        self.register_buffer("serotonin", torch.tensor(float(serotonin)))
        self.register_buffer("norepinephrine", torch.tensor(float(norepinephrine)))
        self.register_buffer("acetylcholine", torch.tensor(float(acetylcholine)))
        # 目标值(外部信号设定)
        self.register_buffer("_target_dopamine", torch.tensor(float(dopamine)))
        self.register_buffer("_target_serotonin", torch.tensor(float(serotonin)))
        self.register_buffer("_target_norepinephrine", torch.tensor(float(norepinephrine)))
        self.register_buffer("_target_acetylcholine", torch.tensor(float(acetylcholine)))
        self.ema_alpha = ema_alpha

    def set_targets(
        self,
        dopamine: Optional[float] = None,
        serotonin: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """设置目标调质水平(由外部信号驱动,如奖励/错误反馈)。"""
        if dopamine is not None:
            self._target_dopamine.fill_(max(0.0, min(1.0, dopamine)))
        if serotonin is not None:
            self._target_serotonin.fill_(max(0.0, min(1.0, serotonin)))
        if norepinephrine is not None:
            self._target_norepinephrine.fill_(max(0.0, min(1.0, norepinephrine)))
        if acetylcholine is not None:
            self._target_acetylcholine.fill_(max(0.0, min(1.0, acetylcholine)))

    def step(self) -> None:
        """EMA 趋近目标值(调质不会突变,而是缓慢调整)。"""
        self.dopamine += self.ema_alpha * (self._target_dopamine - self.dopamine)
        self.serotonin += self.ema_alpha * (self._target_serotonin - self.serotonin)
        self.norepinephrine += self.ema_alpha * (self._target_norepinephrine - self.norepinephrine)
        self.acetylcholine += self.ema_alpha * (self._target_acetylcholine - self.acetylcholine)

    def get_temp_gain(self) -> float:
        """注意力温度增益(NE 驱动) × focus_gain(ACh 驱动)。"""
        # NE: [0.5, 1.5],ACh: [0.6, 1.4],两者相乘组合调制
        ne_gain = 0.5 + float(self.norepinephrine) * 1.0
        ach_gain = 0.6 + float(self.acetylcholine) * 0.8
        return ne_gain * ach_gain

    def get_ffn_gain(self) -> float:
        """FFN 输出增益(DA 驱动)。高 DA→FFN 强化(奖励信号)。"""
        return 0.5 + float(self.dopamine)

    def get_refractory_multiplier(self) -> float:
        """不应期倍数(5-HT 驱动)。高 5-HT→长不应期(满足,不易再激活)。"""
        return 0.5 + float(self.serotonin) * 1.0

    def get_field_write_scale(self) -> float:
        """场写入强度(NE 驱动)。高 NE→强写入(警觉,强信号)。"""
        return 0.5 + float(self.norepinephrine) * 1.0


# ─────────────────────────────────────────────────────────────────────
# 第 2 项:生命周期平面(STDP / coaction / maturity trackers)
# ─────────────────────────────────────────────────────────────────────

class _STDPTracker(nn.Module):
    """STDP 局部学习跟踪器(吸收 stdp.py 的核心逻辑)。

    记录每个神经元的发放时序(field_vector),用于:
    1. 推理时:field 时序差驱动 yin gain 调制(已在 TaijiBlock 算子内实现)
    2. 睡眠巩固时:pair-wise STDP 强化/修剪跨神经元连接

    本跟踪器只记录发放历史,不直接更新权重。权重更新由 TaijiArchitecture
    的 sleep_consolidate() 方法在离线阶段执行。

    公式(吸收 STDPRule):
        LTP: Δw = η⁺ · exp(-Δt / τ⁺), Δt = t_post - t_pre > 0 (pre 先于 post)
        LTD: Δw = -η⁻ · exp(Δt / τ⁻), Δt < 0 (post 先于 pre)
        相似度门控:cos(field_vector_pre, field_vector_post) > threshold
    """

    def __init__(
        self,
        eta_plus: float = 0.01,
        eta_minus: float = 0.005,
        tau_plus: float = 2.0,
        tau_minus: float = 2.0,
        similarity_threshold: float = 0.0,
        history_maxlen: int = 8,
    ):
        super().__init__()
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.similarity_threshold = similarity_threshold
        # {neuron_idx: deque([(step, field_vector), ...])}
        self._firing_history: Dict[int, deque] = {}

    def record_firing(self, neuron_idx: int, step: int,
                      field_vector: torch.Tensor) -> None:
        """记录一次发放(field_vector 应为 detach 的场空间向量)。"""
        if neuron_idx not in self._firing_history:
            self._firing_history[neuron_idx] = deque(maxlen=8)
        self._firing_history[neuron_idx].append(
            (step, field_vector.detach().clone())
        )

    def clear(self) -> None:
        """清空发放历史(每次 forward 开始时调用)。"""
        self._firing_history.clear()

    def compute_pair_update(
        self,
        pre_idx: int, post_idx: int,
        pre_step: int, post_step: int,
        pre_vec: torch.Tensor, post_vec: torch.Tensor,
    ) -> float:
        """计算单次 pair 的 STDP 权重更新量。

        Returns:
            Δw > 0: LTP(pre 先于 post,方向一致→强化)
            Δw < 0: LTD(post 先于 pre,方向一致→减弱)
            Δw = 0: 相似度不足,门控跳过
        """
        sim = float(F.cosine_similarity(
            pre_vec.flatten().unsqueeze(0),
            post_vec.flatten().unsqueeze(0),
            dim=-1
        ).item())
        # 门控:相似度幅值不足则跳过(用 abs 让反向相似度也参与,Δw 符号由 sim 决定)
        if abs(sim) < self.similarity_threshold:
            return 0.0
        delta_t = post_step - pre_step
        if delta_t > 0:
            # LTP: pre 先于 post(sim>0 强化,sim<0 反向减弱)
            return self.eta_plus * math.exp(-delta_t / self.tau_plus) * sim
        elif delta_t < 0:
            # LTD: post 先于 pre(sim>0 减弱,sim<0 反向强化)
            return -self.eta_minus * math.exp(delta_t / self.tau_minus) * sim
        return 0.0


class _CoactivationTracker(nn.Module):
    """共激活跟踪器(吸收 topology.py 的 coaction 逻辑)。

    记录哪些神经元在同一 forward 中同时激活,用于:
    1. 相位动力学(Kuramoto 演化的耦合强度调制)
    2. 睡眠巩固(强化经常共激活的连接)

    实现:EMA 累积 pair-wise 共激活计数,slow EMA 反映长期趋势。
    """

    def __init__(self, num_neurons: int, ema_alpha: float = 0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.ema_alpha = ema_alpha
        # pair-wise 共激活强度矩阵(对称)
        self.register_buffer(
            "coactivation_matrix",
            torch.zeros(num_neurons, num_neurons),
        )
        # 本轮激活的神经元索引集合
        self._current_round_active: List[int] = []

    def update(self, active_indices: List[int]) -> None:
        """更新共激活矩阵(每轮 forward 结束调用)。"""
        self._current_round_active = list(active_indices)
        if len(active_indices) < 2:
            return
        # EMA 更新:活跃 pair += alpha,非活跃 pair *= (1-alpha)
        decay = 1.0 - self.ema_alpha
        self.coactivation_matrix *= decay
        for i in active_indices:
            for j in active_indices:
                if i != j:
                    self.coactivation_matrix[i, j] += self.ema_alpha

    def get_coactivation(self, idx_i: int, idx_j: int) -> float:
        """获取 pair 共激活强度(用于 Kuramoto 耦合调制)。"""
        return float(self.coactivation_matrix[idx_i, idx_j].item())

    def reset_round(self) -> None:
        """重置本轮激活记录(每轮 forward 开始调用)。"""
        self._current_round_active = []


class _MaturityTracker(nn.Module):
    """成熟度跟踪器(吸收 lifecycle.py 的 maturity 逻辑)。

    每个神经元有成熟度 ∈ [0, 1]:
    - 0.0: 幼稚态(新生),共振权重低(0.1),避免噪声主导
    - 1.0: 成熟态,共振权重 1.0,正常参与

    成熟度随激活次数累积增长(EMA),模拟"神经元在使用中成熟"。
    """

    def __init__(self, num_neurons: int, maturation_rate: float = 0.01,
                 initial_maturity: float = 0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.maturation_rate = maturation_rate
        self.register_buffer(
            "maturity",
            torch.full((num_neurons,), float(initial_maturity)),
        )

    def update(self, active_indices: List[int]) -> None:
        """更新成熟度(每轮 forward 结束调用)。"""
        for i in active_indices:
            # EMA 向 1.0 趋近(永远不达到 1.0,但渐进)
            self.maturity[i] += self.maturation_rate * (1.0 - self.maturity[i])

    def get_resonance_weight(self, neuron_idx: int) -> float:
        """获取共振权重(幼稚态低,成熟态 1.0)。"""
        # 幼稚态 0.1,成熟态 1.0(线性映射)
        return float(0.1 + 0.9 * self.maturity[neuron_idx].item())


# ─────────────────────────────────────────────────────────────────────
# 第 3 项:Phasor 模块(可微 Kuramoto + theta-gamma 嵌套)
# ─────────────────────────────────────────────────────────────────────

class _PhasorOscillator(nn.Module):
    """可微相位振荡器(吸收 phasor.py + continuous.py 的核心)。

    相位 = 2D 单位向量 p_i = (cosθ_i, sinθ_i),可微 Kuramoto 演化:
        Δθ_i = ω_i·dt + (K/N)·Σ_j sin(θ_j-θ_i)·c_ij
        p_i ← normalize(R(Δθ_i)·p_i)

    binding_i = mean_j (p_i·p_j) = mean_j cos(θ_i-θ_j)  ∈ [-1, 1]
    activation_i = σ(β·(binding_i - b0))                ∈ (0, 1)

    theta-gamma 嵌套(人脑启发):
    - theta: 慢振荡(4-8Hz),相位调制 gamma 振幅包络
    - gamma: 快振荡(30-100Hz),即神经元的本征相位
    - activation_modulated = activation × (1 + amp·cos(theta_phase))

    演化在每层 forward 时推进一次(dt = 1/num_layers,模拟跨层时间步进)。
    """

    def __init__(
        self,
        num_neurons: int,
        omega_init: float = math.pi / 4,
        coupling_init: float = 0.05,
        binding_scale: float = 0.3,
        act_temp: float = 4.0,
        act_offset: float = 0.0,
        # theta-gamma 嵌套参数
        theta_omega: float = 0.5,
        theta_amp: float = 0.2,
        theta_init: float = 0.0,
        dt: float = 0.2,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.binding_scale = binding_scale
        self.act_temp = act_temp
        self.act_offset = act_offset
        self.dt = dt

        # 可学习参数
        # phasors: [N, 2] 2D 单位相位向量(状态,通过 Kuramoto 演化推进)
        #   注:phasors 是动力学状态(buffer),不是梯度学习对象。
        #   学习信号通过 omega/coupling_k 流入(它们控制演化)。
        #   这样避免了 forward 中 in-place 更新 phasors 与 autograd 冲突。
        # 随机初始相位(打破均匀分布对称性,否则 binding_i 全相同→激活全同)
        init_phases = torch.rand(num_neurons) * 2 * math.pi
        self.register_buffer("phasors", torch.stack([
            torch.cos(init_phases), torch.sin(init_phases)
        ], dim=1))  # [N, 2]
        # 自然频率(每神经元的本征振荡频率,可学)
        self.omega = nn.Parameter(torch.full((num_neurons,), float(omega_init)))
        # Kuramoto 全局耦合强度(可学)
        self.coupling_k = nn.Parameter(torch.tensor(float(coupling_init)))

        # theta 慢振荡状态(全局,所有神经元共享)
        self.theta_omega = theta_omega
        self.theta_amp = theta_amp
        self.register_buffer("theta_phase", torch.tensor(float(theta_init)))
        self.register_buffer("_t", torch.zeros(1))  # 全局时间步

    def _evolve(self, coactivation: Optional[_CoactivationTracker] = None) -> torch.Tensor:
        """可微 Kuramoto 演化一步,返回新相位 [N, 2]。"""
        # clone buffer 以避免 in-place copy_ 破坏 autograd 保存的视图
        # (self.phasors 是 buffer,但 p[:,0]*cos_d 等运算会保存 p 的视图给 backward,
        #  之后 self.phasors.copy_ 会改动同一存储→version 冲突)
        p = self.phasors.clone()  # [N, 2]
        N = self.num_neurons
        # sin(θ_j - θ_i) = p_i[0]·p_j[1] - p_i[1]·p_j[0](可微叉积)
        # dets[i, j] = sin(θ_j - θ_i)
        dets = (
            p[:, 0:1] * p[:, 1:2].t() - p[:, 1:2] * p[:, 0:1].t()
        )  # [N, N]
        if coactivation is not None:
            c = coactivation.coactivation_matrix.detach()
            # 对角线置 0(自身不耦合)
            c = c - torch.diag(torch.diag(c))
            dets = dets * c
        # Δθ_i = ω_i·dt + (K/N)·Σ_j sin(θ_j-θ_i)·c_ij
        dtheta = self.omega * self.dt + (self.coupling_k / N) * dets.sum(dim=1)
        # 旋转:p_new = R(Δθ)·p
        cos_d = torch.cos(dtheta)
        sin_d = torch.sin(dtheta)
        new_x = p[:, 0] * cos_d - p[:, 1] * sin_d
        new_y = p[:, 0] * sin_d + p[:, 1] * cos_d
        new_p = F.normalize(torch.stack([new_x, new_y], dim=1), dim=-1)
        return new_p

    def _binding(self, p: torch.Tensor,
                 coactivation: Optional[_CoactivationTracker] = None) -> torch.Tensor:
        """计算 binding_i = mean_j cos(θ_i - θ_j) ∈ [-1, 1]。"""
        N = p.shape[0]
        if N < 2:
            return torch.ones(N, device=p.device)
        sim = p @ p.t()  # [N, N] cos(θ_i-θ_j)
        if coactivation is not None:
            c = coactivation.coactivation_matrix.detach()
            c = c - torch.diag(torch.diag(c))
            sim = sim * c
        # 排除自身的平均
        b = (sim.sum(dim=1) - torch.diag(sim)) / max(N - 1, 1)
        return b  # [N]

    def _theta_envelope(self) -> float:
        """theta 包络:1 + amp·cos(theta_phase)。无嵌套时恒 1.0。"""
        if self.theta_omega == 0.0:
            return 1.0
        return 1.0 + self.theta_amp * math.cos(float(self.theta_phase.item()))

    def forward(
        self,
        coactivation: Optional[_CoactivationTracker] = None,
        evolve: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向:计算相位、激活、theta 包络。

        Args:
            coactivation: 共激活跟踪器(调制 Kuramoto 耦合)
            evolve: 是否推进相位演化(True=训练/在线,False=只读)

        Returns:
            (phasors [N, 2], activations [N], theta_envelope scalar)
            activations ∈ (0, 1),已被 theta 包络调制

        注意:无论 evolve 取值,前向都会计算一次 Kuramoto 演化以保证
        omega/coupling_k 处于梯度图(可学)。evolve=False 时只不提交状态。
        """
        # 始终计算演化,保证 omega/coupling_k 在梯度图中
        new_p = self._evolve(coactivation)
        if evolve:
            # 提交相位状态(in-place,copy_ 不参与梯度,但 new_p 已参与下游计算)
            with torch.no_grad():
                self.phasors.copy_(new_p.detach())
            p = new_p
            # theta 相位推进
            if self.theta_omega != 0.0:
                self.theta_phase += self.theta_omega * self.dt
                self._t += self.dt
        else:
            # 只读模式:用演化后的相位计算 binding,但不提交状态
            p = new_p

        binding = self._binding(p, coactivation)  # [N]
        # 激活:σ(β·(binding - b0))
        activations = torch.sigmoid(
            self.act_temp * (binding - self.act_offset)
        )  # [N] ∈ (0, 1)
        # theta-gamma 嵌套:gamma 振幅 × theta 包络
        theta_env = self._theta_envelope()
        activations = activations * theta_env
        # 限幅到 (0, 1)
        activations = activations.clamp(1e-4, 1.0 - 1e-4)
        return p, activations, torch.tensor(theta_env)

    def get_phase_for_block(self, phasors: torch.Tensor, neuron_idx: int) -> float:
        """提取第 i 个神经元的相位角(弧度),供 TaijiBlock 的 yin 流旋转用。"""
        p_i = phasors[neuron_idx]
        return float(torch.atan2(p_i[1], p_i[0]).item())


# ─────────────────────────────────────────────────────────────────────
# 第 4 项:hard top-K routing + STE
# ─────────────────────────────────────────────────────────────────────

def _ste_topk(soft_weights: torch.Tensor, k: int) -> torch.Tensor:
    """Straight-Through Estimator top-K。

    前向:hard top-K(保留 top-K,其余置 0)
    反向:梯度穿透 soft_weights(STE)

    Args:
        soft_weights: [B, P] soft 权重(softmax over P)
        k: top-K 数量

    Returns:
        hard_weights: [B, P] hard top-K 权重(前向 hard,反向 soft)
    """
    if k <= 0 or k >= soft_weights.shape[-1]:
        return soft_weights
    # 前向:hard top-K
    topk_vals, topk_idx = soft_weights.topk(k, dim=-1)
    hard = torch.zeros_like(soft_weights)
    hard.scatter_(1, topk_idx, topk_vals)
    # 归一化 hard(行和=1)
    hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-6)
    # STE:前向 hard,反向 soft
    return hard.detach() + soft_weights - soft_weights.detach()


class _SparseRouter(nn.Module):
    """per-sample sparse router(吸收 ensemble.py 的 SparseRouter + STE top-K)。

    输入:P 个神经元的 pooled 表征 [B, P, hidden]
    输出:per-sample 路由权重 [B, P]
    - hard_top_k=0:soft routing(softmax over P)
    - hard_top_k>0:hard top-K + STE(前向 hard 选择,反向梯度穿透)
    """

    def __init__(self, hidden_size: int, num_neurons: int,
                 hard_top_k: int = 0, router_dropout: float = 0.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.hard_top_k = hard_top_k
        self.router_proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.router_proj.weight, std=hidden_size ** -0.5)
        self.dropout = nn.Dropout(router_dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, P, hidden] → weights: [B, P]。"""
        pooled = self.dropout(pooled)
        logits = self.router_proj(pooled).squeeze(-1)  # [B, P]
        soft_weights = F.softmax(logits, dim=-1)  # [B, P]
        if self.hard_top_k > 0 and self.hard_top_k < self.num_neurons:
            # hard top-K + STE
            return _ste_topk(soft_weights, self.hard_top_k)
        return soft_weights


# ─────────────────────────────────────────────────────────────────────
# TaijiPopulationLayer(吸收 ensemble + field + neuron 三层)
# ─────────────────────────────────────────────────────────────────────

class TaijiPopulationLayer(nn.Module):
    """Taiji 群体层 —— 架构级核心创新。

    每层内部并行执行:
    1. P 个 TaijiBlock 并行(yang+yin 双流,field-native)
    2. 层内场共振(field_write → field_state → field_read)
    3. 相位绑定(_PhasorOscillator 可微 Kuramoto + theta-gamma)
    4. E/I 路由(算子内原生,excite→yang / inhibit→yin)
    5. per-sample sparse routing(hard top-K + STE)
    6. 不应期调度(refractory counter,受 5-HT 调质调制)
    7. STDP 局部学习(field 时序差,算子内原生)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        field_dim: int,
        num_neurons: int = 4,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        # 相位绑定参数(_PhasorOscillator)
        omega_init: float = math.pi / 4,
        coupling_init: float = 0.05,
        act_temp: float = 4.0,
        theta_omega: float = 0.5,
        theta_amp: float = 0.2,
        # 路由参数(hard top-K + STE)
        hard_top_k: int = 0,
        # 不应期
        refractory_steps: int = 2,
        # STDP
        stdp_strength: float = 0.1,
        # 生命周期 trackers(由 TaijiArchitecture 注入,所有层共享)
        stdp_tracker: Optional[_STDPTracker] = None,
        coaction_tracker: Optional[_CoactivationTracker] = None,
        maturity_tracker: Optional[_MaturityTracker] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_dim = field_dim
        self.num_neurons = num_neurons
        self.refractory_steps = refractory_steps

        # ── P 个并行 TaijiBlock(群体) ──
        self.taiji_blocks = nn.ModuleList([
            TaijiBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
                field_dim=field_dim,
                refractory_steps=refractory_steps,
                stdp_strength=stdp_strength,
            )
            for _ in range(num_neurons)
        ])

        # ── 层内场状态(first-class) ──
        self.register_buffer("field_state", torch.zeros(1, field_dim), persistent=False)
        self.W_cond = nn.Parameter(torch.randn(field_dim, field_dim) * 0.02)

        # ── 相位振荡器(第 3 项) ──
        self.phasor = _PhasorOscillator(
            num_neurons=num_neurons,
            omega_init=omega_init,
            coupling_init=coupling_init,
            act_temp=act_temp,
            theta_omega=theta_omega,
            theta_amp=theta_amp,
        )

        # ── per-sample sparse router(第 4 项) ──
        self.router = _SparseRouter(
            hidden_size=hidden_size,
            num_neurons=num_neurons,
            hard_top_k=hard_top_k,
        )

        # ── 场写入投影 ──
        self.field_write_proj = nn.ModuleList([
            nn.Linear(hidden_size, field_dim, bias=False)
            for _ in range(num_neurons)
        ])

        # ── E/I 路由投影(把激活强度投影成 excite_signal [B, hidden]) ──
        self.excite_proj = nn.Linear(1, hidden_size, bias=False)

        # ── 融合门控 ──
        self.fusion_norm = RMSNorm(hidden_size, rms_norm_eps)

        # ── 生命周期 trackers(共享引用) ──
        self.stdp_tracker = stdp_tracker
        self.coaction_tracker = coaction_tracker
        self.maturity_tracker = maturity_tracker

        # ── 不应期状态(每神经元独立) ──
        self.register_buffer(
            "refractory_counters",
            torch.zeros(num_neurons, dtype=torch.long),
        )

    def reset_field(self, batch_size: int = 1) -> None:
        """重置层内场状态。"""
        dev = self.W_cond.device
        self.field_state = torch.zeros(batch_size, self.field_dim, device=dev)

    def enter_refractory(self, neuron_idx: int, multiplier: float = 1.0) -> None:
        """让指定神经元进入不应期。"""
        steps = max(1, int(self.refractory_steps * multiplier))
        self.refractory_counters[neuron_idx] = steps

    def tick_refractory(self) -> None:
        """不应期计数递减(每层 forward 结束调用)。"""
        self.refractory_counters = torch.clamp(
            self.refractory_counters - 1, min=0
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        cross_layer_field: Optional[torch.Tensor] = None,
        refractory_multiplier: float = 1.0,
        field_write_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Taiji 群体层前向。

        Args:
            x: [B, L, hidden]
            mask: causal mask [1,1,L,L]
            temp_gain: 注意力温度增益(神经调质 NE/ACh 驱动)
            ffn_gain: FFN 输出增益(神经调质 DA 驱动)
            cross_layer_field: [B, D] 上一层传导来的场状态
            refractory_multiplier: 不应期倍数(5-HT 驱动)
            field_write_scale: 场写入强度(NE 驱动)

        Returns:
            (fused_output [B, L, hidden], field_state [B, D])
        """
        bsz, seqlen, _ = x.shape

        # ── 场状态初始化 + 跨层场传导融合 ──
        self.reset_field(bsz)
        if cross_layer_field is not None:
            if cross_layer_field.shape[0] != bsz:
                cross_layer_field = cross_layer_field.expand(bsz, -1)
            field = self.field_state + cross_layer_field
        else:
            field = self.field_state
        field = F.normalize(field, dim=-1)

        # ── 相位绑定:计算 P 个神经元的激活强度 ──
        phasors, activations, theta_env = self.phasor(
            coactivation=self.coaction_tracker, evolve=self.training
        )
        # activations: [P] → 广播到 [B, P]
        activations_b = activations.unsqueeze(0).expand(bsz, -1)  # [B, P]

        # ── P 个 TaijiBlock 并行前向 ──
        block_outputs: List[torch.Tensor] = []
        active_indices: List[int] = []
        for i, block in enumerate(self.taiji_blocks):
            # 不应期检查
            in_refractory = bool(self.refractory_counters[i].item() > 0)
            if in_refractory:
                # 不应期内:yin 流关闭,只走 yang 流(算子内退化路径)
                # 仍然前向(保留 token 表征),但不参与场写入
                out, _, _ = block(
                    x, mask=mask, temp_gain=temp_gain, ffn_gain=ffn_gain,
                    field_state=None, phase=None,  # 退化:无场共振
                    excite_signal=None, inhibit_signal=None,
                )
                block_outputs.append(out)
                continue

            active_indices.append(i)
            phase_i = self.phasor.get_phase_for_block(phasors, i)
            # E/I 信号:激活强度作为 excite_signal(强激活→yang 放大)
            # excite_signal 期望 [B, hidden]
            act_i = activations[i].detach()
            if act_i.item() > 0.3:  # 激活阈值
                excite_sig = self.excite_proj(
                    act_i.view(1, 1).expand(bsz, 1)
                )  # [B, hidden]
            else:
                excite_sig = None

            out, _, _ = block(
                x, mask=mask, temp_gain=temp_gain, ffn_gain=ffn_gain,
                field_state=field, phase=phase_i,
                excite_signal=excite_sig, inhibit_signal=None,
            )
            block_outputs.append(out)

        # ── per-sample sparse routing ──
        pooled = torch.stack([
            out.mean(dim=1) for out in block_outputs
        ], dim=1)  # [B, P, hidden]
        router_weights = self.router(pooled)  # [B, P]

        # ── 场写入:P 个神经元 → 层内场 ──
        new_field = torch.zeros_like(field)
        field_vectors: List[torch.Tensor] = []
        for i, out in enumerate(block_outputs):
            if i not in active_indices:
                field_vectors.append(torch.zeros(bsz, self.field_dim, device=field.device))
                continue
            pooled_i = out.mean(dim=1)  # [B, hidden]
            vec = self.field_write_proj[i](pooled_i)  # [B, D]
            vec = F.normalize(vec, dim=-1)
            field_vectors.append(vec)
            # 写入权重 = 激活 × 路由权重 × 成熟度 × 场写入调质
            maturity_w = (
                self.maturity_tracker.get_resonance_weight(i)
                if self.maturity_tracker is not None else 1.0
            )
            write_weight = (
                activations_b[:, i] * router_weights[:, i] * maturity_w * field_write_scale
            )  # [B]
            new_field = new_field + write_weight.unsqueeze(-1) * vec

            # STDP 记录发放(供睡眠巩固,仅训练时)
            if self.stdp_tracker is not None and self.training:
                self.stdp_tracker.record_firing(
                    neuron_idx=i, step=0, field_vector=vec[0].detach()
                )

        new_field = F.normalize(new_field, dim=-1)
        self.field_state = new_field.detach()

        # 共激活更新(仅训练时,eval 时保持矩阵稳定以保证确定性推理)
        if self.coaction_tracker is not None and self.training:
            self.coaction_tracker.update(active_indices)

        # ── 融合输出:per-sample 加权 P 个神经元 ──
        fusion_weights = activations_b * router_weights  # [B, P]
        # 不应期内的神经元权重置 0
        for i in range(self.num_neurons):
            if i not in active_indices:
                fusion_weights[:, i] = 0.0
        fusion_weights = fusion_weights / (
            fusion_weights.sum(dim=-1, keepdim=True) + 1e-6
        )

        fused = torch.zeros_like(x)
        for i, out in enumerate(block_outputs):
            w = fusion_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            fused = fused + w * out

        fused = self.fusion_norm(fused)

        # 不应期计数递减
        self.tick_refractory()

        # 让 top-K 激活神经元进入不应期(信息轮替)
        # 只在训练时执行(推理时保持稳定)
        if self.training and active_indices:
            # 按激活强度排序,前半进入不应期
            sorted_active = sorted(
                active_indices,
                key=lambda i: float(activations[i].item()),
                reverse=True
            )
            k = max(1, len(sorted_active) // 2)
            for i in sorted_active[:k]:
                self.enter_refractory(i, multiplier=refractory_multiplier)

        # 成熟度更新(仅训练时)
        if self.maturity_tracker is not None and self.training:
            self.maturity_tracker.update(active_indices)

        return fused, new_field


# ─────────────────────────────────────────────────────────────────────
# TaijiArchitecture(完整架构)
# ─────────────────────────────────────────────────────────────────────

class TaijiArchitecture(nn.Module):
    """Taiji 架构级模型 —— 全栈内化群体机制。

    架构组成:
    1. SensoryPlane:embedding + position
    2. TaijiPopulationLayer × N:层叠群体计算单元
    3. CrossLayerField:跨层场传导
    4. NeuromodulatorPlane:调质平面(第 1 项)
    5. LifecyclePlane:生命周期平面(第 2 项)
    6. OutputPlane:lm_head
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        field_dim: int,
        num_neurons_per_layer: int = 4,
        max_seq_len: int = 2048,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        # 群体参数
        omega_init: float = math.pi / 4,
        coupling_init: float = 0.05,
        act_temp: float = 4.0,
        theta_omega: float = 0.5,
        theta_amp: float = 0.2,
        hard_top_k: int = 0,
        refractory_steps: int = 2,
        stdp_strength: float = 0.1,
        cross_layer_diffusion: float = 0.3,
        # 调质参数(第 1 项)
        neuromodulator: Optional[_NeuromodulatorState] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.field_dim = field_dim

        # ── SensoryPlane ──
        self.shared_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.input_norm = RMSNorm(hidden_size, rms_norm_eps)

        # ── 调质平面(第 1 项) ──
        self.neuromodulator = neuromodulator or _NeuromodulatorState()

        # ── 生命周期平面(第 2 项,所有层共享) ──
        self.stdp_tracker = _STDPTracker()
        self.coaction_tracker = _CoactivationTracker(
            num_neurons=num_neurons_per_layer
        )
        self.maturity_tracker = _MaturityTracker(
            num_neurons=num_neurons_per_layer
        )

        # ── TaijiPopulationLayer × N ──
        self.layers = nn.ModuleList([
            TaijiPopulationLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                field_dim=field_dim,
                num_neurons=num_neurons_per_layer,
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
                omega_init=omega_init,
                coupling_init=coupling_init,
                act_temp=act_temp,
                theta_omega=theta_omega,
                theta_amp=theta_amp,
                hard_top_k=hard_top_k,
                refractory_steps=refractory_steps,
                stdp_strength=stdp_strength,
                stdp_tracker=self.stdp_tracker,
                coaction_tracker=self.coaction_tracker,
                maturity_tracker=self.maturity_tracker,
            )
            for _ in range(num_layers)
        ])

        # ── 跨层场传导 ──
        self.cross_layer_diffusion = cross_layer_diffusion
        self.field_transfer = nn.Linear(field_dim, field_dim, bias=False)
        nn.init.eye_(self.field_transfer.weight)

        # ── OutputPlane ──
        self.output_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight  # tied

    def set_neuromodulator_targets(
        self,
        dopamine: Optional[float] = None,
        serotonin: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """设置调质目标值(由外部信号驱动,如奖励/错误反馈)。"""
        self.neuromodulator.set_targets(
            dopamine=dopamine,
            serotonin=serotonin,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

    def step_neuromodulator(self) -> None:
        """推进调质 EMA(每次 forward 前调用)。"""
        self.neuromodulator.step()

    def reset_lifecycle(self) -> None:
        """重置生命周期状态(每次 forward 开始调用)。"""
        self.stdp_tracker.clear()
        self.coaction_tracker.reset_round()
        # 重置所有层的不应期
        for layer in self.layers:
            layer.refractory_counters.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        return_field_history: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Taiji 架构前向。

        Args:
            input_ids: [B, L] token ids
            return_field_history: 若 True,返回每层场状态历史

        Returns:
            dict with logits, final_field, (可选 field_history)
        """
        bsz, seqlen = input_ids.shape

        # ── 调质 EMA 推进(第 1 项) ──
        self.step_neuromodulator()
        temp_gain = self.neuromodulator.get_temp_gain()
        ffn_gain = self.neuromodulator.get_ffn_gain()
        refractory_mult = self.neuromodulator.get_refractory_multiplier()
        field_write_scale = self.neuromodulator.get_field_write_scale()

        # ── 生命周期重置(第 2 项) ──
        self.reset_lifecycle()

        # ── SensoryPlane ──
        h = self.shared_embedding(input_ids)
        pos = torch.arange(seqlen, device=h.device)
        h = h + self.pos_embedding(pos).unsqueeze(0)
        h = self.input_norm(h)

        # Causal mask
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"),
                device=h.device, dtype=h.dtype,
            )
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = None

        # ── 跨层场传导 ──
        cross_field = None
        field_history: List[torch.Tensor] = []

        for layer in self.layers:
            h, field = layer(
                h, mask=mask,
                temp_gain=temp_gain, ffn_gain=ffn_gain,
                cross_layer_field=cross_field,
                refractory_multiplier=refractory_mult,
                field_write_scale=field_write_scale,
            )
            if return_field_history:
                field_history.append(field.detach())
            cross_field = self.cross_layer_diffusion * self.field_transfer(field)

        # ── OutputPlane ──
        h = self.output_norm(h)
        logits = self.lm_head(h)

        result = {
            "logits": logits,
            "final_field": field,
            "neuromodulator_state": {
                "dopamine": float(self.neuromodulator.dopamine),
                "serotonin": float(self.neuromodulator.serotonin),
                "norepinephrine": float(self.neuromodulator.norepinephrine),
                "acetylcholine": float(self.neuromodulator.acetylcholine),
            },
        }
        if return_field_history:
            result["field_history"] = field_history
        return result

    def sleep_consolidate(self, recent_fields: List[torch.Tensor]) -> Dict[str, float]:
        """睡眠巩固(离线阶段,第 2 项 STDP 应用)。

        Args:
            recent_fields: 近期高共振场状态列表(用于回放)

        Returns:
            统计信息(更新的 pair 数、平均 Δw 等)
        """
        # 应用 STDP:遍历发放历史中的 pair
        updates_applied = 0
        total_delta = 0.0
        history = self.stdp_tracker._firing_history
        for pre_idx, pre_records in history.items():
            for post_idx, post_records in history.items():
                if pre_idx == post_idx:
                    continue
                for pre_step, pre_vec in pre_records:
                    for post_step, post_vec in post_records:
                        delta = self.stdp_tracker.compute_pair_update(
                            pre_idx, post_idx, pre_step, post_step,
                            pre_vec, post_vec,
                        )
                        if abs(delta) > 1e-6:
                            # 这里 delta 只统计,实际权重更新需要 side_channels
                            # (Taiji 架构暂无显式 side_channels,STDP 信号
                            #  主要通过算子内的 stdp_strength 影响 yin gain)
                            updates_applied += 1
                            total_delta += delta
        return {
            "pairs_updated": updates_applied,
            "total_delta_w": total_delta,
            "mean_delta_w": total_delta / max(updates_applied, 1),
        }


__all__ = [
    "TaijiPopulationLayer",
    "TaijiArchitecture",
    "_NeuromodulatorState",
    "_PhasorOscillator",
    "_SparseRouter",
    "_STDPTracker",
    "_CoactivationTracker",
    "_MaturityTracker",
    "_ste_topk",
]
