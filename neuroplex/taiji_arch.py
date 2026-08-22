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
               ├─ 相位绑定(continuous_resonance 原生)
               ├─ E/I 路由(算子内原生,excite→yang / inhibit→yin)
               ├─ per-sample sparse routing
               ├─ 不应期调度(refractory counter)
               └─ STDP 局部学习(field 时序差,算子内原生)

    群体性的新定义:
        旧:独立 neuron 通过 ensemble 协调
        新:层内多神经元原生并行 + 跨层场传导
        "群体是能力单位"的精神保留,表达方式从外挂调度下沉到架构层原生。

最小原型(本文件):
    - TaijiPopulationLayer:单层群体计算单元(吸收 ensemble+field+neuron 三层)
    - TaijiArchitecture:层叠 + 跨层场传导 + 输出
    - 简化项:phase 用线性递增、sparse routing 用 soft routing、
      调质用 temp_gain/ffn_gain 参数、生命周期预留接口

接口契约:
    forward(input_ids) -> logits
    完全独立的架构,不依赖 ResonanceNeuron/Ensemble/Field。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.taiji import TaijiBlock
from neuroplex.layers import RMSNorm


class _PhaseBinder(nn.Module):
    """相位绑定器(简化版,吸收 continuous.py 的核心逻辑)。

    计算 P 个神经元的激活强度(连续替代离散不应期硬门):
        binding_i = cos(phasor_i - mean_phasor)   ∈ [-1, 1]
        a_i = σ(β · (binding_i - b0))             ∈ (0, 1)

    同相群体 binding→+1 → a→1(强参与);异相 → -1 → a→0(退场)。
    连续、可微、无硬跳变。

    简化:本原型用线性递增相位(phasor_i = i · Δφ),后续可换 Phasor 模块。
    """

    def __init__(self, num_neurons: int, phase_delta: float = 0.5,
                 act_temp: float = 4.0, act_offset: float = 0.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.phase_delta = phase_delta  # 相邻神经元相位差
        self.act_temp = act_temp
        self.act_offset = act_offset
        # 每个神经元的固定相位(后续可换成可学习或动态 Phasor)
        phases = torch.arange(num_neurons).float() * phase_delta
        self.register_buffer("phases", phases)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (phases [P], activations [P])。"""
        # 相位绑定:每个神经元与群体平均相位的 cos 距离
        mean_phase = self.phases.mean()
        binding = torch.cos(self.phases - mean_phase)  # [P] ∈ [-1, 1]
        activations = torch.sigmoid(
            self.act_temp * (binding - self.act_offset)
        )  # [P] ∈ (0, 1)
        return self.phases, activations


class _SparseRouter(nn.Module):
    """per-sample soft routing(简化版,吸收 ensemble.py 的 SparseRouter 核心)。

    每个样本独立选择 top-K 神经元,但用 soft 权重(可微)。
    本原型用基于 round1 响应的 softmax 路由,后续可换 hard top-K + STE。

    输入:P 个神经元的 pooled 表征 [B, P, hidden]
    输出:per-sample 路由权重 [B, P](softmax over P)
    """

    def __init__(self, hidden_size: int, num_neurons: int, router_top_k: int = 0):
        super().__init__()
        self.num_neurons = num_neurons
        self.router_top_k = router_top_k  # 0 = soft all, >0 = top-K soft
        self.router_proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.router_proj.weight, std=hidden_size ** -0.5)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, P, hidden] → weights: [B, P] (softmax over P)。

        per-sample 独立路由:每个样本根据自己的响应模式选神经元。
        """
        logits = self.router_proj(pooled).squeeze(-1)  # [B, P]
        if self.router_top_k > 0 and self.router_top_k < self.num_neurons:
            # soft top-K:保留 top-K 的 logits,其余设 -inf
            topk_vals, topk_idx = logits.topk(self.router_top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            logits = mask
        return F.softmax(logits, dim=-1)  # [B, P]


class TaijiPopulationLayer(nn.Module):
    """Taiji 群体层 —— 架构级核心创新(吸收 ensemble+field+neuron 三层职责)。

    每层内部并行执行:
    1. P 个 TaijiBlock 并行(yang+yin 双流,field-native)
    2. 层内场共振(field_write → field_state → field_read)
    3. 相位绑定(连续激活强度,替代离散不应期)
    4. E/I 路由(算子内原生,excite→yang / inhibit→yin)
    5. per-sample sparse routing
    6. 不应期调度(TaijiBlock 自带 refractory_counter)
    7. STDP 局部学习(field 时序差,算子内原生)

    群体性来源:
    - 层内场共享(所有 P 个 TaijiBlock 读写同一 field_state)
    - 相位绑定(群体平均相位驱动激活)
    - per-sample routing(每个样本独立选神经元子集)
    - 跨层场传导(由 TaijiArchitecture 的 spatial_diffuser 处理)
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
        # 相位绑定参数
        phase_delta: float = 0.5,
        act_temp: float = 4.0,
        # 路由参数
        router_top_k: int = 0,
        # 不应期
        refractory_steps: int = 2,
        # STDP
        stdp_strength: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_dim = field_dim
        self.num_neurons = num_neurons

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

        # ── 层内场状态(first-class,吸收 ResonanceField) ──
        # [B, D] 每样本独立场(无跨样本污染)
        self.register_buffer("field_state", torch.zeros(field_dim), persistent=False)
        # W_cond:场条件化门(吸收 field.py 的 W_cond)
        self.W_cond = nn.Parameter(torch.randn(field_dim, field_dim) * 0.02)

        # ── 相位绑定器(吸收 continuous.py) ──
        self.phase_binder = _PhaseBinder(
            num_neurons=num_neurons,
            phase_delta=phase_delta,
            act_temp=act_temp,
        )

        # ── per-sample sparse router(吸收 ensemble.py SparseRouter) ──
        self.router = _SparseRouter(
            hidden_size=hidden_size,
            num_neurons=num_neurons,
            router_top_k=router_top_k,
        )

        # ── 场写入投影(吸收 neuron.py field_write) ──
        # 每个 TaijiBlock 独立的 field_write 投影
        self.field_write_proj = nn.ModuleList([
            nn.Linear(hidden_size, field_dim, bias=False)
            for _ in range(num_neurons)
        ])

        # ── 融合门控(per-sample 加权融合 P 个神经元输出) ──
        self.fusion_norm = RMSNorm(hidden_size, rms_norm_eps)

        # ── 调质接口(预留,由 TaijiArchitecture 注入) ──
        # temp_gain/ffn_gain 由架构级 NeuromodulatorState 计算,层共享

    def reset_field(self, batch_size: int = 1) -> None:
        """重置层内场状态(每次 forward 前调用)。

        总是创建 [B, D] 形状(即使 B=1),避免广播歧义。
        """
        dev = self.W_cond.device
        self.field_state = torch.zeros(batch_size, self.field_dim, device=dev)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        cross_layer_field: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Taiji 群体层前向。

        Args:
            x: [B, L, hidden]
            mask: causal mask [1,1,L,L]
            temp_gain: 注意力温度增益(神经调质驱动)
            ffn_gain: FFN 输出增益(神经调质驱动)
            cross_layer_field: [B, D] 上一层传导来的场状态(跨层场传导)
                None 时只用层内场;非 None 时与层内场融合

        Returns:
            (fused_output [B, L, hidden], field_state [B, D])
            field_state 供下一层跨层传导使用
        """
        bsz, seqlen, _ = x.shape

        # ── 场状态初始化 + 跨层场传导融合 ──
        # 每次 forward 重置层内场(跨 forward 独立,避免样本间污染)
        # 层内场累积只在本次 forward 内发生(层叠结构中场是层内状态)
        self.reset_field(bsz)
        if cross_layer_field is not None:
            # 跨层场传导:上一层场 + 本层场(简化为相加后归一化)
            if cross_layer_field.shape[0] != bsz:
                cross_layer_field = cross_layer_field.expand(bsz, -1)
            field = self.field_state + cross_layer_field
        else:
            field = self.field_state
        field = F.normalize(field, dim=-1)  # L2 归一化(吸收 field.py 的归一化)

        # ── 相位绑定:计算 P 个神经元的激活强度 ──
        phases, activations = self.phase_binder()  # [P], [P]
        # activations: [P] → 广播到 [B, P]
        activations_b = activations.unsqueeze(0).expand(bsz, -1)  # [B, P]

        # ── P 个 TaijiBlock 并行前向(yang+yin 双流) ──
        block_outputs: List[torch.Tensor] = []
        for i, block in enumerate(self.taiji_blocks):
            phase_i = float(phases[i].item())
            # E/I 信号:简化原型用 activations 作为 excite_signal
            # (强激活→yang 放大),inhibit_signal 暂不传(算子内中性)
            excite_sig = activations[i].unsqueeze(0).expand(bsz, -1) \
                if activations[i].item() > 0.5 else None
            # 注意:excite_signal 期望 [B, hidden],这里用激活值广播
            # (简化,后续可换专门的 E/I 路由器)
            if excite_sig is not None:
                excite_sig = excite_sig.unsqueeze(-1).expand(-1, -1, self.hidden_size)
                # 取第一个样本的 hidden 维度广播(原型简化)
                excite_sig = excite_sig[:, 0, :]  # [B, hidden]

            out, _, _ = block(
                x, mask=mask, temp_gain=temp_gain, ffn_gain=ffn_gain,
                field_state=field, phase=phase_i,
                excite_signal=excite_sig, inhibit_signal=None,
            )
            block_outputs.append(out)  # [B, L, hidden]

        # ── per-sample sparse routing ──
        # 用每个 block 的 pooled 表征做路由
        pooled = torch.stack([
            out.mean(dim=1)  # [B, hidden]
            for out in block_outputs
        ], dim=1)  # [B, P, hidden]
        router_weights = self.router(pooled)  # [B, P]

        # ── 场写入:P 个神经元 → 层内场(吸收 ensemble 的 field.write) ──
        # 每 neuron 写入自己的 field_vector,L2 归一化后按激活×路由权重累加
        new_field = torch.zeros_like(field)
        for i, out in enumerate(block_outputs):
            # field_vector:pool 序列 → 投影到 field_dim
            pooled_i = out.mean(dim=1)  # [B, hidden]
            vec = self.field_write_proj[i](pooled_i)  # [B, D]
            vec = F.normalize(vec, dim=-1)  # L2 归一化(场写入契约)
            # 写入权重 = 激活 × 路由权重(每样本独立)
            write_weight = activations_b[:, i] * router_weights[:, i]  # [B]
            new_field = new_field + write_weight.unsqueeze(-1) * vec
        # 场更新:L2 归一化(防止单一 neuron 方向主导 magnitude)
        new_field = F.normalize(new_field, dim=-1)
        self.field_state = new_field.detach()  # 更新层内场状态(供下轮/跨层用)

        # ── 融合输出:per-sample 加权 P 个神经元 ──
        # 权重 = 激活 × 路由权重,归一化
        fusion_weights = activations_b * router_weights  # [B, P]
        fusion_weights = fusion_weights / (
            fusion_weights.sum(dim=-1, keepdim=True) + 1e-6
        )  # [B, P]

        fused = torch.zeros_like(x)
        for i, out in enumerate(block_outputs):
            w = fusion_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            fused = fused + w * out

        fused = self.fusion_norm(fused)
        return fused, new_field


class TaijiArchitecture(nn.Module):
    """Taiji 架构级模型 —— 全栈内化群体机制(替代整个 NeuroPlex 群体栈)。

    架构组成:
    1. SensoryPlane:embedding + population_adapter
    2. TaijiPopulationLayer × N:层叠群体计算单元
    3. CrossLayerField:跨层场传导(spatial_diffuser)
    4. OutputPlane:lm_head

    群体性(层内并行 + 跨层场传导):
    - 每层 P 个 TaijiBlock 并行,共享层内场
    - 跨层场传导:上一层 field_state 经 spatial_diffuser 传到下一层
    - 相位绑定驱动群体激活
    - per-sample sparse routing 控制每样本的神经元子集

    退化行为:
    - num_neurons=1:退化为单神经元(无群体),TaijiPopulationLayer 等价于
      单个 TaijiBlock + field I/O,类似 ResonanceNeuron 单体
    - field_dim=None(未实现路径):退化为纯 yang 流(无场共振)
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
        phase_delta: float = 0.5,
        router_top_k: int = 0,
        refractory_steps: int = 2,
        stdp_strength: float = 0.1,
        # 跨层场传导
        cross_layer_diffusion: float = 0.3,
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
                phase_delta=phase_delta,
                act_temp=4.0,
                router_top_k=router_top_k,
                refractory_steps=refractory_steps,
                stdp_strength=stdp_strength,
            )
            for _ in range(num_layers)
        ])

        # ── 跨层场传导(spatial_diffuser 简化版) ──
        # 上一层 field → 下一层 field 的扩散系数
        self.cross_layer_diffusion = cross_layer_diffusion
        # 场投影(不同层 field_dim 可能不同,这里同 dim 用 Identity)
        self.field_transfer = nn.Linear(field_dim, field_dim, bias=False)
        nn.init.eye_(self.field_transfer.weight)  # 初始恒等(零回归)

        # ── OutputPlane ──
        self.output_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 权重共享(tied embedding,标准 LLaMA 风格)
        self.lm_head.weight = self.shared_embedding.weight

        # ── 调质平面(预留接口,后续接 NeuromodulatorState) ──
        self._temp_gain = 1.0
        self._ffn_gain = 1.0

    def set_neuromodulator_gains(self, temp_gain: float = 1.0,
                                 ffn_gain: float = 1.0) -> None:
        """设置神经调质驱动的 gain(后续接 NeuromodulatorState)。"""
        self._temp_gain = temp_gain
        self._ffn_gain = ffn_gain

    def forward(
        self,
        input_ids: torch.Tensor,
        return_field_history: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Taiji 架构前向。

        Args:
            input_ids: [B, L] token ids
            return_field_history: 若 True,返回每层场状态历史(诊断用)

        Returns:
            dict with:
            - logits: [B, L, vocab]
            - final_field: [B, D] 最终层场状态
            - field_history: List[[B, D]] 每层场状态(若 return_field_history)
        """
        bsz, seqlen = input_ids.shape

        # ── SensoryPlane ──
        h = self.shared_embedding(input_ids)  # [B, L, hidden]
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

        # ── 跨层场传导初始化 ──
        cross_field = None  # 第一层无跨层输入
        field_history: List[torch.Tensor] = []

        # ── TaijiPopulationLayer × N(层叠 + 跨层场传导) ──
        for layer in self.layers:
            h, field = layer(
                h, mask=mask,
                temp_gain=self._temp_gain, ffn_gain=self._ffn_gain,
                cross_layer_field=cross_field,
            )
            if return_field_history:
                field_history.append(field.detach())
            # 跨层场传导:本层场经扩散传到下一层
            cross_field = self.cross_layer_diffusion * self.field_transfer(field)

        # ── OutputPlane ──
        h = self.output_norm(h)
        logits = self.lm_head(h)  # [B, L, vocab]

        result = {
            "logits": logits,
            "final_field": field,
        }
        if return_field_history:
            result["field_history"] = field_history
        return result


__all__ = ["TaijiPopulationLayer", "TaijiArchitecture", "_PhaseBinder", "_SparseRouter"]
