"""
Taiji 算子 —— 群体神经元架构的原生底层算子（替代 TransformerBlock）

设计动机（详见 plans/active/TAIJI_OPERATOR_DESIGN.md）：
    TransformerBlock 在 ResonanceNeuron 群体架构下有四个结构性瓶颈：
    1. field 注入是外挂残差（field_read_layers 在 block 外加），field 无法
       调制 token 级 attention 的 K/V → 群体共振信号不能真正进入注意力
    2. ensemble 多轮共振每轮重算全 block，KV cache 在 field 变化时失效
    3. 相位绑定 a_i = σ(β·(binding_i-b0)) 只在 ensemble 层调控激活强度，
       phase 不进入神经元内部计算
    4. side_channels（excite/inhibit）也是外挂投影，不进入 block 内部

核心思想（太极轮转）：
    用复值旋转（yin-yang 互补双相）统一 token-attention 与 field-resonance，
    让 phase、field、side-signal 全部成为算子内部的原生调制量。

    - hidden 拆成实部/虚部双流 (h_r, h_i)
    - yang 流（实部）：causal token attention（局部时序，对应 TransformerBlock basal）
    - yin 流（虚部）：field-coupled global resonance（全局共振，对应 apical 但 field-native）
    - 太极门控：可学习 sigmoid 互斥融合两路（yang 强时 yin 让位，反之亦然）
    - phase 作为旋转角直接进入 yin 流的复值旋转（相位绑定进入算子内部）
    - field_state 进入 yin 流的 K/V 相位项（field-native conditioning）

接口契约（与 TransformerBlock.forward 完全对齐，可直接替换）：
    forward(x, mask, kv_cache, use_cache, temp_gain, ffn_gain, field_state,
            return_attn_weights, phase)
        -> (x_out, new_kv_cache, attn_weights)

    新增可选 phase 参数（None 时退化为纯 yang 流，行为接近 TransformerBlock）。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.layers import RMSNorm, RotaryEmbedding, apply_rotary_emb, SwiGLU


class _YangAttention(nn.Module):
    """Yang 流注意力：causal token-token attention（局部时序）。

    复用 layers.py 的 RoPE + GQA 模式，但简化为单次实现，便于后续 kernel 化。
    与 GroupedQueryAttention 的区别：内部预留 phase 调制接口（本原型 phase
    主要进 yin 流，yang 流保持标准 causal 语义）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        # 复用 layers.py 的 RoPE（线程安全 + LRU 缓存）
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temp_gain: float = 1.0,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args: x [B, L, hidden]; mask [1,1,L,L] 或 None。"""
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # RoPE
        sin, cos = self.rope(xq, seqlen)
        xq, xk = apply_rotary_emb(xq, xk, sin, cos)

        # S9: temp_gain 门控注意力温度（norepinephrine 驱动）
        if temp_gain != 1.0:
            xq = xq * temp_gain

        # GQA: 扩展 KV heads
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=2)

        # [B, heads, L, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        is_causal = mask is None and seqlen > 1
        attn_weights = None

        try:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask, is_causal=is_causal,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        except Exception:
            # 手动 fallback（与 GroupedQueryAttention 一致）
            scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            elif seqlen > 1:
                causal = torch.triu(
                    torch.full((seqlen, seqlen), float("-inf"),
                               device=x.device, dtype=x.dtype),
                    diagonal=1,
                )
                scores = scores + causal
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)
            if self.training:
                attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, xv)

        if return_attn_weights and attn_weights is None:
            # 仅兼容性对齐实验需要时重算一次（开销大，默认不开）
            with torch.no_grad():
                scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
                if mask is not None:
                    scores = scores + mask
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), attn_weights


class _YinResonance(nn.Module):
    """Yin 流共振：field-coupled global resonance（全局共振，field-native）。

    这是 taiji 算子相对 TransformerBlock 的核心增量：field_state 不再是
    外挂残差，而是直接进入 K/V 的相位项，让群体共振信号原生调制 token 表征。

    数学：
        Q = Wq · h                       # [B, L, hidden]
        K_field = Wk_field · field_state # [B, 1, hidden]（field → 单 token KV）
        V_field = Wv_field · field_state
        # phase 旋转（phase 进入算子内部，对应 ensemble 的相位绑定）
        K_field_rot = K_field · e^{iφ}   # 复值旋转，φ = phase
        # cross-attention（无 causal，field 是全局反馈）
        yin_out = softmax(Q · K_field_rot^T / √d) · V_field

    phase=None 时退化为实数 cross-attention（无旋转），行为接近 dendritic
    apical 路径但 field-native（field 直接进 K/V 而非外挂）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        field_dim: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Q from x（token-side）
        self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        # K/V from field_state（field-side，跨空间投影）
        self.wk_field = nn.Linear(field_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wv_field = nn.Linear(field_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        field_state: torch.Tensor,
        phase: Optional[float] = None,
        temp_gain: float = 1.0,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            x: [B, L, hidden]
            field_state: [B, D] 或 [B, S, D]（全局场状态）
            phase: 旋转角（弧度），None 退化为实数 cross-attention
        """
        bsz, seqlen, _ = x.shape

        # field_state → [B, kv_len, D]
        if field_state.dim() == 2:
            fs = field_state.unsqueeze(1)  # [B, 1, D]
        else:
            fs = field_state
        kv_len = fs.shape[1]

        xq = self.wq(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = self.wk_field(fs).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        xv = self.wv_field(fs).view(bsz, kv_len, self.num_kv_heads, self.head_dim)

        # S9: temp_gain 门控
        if temp_gain != 1.0:
            xq = xq * temp_gain

        # ── phase 旋转（taiji 核心：phase 进入算子内部）──
        # 对 K 和 V 施加复值旋转 e^{iφ} = (cos φ, sin φ)：
        #   K_rot_real = K_real · cos φ - K_imag · sin φ
        #   K_rot_imag = K_real · sin φ + K_imag · cos φ
        # （V 同理）
        # 但 K/V 是实数张量 → 把 head_dim 拆成两半当作 (real, imag) 对，
        # 这样无需引入真正的复值张量，纯实数运算即可表达旋转。
        #
        # 为什么同时旋转 K 和 V（而非只旋转 K）：
        #   - 只旋转 K：<Q, K_rot> 依赖 phase → attention 权重随 phase 变化
        #     但当 kv_len=1（单 token field）时 softmax 恒为 1.0，K 旋转无效。
        #   - 同时旋转 V：attention 输出 = softmax · V_rot，即使 softmax 不变
        #     （kv_len=1），V 旋转仍改变输出 → phase 在单 token field 下也生效。
        #   - K 旋转负责"选择性调制"（多 token field 下的相位路由），
        #     V 旋转负责"内容调制"（任何 kv_len 下的相位编码），
        #     两者互补，让 phase 在所有 field 形状下都有调制能力。
        #
        # Q 不旋转：保持 <Q, K_rot> 对 phase 的依赖（Q 旋转会让 K 旋转的内积
        # 不变性复现，phase 在 K 侧失效）。
        if phase is not None and phase != 0.0:
            half = self.head_dim // 2
            if half > 0:
                cos_p = math.cos(phase)
                sin_p = math.sin(phase)
                # K 旋转
                k_r, k_i = xk[..., :half], xk[..., half:2 * half]
                k_rot_r = k_r * cos_p - k_i * sin_p
                k_rot_i = k_r * sin_p + k_i * cos_p
                # V 旋转（内容调制，单 token field 下也生效）
                v_r, v_i = xv[..., :half], xv[..., half:2 * half]
                v_rot_r = v_r * cos_p - v_i * sin_p
                v_rot_i = v_r * sin_p + v_i * cos_p
                if self.head_dim % 2 == 0:
                    xk = torch.cat([k_rot_r, k_rot_i], dim=-1)
                    xv = torch.cat([v_rot_r, v_rot_i], dim=-1)
                else:
                    xk = torch.cat([k_rot_r, k_rot_i, xk[..., 2 * half:]], dim=-1)
                    xv = torch.cat([v_rot_r, v_rot_i, xv[..., 2 * half:]], dim=-1)
                # Q 不旋转（保持 <Q, K_rot> 对 phase 的依赖）

        # GQA: 扩展 KV heads
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=2)

        xq = xq.transpose(1, 2)  # [B, heads, L, head_dim]
        xk = xk.transpose(1, 2)  # [B, heads, kv_len, head_dim]
        xv = xv.transpose(1, 2)

        # Cross-attention（无 causal mask，field 是全局反馈）
        attn_weights = None
        try:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, is_causal=False,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        except Exception:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)
            if self.training:
                attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, xv)

        if return_attn_weights and attn_weights is None:
            with torch.no_grad():
                scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), attn_weights


class TaijiBlock(nn.Module):
    """Taiji 算子块（替代 TransformerBlock）。

    太极轮转：yang 流（causal token attention）+ yin 流（field-coupled
    resonance）+ 太极门控融合 + SwiGLU FFN。

    退化行为（向后兼容）：
        - field_state=None 且 phase=None：yin 流不激活，退化为标准
          causal TransformerBlock（yang + FFN），行为与 TransformerBlock
          dendritic=False 等价。

    接口与 TransformerBlock.forward 完全对齐（新增可选 phase 参数），
    可直接在 ResonanceNeuron 中替换 self.layers 的构造。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
        dropout: float = 0.0,
        field_dim: Optional[int] = None,
        # S11 长上下文参数（yang 流保留兼容，本原型暂不实现 sink/sliding）
        attention_sink_size: int = 0,
        sliding_window_size: int = 0,
        # taiji 特有
        yin_num_heads: Optional[int] = None,
        taiji_gate_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_dim = field_dim

        # ── Yang 流：causal token attention ──
        self.yang_attn = _YangAttention(
            hidden_size, num_heads, num_kv_heads, bias=bias, dropout=dropout,
        )
        self.yang_norm = RMSNorm(hidden_size, rms_norm_eps)

        # ── Yin 流：field-coupled resonance（仅 field_dim 给定才创建）──
        self.has_yin = field_dim is not None
        if self.has_yin:
            self.yin_resonance = _YinResonance(
                hidden_size,
                yin_num_heads or num_heads,
                field_dim=field_dim,
                num_kv_heads=num_kv_heads,
                bias=bias,
                dropout=dropout,
            )
            self.yin_norm = RMSNorm(hidden_size, rms_norm_eps)
            # 太极门控：可学习 sigmoid，决定每位置信任 yang 还是 yin
            # gate → 0.5: 两路均衡；>0.5: 偏 yang；<0.5: 偏 yin
            # 初始化为 taiji_gate_init（默认 0 → sigmoid(0)=0.5 均衡起点）
            self.taiji_gate = nn.Linear(hidden_size, 1, bias=True)
            nn.init.constant_(self.taiji_gate.bias, taiji_gate_init)
        else:
            self.yin_resonance = None
            self.yin_norm = None
            self.taiji_gate = None

        # ── FFN（复用 layers.SwiGLU，保持 ffn_gain 调质接口）──
        self.feed_forward = SwiGLU(hidden_size, intermediate_size)
        self.ffn_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.resid_dropout = nn.Dropout(dropout)

        # 标记（与 TransformerBlock 兼容）
        self.dendritic = self.has_yin  # has_yin 时行为类似 dendritic=True

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        field_state: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        # taiji 特有（可选，向后兼容 TransformerBlock 调用）
        phase: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Taiji 块前向。

        Args:
            x: [B, L, hidden]
            mask: [1,1,L,L] causal mask（yang 流用）
            kv_cache: 保留接口兼容（本原型 yang 流不实现 cache，返回 None）
            use_cache: 同上
            temp_gain: S9 注意力温度增益（yang + yin 共享）
            ffn_gain: S9 FFN 输出增益
            field_state: [B, D] 或 [B, S, D] 全局场状态（yin 流 KV 来源）
            return_attn_weights: 返回 yang 流 attention 权重（兼容性对齐实验）
            phase: yin 流旋转角（弧度），None 退化为实数 cross-attention

        Returns:
            (x_out, new_kv_cache, attn_weights)
            与 TransformerBlock.forward 返回结构对齐。
        """
        # ── Yang 流：causal token attention ──
        yang_out, attn_weights = self.yang_attn(
            self.yang_norm(x), mask=mask,
            temp_gain=temp_gain, return_attn_weights=return_attn_weights,
        )

        # ── Yin 流：field-coupled resonance（field_state 非 None 才激活）──
        if self.has_yin and field_state is not None:
            yin_out, _ = self.yin_resonance(
                self.yin_norm(x), field_state=field_state,
                phase=phase, temp_gain=temp_gain,
                return_attn_weights=return_attn_weights,
            )
            # 太极门控融合：gate ∈ (0,1) 决定 yang/yin 权重
            # gate = sigmoid(taiji_gate(x))  → yang 权重 = gate, yin 权重 = 1-gate
            # （互斥融合：yang 强时 yin 让位，反之亦然）
            gate = torch.sigmoid(self.taiji_gate(x))  # [B, L, 1]
            fused = gate * yang_out + (1.0 - gate) * yin_out
            x = x + self.resid_dropout(fused)
        else:
            # 退化路径：无 yin 流（field_state=None 或未配置 field_dim）
            x = x + self.resid_dropout(yang_out)

        # ── FFN（与 TransformerBlock 一致）──
        x = x + self.resid_dropout(self.feed_forward(self.ffn_norm(x), gain=ffn_gain))

        # kv_cache 接口兼容（本原型不实现，返回 None）
        return x, None, attn_weights


__all__ = ["TaijiBlock", "_YangAttention", "_YinResonance"]
