"""
核心层实现
LLaMA 3 风格: RMSNorm + RoPE + GQA + SwiGLU + Pre-Norm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMSNorm — 比 LayerNorm 更快更稳定"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    """RoPE — 旋转位置编码（预计算 + 缓存限制，防止内存泄漏）"""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # 频率: 1 / (theta ^ (2i/dim))
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        # 使用 OrderedDict 作为 LRU 缓存，最多保留 4 个条目
        from collections import OrderedDict
        import threading as _threading
        self._cache = OrderedDict()
        self._max_cache_size = 4
        self._cache_lock = _threading.Lock()

    def _get_sin_cos(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(pos, self.freqs.to(device))
        result = (
            torch.sin(angles).to(dtype),
            torch.cos(angles).to(dtype),
        )

        with self._cache_lock:
            self._cache[key] = result
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        return result

    def forward(self, x: torch.Tensor, seq_len: int):
        return self._get_sin_cos(seq_len, x.device, x.dtype)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor,
    sin: torch.Tensor, cos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 应用旋转编码"""
    # xq, xk: [batch, seq, heads, head_dim]
    # sin, cos: [seq, head_dim/2]
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]

    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim/2]
    cos = cos.unsqueeze(0).unsqueeze(2)

    q_out = torch.stack([xq_r * cos - xq_i * sin,
                         xq_r * sin + xq_i * cos], dim=-1).flatten(-2)
    k_out = torch.stack([xk_r * cos - xk_i * sin,
                         xk_r * sin + xk_i * cos], dim=-1).flatten(-2)
    return q_out.type_as(xq), k_out.type_as(xk)


class GroupedQueryAttention(nn.Module):
    """GQA — 分组查询注意力，省显存效果好

    S11: 支持 attention sink + 滑动窗口（StreamingLLM 启发）。
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 dropout: float = 0.0, bias: bool = False,
                 attention_sink_size: int = 0, sliding_window_size: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # S11: attention sink + 滑动窗口配置
        # sink_size > 0 且 window_size > 0 时启用长上下文 KV cache 管理
        # KV cache 保留前 sink_size 个 token（锚点）+ 最近 window_size 个 token（滑动窗口）
        # 总 KV cache 长度上限 = sink_size + window_size
        # 0 = 关闭（向后兼容，KV cache 无限增长）
        self.attention_sink_size = attention_sink_size
        self.sliding_window_size = sliding_window_size
        self.kv_cache_max_len = attention_sink_size + sliding_window_size

    def _evict_kv_cache(
        self, xk: torch.Tensor, xv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S11: 滑动窗口 KV cache 驱逐。

        当 KV cache 超过 max_len 时，保留前 sink_size 个 token + 最近 window_size 个 token，
        驱逐中间的旧 token。这保持 sink 锚点不变，滑动窗口向前推进。

        Args:
            xk: [B, S, num_kv_heads, head_dim] 完整 KV cache keys
            xv: [B, S, num_kv_heads, head_dim] 完整 KV cache values

        Returns:
            (xk_evicted, xv_evicted): 驱逐后的 KV cache，长度 = sink_size + window_size
        """
        sink_size = self.attention_sink_size
        window_size = self.sliding_window_size
        max_len = self.kv_cache_max_len

        cur_len = xk.shape[1]
        if cur_len <= max_len:
            return xk, xv  # 未超限，无需驱逐

        # 保留前 sink_size + 最近 window_size
        sink_k = xk[:, :sink_size]
        sink_v = xv[:, :sink_size]
        window_k = xk[:, -window_size:]
        window_v = xv[:, -window_size:]
        return torch.cat([sink_k, window_k], dim=1), torch.cat([sink_v, window_v], dim=1)

    def forward(
        self, x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        temp_gain: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """S9: temp_gain 门控注意力温度（norepinephrine 驱动）。

        temp_gain > 1 → xq 放大 → logits 放大 → softmax 更尖锐（高警觉，聚焦）
        temp_gain < 1 → xq 缩小 → logits 缩小 → softmax 更分散（低警觉，泛化）
        temp_gain = 1 → 标准注意力（向后兼容）

        S11: kv_cache 启用时，若配置了 attention_sink_size + sliding_window_size，
        会自动驱逐旧 token 保持 KV cache 长度上限 = sink_size + window_size。
        """
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # S9: norepinephrine 门控注意力温度（缩放 query 等价于缩放 logits）
        if temp_gain != 1.0:
            xq = xq * temp_gain

        # RoPE
        start_pos = kv_cache[0].shape[1] if kv_cache is not None else 0
        sin, cos = self.rope(x, seqlen + start_pos)
        xq, xk = apply_rotary_emb(xq, xk, sin[start_pos:], cos[start_pos:])

        # KV Cache
        if kv_cache is not None:
            xk = torch.cat([kv_cache[0], xk], dim=1)
            xv = torch.cat([kv_cache[1], xv], dim=1)
            # S11: 滑动窗口驱逐（KV cache 超限时保留 sink + window）
            if self.kv_cache_max_len > 0:
                xk, xv = self._evict_kv_cache(xk, xv)
        new_kv_cache = (xk, xv) if use_cache else None

        # GQA: 扩展 KV heads（PyTorch 2.5+ 原生支持 GQA，旧版本需要手动扩展）
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=2)

        # 转换为 [batch, heads, seq, dim] 格式
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # S11: 滑动窗口启用时，mask 需要适配驱逐后的 KV cache 长度
        # 当驱逐发生时，KV cache 长度可能小于 start_pos + seqlen，
        # 此时不能再用标准 causal mask（会因维度不匹配报错）
        # 解决：滑动窗口模式下禁用 causal mask（依赖 KV cache 顺序保证因果性）
        # 训练时（无 kv_cache）不受影响
        if self.kv_cache_max_len > 0 and kv_cache is not None and mask is not None:
            # 滑动窗口 + KV cache 模式：mask 维度可能不匹配，禁用 mask
            # 注意：这只在推理时（use_cache=True）发生，训练时 mask 仍然生效
            mask = None

        # Flash Attention（PyTorch 2.0+ 自动调度到 FlashAttention-2/3 或 Memory-Efficient）
        # 比手动 matmul 快 2-4x，内存减少 50-70%
        is_causal = (mask is not None) and (seqlen > 1)
        try:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                is_causal=is_causal,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        except Exception:
            # 回退到手动 attention（兼容旧版 PyTorch）
            scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)
            if self.training:
                scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), new_kv_cache


class SwiGLU(nn.Module):
    """SwiGLU — 门控激活，比 GELU 效果好"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """S9: gain 门控 FFN 输出强度（dopamine 驱动）。

        gain > 1 → FFN 输出增强（奖励信号，强化重要特征通过）
        gain < 1 → FFN 输出衰减（惩罚信号，弱化非重要特征）
        gain = 1 → 标准 FFN（向后兼容）

        gain 作用于残差路径（FFN 输出），不影响 Pre-Norm 的输入。
        """
        out = self.w2(F.silu(self.w_gate(x)) * self.w1(x))
        if gain != 1.0:
            out = out * gain
        return out


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 块（支持 S10 树突化扩展）。

    dendritic=False（默认）: 标准 Transformer 块（向后兼容）
    dendritic=True: 树突化块
        - Basal 路径: 标准 attention + FFN（自下而上）
        - Apical 路径: cross-attention（Q=x, KV=field_state，自上而下）
        - 胞体整合: 预测编码（error = basal - apical_prediction，误差驱动校正）
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, rms_norm_eps: float = 1e-5, bias: bool = False,
                 dropout: float = 0.0,
                 dendritic: bool = False, apical_kv_dim: Optional[int] = None,
                 attention_sink_size: int = 0, sliding_window_size: int = 0):
        super().__init__()
        self.dendritic = dendritic

        # ── Basal 路径（始终存在，标准 Transformer）──
        # S11: 传入 attention_sink_size + sliding_window_size 启用长上下文
        self.attention = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads,
            dropout=dropout, bias=bias,
            attention_sink_size=attention_sink_size,
            sliding_window_size=sliding_window_size,
        )
        self.attention_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.feed_forward = SwiGLU(hidden_size, intermediate_size)
        self.ffn_norm = RMSNorm(hidden_size, rms_norm_eps)
        # 残差 dropout：防止过拟合（社区规范 SmolLM/GPT-2 用 0.1）
        self.resid_dropout = nn.Dropout(dropout)

        # ── Apical 路径（树突化时创建）──
        # 人脑启发：顶树突接收皮层反馈（类比 field_state 集体意识场）
        # Apical cross-attention: Q 来自当前层输入，KV 来自 field_state
        if dendritic:
            assert apical_kv_dim is not None, "dendritic=True 需要 apical_kv_dim（field_dim）"
            self.apical_num_heads = num_heads
            self.apical_head_dim = hidden_size // num_heads
            self.apical_num_kv_heads = num_kv_heads
            self.apical_num_queries_per_kv = num_heads // num_kv_heads
            self.apical_scale = self.apical_head_dim ** -0.5

            # Q 投影（来自 x，与 basal 共享输入但独立参数）
            self.apical_wq = nn.Linear(hidden_size, num_heads * self.apical_head_dim, bias=bias)
            # KV 投影（来自 field_state，跨空间投影）
            self.apical_wk = nn.Linear(apical_kv_dim, num_kv_heads * self.apical_head_dim, bias=bias)
            self.apical_wv = nn.Linear(apical_kv_dim, num_kv_heads * self.apical_head_dim, bias=bias)
            # 输出投影
            self.apical_wo = nn.Linear(num_heads * self.apical_head_dim, hidden_size, bias=False)

            # Apical 的 Pre-Norm
            self.apical_norm = RMSNorm(hidden_size, rms_norm_eps)
            self.apical_attn_dropout = nn.Dropout(dropout)

            # 胞体整合：预测编码门控
            # gate 决定每位置信任 basal 还是 apical 预测
            # gate → 0: 信任 basal（标准 Transformer 行为）
            # gate → 1: 信任 apical 预测（强反馈驱动）
            self.somatic_gate = nn.Linear(hidden_size, 1, bias=True)
            # 预测误差缩放（可学习，初始小值保证训练稳定）
            self.error_scale = nn.Parameter(torch.tensor(0.1))

    def _apical_cross_attention(
        self, x: torch.Tensor, field_state: torch.Tensor, temp_gain: float = 1.0,
    ) -> torch.Tensor:
        """Apical cross-attention: Q from x, KV from field_state.

        Args:
            x: [B, L, hidden] 当前层 basal 输入（已过 apical_norm）
            field_state: [B, D] 或 [B, S, D] 全局场状态（自上而下反馈）
            temp_gain: 注意力温度增益（与 basal 共享调质）

        Returns:
            apical_out: [B, L, hidden] apical 路径输出
        """
        bsz, seqlen, _ = x.shape

        # field_state: [B, D] → [B, 1, D]（单 token KV，全局上下文）
        if field_state.dim() == 2:
            fs = field_state.unsqueeze(1)  # [B, 1, D]
        else:
            fs = field_state  # [B, S, D]
        kv_len = fs.shape[1]

        # Q from x, K/V from field_state
        xq = self.apical_wq(x).view(bsz, seqlen, self.apical_num_heads, self.apical_head_dim)
        xk = self.apical_wk(fs).view(bsz, kv_len, self.apical_num_kv_heads, self.apical_head_dim)
        xv = self.apical_wv(fs).view(bsz, kv_len, self.apical_num_kv_heads, self.apical_head_dim)

        # S9: temp_gain 门控注意力温度
        if temp_gain != 1.0:
            xq = xq * temp_gain

        # GQA: 扩展 KV heads
        if self.apical_num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.apical_num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.apical_num_queries_per_kv, dim=2)

        # [B, heads, seq, dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Cross-attention（无 causal mask，KV 是全局反馈）
        try:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                is_causal=False,  # cross-attention 不需要 causal
                dropout_p=self.apical_attn_dropout.p if self.training else 0.0,
            )
        except Exception:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.apical_scale
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)
            if self.training:
                scores = self.apical_attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.apical_wo(output)

    def forward(
        self, x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        field_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """S9+S10: 神经调质门控 + 树突化扩展。

        - temp_gain/ffn_gain: S9 神经调质信号（norepinephrine/dopamine）
        - field_state: S10 树突化 apical 路径的 KV 来源（全局场状态）
          - None 或 dendritic=False: 走标准 basal 路径（向后兼容）
          - 非 None 且 dendritic=True: basal + apical + 预测编码整合
        """
        # ── Basal 路径: 标准 attention + FFN ──
        h, new_kv_cache = self.attention(
            self.attention_norm(x), mask, kv_cache, use_cache, temp_gain=temp_gain,
        )
        x = x + self.resid_dropout(h)
        x = x + self.resid_dropout(self.feed_forward(self.ffn_norm(x), gain=ffn_gain))

        # ── Apical 路径: 树突化 cross-attention + 预测编码整合 ──
        if self.dendritic and field_state is not None:
            # Apical cross-attention（Q=x, KV=field_state）
            h_apical = self._apical_cross_attention(
                self.apical_norm(x), field_state, temp_gain=temp_gain,
            )

            # 预测编码整合（胞体整合）
            # apical_prediction = x + h_apical（apical 残差预测）
            # error = basal_output - apical_prediction（预测误差）
            # gate 决定误差校正强度
            apical_prediction = x + h_apical
            error = x - apical_prediction  # = -h_apical（简化形式）
            gate = torch.sigmoid(self.somatic_gate(x))  # [B, L, 1]
            # 误差校正：x = x - error_scale * gate * error
            # gate→0: 不校正（信任 basal）；gate→1: 强校正（信任 apical）
            x = x - self.error_scale * gate * error

        return x, new_kv_cache
