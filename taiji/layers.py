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
    """GQA — 分组查询注意力，省显存效果好"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 dropout: float = 0.0, bias: bool = False):
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
        new_kv_cache = (xk, xv) if use_cache else None

        # GQA: 扩展 KV heads（PyTorch 2.5+ 原生支持 GQA，旧版本需要手动扩展）
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=2)

        # 转换为 [batch, heads, seq, dim] 格式
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

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
    """Pre-Norm Transformer 块"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, rms_norm_eps: float = 1e-5, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.attention = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads,
                                               dropout=dropout, bias=bias)
        self.attention_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.feed_forward = SwiGLU(hidden_size, intermediate_size)
        self.ffn_norm = RMSNorm(hidden_size, rms_norm_eps)
        # 残差 dropout：防止过拟合（社区规范 SmolLM/GPT-2 用 0.1）
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """S9: temp_gain / ffn_gain 接收神经调质信号，注入 Transformer 内部计算。

        - temp_gain: 传给 GroupedQueryAttention（norepinephrine 驱动注意力温度）
        - ffn_gain: 传给 SwiGLU（dopamine 驱动 FFN 输出强度）
        - 两者默认 1.0，向后兼容标准 Transformer
        """
        # Attention + Residual (Pre-Norm)
        h, new_kv_cache = self.attention(
            self.attention_norm(x), mask, kv_cache, use_cache, temp_gain=temp_gain,
        )
        x = x + self.resid_dropout(h)
        # FFN + Residual (Pre-Norm)
        x = x + self.resid_dropout(self.feed_forward(self.ffn_norm(x), gain=ffn_gain))
        return x, new_kv_cache
