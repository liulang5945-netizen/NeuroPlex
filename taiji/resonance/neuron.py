"""Resonance neuron — wraps a ModelSelfBackbone with field I/O.

Each neuron:
1. Receives shared base embeddings (Level 0 → Level 1)
2. Projects through its own adapter into a per-neuron concept space
3. Processes through a standard Transformer (layers.py, zero changes)
4. Writes a normalised field vector (Level 1 → Level 2)
5. Reads field state for conditioning (Level 2 → Level 1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.config import ModelConfig
from taiji.layers import RMSNorm, TransformerBlock

from .config import NeuronConfig


class ResonanceNeuron(nn.Module):
    """A single resonance neuron — independent Transformer + field interface.

    Built on top of the existing TransformerBlock and RMSNorm from layers.py
    (zero changes to existing code).  Adds:
    - embed_adapter: shared embedding → neuron's internal dimension
    - field_write: final hidden → field vector (L2-normalised)
    - field_read_layers: field state → per-layer conditioning residual
    - lm_head: vocab projection (for PPL evaluation / pretraining)
    - fingerprint: frozen direction vector for prescreening (future)
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()
        self.config = neuron_config
        c = neuron_config

        # ── 神经元类型（人脑启发：兴奋性/抑制性分化）──
        # excitatory: 对场做正向贡献（默认）
        # inhibitory: 对场做负向贡献，抑制过度共振
        self.neuron_type = c.neuron_type

        # ── 不应期状态（人脑启发：refractory period）──
        # 写入场后进入不应期，refractory_cooldown 轮内只能读场不能写
        # 由 ensemble 调度，防止强神经元垄断场
        self.register_buffer("refractory_counter", torch.zeros(1, dtype=torch.long))

        # ── Embedding adapter (P0: shared base → per-neuron concept space) ──
        self.embed_adapter = nn.Linear(c.base_embed_dim, c.hidden_size, bias=False)

        # ── Transformer body (reuses layers.py, zero changes) ──
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_attention_heads,
                num_kv_heads=c.num_key_value_heads,
                intermediate_size=c.intermediate_size,
                rms_norm_eps=c.rms_norm_eps,
                bias=c.attention_bias,
            )
            for _ in range(c.num_hidden_layers)
        ])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

        # ── Field write projection ──
        self.field_write = nn.Linear(c.hidden_size, c.field_dim, bias=False)

        # Attention pooling for field write (v2)
        # Learn a query that pools over all positions instead of only using the last token.
        # This surfaces the most salient conceptual content for field communication.
        self.field_pool_query = nn.Parameter(torch.randn(c.hidden_size) * 0.02)
        self.field_pool_scale = c.hidden_size ** -0.5

        # ── Field read projections (one per layer, for conditioning) ──
        self.field_read_layers = nn.ModuleList([
            nn.Linear(c.field_dim, c.hidden_size, bias=False)
            for _ in range(c.num_hidden_layers)
        ])

        # Position gate for field read (v2)
        # Each position decides how much field conditioning to absorb,
        # replacing the old broadcast (same vector to all positions).
        self.field_read_gate = nn.Linear(c.hidden_size, 1, bias=True)

        # ── Language modelling head（低秩分解：共享基 + per-neuron 残差）──
        # 人脑启发：所有神经元共享基础"语言能力"（W_base），
        # 但每个神经元有个体表达偏好（U_i @ V_i 低秩残差）
        # logits = W_base(h) + V_i(U_i(h))
        # 兼容旧 ckpt：lm_head_rank=0 时使用传统 per-neuron lm_head
        if c.lm_head_rank > 0:
            # 低秩模式：W_base 由外部注入（共享），U_i/V_i 是 per-neuron
            self.lm_head_delta_u = nn.Linear(c.hidden_size, c.lm_head_rank, bias=False)
            self.lm_head_delta_v = nn.Linear(c.lm_head_rank, c.vocab_size, bias=False)
            # W_base 不在此处创建（由 Cortex/Ensemble 统一注入共享实例）
            # 初始时 W_base=None，forward 时若 W_base 为 None 则只用残差（用于独立训练）
            self.lm_head_base: Optional[nn.Module] = None
            # 初始化残差为小值，训练初期 logits ≈ W_base(h)
            nn.init.normal_(self.lm_head_delta_u.weight, std=0.01)
            nn.init.normal_(self.lm_head_delta_v.weight, std=0.01)
        else:
            # 传统模式：per-neuron 完整 lm_head
            self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

        # ── Direction fingerprint (frozen, for future prescreening) ──
        self.register_buffer("fingerprint", torch.zeros(c.hidden_size))

        # ── Side channel interface（人脑启发：兴奋/抑制双通道）──
        # excite_channels: 正向调制（兴奋性突触，类比谷氨酸能）
        # inhibit_channels: 负向调制（抑制性突触，类比 GABA 能）
        # 每个 peer 可同时拥有两种通道，由 STDP 学习决定哪种占主导
        self.excite_channels = nn.ModuleDict()
        self.inhibit_channels = nn.ModuleDict()
        # 保留 side_channels 别名以兼容旧代码（指向 excite_channels）
        # 注意：旧代码通过 self.side_channels 访问的，现在统一指向 excite_channels

        # v1 compatibility: use last-token write + broadcast read (for old ckpts)
        self.v1_compat: bool = False

    @property
    def side_channels(self) -> nn.ModuleDict:
        """兼容旧代码：side_channels 现在统一指向 excite_channels。"""
        return self.excite_channels

    def set_shared_lm_head(self, shared_base: nn.Module) -> None:
        """注入共享 lm_head 基矩阵（由 Cortex/Ensemble 统一调用）。

        低秩分解模式下，W_base 是所有神经元共享的，
        由外部创建一次后注入到每个神经元。
        """
        if self.config.lm_head_rank > 0:
            self.lm_head_base = shared_base
        else:
            raise RuntimeError(
                "set_shared_lm_head 仅在 lm_head_rank > 0（低秩模式）时可用"
            )

    def compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        """计算 logits，支持低秩分解和传统两种模式。"""
        if self.config.lm_head_rank > 0:
            # 低秩模式：logits = W_base(h) + V_i(U_i(h))
            delta = self.lm_head_delta_v(self.lm_head_delta_u(h))
            if self.lm_head_base is not None:
                return self.lm_head_base(h) + delta
            return delta
        else:
            # 传统模式
            return self.lm_head(h)

    @property
    def is_inhibitory(self) -> bool:
        """是否为抑制性神经元。"""
        return self.neuron_type == "inhibitory"

    @property
    def in_refractory(self) -> bool:
        """是否处于不应期（不能写入场）。"""
        return bool(self.refractory_counter.item() > 0)

    def enter_refractory(self, multiplier: float = 1.0) -> None:
        """写入场后调用，进入不应期。

        P1-2: multiplier 由 NeuromodulatorState.get_refractory_multiplier 提供，
        高血清素 → 不应期更长（满足，不易再激活）。
        """
        cooldown = max(1, int(self.config.refractory_cooldown * multiplier))
        self.refractory_counter.fill_(cooldown)

    def tick_refractory(self) -> None:
        """每轮共振结束时调用，递减不应期计数器。"""
        if self.refractory_counter.item() > 0:
            self.refractory_counter -= 1

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        field_state: Optional[torch.Tensor] = None,
        round_num: int = 1,
        return_logits: bool = False,
        side_signals: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the neuron.

        Args:
            shared_embeddings: [B, L, base_embed_dim] from the shared embedding table
            field_state: [D] current field state vector (from round 2 onward)
            round_num: current resonance round (1 = independent, 2+ = conditioned)
            return_logits: if True, also return lm_head logits (for PPL)
            side_signals: optional {neuron_id: vector} for side-channel communication

        Returns:
            dict with keys:
            - field_vector: [B, D] L2-normalised write vector
            - hidden_before_write: [B, hidden] for diversity loss
            - logits: [B, L, vocab] (only if return_logits=True)
        """
        # ── Step 1: Embedding adapter (shared base → neuron concept space) ──
        h = self.embed_adapter(shared_embeddings)  # [B, L, hidden]

        # ── Step 2: Transformer layers + field conditioning ──
        bsz, seqlen, _ = h.shape

        for i, block in enumerate(self.layers):
            # Standard transformer block (layers.py, zero changes)
            h_normed = block.attention_norm(h)
            attn_out, _ = block.attention(h_normed)
            h = h + attn_out
            h = h + block.feed_forward(block.ffn_norm(h))

            # Field conditioning (round 2+ only)
            if field_state is not None and round_num > 1:
                conditioning = self.field_read_layers[i](field_state)  # [D] → [hidden]
                if conditioning.dim() == 1:
                    conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1,1,H] -> all B,L
                else:
                    conditioning = conditioning.unsqueeze(1)  # [B,1,H] -> over seq
                if self.v1_compat:
                    # v1: broadcast same conditioning to all positions
                    h = h + conditioning
                else:
                    # v2: per-position gated read
                    gate = torch.sigmoid(self.field_read_gate(h))  # [B, L, 1]
                    h = h + gate * conditioning

            # Side channel injection（人脑启发：兴奋/抑制双通道调制）
            # excite: 正向残差，h += 0.1 * excite_proj(signal)
            # inhibit: 负向残差，h -= 0.1 * inhibit_proj(signal)
            # 双通道并存使神经元能学习"何时被 peer 激活、何时被 peer 抑制"
            if side_signals is not None:
                for src_id, signal in side_signals.items():
                    key = str(src_id)
                    if key in self.excite_channels:
                        h = h + 0.1 * self.excite_channels[key](signal)
                    if key in self.inhibit_channels:
                        h = h - 0.1 * self.inhibit_channels[key](signal)

        # ── Step 3: Final norm ──
        h = self.norm(h)

        # ── Step 4: Field write ──
        # 人脑启发：抑制性神经元的 field_vector 取反，对场做负向贡献
        # 这样 field.write(nid, v) 时 v 已携带正负号，field 无需感知 neuron_type
        if self.v1_compat:
            # v1: last-token write (matches old checkpoint training distribution)
            hidden_last = h[:, -1, :]  # [B, hidden]
            v_raw = self.field_write(hidden_last)  # [B, D]
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            if self.is_inhibitory:
                v = -v  # 抑制性神经元取反
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": hidden_last,
            }
        else:
            # v2: attention-pooled field write
            attn_scores = torch.matmul(h, self.field_pool_query) * self.field_pool_scale  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L]
            pooled = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # [B, hidden]
            v_raw = self.field_write(pooled)  # [B, D]
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            if self.is_inhibitory:
                v = -v  # 抑制性神经元取反
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": pooled,
                "field_attn_weights": attn_weights,
            }

        # ── Step 5: Optional logits (for PPL evaluation) ──
        if return_logits:
            result["logits"] = self.compute_logits(h)  # [B, L, vocab]

        return result

    def freeze_fingerprint(self) -> None:
        """Compute and freeze the direction fingerprint.

        Fingerprint = L2-normalised mean of field_write weight rows.
        Used for lightweight prescreening (P1).
        """
        with torch.no_grad():
            fp = self.field_write.weight.mean(dim=0)  # [hidden]
            self.fingerprint.copy_(fp / (fp.norm() + 1e-8))

    def establish_side_channel(
        self,
        peer_id: int,
        channel_type: str = "excite",
    ) -> None:
        """Create a side channel to a frequently co-active peer neuron.

        人脑启发：每个 peer 关系可同时拥有兴奋性和抑制性通道，
        由 STDP 学习决定哪种占主导。

        Args:
            peer_id: peer 神经元 ID
            channel_type: "excite"（兴奋性，正向调制）或 "inhibit"（抑制性，负向调制）
        """
        key = str(peer_id)
        target_dict = self.excite_channels if channel_type == "excite" else self.inhibit_channels
        if key not in target_dict:
            target_dict[key] = nn.Linear(
                self.config.field_dim, self.config.hidden_size, bias=False
            )

    @torch.no_grad()
    def select_top_k_peers(
        self,
        peer_fingerprints: Dict[int, torch.Tensor],
        k: int = 8,
        sim_range: tuple = (0.3, 0.7),
    ) -> list:
        """选择 top-K 互补 peer 建立侧信道（人脑启发：稀疏连接）。

        人脑每个神经元约 10^4 突触而非全连接，态极类比限制为 top-K。
        选择标准：fingerprint cosine 相似度在 [sim_range] 区间的 peer
        （太相似=冗余，太正交=无关，中间=互补）。

        Args:
            peer_fingerprints: {peer_id: fingerprint_tensor}
            k: 最多选择 K 个 peer
            sim_range: 互补相似度区间

        Returns:
            选中的 peer_id 列表
        """
        if self.fingerprint.norm() < 1e-8:
            return []  # fingerprint 未冻结

        fp_norm = self.fingerprint / (self.fingerprint.norm() + 1e-8)
        candidates = []
        for pid, pfp in peer_fingerprints.items():
            if pfp.norm() < 1e-8:
                continue
            pfp_norm = pfp / (pfp.norm() + 1e-8)
            sim = float(torch.dot(fp_norm, pfp_norm).item())
            if sim_range[0] <= sim <= sim_range[1]:
                candidates.append((pid, sim))

        # 按相似度降序选 top-K（更接近上界的更"互补"）
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in candidates[:k]]

    def establish_top_k_channels(
        self,
        peer_fingerprints: Dict[int, torch.Tensor],
        k: int = 8,
        sim_range: tuple = (0.3, 0.7),
    ) -> list:
        """自动选择 top-K peer 并建立兴奋/抑制双通道。"""
        selected = self.select_top_k_peers(peer_fingerprints, k, sim_range)
        for pid in selected:
            self.establish_side_channel(pid, "excite")
            self.establish_side_channel(pid, "inhibit")
        return selected

    def prune_weak_channels(self, threshold: float = 0.01) -> int:
        """修剪弱 side_channels（人脑启发：用进废退）。

        检查每个 channel 的权重范数，低于 threshold 的删除。
        返回修剪数量。
        """
        pruned = 0
        for channel_dict in [self.excite_channels, self.inhibit_channels]:
            to_remove = []
            for key, linear in channel_dict.items():
                if linear.weight.norm().item() < threshold:
                    to_remove.append(key)
            for key in to_remove:
                del channel_dict[key]
                pruned += 1
        return pruned

    @torch.no_grad()
    def quick_probe(self, shared_embeddings: torch.Tensor) -> torch.Tensor:
        """Lightweight forward pass for prescreening (skip full Transformer).

        Runs only the adapter + field_write, no Transformer layers.
        Returns a rough field_vector direction for candidate selection.
        """
        h = self.embed_adapter(shared_embeddings)
        # Use mean pooling over sequence as a rough representation
        h_pooled = h.mean(dim=1)  # [B, hidden]
        v_raw = self.field_write(h_pooled)
        return v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
