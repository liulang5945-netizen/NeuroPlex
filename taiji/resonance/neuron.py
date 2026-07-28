"""Resonance neuron — wraps a backbone with field I/O.

Each neuron:
1. Receives shared base embeddings (Level 1: shared 256K → 512)
2. Projects through its own adapter into a per-neuron concept space
3. Processes through a standard Transformer (layers.py, zero changes)
4. Writes a normalised field vector (Level 2 → Level 3)
5. Reads field state for conditioning (Level 3 → Level 2)

注意：neuron 不再拥有独立的 nn.Embedding。所有 neuron 共享
一张外部 nn.Embedding(256000, 512)（Layer 1 共享感官层）。
TokenTranslator 将域 tokenizer 的输出映射到通用 token ID，
再查共享嵌入表。
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
    - domain_prototype: EMA-updated data-driven典型响应向量 (for L2 prototype routing)
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

        # ── Embedding adapter (shared base → per-neuron concept space) ──
        # neuron 不再拥有独立 embedding 表。所有 neuron 共享一张
        # nn.Embedding(256000, 512)（Layer 1 共享感官层），由外部传入。
        # embed_adapter 是 per-neuron 的独立映射，保留神经元个性。
        self.embed_adapter = nn.Linear(c.base_embed_dim, c.hidden_size, bias=False)

        # ── 多模态投影层（P8 预留）──
        # 非文本模态（图像 patch / 音频 frame）的连续特征投影到 base_embed_dim，
        # 再走 embed_adapter → Transformer body（与文本路径共用 forward）。
        # 每个模态独立投影层，避免模态间干扰。
        # text 模态不需要投影（外部 shared_embedding 已完成查表）。
        # 离散 token id（VQ-VAE codebook 索引）走外部 shared_embedding。
        self.mm_projections = nn.ModuleDict()  # {modality: Linear(raw_dim, base_embed_dim)}
        # 多模态输出头：每个模态独立 lm_head，映射到 codec vocab（codebook size）。
        # 与文本 lm_head 分离，避免模态间词汇干扰。
        self.mm_lm_heads = nn.ModuleDict()  # {modality: Linear(hidden_size, codec_vocab_size)}

        # ── Transformer body (reuses layers.py, zero changes) ──
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_attention_heads,
                num_kv_heads=c.num_key_value_heads,
                intermediate_size=c.intermediate_size,
                rms_norm_eps=c.rms_norm_eps,
                bias=c.attention_bias,
                dropout=c.dropout,
            )
            for _ in range(c.num_hidden_layers)
        ])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

        # ── Field write projection ──
        self.field_write = nn.Linear(c.hidden_size, c.field_dim, bias=False)

        # 突触投影（Field Projector）：不同规格 field_dim → 统一场空间
        # 模拟人脑突触可塑性（LTP/LTD）：不同类型神经元通过突触连接到统一网络
        # None 或 == field_dim 时为 Identity（向后兼容，不影响现有训练）
        effective_field_dim = c.unified_field_dim if c.unified_field_dim is not None else c.field_dim
        if c.unified_field_dim is not None and c.unified_field_dim != c.field_dim:
            self.field_projector = nn.Linear(c.field_dim, c.unified_field_dim, bias=False)
        else:
            self.field_projector = nn.Identity()

        # Attention pooling for field write (v2)
        # Learn a query that pools over all positions instead of only using the last token.
        # This surfaces the most salient conceptual content for field communication.
        self.field_pool_query = nn.Parameter(torch.randn(c.hidden_size) * 0.02)
        self.field_pool_scale = c.hidden_size ** -0.5

        # ── Field read projections (one per layer, for conditioning) ──
        self.field_read_layers = nn.ModuleList([
            nn.Linear(effective_field_dim, c.hidden_size, bias=False)
            for _ in range(c.num_hidden_layers)
        ])

        # Position gate for field read (v2)
        # Each position decides how much field conditioning to absorb,
        # replacing the old broadcast (same vector to all positions).
        self.field_read_gate = nn.Linear(c.hidden_size, 1, bias=True)

        # ── Language modelling head ──
        # P7: 每 neuron 自带完整独立 lm_head [hidden, domain_vocab]
        # 域专用 vocab (10k-20k) 让独立 lm_head 参数量可控 (5-10M)
        # lm_head_rank > 0 保留用于实验性低秩训练（非共享，per-neuron only）
        if c.lm_head_rank > 0:
            # 低秩模式：U_i/V_i 是 per-neuron 低秩分解（实验性）
            self.lm_head_delta_u = nn.Linear(c.hidden_size, c.lm_head_rank, bias=False)
            self.lm_head_delta_v = nn.Linear(c.lm_head_rank, c.vocab_size, bias=False)
            nn.init.normal_(self.lm_head_delta_u.weight, std=0.01)
            nn.init.normal_(self.lm_head_delta_v.weight, std=0.01)
        else:
            # P7 默认：per-neuron 独立 lm_head
            self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=c.hidden_size ** -0.5)

        # ── Domain prototype (EMA updated, for L2 prototype routing) ──
        # 数据驱动典型响应向量，768 维 hidden_size 空间
        # 训练时每轮 EMA 更新，推理时用 cosine 相似度做轻量路由
        self.register_buffer("domain_prototype", torch.zeros(c.hidden_size))
        self.register_buffer("proto_ema_decay", torch.tensor(0.99))

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

    def establish_side_channel(
        self,
        peer_id: str,
        peer_neuron: "ResonanceNeuron",
        channel_type: str = "excite",
        init_std: float = 0.01,
    ):
        """建立一条指向 peer 神经元的突触通道。

        Args:
            peer_id: peer 神经元标识。
            peer_neuron: peer 神经元实例，用于读取其 field_dim。
            channel_type: "excite" 或 "inhibit"。
            init_std: 通道权重初始化标准差。
        """
        src_dim = peer_neuron.config.field_dim
        dst_dim = self.config.hidden_size
        channel = nn.Linear(src_dim, dst_dim, bias=False)
        nn.init.normal_(channel.weight, std=init_std)

        if channel_type == "excite":
            self.excite_channels[peer_id] = channel
        elif channel_type == "inhibit":
            self.inhibit_channels[peer_id] = channel
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

    def compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        """计算 logits。

        P7 默认 (lm_head_rank=0): 独立 lm_head 直出。
        实验性 (lm_head_rank>0): per-neuron 低秩分解 V_i(U_i(h))。
        """
        if self.config.lm_head_rank > 0:
            return self.lm_head_delta_v(self.lm_head_delta_u(h))
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

    def register_modality_projection(self, modality: str, raw_dim: int) -> None:
        """P8: 注册非文本模态的投影层。

        图像/音频等连续特征需先投影到 base_embed_dim，再走 embed_adapter → Transformer。
        每个 neuron 对每个模态有独立投影层（per-neuron 个性）。

        Args:
            modality: 模态名（"image"/"audio"/"video"）。
            raw_dim: 原始特征维度（如 VQ-VAE codebook dim、EnCodec frame dim）。
        """
        if modality == "text":
            return  # text 模态走外部 shared_embedding，不需要投影
        if modality in self.mm_projections:
            return  # 已注册，幂等
        self.mm_projections[modality] = nn.Linear(raw_dim, self.config.base_embed_dim, bias=False)
        nn.init.normal_(self.mm_projections[modality].weight, std=self.config.base_embed_dim ** -0.5)

    def register_modality_lm_head(self, modality: str, vocab_size: int) -> None:
        """P8: 注册非文本模态的输出头（lm_head）。

        每个模态独立输出头，映射到 codec vocab（codebook size）。
        与文本 lm_head 分离，避免模态间词汇干扰。

        Args:
            modality: 模态名（"image"/"audio"/"video"）。
            vocab_size: codec codebook 大小。
        """
        if modality == "text":
            return  # text 模态走自带的 lm_head
        if modality in self.mm_lm_heads:
            return  # 已注册，幂等
        self.mm_lm_heads[modality] = nn.Linear(
            self.config.hidden_size, vocab_size, bias=False,
        )
        nn.init.normal_(self.mm_lm_heads[modality].weight,
                        std=self.config.hidden_size ** -0.5)

    def auto_register_modalities(self, tokenizer_hub) -> None:
        """P8: 自动注册所有已注册到 TokenizerHub 的非文本模态。

        从 TokenizerHub 获取所有模态编码器，自动注册投影层和输出头。
        新增模态或新增 neuron 时无需手动修改代码。

        Args:
            tokenizer_hub: TokenizerHub 实例
        """
        for modality in tokenizer_hub.list_modalities():
            encoder = tokenizer_hub.modal_encoders.get(modality)
            if encoder is None:
                continue

            # 获取 latent_dim（codebook 维度）
            latent_dim = 256
            if hasattr(encoder, "model") and hasattr(encoder.model, "quantizer"):
                if hasattr(encoder.model.quantizer, "codebook"):
                    latent_dim = encoder.model.quantizer.codebook.weight.shape[-1]

            # 获取 vocab_size（codebook size）
            vocab_size = encoder.vocab_size()

            # 注册投影层和输出头
            self.register_modality_projection(modality, raw_dim=latent_dim)
            self.register_modality_lm_head(modality, vocab_size=vocab_size)

    def compute_mm_logits(self, h: torch.Tensor, modality: str) -> torch.Tensor:
        """P8: 计算指定模态的 logits。

        Args:
            h: [B, L, hidden_size] transformer 输出
            modality: 模态名

        Returns:
            logits: [B, L, vocab_size]
        """
        if modality == "text":
            return self.compute_logits(h)
        if modality not in self.mm_lm_heads:
            raise ValueError(
                f"模态 '{modality}' 未注册输出头，请先调用 register_modality_lm_head"
            )
        return self.mm_lm_heads[modality](h)

    def encode_multimodal_input(
        self,
        features: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """P8: 把非文本模态的连续特征编码为 shared_emb.

        与外部 shared_embedding 输出同构（[B, L, base_embed_dim]），可直接送入 forward()。
        需先调用 register_modality_projection(modality, raw_dim) 注册投影层。

        支持连续特征输入：
        - 连续特征 [B, L, raw_dim] float → mm_projections[modality] 投影

        注意：离散 token id（VQ-VAE codebook 索引）应走外部 shared_embedding，
        不再由 neuron 内部处理。

        Args:
            features: [B, L, raw_dim] float (连续特征)
            modality: 模态名（"image"/"audio"/"video"）

        Returns:
            shared_embeddings: [B, L, base_embed_dim]
        """
        if modality == "text":
            raise ValueError("text 模态请使用外部 shared_embedding(general_ids)")

        # 连续特征路径：投影到 base_embed_dim
        if features.dim() != 3:
            raise ValueError(
                f"多模态输入应为 [B, L, raw_dim] float，got {features.shape}"
            )

        if modality not in self.mm_projections:
            raise ValueError(
                f"模态 '{modality}' 未注册投影层，请先调用 register_modality_projection('{modality}', raw_dim)"
            )
        proj = self.mm_projections[modality]
        return proj(features.float())

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        field_state: Optional[torch.Tensor] = None,
        round_num: int = 1,
        return_logits: bool = False,
        side_signals: Optional[Dict[int, torch.Tensor]] = None,
        mm_logits_modality: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the neuron.

        Args:
            shared_embeddings: [B, L, base_embed_dim] from the shared embedding table
                               (外部 shared_embedding(general_ids) 生成)
            field_state: [D] current field state vector (from round 2 onward)
            round_num: current resonance round (1 = independent, 2+ = conditioned)
            return_logits: if True, also return lm_head logits (for PPL)
            side_signals: optional {neuron_id: vector} for side-channel communication
            mm_logits_modality: if set, 返回该模态的 lm_head logits（而非文本 lm_head）。
                                优先级高于 return_logits（return_logits 文本，mm_logits_modality 多模态）。

        Returns:
            dict with keys:
            - field_vector: [B, D] L2-normalised write vector
            - hidden_before_write: [B, hidden] for diversity loss
            - logits: [B, L, vocab] (only if return_logits=True or mm_logits_modality set)
        """
        # ── Step 1: Embedding adapter (shared base → neuron concept space) ──
        h = self.embed_adapter(shared_embeddings)  # [B, L, hidden]

        # ── Step 2: Transformer layers + field conditioning ──
        bsz, seqlen, _ = h.shape

        # Causal mask（修复双向注意力 bug）:
        # GQA.is_causal = (mask is not None) and (seqlen > 1)
        # 之前没传 mask → is_causal=False → 双向注意力 → 训练时偷看未来 token
        # → teacher-forcing 虚高（100%）、自回归崩溃（死循环重复）
        # 修复：传标准下三角 causal mask，确保位置 K 只看到 0..K
        if seqlen > 1:
            causal_mask = torch.full(
                (1, 1, seqlen, seqlen), float('-inf'),
                device=h.device, dtype=h.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
        else:
            causal_mask = None

        for i, block in enumerate(self.layers):
            # Standard transformer block (layers.py, zero changes)
            h_normed = block.attention_norm(h)
            attn_out, _ = block.attention(h_normed, mask=causal_mask)
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

        # ── Step 3: Side-channel modulation (per-pair synaptic projection) ──
        # side_signals: {peer_id: peer_field_vector [B, peer_field_dim]}
        # 通过 excite/inhibit_channels 把 peer 的 field_vector 投影到本神经元 hidden 空间，
        # 以乘性 gate 调制 h（加性调制会被 LayerNorm 抵消，乘性不会）。
        if side_signals:
            excite_sum = None
            inhibit_sum = None
            for peer_id, sig in side_signals.items():
                if peer_id in self.excite_channels:
                    proj = self.excite_channels[peer_id](sig)  # [B, hidden_size]
                    excite_sum = proj if excite_sum is None else excite_sum + proj
                if peer_id in self.inhibit_channels:
                    proj = self.inhibit_channels[peer_id](sig)
                    inhibit_sum = proj if inhibit_sum is None else inhibit_sum + proj

            # 乘性调制：gate = 1 + tanh(proj)，初始 ≈ 1.0（不改变基线）
            # excite 增强维度（gate > 1），inhibit 抑制维度（gate < 1）
            if excite_sum is not None:
                gate = 1.0 + torch.tanh(excite_sum.unsqueeze(1))  # [B, L, hidden]
                h = h * gate
            if inhibit_sum is not None:
                gate = 1.0 - torch.tanh(inhibit_sum.unsqueeze(1))  # [B, L, hidden]
                h = h * gate

        # ── Step 4: Final norm ──
        h = self.norm(h)

        # ── Step 5: Field write ──
        # P0#3: 不再对抑制性神经元取反（v=-v）。
        # field_vector 始终为正方向；抑制效果由 field.write_inhibit() 的
        # 乘法掩码实现（divisive inhibition，GABA-like）。
        if self.v1_compat:
            # v1: last-token write (matches old checkpoint training distribution)
            hidden_last = h[:, -1, :]  # [B, hidden]
            v_raw = self.field_write(hidden_last)  # [B, field_dim]
            v_raw = self.field_projector(v_raw)  # [B, effective_field_dim] 突触投影
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": hidden_last,
            }
        else:
            # v2: attention-pooled field write
            attn_scores = torch.matmul(h, self.field_pool_query) * self.field_pool_scale  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L]
            pooled = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # [B, hidden]
            v_raw = self.field_write(pooled)  # [B, field_dim]
            v_raw = self.field_projector(v_raw)  # [B, effective_field_dim] 突触投影
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": pooled,
                "field_attn_weights": attn_weights,
            }

        # ── Step 5: Optional logits (for PPL evaluation) ──
        if return_logits:
            result["logits"] = self.compute_logits(h)  # [B, L, vocab]
        if mm_logits_modality is not None:
            result["logits"] = self.compute_mm_logits(h, mm_logits_modality)  # [B, L, mm_vocab]

        return result

    def update_domain_prototype(self, pooled: torch.Tensor) -> None:
        """用真实处理样本后的 hidden state 更新 domain prototype（EMA）。

        Prototype 是数据驱动的典型响应向量，非权重统计量。
        训练时每轮调用，EMA 平滑跟踪 neuron 的典型响应模式。
        """
        with torch.no_grad():
            target = pooled.detach().mean(dim=0) if pooled.dim() == 2 else pooled.detach()
            self.domain_prototype.mul_(self.proto_ema_decay.item()).add_(
                target, alpha=(1.0 - self.proto_ema_decay.item())
            )
            # 归一化保持单位球面
            self.domain_prototype.div_(self.domain_prototype.norm() + 1e-8)

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
        v_raw = self.field_projector(v_raw)  # 突触投影
        return v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
