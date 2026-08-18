"""Neuron specification configurations for the resonance field architecture.

Three standard specs (compact / standard / expert), each designed
as the smallest configuration that can independently understand
text while staying trainable on CPU (see docs Chapter 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class NeuronConfig:
    """Configuration for a single resonance neuron."""

    # ── Transformer body ──
    hidden_size: int = 768
    num_hidden_layers: int = 10
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2304
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    # 正则化：dropout 防止过拟合（社区规范：0.1 为小模型常用值）
    # 0.0 = 关闭（向后兼容旧 ckpt）
    dropout: float = 0.0

    # ── S11: 长上下文（attention sink + 滑动窗口）──
    # StreamingLLM 启发：保留前 K 个 token 的 KV 作为"注意力锚点"，
    # 滑动窗口处理新 token，实现近 O(1) 推理时长上下文。
    # - attention_sink_size: 保留的前 K 个 sink token（典型 4）
    # - sliding_window_size: 滑动窗口大小（典型 1024-2048）
    # - 两者之和为 KV cache 最大长度（sink_size + window_size）
    # - 0 = 关闭（向后兼容，KV cache 无限增长直到显存溢出）
    attention_sink_size: int = 0
    sliding_window_size: int = 0

    # ── C4: 场读入模式（field_read_mode）──
    # 控制 field_state 如何调制神经元内部表示：
    # - "additive": h = h + gate * conditioning（加性残差，向后兼容默认）
    # - "multiplicative": h = h * (1 + gate * tanh(conditioning))（乘性门控，场调制表示方向）
    # - "predictive": h = h - gate * (h - conditioning)（预测编码，场作为自上而下预测）
    #
    # 乘性门控让 field_state 能真正调制表示的幅度/方向，而非仅加性偏移。
    # 预测编码与 S10 树突化 apical 路径模式一致（误差驱动校正）。
    field_read_mode: str = "additive"

    # ── C6: 多头 field write（num_field_heads）──
    # 场写入的语义切面数量。当前单 query 只能写入一个语义切面（如"主题"），
    # 多头让每个 neuron 同时写入多个语义维度（"主题"+"情感"+"结构"）。
    #
    # - 1 = 单 query attention pooling（向后兼容默认，走原 v2 路径）
    # - K>1 = K 个独立 query + K 个独立 field_write 投影 + 门控聚合
    #
    # 多头机制：
    # 1. K 个 query 各自 attention pooling → K 个 pooled 向量（捕捉不同语义切面）
    # 2. K 个独立 field_write 线性投影 → K 个 raw field 向量（每个 head 学不同写入方向）
    # 3. softmax 门控聚合 → 单个 field 向量（模型动态选择每个切面权重）
    #
    # 参数量：K=1 时不变；K>1 时增加 (K-1)×hidden×field_dim + K×hidden（gate）
    # 上限提升：场信息带宽 ×K，与 C7 空间扩散协同（更丰富的场写入→更有意义的扩散）
    num_field_heads: int = 1

    # ── Embedding (per-neuron, domain-specific tokenizer) ──
    # P7: 每 neuron 独立 embedding + 独立 lm_head，域专用 vocab
    # vocab_size 由域 tokenizer 决定（zh=20k, en=16k, code=12k, math=10k）
    # 默认 256000 仅用于兼容旧 ckpt；新 neuron 应显式传入域 vocab
    vocab_size: int = 256000
    base_embed_dim: int = 512

    # ── Field interface ──
    field_dim: int = 4096
    # 突触投影（Field Projector）：不同规格 field_dim 通过可学习投影层映射到统一场空间
    # 模拟人脑突触可塑性（LTP/LTD）：不同类型神经元通过突触连接到统一网络
    # None = 不投影（向后兼容，field_dim 即场维度）
    # 设为统一值（如4096）时，field_projector: Linear(field_dim → unified_field_dim)
    unified_field_dim: Optional[int] = None

    # ── C12: 评分投影（score_dim）──
    # 共振分数可比性修复：大神经元（shared_expert）系统性主导场状态方向，
    # 导致小神经元 cosine 评分偏低，共振退化成"大神经元主导"。
    #
    # 评分空间与写入空间分离：
    # - neuron 加 score_proj: Linear(field_dim → score_dim)，输出 score_vec
    # - ensemble 持有 field_score_proj: Linear(field_dim → score_dim)，投影场状态
    # - 评分 = cosine(score_vec, field_score_proj(loo_state))
    #
    # 投影头通过 CE loss 学习（scores → weights → fused_logits → CE），
    # balance_loss 约束公平性（防止投影头退化成大神经元偏向）。
    # 可选 contrastive_loss（forward_train 传入 targets 时）：共振分与 NLL 排序对齐。
    #
    # - None = 不投影（向后兼容，评分用原始 field_vector cosine）
    # - int = 评分投影维度（推荐 256，参数量 5×4096×256≈5.2M + 1×4096×256≈1M）
    score_dim: Optional[int] = None

    # ── C5: 多原型混合（num_prototypes）──
    # 单 EMA 原型只能跟踪单一模式，多原型覆盖多子分布。
    # - 1 = 单 EMA 原型（向后兼容，原 domain_prototype 行为）
    # - K>1 = K 个原型 + 在线聚类（胜者 EMA 更新，路由取 max cosine）
    # 上限提升：domain 内多主题/多风格数据时，单原型会"模糊"，
    # 多原型让每个子模式有独立代表，路由更精确。
    num_prototypes: int = 1

    # ── Domain extension (0 = disabled) ──
    num_domain_concepts: int = 0

    # ── Metadata ──
    spec: str = "standard"
    neuron_id: Optional[str] = None

    # ── 神经元类型（人脑启发：兴奋性/抑制性分化）──
    # excitatory: 默认，对场做正向贡献（类比谷氨酸能）
    # inhibitory: 对场做负向贡献，抑制过度共振（类比 GABA 能）
    # 约 20% 神经元应为 inhibitory，由 CoactivationTracker 自动转化过度兴奋的神经元
    # C1: 扩展为多亚型（生物学启发）
    # - "excitatory"（默认，向后兼容）: 标准兴奋性，正向场写入
    # - "excitatory_pv": PV+ 快速放电，强但短促场写入（高 gain + 短不应期）
    # - "excitatory_som": SOM+ 抑制树突，定向调制（写入 + 弱抑制近邻）
    # - "excitatory_vip": VIP+ 去抑制，解除 SOM 抑制（写入 + 增强 VIP 目标）
    # - "inhibitory": 标准抑制性，负向场写入（write_inhibit）
    # 上限提升：不同亚型有不同场写入行为，更丰富的协作动态
    neuron_type: Literal[
        "excitatory", "excitatory_pv", "excitatory_som", "excitatory_vip", "inhibitory"
    ] = "excitatory"

    # ── 不应期配置（人脑启发：refractory period）──
    # 写入场后进入不应期，rounds_cooldown 轮内只能读场不能写
    # 防止强神经元垄断场，强制信息轮替
    refractory_cooldown: int = 2

    # ── lm_head 配置 ──
    # P7: 默认 lm_head_rank=0（独立 lm_head，从零训练）
    # 每 neuron 自带完整 lm_head [hidden, vocab]，不共享 W_base
    # 域专用 vocab (10k-20k) 让独立 lm_head 参数量可控 (5-10M)
    # lm_head_rank > 0 保留用于实验性低秩训练（非共享）
    lm_head_rank: int = 0

    # ── S10: 树突化（Dendritic）配置 ──
    # 人脑启发：锥体神经元 basal/apical 树突分离
    # - basal: 标准 attention + FFN（自下而上，处理输入）
    # - apical: cross-attention（Q=x, KV=field_state，自上而下反馈）
    # - 胞体整合: 预测编码（error = basal - apical_prediction）
    # False = 标准 Transformer 块（向后兼容，默认）
    # True = 树突化块，apical 路径接收 field_state 做独立计算
    dendritic_enabled: bool = False
    # apical cross-attention 的 KV 来源维度（= field_dim，由 neuron 构建时传入）
    # None 时用 field_dim（向后兼容）
    apical_kv_dim: Optional[int] = None

    # ── Approximate parameter count (excluding shared embedding) ──
    @property
    def approx_params_m(self) -> float:
        """Rough parameter count in millions (transformer body + field projections)."""
        d = self.hidden_size
        n = self.num_hidden_layers
        h = self.num_attention_heads
        # Per-layer: 4*(d^2) for Q/K/V/O (GQA saves on K/V) + 3*(d*intermediate) for SwiGLU
        kv_ratio = self.num_key_value_heads / self.num_attention_heads
        attn_params = d * d * (2 + 2 * kv_ratio)  # Q(1) + K(kv_ratio) + V(kv_ratio) + O(1)
        ffn_params = 3 * d * self.intermediate_size
        per_layer = attn_params + ffn_params
        # Norm params are negligible
        body = n * per_layer
        # Embed adapter: base_embed_dim -> hidden_size
        adapter = self.base_embed_dim * d
        # Field write: hidden_size -> field_dim
        field_w = d * self.field_dim
        # Field read: field_dim -> hidden_size per layer
        field_r = n * self.field_dim * d
        # LM head: hidden_size -> vocab_size (per-neuron, not shared)
        lm_head = d * self.vocab_size
        total = body + adapter + field_w + field_r + lm_head
        return total / 1_000_000


# ── Standard specs ──

COMPACT = NeuronConfig(
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1536,
    spec="compact",
    # P8: field_dim proportional to hidden_size (512 × 4 = 2048)
    # Reduces communication overhead from 27.5% to 16%,
    # within the 20-33% target range for all specs.
    field_dim=2048,
)

STANDARD = NeuronConfig(
    hidden_size=768,
    num_hidden_layers=10,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2304,
    spec="standard",
    field_dim=3072,  # P8: proportional to hidden_size (20% communication, within 20-33% target)
)


FOUNDATION = NeuronConfig(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    num_key_value_heads=2,
    intermediate_size=1152,
    spec='foundation',
    # H9: unified field_dim=4096 across all v3 neurons.
    field_dim=4096,
)

EXPERT = NeuronConfig(
    hidden_size=1024,
    num_hidden_layers=14,
    num_attention_heads=16,
    num_key_value_heads=4,
    intermediate_size=3072,
    spec="expert",
    # Existing code/math checkpoints use field_dim=4096.
    field_dim=4096,
)

# ── Tiny test config (for smoke-testing the code) ──

TINY_TEST = NeuronConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    field_dim=512,
    spec="tiny_test",
    # 注意：TINY_TEST 是唯一 field_dim != 4096 的 spec，仅用于单元测试，
    # 严禁用于生产 cortex（会触发 cortex.py 的 field_dim 一致性错误）。
)


# ============================================================================
# 全局默认 spec（神经新生 / fallback / 蒸馏的统一入口）
# ============================================================================
# 所有新创建的生产 neuron 都应使用此 spec，避免硬编码不一致。
# 修改时同步：scripts/training/train_neuron.py
DEFAULT_NEURON_SPEC = "compact"  # COMPACT: hidden=512, layers=6, ~85M/neuron


# ============================================================================
# P7: 域专用 tokenizer vocab 映射
# ============================================================================
# 每 neuron 用域专用 tokenizer，独立 embedding + 独立 lm_head
# vocab 大小由域 tokenizer 决定（taiji/domains/{domain}/sp_{domain}.model）
# general 域复用 en tokenizer（16k），避免重新训 tokenizer
# T12: zh vocab 20K → 50K（词表库热插拔升级，upgrade_tokenizer.py 训练 + hot_swap_vocab.py 迁移 ckpt）
DOMAIN_VOCAB_SIZES = {
    "zh": 50000,
    "en": 16000,
    "code": 12000,
    "math": 10000,
    "general": 16000,  # general 复用 en tokenizer
}

# general 域实际使用的 tokenizer 名（用于 tokenizer 加载）
GENERAL_TOKENIZER_DOMAIN = "en"


def get_default_neuron_config(spec: str = None) -> "NeuronConfig":
    """根据 spec 名称返回 NeuronConfig 实例。

    Args:
        spec: spec 名称 ("compact" / "foundation" / "standard" / "expert")
              None 表示使用 DEFAULT_NEURON_SPEC

    Returns:
        对应的 NeuronConfig 实例（不是引用，是独立实例）
    """
    if spec is None:
        spec = DEFAULT_NEURON_SPEC
    spec_map = {
        "compact": COMPACT,
        "foundation": FOUNDATION,
        "standard": STANDARD,
        "expert": EXPERT,
    }
    if spec not in spec_map:
        raise ValueError(
            f"未知 spec: {spec}. 可选: {list(spec_map.keys())}. "
            f"默认: {DEFAULT_NEURON_SPEC}"
        )
    # 返回独立副本，避免修改全局常量
    base = spec_map[spec]
    from dataclasses import replace
    return replace(base)


def get_domain_neuron_config(domain: str, spec: str = None) -> "NeuronConfig":
    """P7: 返回域专用 NeuronConfig（vocab_size 对齐域 tokenizer）。

    Args:
        domain: 域名 ("zh" / "en" / "code" / "math" / "general")
        spec: spec 名称（None 用 DEFAULT_NEURON_SPEC）

    Returns:
        NeuronConfig 实例，vocab_size 已设置为域 tokenizer 大小
    """
    if domain not in DOMAIN_VOCAB_SIZES:
        raise ValueError(
            f"未知 domain: {domain}. 可选: {list(DOMAIN_VOCAB_SIZES.keys())}"
        )
    cfg = get_default_neuron_config(spec)
    cfg.vocab_size = DOMAIN_VOCAB_SIZES[domain]
    cfg.neuron_id = domain
    return cfg
