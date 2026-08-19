"""Resonance neuron — wraps a backbone with field I/O.

Each neuron:
1. Receives shared base embeddings (Level 1: shared 256K → 512)
2. Projects through its own adapter into a per-neuron concept space
3. Processes through a standard Transformer (layers.py, zero changes)
4. Writes a normalised field vector (Level 2 → Level 3)
5. Reads field state for conditioning (Level 3 → Level 2)

注意：neuron 不再拥有独立的 nn.Embedding。所有 neuron 共享
一张外部 nn.Embedding(256000, 512)（Layer 1 共享感官层）。
P7 对齐通过 build_position_alignment（字符 span 重叠）实现
general/domain token 映射，再查共享嵌入表。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.layers import RMSNorm, TransformerBlock

from .config import NeuronConfig


class LoraPair(nn.Module):
    """低秩适配对（C16，2026-08-08）：y += scale * BAx。

    B 初始化为 0 → LoRA 分支初始输出恒 0 → 个体生成能力零破坏起点；
    训练中低秩增量 ΔW = scale·BA 适配协作（collab 训练不再直接微调 body）。
    """

    def __init__(self, in_dim: int, rank: int, alpha: Optional[float] = None):
        super().__init__()
        self.rank = rank
        # alpha/r 风格缩放（alpha 默认 = rank，即 scale=1.0，可调）
        self.scale = (alpha if alpha is not None else rank) / rank
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, in_dim, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x)) * self.scale


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

    def __init__(self, neuron_config: NeuronConfig, shared_lm_head: Optional[nn.Linear] = None):
        super().__init__()
        self.config = neuron_config
        c = neuron_config

        # ── C24（2026-08-09）：判定头（general 256K 空间）──
        # 双头架构：lm_head（域词表，生成）+ judge_lm_head（general 256K，判定）。
        # 背景：C20 判定 5/5 依赖所有 neuron 共享 general 256K 空间（投影 NLL 可比）；
        # C24 换域头后 native NLL 跨 neuron 不可比（en 16K 英文词表对英文回合 NLL 恒低
        # → quality_logit 膨胀常数头 → 判定退化）。双头方案：域头负责生成（C24 目标），
        # judge_lm_head 保留 general 空间判定信号（C20 信号链，天然可比）。
        # 由 loader/train 脚本注入（不在 config 中声明，避免破坏既有 ckpt 加载）。
        self.judge_lm_head: Optional[nn.Linear] = None
        # 判定空间统一化（2026-08-14）：判定头全局唯一（general 256K 空间，可比），
        # hidden≠判定空间维度（512）的 neuron（std0 768 / hub 1024）经 judge_proj
        # （hidden→512 小投影）适配后挂同一共享判定头——新 neuron 无论规格都获得
        # 可比判定能力，无需 per-neuron 131M 大判定头。None = 恒等（compact 512）。
        # 分工：judge_lm_head（标尺）全局唯一·冻结；judge_proj（翻译）每 neuron
        # 独立·可训练（协作/睡眠巩固阶段被共同塑造）。
        self.judge_proj: Optional[nn.Linear] = None

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
        # mm_lm_heads（独立 codebook 输出头）2026-08-07 已废弃——多模态输出统一走共享 general lm_head

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
                dendritic=c.dendritic_enabled,
                apical_kv_dim=(c.apical_kv_dim or c.field_dim) if c.dendritic_enabled else None,
                # S11: 长上下文 attention sink + 滑动窗口
                attention_sink_size=c.attention_sink_size,
                sliding_window_size=c.sliding_window_size,
            )
            for _ in range(c.num_hidden_layers)
        ])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.dendritic_enabled = c.dendritic_enabled
        # C4: 场读入模式
        self.field_read_mode = c.field_read_mode

        # ── C16: LoRA 适配器（collab 训练保护 body 个体能力）──
        # 启用后尾层 body 冻结，只训低秩增量 BA（B 初始 0 → 零破坏起点）。
        # 嵌套 ModuleDict：{str(层索引): ModuleDict({attn/ffn/blk: LoraPair})}
        self.lora_adapters = nn.ModuleDict()
        self.lora_enabled = False
        self.lora_layers: List[int] = []

        # ── Field write projection ──
        # C6: 多头 field write（num_field_heads > 1 时启用）
        # 单 query 只能写一个语义切面；多头让 neuron 同时表达"主题"+"情感"+"结构"等多个维度
        self.num_field_heads = c.num_field_heads
        self.field_pool_scale = c.hidden_size ** -0.5

        if c.num_field_heads > 1:
            # C6: 多头路径
            # K 个独立 query 捕捉不同语义切面
            self.field_pool_queries = nn.Parameter(
                torch.randn(c.num_field_heads, c.hidden_size) * 0.02
            )
            # K 个独立 field_write 投影（每个 head 学不同写入方向，强制多样性）
            self.field_write_heads = nn.ModuleList([
                nn.Linear(c.hidden_size, c.field_dim, bias=False)
                for _ in range(c.num_field_heads)
            ])
            # 门控聚合：从 pooled 特征动态选择每个 head 的权重
            self.field_gate = nn.Linear(c.hidden_size, c.num_field_heads, bias=True)
            # 保留 field_write=None 标记（兼容旧代码访问）
            self.field_write = None
        else:
            # 向后兼容：单 query v2 路径
            self.field_write = nn.Linear(c.hidden_size, c.field_dim, bias=False)
            self.field_pool_query = nn.Parameter(torch.randn(c.hidden_size) * 0.02)

        # 突触投影（Field Projector）：不同规格 field_dim → 统一场空间
        # 模拟人脑突触可塑性（LTP/LTD）：不同类型神经元通过突触连接到统一网络
        # None 或 == field_dim 时为 Identity（向后兼容，不影响现有训练）
        effective_field_dim = c.unified_field_dim if c.unified_field_dim is not None else c.field_dim
        if c.unified_field_dim is not None and c.unified_field_dim != c.field_dim:
            self.field_projector = nn.Linear(c.field_dim, c.unified_field_dim, bias=False)
        else:
            self.field_projector = nn.Identity()

        # ── C12: 评分投影头（score_dim）──
        # 评分空间与写入空间分离：neuron 的 field_vector 经 score_proj 投影到共享评分空间，
        # 与 field_score_proj(loo_state) 算 cosine。让大神经元对场状态方向的主导
        # 被 field_score_proj 抵消，小神经元能获得公平的共振分。
        # None = 不投影（向后兼容，评分用原始 field_vector）
        self.score_dim = c.score_dim
        if c.score_dim is not None:
            self.score_proj = nn.Linear(effective_field_dim, c.score_dim, bias=False)
            nn.init.normal_(self.score_proj.weight, std=effective_field_dim ** -0.5)
        else:
            self.score_proj = None

        # ── C15: 预测质量 head（2026-08-08，D 方案：预测质量路由）──
        # 替代 C13/C14 的 domain_score_head（域标签判别，三次迭代均失败：
        # ① math/code 文本是英文 → en 的"英文性"覆盖它们，判别任务不对称；
        # ② 各判别器输入自己的 round 1 表征（不同空间）→ softmax 不可比；
        # ③ softmax CE 只约束排序不约束尺度 → lr 高时 logit 膨胀作弊）。
        # quality_head 不再预测"属于哪个域"，而是预测"我对当前样本的预测质量"
        # （round 1 独立前向，无场注入）。监督目标 = 各 neuron 的真实 NLL 排序
        # （ensemble.contrastive_loss：softmax(quality/temp) 对齐 softmax(-NLL/tau)）
        # ——NLL 是客观预测质量，训练时可得，推理时 quality_logit 直接可用。
        # 结构与 C14 MLP 相同（mean 全局分布 + max 强信号 token，2 层非线性）。
        self.quality_head = nn.Sequential(
            nn.Linear(c.hidden_size * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1, bias=False),
        )
        # 第一层 xavier（GELU MLP 标准）；末层小初始化（logit 量级 ~0.1-1）
        for m in self.quality_head:
            if isinstance(m, nn.Linear) and m.out_features > 1:
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

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
        if shared_lm_head is not None:
            # 统一输出空间（2026-08-06）：所有 neuron 共享同一个 general 256K lm_head，
            # 直接预测通用 token（无词库转译投影稀释）→ 路由置信度信号保留。
            # hidden_size 由外部保证与共享 lm_head 输入一致。
            self.lm_head = shared_lm_head
        elif c.lm_head_rank > 0:
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

        # ── C5: 多原型混合（K 个 EMA 原型 + 在线聚类）──
        # num_prototypes=1 时退化为单 EMA 原型（向后兼容，使用 domain_prototype）
        # num_prototypes=K>1 时启用 K 个原型，胜者 EMA 更新（在线 k-means）
        # 路由时取 max cosine（与最近原型的相似度），覆盖多子分布
        self.num_prototypes = max(1, c.num_prototypes)
        if self.num_prototypes > 1:
            # K 个原型，随机初始化（单位球面）
            prototypes = torch.randn(self.num_prototypes, c.hidden_size)
            prototypes = F.normalize(prototypes, dim=-1)
            self.register_buffer("domain_prototypes", prototypes)
            # 原型使用计数（用于初始化阶段强制分配，避免死原型）
            self.register_buffer("proto_counts", torch.zeros(self.num_prototypes))
        else:
            self.domain_prototypes = None  # 单原型模式用 domain_prototype

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

        # Auxiliary-loss-free balancing 运行时统计（不持久化）
        # _channel_usage[peer_id] = 最近一次 forward 的 |proj*scale+bias| 均值
        # 用于启发式 bias 更新：低 usage channel → 正 bias，高 usage → 负 bias
        self._channel_usage: Dict[str, float] = {}

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
        init_scale: float = 50.0,
    ):
        """建立一条指向 peer 神经元的突触通道。

        Args:
            peer_id: peer 神经元标识。
            peer_neuron: peer 神经元实例，用于读取其 field_dim。
            channel_type: "excite" 或 "inhibit"。
            init_std: 通道权重初始化标准差。
            init_scale: 可学习缩放因子初始值（放大 proj 信号使 tanh 饱和）。
                        proj_mean ≈ 0.008，scale=50 → tanh(0.4) ≈ 0.38（有效调制）。
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

        # 可学习缩放因子（放大 proj 信号使 tanh 产生有效 gate）
        scale_param = nn.Parameter(torch.tensor(init_scale))
        if channel_type == "excite":
            self.register_parameter(f"excite_scale_{peer_id}", scale_param)
        else:
            self.register_parameter(f"inhibit_scale_{peer_id}", scale_param)

        # Auxiliary-loss-free balancing bias（非梯度更新，启发式驱动）
        # 低利用率的通道获得正 bias，增强其调制效果
        bias_buf = torch.zeros(1)
        if channel_type == "excite":
            self.register_buffer(f"excite_bias_{peer_id}", bias_buf)
        else:
            self.register_buffer(f"inhibit_bias_{peer_id}", bias_buf)

    def update_channel_bias(self, update_rate: float = 0.1) -> Dict[str, float]:
        """Auxiliary-loss-free balancing 启发式 bias 更新（借鉴 DeepSeek V3）。

        根据最近一次 forward 的 channel usage 统计，调整 bias：
        - 低 usage channel → 正 bias（鼓励激活，增加 |proj*scale+bias|）
        - 高 usage channel → 负 bias（抑制过度激活）
        - 更新幅度 = update_rate * (avg_usage - channel_usage)

        非梯度更新，不参与反向传播。每 N 步调用一次（N=50 推荐）。

        Args:
            update_rate: bias 更新步长（典型 0.1）

        Returns:
            Dict[channel_key, bias_delta] 用于日志记录
        """
        if not self._channel_usage:
            return {}

        usages = list(self._channel_usage.values())
        avg_usage = sum(usages) / len(usages)
        deltas: Dict[str, float] = {}

        for key, usage in self._channel_usage.items():
            # 偏离平均的 channel 获得补偿 bias
            delta = update_rate * (avg_usage - usage)
            if abs(delta) < 1e-6:
                continue

            ch_type, peer_id = key.split(":", 1)
            bias_name = f"{ch_type}_bias_{peer_id}"
            bias_buf = getattr(self, bias_name, None)
            if bias_buf is not None:
                # 直接修改 buffer（非梯度更新）
                bias_buf.data.add_(delta)
                deltas[key] = delta

        return deltas

    def get_channel_usage_stats(self) -> Dict[str, float]:
        """获取当前 channel usage 统计（用于日志/诊断）。"""
        return dict(self._channel_usage)

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
        """是否为抑制性神经元（标准 inhibitory 或 SOM+ 亚型）。"""
        # C1: SOM+ 也有抑制效果（定向抑制树突）
        return self.neuron_type in ("inhibitory", "excitatory_som")

    @property
    def is_excitatory(self) -> bool:
        """是否为兴奋性神经元（含 PV+/VIP+ 亚型）。"""
        return self.neuron_type in ("excitatory", "excitatory_pv", "excitatory_vip")

    @property
    def write_gain(self) -> float:
        """C1: 场写入增益（不同亚型有不同增益）。

        - excitatory: 1.0（标准）
        - excitatory_pv (PV+): 1.5（快速放电，强写入）
        - excitatory_som (SOM+): 0.8（弱写入，主要做抑制）
        - excitatory_vip (VIP+): 1.2（增强写入，去抑制效果）
        - inhibitory: 1.0（标准抑制）
        """
        gains = {
            "excitatory": 1.0,
            "excitatory_pv": 1.5,
            "excitatory_som": 0.8,
            "excitatory_vip": 1.2,
            "inhibitory": 1.0,
        }
        return gains.get(self.neuron_type, 1.0)

    @property
    def refractory_multiplier(self) -> float:
        """C1: 不应期长度乘数（不同亚型有不同不应期）。

        - excitatory: 1.0（标准）
        - excitatory_pv (PV+): 0.5（快速恢复，短不应期）
        - excitatory_som (SOM+): 1.5（长不应期，持续调制）
        - excitatory_vip (VIP+): 0.8（较快恢复）
        - inhibitory: 2.0（最长不应期，强抑制后需长恢复）
        """
        mults = {
            "excitatory": 1.0,
            "excitatory_pv": 0.5,
            "excitatory_som": 1.5,
            "excitatory_vip": 0.8,
            "inhibitory": 2.0,
        }
        return mults.get(self.neuron_type, 1.0)

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

    def auto_register_modalities(self, tokenizer_hub) -> None:
        """2026-08-07 收敛后：自动注册所有已注册到 TokenizerHub 的非文本模态的**输入投影**。

        从 TokenizerHub 获取所有模态编码器，自动注册投影层（raw 特征 → shared 空间）。
        mm_lm_heads（独立 codebook 输出头）已废弃——多模态输出统一走共享 general
        lm_head，与文本同构（输入输出都在 shared/general 空间，消除空间割裂）。

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

            # 只注册输入投影（mm_lm_heads 已废弃）
            self.register_modality_projection(modality, raw_dim=latent_dim)

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
        return_judge_logits: bool = False,
        side_signals: Optional[Dict[int, torch.Tensor]] = None,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        return_intermediate: bool = False,
        return_quality_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the neuron.

        Args:
            shared_embeddings: [B, L, base_embed_dim] from the shared embedding table
                               (外部 shared_embedding(general_ids) 生成)
            field_state: [D] current field state vector (from round 2 onward)
            round_num: current resonance round (1 = independent, 2+ = conditioned)
            return_logits: if True, also return lm_head logits (for PPL)
            return_judge_logits: C24 if True and judge_lm_head set, also return
                                 general 256K judge logits（判定空间可比信号）
            side_signals: optional {neuron_id: vector} for side-channel communication
            temp_gain: S9 注意力温度增益（norepinephrine 驱动）。
                       >1 聚焦（logits 尖锐），<1 泛化（logits 分散），1.0 标准。
            ffn_gain: S9 FFN 输出增益（dopamine 驱动）。
                      >1 强化（奖励），<1 衰减（惩罚），1.0 标准。
            return_intermediate: R7 若 True，返回每层 hidden 和 attention 权重
                                 （用于兼容性表示对齐实验）。默认 False（向后兼容）。
            return_quality_tokens: 临时逐位置路由实验开关。为 True 时，quality_head
                                    读取 token hidden 并返回逐位置 logit；默认 False
                                    保持原有回合级 quality_head 行为。

        Returns:
            dict with keys:
            - field_vector: [B, D] L2-normalised write vector
            - hidden_before_write: [B, hidden] for diversity loss
            - logits: [B, L, vocab] (only if return_logits=True)
            - judge_logits: [B, L, 256000] (only if return_judge_logits=True and judge_lm_head set)
            - intermediate_hidden: [B, num_layers, L, hidden] (only if return_intermediate=True)
            - attn_weights: [B, num_layers, num_heads, L, L] (only if return_intermediate=True)
        """
        # ── Step 1: Embedding adapter (shared base → neuron concept space) ──
        h = self.embed_adapter(shared_embeddings)  # [B, L, hidden]

        # ── Step 2: Transformer layers + field conditioning ──
        bsz, seqlen, _ = h.shape

        # 兼容性表示对齐：按需收集中间表示，默认关闭
        layer_hiddens = []
        layer_attns = []

        # Causal mask（修复双向注意力 bug）:
        # GQA.is_causal = (mask is not None) and (seqlen > 1)
        # 之前没传 mask → is_causal=False → 双向注意力 → 训练时偷看未来 token
        # → full-sequence forcing 虚高（100%）、自回归崩溃（死循环重复）
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
            # S9: temp_gain/ffn_gain 注入 Transformer 内部（神经调质门控 attention/FFN）
            # S10: dendritic=True 时，block.forward 内部完成 basal + apical + 预测编码整合
            #   - dendritic=False: field_state 被忽略（走标准 basal 路径）
            #   - dendritic=True: field_state 作为 apical KV，参与 cross-attention
            lora_on = self.lora_enabled and i in self.lora_layers
            if self.dendritic_enabled and field_state is not None:
                # 树突化：直接调用 block.forward，内部处理 basal + apical
                h_in = h
                h, _, attn_w = block(
                    h, mask=causal_mask, temp_gain=temp_gain, ffn_gain=ffn_gain,
                    field_state=field_state, return_attn_weights=return_intermediate,
                )
                # C16: 整块 LoRA（dendritic 路径无法拆开注入，退化到块级低秩残差）
                if lora_on:
                    h = h + self.lora_adapters[str(i)]["blk"](h_in)
            else:
                # 标准路径（向后兼容）
                h_normed = block.attention_norm(h)
                attn_out, _, attn_w = block.attention(
                    h_normed, mask=causal_mask, temp_gain=temp_gain,
                    return_attn_weights=return_intermediate,
                )
                h = h + attn_out
                # C16: attention 低秩增量（作用于 norm 后输入 → 整层 attention 修正）
                if lora_on:
                    h = h + self.lora_adapters[str(i)]["attn"](h_normed)
                h_ffn_in = block.ffn_norm(h)
                h = h + block.feed_forward(h_ffn_in, gain=ffn_gain)
                # C16: FFN 低秩增量
                if lora_on:
                    h = h + self.lora_adapters[str(i)]["ffn"](h_ffn_in)

            # Field conditioning (round 2+ only)
            if field_state is not None and round_num > 1:
                conditioning = self.field_read_layers[i](field_state)  # [D] → [hidden]
                if conditioning.dim() == 1:
                    conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1,1,H] -> all B,L
                else:
                    conditioning = conditioning.unsqueeze(1)  # [B,1,H] -> over seq

                if self.v1_compat:
                    # v1: broadcast same conditioning to all positions (加性)
                    h = h + conditioning
                else:
                    # C4: 根据 field_read_mode 选择调制方式
                    gate = torch.sigmoid(self.field_read_gate(h))  # [B, L, 1]

                    if self.field_read_mode == "multiplicative":
                        # 乘性门控：h = h * (1 + gate * tanh(conditioning))
                        # 场状态调制表示的幅度/方向，而非仅加性偏移
                        # tanh 限制 conditioning 在 [-1,1]，防止乘性爆炸
                        modulation = torch.tanh(conditioning)  # [B, L, H]
                        h = h * (1.0 + gate * modulation)

                    elif self.field_read_mode == "predictive":
                        # 预测编码：h = h - gate * (h - conditioning)
                        # field_state 作为自上而下预测，误差驱动校正
                        # 与 S10 树突化 apical 路径模式一致
                        error = h - conditioning  # [B, L, H] 预测误差
                        h = h - gate * error

                    else:  # "additive"（默认，向后兼容）
                        # 加性门控：h = h + gate * conditioning
                        h = h + gate * conditioning

            # 兼容性表示对齐：收集该层输出
            if return_intermediate:
                layer_hiddens.append(h)
                layer_attns.append(attn_w)

        # ── Step 3: Final norm ──
        # side_channels 移到 norm 之后，避免 RMSNorm 抵消乘性调制
        h = self.norm(h)

        # ── Step 4: Side-channel modulation (after norm, directly affects logits) ──
        # side_signals: {peer_id: peer_field_vector [B, peer_field_dim]}
        # 通过 excite/inhibit_channels 把 peer 的 field_vector 投影到本神经元 hidden 空间，
        # 以乘性 gate 调制 h（移到 norm 后，避免被 RMSNorm 抵消）。
        # Auxiliary-loss-free balancing: 每条 channel 有可学习 scale + 启发式 bias
        # usage 统计：记录 |scaled_proj| 均值，供启发式 bias 更新使用
        if side_signals:
            excite_sum = None
            inhibit_sum = None
            for peer_id, sig in side_signals.items():
                if peer_id in self.excite_channels:
                    proj = self.excite_channels[peer_id](sig)  # [B, hidden_size]
                    # 可学习缩放因子 + 启发式 bias（Auxiliary-loss-free balancing）
                    scale = getattr(self, f"excite_scale_{peer_id}", None)
                    bias = getattr(self, f"excite_bias_{peer_id}", None)
                    if scale is not None:
                        proj = proj * scale + (bias if bias is not None else 0.0)
                        # 记录 usage = |scaled_proj|.mean()（detached，不参与梯度）
                        self._channel_usage[f"excite:{peer_id}"] = (
                            proj.detach().abs().mean().item()
                        )
                    excite_sum = proj if excite_sum is None else excite_sum + proj
                if peer_id in self.inhibit_channels:
                    proj = self.inhibit_channels[peer_id](sig)
                    scale = getattr(self, f"inhibit_scale_{peer_id}", None)
                    bias = getattr(self, f"inhibit_bias_{peer_id}", None)
                    if scale is not None:
                        proj = proj * scale + (bias if bias is not None else 0.0)
                        self._channel_usage[f"inhibit:{peer_id}"] = (
                            proj.detach().abs().mean().item()
                        )
                    inhibit_sum = proj if inhibit_sum is None else inhibit_sum + proj

            # 乘性调制：gate = 1 + tanh(proj * scale + bias)
            # scale=50 使 proj_mean=0.008 → 0.4，tanh(0.4)≈0.38（有效调制）
            if excite_sum is not None:
                gate = 1.0 + torch.tanh(excite_sum.unsqueeze(1))  # [B, L, hidden]
                h = h * gate
            if inhibit_sum is not None:
                gate = 1.0 - torch.tanh(inhibit_sum.unsqueeze(1))  # [B, L, hidden]
                h = h * gate

        # ── Step 5: Field write ──
        # P0#3: 不再对抑制性神经元取反（v=-v）。
        # field_vector 始终为正方向；抑制效果由 field.write_inhibit() 的
        # 乘法掩码实现（divisive inhibition，GABA-like）。
        #
        # C8: field_confidence — 保留幅度作置信度
        # 当前 L2 归一化丢弃幅度，高置信度 neuron 和低置信度 neuron 写入幅度相同。
        # C8 让 neuron 自身产生置信度（attention entropy 方案）：
        # - entropy = -Σ(p·log p)，完全聚焦 entropy=0，均匀分布 entropy=log(L)
        # - confidence = 1 - entropy/log(L) ∈ [0, 1]
        # - 高聚焦（neuron 对输入有明确判断）→ 高 confidence → 写入幅度大
        # - 低聚焦（neuron 不确定）→ 低 confidence → 写入幅度小
        # 方向仍 L2 归一化（保持 cosine similarity 评分），幅度通过 scale 调制
        if self.v1_compat:
            # v1: last-token write (matches old checkpoint training distribution)
            # v1_compat 与 num_field_heads>1 互斥（v1 是旧 ckpt 兼容模式）
            hidden_last = h[:, -1, :]  # [B, hidden]
            v_raw = self.field_write(hidden_last)  # [B, field_dim]
            v_raw = self.field_projector(v_raw)  # [B, effective_field_dim] 突触投影
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)
            # v1 无 attention pooling，用 hidden norm 作置信度（强激活=高置信度）
            field_confidence = torch.sigmoid(
                hidden_last.norm(dim=-1) / math.sqrt(self.config.hidden_size)
            )  # [B] ∈ [0, 1]
            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": hidden_last,
                "field_confidence": field_confidence,
            }
        elif self.num_field_heads > 1:
            # C6: 多头 attention pooling + 门控聚合
            # K 个独立 query 各自 attention pooling，捕捉不同语义切面
            attn_scores = torch.matmul(
                h, self.field_pool_queries.T
            ) * self.field_pool_scale  # [B, L, K]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L, K] softmax over L
            pooled = torch.einsum('blk,blh->bkh', attn_weights, h)  # [B, K, hidden]

            # K 个独立 field_write 投影（每个 head 学不同写入方向）
            v_raw_k = torch.stack([
                head(pooled[:, k]) for k, head in enumerate(self.field_write_heads)
            ])  # [K, B, field_dim]

            # 门控聚合：从 pooled 均值动态选择每个 head 的权重
            gate = torch.softmax(
                self.field_gate(pooled.mean(dim=1)), dim=-1
            )  # [B, K]
            v_raw = torch.einsum('bk,kbd->bd', gate, v_raw_k)  # [B, field_dim]

            v_raw = self.field_projector(v_raw)  # [B, effective_field_dim] 突触投影
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)

            # C8: 多头 confidence — 每 head 独立 entropy，gate 加权聚合
            L = h.shape[1]
            max_entropy = math.log(L) if L > 1 else 1.0
            # 每 head 的 attention entropy: -Σ(p·log p) over L
            entropy_per_head = -(
                attn_weights * (attn_weights + 1e-8).log()
            ).sum(dim=1)  # [B, K]
            confidence_per_head = 1.0 - entropy_per_head / max_entropy  # [B, K]
            field_confidence = (gate * confidence_per_head).sum(dim=-1)  # [B]

            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": pooled.mean(dim=1),  # 平均 pooling 用于 domain_prototype 更新
                "field_attn_weights": attn_weights.mean(dim=-1),  # [B, L] 平均 attention（诊断用）
                "field_gate": gate,  # [B, K] 门控权重（诊断用）
                "field_confidence": field_confidence,  # [B] 置信度（C8）
            }
        else:
            # v2: attention-pooled field write（向后兼容，单 query）
            attn_scores = torch.matmul(h, self.field_pool_query) * self.field_pool_scale  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L]
            pooled = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # [B, hidden]
            v_raw = self.field_write(pooled)  # [B, field_dim]
            v_raw = self.field_projector(v_raw)  # [B, effective_field_dim] 突触投影
            v = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)

            # C8: attention entropy → confidence
            # 完全聚焦（entropy=0）→ confidence=1，均匀分布（entropy=log L）→ confidence=0
            L = h.shape[1]
            max_entropy = math.log(L) if L > 1 else 1.0
            entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1)  # [B]
            field_confidence = 1.0 - entropy / max_entropy  # [B] ∈ [0, 1]

            result: Dict[str, torch.Tensor] = {
                "field_vector": v,
                "hidden_before_write": pooled,
                "field_attn_weights": attn_weights,
                "field_confidence": field_confidence,  # [B] 置信度（C8）
            }

        # ── C12: 评分投影向量 ──
        # score_proj 将 field_vector 投影到共享评分空间，与 field_score_proj(loo_state) 算 cosine
        # 让评分学习与写入学习解耦，大神经元对场方向的主导被 field_score_proj 抵消
        if self.score_proj is not None:
            score_vec = self.score_proj(v_raw)  # [B, score_dim]
            score_vec = F.normalize(score_vec, dim=-1)
            result["score_vec"] = score_vec

        # ── Step 5: Optional logits (for PPL evaluation) ──
        # 2026-08-07 收敛：多模态输出统一走共享 general lm_head（compute_logits）。
        # 旧的 mm_lm_heads 独立 codebook 头已废弃（与"共享 general 256K 头"架构矛盾，
        # 输入投影进 shared 空间、输出却跳去独立 codebook 空间，输入输出割裂）。
        if return_logits:
            result["logits"] = self.compute_logits(h)  # [B, L, vocab]

        # ── C24: 判定头 logits（general 256K 空间，判定信号可比）──
        # judge_lm_head 存在时输出 general 空间 logits——C20 判定 5/5 的信号链
        # （所有 neuron 共享 general 256K 头 → 投影 NLL 跨 neuron 可比）。
        # judge_proj 存在时先投影到判定空间维度（hidden≠512 的 neuron 适配）。
        if return_judge_logits and self.judge_lm_head is not None:
            h_judge = self.judge_proj(h) if self.judge_proj is not None else h
            result["judge_logits"] = self.judge_lm_head(h_judge)  # [B, L, 256000]

        # ── C15: 预测质量 logit（2026-08-08，D 方案）──
        # 只在 round 1 计算：round 1 独立前向（无 field_state 注入、无 side_signals），
        # h 是纯自身能力输出 → quality_logit 梯度只流向自身 neuron（无跨 neuron 泄漏）。
        # round 2+ 的 h 经场条件化/侧通道调制（携带他人信息），不适合做质量评估。
        # mean/max 双 pooling → MLP（mean 全局分布 + max 强信号 token）。
        # 监督由 ensemble.contrastive_loss 提供（对齐 per-neuron NLL 排序）。
        if round_num == 1 and self.quality_head is not None:
            if return_quality_tokens:
                result["quality_token_logits"] = self.quality_head(h)  # [B, L, 1]
            else:
                h_pool = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # [B, 2H]
                result["quality_logit"] = self.quality_head(h_pool)  # [B, 1]

        # ── 兼容性表示对齐：中间表示 ──
        if return_intermediate:
            result["intermediate_hidden"] = torch.stack(layer_hiddens, dim=1)  # [B, n_layers, L, hidden]
            # attn_w: [B, num_heads, L, L] per layer；None（如 seqlen<=1）时跳过
            valid_attns = [a for a in layer_attns if a is not None]
            if valid_attns:
                result["attn_weights"] = torch.stack(valid_attns, dim=1)  # [B, n_layers, num_heads, L, L]
            else:
                result["attn_weights"] = None

        return result

    def update_domain_prototype(self, pooled: torch.Tensor) -> None:
        """用真实处理样本后的 hidden state 更新 domain prototype（EMA）。

        Prototype 是数据驱动的典型响应向量，非权重统计量。
        训练时每轮调用，EMA 平滑跟踪 neuron 的典型响应模式。

        C5: 多原型模式（num_prototypes>1）时，用在线 k-means：
        - 找到与 target 最接近的原型（胜者）
        - 仅更新胜者原型（EMA）
        - 路由时取 max cosine（与最近原型的相似度）
        """
        with torch.no_grad():
            target = pooled.detach().mean(dim=0) if pooled.dim() == 2 else pooled.detach()
            target_norm = F.normalize(target.unsqueeze(0), dim=-1).squeeze(0)  # 单位球面

            if self.num_prototypes > 1 and self.domain_prototypes is not None:
                # C5: 多原型在线 k-means
                # 计算与所有原型的相似度，选胜者
                sims = F.cosine_similarity(
                    target_norm.unsqueeze(0), self.domain_prototypes, dim=-1
                )  # [K]
                winner_idx = int(sims.argmax().item())

                # 初始化阶段：若原型未被使用过，直接赋值（避免死原型）
                if self.proto_counts[winner_idx] < 1.0:
                    self.domain_prototypes[winner_idx] = target_norm
                else:
                    # EMA 更新胜者原型
                    decay = self.proto_ema_decay.item()
                    self.domain_prototypes[winner_idx].mul_(decay).add_(
                        target_norm, alpha=(1.0 - decay)
                    )
                    self.domain_prototypes[winner_idx] = F.normalize(
                        self.domain_prototypes[winner_idx], dim=-1
                    )
                self.proto_counts[winner_idx] += 1.0
            else:
                # 单原型模式（向后兼容）
                self.domain_prototype.mul_(self.proto_ema_decay.item()).add_(
                    target, alpha=(1.0 - self.proto_ema_decay.item())
                )
                # 归一化保持单位球面
                self.domain_prototype.div_(self.domain_prototype.norm() + 1e-8)

    # ── DEAD CODE (R17, REMEDIATION_PLAN 2026-08-14)：仅 scripts/archive/
    # _smoke_c6_multihead_field_write.py 调用；生产路径无调用者，保留以存审计证据。──
    @torch.no_grad()
    def quick_probe(self, shared_embeddings: torch.Tensor) -> torch.Tensor:
        """Lightweight forward pass for prescreening (skip full Transformer).

        Runs only the adapter + field_write, no Transformer layers.
        Returns a rough field_vector direction for candidate selection.
        """
        h = self.embed_adapter(shared_embeddings)
        # Use mean pooling over sequence as a rough representation
        h_pooled = h.mean(dim=1)  # [B, hidden]
        if self.num_field_heads > 1:
            # C6: 多头简化路径（用 head 0 做轻量预筛选，完整多头逻辑在 forward 中）
            v_raw = self.field_write_heads[0](h_pooled)
        else:
            v_raw = self.field_write(h_pooled)
        v_raw = self.field_projector(v_raw)  # 突触投影
        return v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8)

    def get_field_write_parameters(self):
        """返回所有 field_write 相关参数（C6 多头兼容）。

        单头模式：返回 self.field_write.parameters()
        多头模式：返回 self.field_write_heads + self.field_gate + self.field_pool_queries
        """
        if self.num_field_heads > 1:
            params = list(self.field_write_heads.parameters())
            params += list(self.field_gate.parameters())
            params.append(self.field_pool_queries)
            return params
        return list(self.field_write.parameters()) + [self.field_pool_query]

    def get_field_read_parameters(self):
        """返回 round2+ 场条件化读取路径的参数（R2, REMEDIATION_PLAN 2026-08-14）。

        审计发现 field_read_layers / field_read_gate 参与推理 round2+ 每层
        条件化（neuron.forward round_num>1 路径），但全仓库训练脚本均未解冻
        → 恒为随机初始化投影。此方法供训练脚本解冻，使场读取成为可学习路径。
        """
        params = list(self.field_read_layers.parameters())
        params += list(self.field_read_gate.parameters())
        return params

    def enable_lora(self, rank: int, layers: Optional[List[int]] = None):
        """C16: 启用尾层 LoRA 适配器（冻结 body，只训低秩增量）。

        Args:
            rank: 低秩维度（默认 16）
            layers: 尾层索引列表；None = 最后 2 层

        适配器 B 初始 0 → LoRA 输出恒 0 → 不改变任何既有 forward 结果（向后兼容）。
        """
        if rank <= 0:
            return
        if layers is None:
            n = len(self.layers)
            layers = list(range(max(0, n - 2), n))
        self.lora_enabled = True
        self.lora_layers = list(layers)
        for i in layers:
            layer_adapters = nn.ModuleDict()
            if self.dendritic_enabled:
                layer_adapters["blk"] = LoraPair(self.config.hidden_size, rank)
            else:
                layer_adapters["attn"] = LoraPair(self.config.hidden_size, rank)
                layer_adapters["ffn"] = LoraPair(self.config.hidden_size, rank)
            self.lora_adapters[str(i)] = layer_adapters
        return self.lora_adapters

    def load_lora(self, sd: dict, layers: Optional[List[int]] = None) -> bool:
        """C26 增量三补：从 ckpt state_dict 恢复沉淀的 LoRA 增量。

        enable_lora 是运行时方法（不写 config），装配重建的 neuron 无
        lora_adapters，strict=False 加载会静默丢弃 lora keys → 沉淀的皮层
        记忆重启即失。检测 sd 含 lora_adapters.* 时：enable_lora（rank 从
        a.weight 推断，层用训练同款尾层默认）+ 加载。

        Args:
            sd: ckpt["state_dict"]（含 lora_adapters.* keys 才有意义）
            layers: 与训练一致（None = 尾层 2 层）

        Returns:
            True=已恢复 LoRA；False=sd 无 lora 或加载失败
        """
        lora_keys = [k for k in sd if k.startswith("lora_adapters.")]
        if not lora_keys:
            return False
        rank = 0
        for k, v in sd.items():
            if k.startswith("lora_adapters.") and k.endswith(".a.weight"):
                rank = max(rank, v.shape[0])
        if len(self.lora_adapters) == 0:
            self.enable_lora(rank if rank > 0 else 16, layers=layers)
        lora_sd = {k[len("lora_adapters."):]: v for k, v in sd.items()
                   if k.startswith("lora_adapters.")}
        try:
            self.lora_adapters.load_state_dict(lora_sd, strict=False)
            return True
        except Exception:
            return False

