"""Cortex — consciousness center via resonance field state.

Phase 4: Replaces the monolithic backbone with ResonanceEnsemble.
The field state IS the consciousness — not a single model's hidden state,
but the collective resonance pattern across domain-specialized neurons.

Architecture (shared embedding):
    Input text → General Tokenizer (256K) → Shared Embedding → ResonanceEnsemble
        ├── zh neuron (via embed_adapter)
        ├── en neuron (via embed_adapter)
        ├── code neuron (via embed_adapter)
        ├── math neuron (via embed_adapter)
        └── general neuron (via embed_adapter)
            ↓
    Resonance Field (shared consciousness) → Field vectors NOW comparable!
            ↓
    Per-neuron lm_head (domain vocab) → Domain-specific output

Usage:
    cortex = Cortex(neurons_dir="data/neurons")
    cortex.set_shared_embedding(embedding)
    cortex.set_tokenizer_hub(hub)
    output = cortex.generate("今天天气怎么样？", max_tokens=256)
"""

from __future__ import annotations

import os
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger("Cortex")

from neuroplex.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig,
)
from neuroplex.resonance.translator import (
    build_position_alignment,
    tokenizer_fingerprint,
)
from neuroplex.resonance.dialogue_format import dialogue_prompt_requires_guard


@dataclass
class TaskSet:
    """C26 增量八：多阶段任务模式链 v2——一个阶段 = 一个任务集（人脑任务集切换）。

    对比 C25-F 首步（generate_staged 的 dict 阶段）：TaskSet 是类型化对象，
    表达"激活哪个 neuron 群体 + 用哪种共振模式 + 质量门约束"——任务集切换
    的真正含义（显式激活子集 + 判定约束），且可组合/可序列化。

    Attributes:
        prompt: 阶段指令。含 "{prev}" 用上一阶段输出填充模板；无 "{prev}"
            且已有 prev 时自动追加（"prompt\n上一阶段输出"）。
        mode: 共振模式 "continuous"（默认，C25-E）/ "executive"（回合级判定）。
        domain: task-set 域约束（None = 模式判定）。
        active_nids: 显式激活的 neuron 子集（任务集切换核心；None = 路由自动）。
            支持 'auto_topK'/'auto_all'/'auto_top1' 字符串模式（稀疏激活）。
        max_tokens: 阶段生成长度覆盖（None = 默认）。
        temperature: 阶段温度覆盖（None = 默认）。
        quality_gate: 阶段质量门（退化检测 → 重试 → 隔离）。默认 True。
        record_memory: 阶段结束后是否把场状态写入记忆库（睡眠固化候选）。
            三重传递的第三重（文本 prev + 场状态 seed_memories + 记忆写入）。
        memory_label: 记忆写入的标签（None = 用 prompt 截断）。
    """
    prompt: str
    mode: str = "continuous"
    domain: Optional[str] = None
    active_nids: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    quality_gate: bool = True
    record_memory: bool = False
    memory_label: Optional[str] = None


class Cortex:
    """Resonance-field-based consciousness center.

    Wraps ResonanceEnsemble with a high-level generate() interface
    compatible with the existing API and agent systems.
    """

    def __init__(
        self,
        neurons_dir: str = "data/neurons",
        device: str = "cpu",
        max_rounds: int = 3,
        shared_embedding: Optional[torch.nn.Embedding] = None,
        general_tokenizer = None,
        neuron_ids: Optional[List[str]] = None,
    ):
        self.device = device
        self.neurons_dir = neurons_dir
        self.max_rounds = max_rounds
        self.is_loaded = False
        # C26 增量四（2026-08-14）：场记忆库引用（懒注入）——generate 未显式
        # 传 memory_vectors 时自动检索 top-k 记忆注入生成（记忆自动调取）。
        self._memory_bank = None

        # ── Load neurons ──
        # neuron_ids 指定装配集合（如对话综合体 ENSEMBLE_DIALOGUE_IDS）；
        # None = 扫描全部 neuron_*.pt（向后兼容）
        self.neurons: Dict[str, ResonanceNeuron] = {}
        # 增删锁（热插拔：add/remove/isolate/revive 串行化；推理读走快照隔离不拿锁）
        self._neurons_lock = threading.RLock()
        self._load_neurons(neuron_ids=neuron_ids)

        # ── Create field and ensemble ──
        if self.neurons:
            # 混合规格协作：不同 field_dim 通过 ensemble 的跨规格投影层统一
            # （embed_adapter 已处理 hidden_size 差异，无需校验 hidden_size）
            # field_dim 取最大值，ensemble 自动为其他规格创建正/反向投影层
            effective_dims = {
                n.config.unified_field_dim if n.config.unified_field_dim is not None else n.config.field_dim
                for n in self.neurons.values()
            }
            field_dim = max(effective_dims)
        else:
            field_dim = 4096
        # R16（REMEDIATION_PLAN 2026-08-14）：场创建时带上真实设备（神经元
        # 所在设备，退化到 cortex.device）。此前恒 CPU —— 全链路 CPU 假设，
        # 神经元参数移 GPU 后场/W_cond 仍留 CPU（device 不匹配隐患）。
        first_neuron = next(iter(self.neurons.values()), None)
        field_device = None
        try:
            field_device = getattr(first_neuron.lm_head, "device", None)
        except Exception:
            pass
        self.field = ResonanceField(dim=field_dim, device=field_device or self.device)
        # P1-1: CoactivationTracker（共激活追踪，供孤立检测+部落分组）
        from neuroplex.resonance.tribal import CoactivationTracker
        self.coaction = CoactivationTracker()
        self.ensemble = ResonanceEnsemble(
            self.neurons, self.field,
            max_rounds=max_rounds,
            coaction=self.coaction,
        )

        # ── Shared embedding (Layer 1: shared sensory) ──
        # nn.Embedding(256000, 512) — ALL neurons share this.
        # Can be hot-swapped for larger vocabs.
        self._shared_embedding: Optional[torch.nn.Embedding] = shared_embedding

        # ── Per-neuron shared embeddings（P7-修复 2026-08-04）──
        # 训练（finetune_cross_spec）时每个神经元从各自 ckpt 加载独立的
        # shared_embedding_state，推理若用单个 data/shared_embedding.pt 则与训练
        # 不一致 → 生成垃圾。此 dict 保存 {nid: nn.Embedding}，_generate_p7 用它
        # 构建 neuron_embeddings（与 eval 路径一致）。为空时回退 _shared_embedding。
        self._neuron_shared_embeddings: Optional[Dict[str, torch.nn.Embedding]] = None

        # ── General tokenizer (256K, hot-swappable I/O protocol) ──
        self._general_sp = general_tokenizer

        # ── S6: Domain→General token 对齐表缓存 ──
        # 消除自回归生成时的 domain→text→general re-encode 往返
        # 格式：{domain_name: {"fp": ..., "alignment": {domain_token_id: [general_token_ids]}}}
        self._domain_to_general_cache: Dict[str, Dict[int, list]] = {}
        # 可编辑词库规则层（AlignmentRules）：人工覆盖自动转译，见 set_alignment_rules
        self._alignment_rules = None

        # ── Domain tokenizer hub ──
        # Manages per-domain tokenizers (zh=20K, en=16K, code=12K, math=10K).
        # Used for domain-specific lm_head targets and decoding.
        self._tokenizer_hub = None

        # ── C19: 回合级 quality 判定 EMA（per-neuron 标准化）──
        # quality_head 输出跨 neuron 不可比（C16b 教训：转译 neuron 基线巨大），
        # _executive_route 用 per-neuron EMA z-score（相对自身水平）聚合域分。
        # 样本数 < WARMUP 时不参与切换（回退纯启发式）；C20 训练 quality_head
        # 期间 EMA 自然积累，成熟后混合信号自动生效。
        self._quality_logit_ema: Dict[str, dict] = {}
        self._quality_ema_warmup = 20
        self._quality_ema_alpha = 0.05

        # ── Legacy tokenizer (for non-P7 paths) ──
        self._tokenizer = None

        # ── Route tracking ──
        self._last_routing: Optional[Dict] = None

        # ── P1-2: NeuromodulatorState（自主进化调质）──
        self._neuromodulator = None

        # ── SleepConsolidator（睡眠巩固，跨会话 replay buffer 连续性）──
        self._sleep_consolidator = None

        # ── P6-3: GammaOscillator ──
        self.gamma_oscillator = None

        # ── P6-4: WorkingMemory ──
        # R13（REMEDIATION_PLAN 2026-08-14）：仅"注册未接入"——set_working_memory
        # 后 generate/_generate_p7/think 均不读取本字段（假接线）；生产对话上下文
        # 由 neuroplex/agent/working_memory（ContextManager 链）承担。保留接口向后
        # 兼容；若未来接 token 级滑动窗口，接入点在 _generate_p7。
        self.working_memory = None

        # ── S12: DialogueState（多轮对话状态管理）──
        # 替代前缀拼接，用 field_state 持久化 + 对话轮次 token
        # None 时 generate() 保持原前缀拼接行为（向后兼容）
        self._dialogue_state = None

        # ── State ──
        self.is_loaded = len(self.neurons) > 0
        print(f"[Cortex] Loaded {len(self.neurons)} neurons, field_dim={field_dim}")
        if self._shared_embedding is not None:
            print(f"[Cortex] Shared embedding: {self._shared_embedding.num_embeddings} × {self._shared_embedding.embedding_dim}")

    def _load_neurons(self, neuron_ids: Optional[List[str]] = None):
        """Load all trained neurons from disk.

        H3 修复：原来硬编码 5 域 ['zh','en','code','math','general']，
        新生 neuron（如 neuron_physics_1.pt）被静默忽略。
        现改为扫描 neurons_dir 下所有 neuron_*.pt 文件动态加载。

        Args:
            neuron_ids: 只装配指定 ID 集合（如对话综合体）；
                None = 扫描全部（向后兼容）。
                注意：跨规格协作时，不同 hidden_size / field_dim 的 neuron
                通过 embed_adapter + ensemble 跨规格投影层兼容，无需同规格。
        """
        import glob
        # 扫描所有 neuron_*.pt（排除 _fieldcond.pt 等非 neuron 文件）
        ckpt_paths = sorted(glob.glob(os.path.join(self.neurons_dir, "neuron_*.pt")))
        # DEAD CODE 标注 (R17, REMEDIATION_PLAN 2026-08-14)：以下 _fieldcond
        # 特判（跳过/优先加载）为孤儿逻辑——仓库内已无任何 *_fieldcond.pt 文件，
        # field_conditioning 已并入 base neuron config（checkpoint 迁移后不再生成
        # 独立变体）。保留以存审计证据；后续清理可整体删除本段。
        for ckpt_path in ckpt_paths:
            name = os.path.basename(ckpt_path)
            # 跳过 fieldcond 版（下面会优先加载）、W_base 等非 neuron 文件
            if "_fieldcond" in name or name.startswith("_"):
                continue
            # 从文件名提取 domain：neuron_{domain}.pt → {domain}
            domain = name[len("neuron_"):-len(".pt")]

            # 只装配指定集合（对话综合体场景：排除 base 版避免污染）
            if neuron_ids is not None and domain not in neuron_ids:
                continue

            # 优先 fieldcond 版本，回退到 base 版本
            fc_path = os.path.join(self.neurons_dir, f"neuron_{domain}_fieldcond.pt")
            load_path = fc_path if os.path.exists(fc_path) else ckpt_path
            if load_path != ckpt_path:
                # 用 fieldcond 版本覆盖
                ckpt_path = load_path

            if not os.path.exists(ckpt_path):
                continue
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg: NeuronConfig = ckpt["neuron_config"]
            sd = ckpt["state_dict"]

            # v3 兼容性处理：旧 ckpt → 新代码结构
            sd = self._migrate_state_dict(sd, cfg)

            neuron = ResonanceNeuron(cfg).to(self.device)
            # H1: auto-detect v1 vs v2 from the actual parameter keys present.
            # v2 neurons carry field_pool_query + field_read_gate; v1
            # checkpoints must run in v1-compat mode.
            has_v2 = {"field_pool_query", "field_read_gate.weight"} <= set(sd.keys())
            neuron.load_state_dict(sd, strict=False)
            neuron.v1_compat = not has_v2
            neuron.eval()
            # C26 增量三补：恢复 ckpt 自带的沉淀 LoRA 增量（strict=False 会静默
            # 丢弃 lora keys → 皮层记忆重启即失；此处显式恢复）
            if neuron.load_lora(sd):
                logger.info("[Cortex] %s 恢复沉淀 LoRA 增量", domain)
            # domain_prototype 由 sleep_engine contrastive phase EMA 更新，
            # 加载时保持初始化值（zeros），训练后自动填充
            self.neurons[domain] = neuron
            n_params = sum(p.numel() for p in neuron.parameters())
            print(f"  [{domain}] {cfg.spec} neuron: {n_params/1e6:.0f}M params")

    def _migrate_state_dict(self, sd: dict, cfg: NeuronConfig) -> dict:
        """v3 兼容性：旧 ckpt → 新代码结构。

        处理：
        1. side_channels.* → excite_channels.*
        2. 其他新增字段（refractory_counter 等）由 strict=False 跳过

        M2 修复：旧 ckpt 含 lm_head.weight 时，原来静默降级到传统模式（lm_head_rank=0），
        导致 W_base 共享机制失效。现改为显式报错，提示用户运行迁移脚本。
        """
        sd_keys = set(sd.keys())

        # 1. lm_head 兼容性检查（M2：报错而非静默降级）
        if "lm_head.weight" in sd_keys and cfg.lm_head_rank > 0:
            raise RuntimeError(
                f"旧 ckpt 含 lm_head.weight 但 cfg.lm_head_rank={cfg.lm_head_rank}（低秩模式）。"
                f"低秩模式需要 lm_head_delta_u/delta_v，不能用传统 lm_head.weight。"
                f"请运行迁移: python scripts/migrate_ckpt_v3.py --enable-low-rank"
            )

        # 2. side_channels → excite_channels 重命名
        side_keys = [k for k in sd_keys if k.startswith("side_channels.")]
        if side_keys:
            for k in side_keys:
                new_k = k.replace("side_channels.", "excite_channels.", 1)
                sd[new_k] = sd.pop(k)
            print(f"  [compat] 重命名 {len(side_keys)} 个 side_channels → excite_channels")

        return sd

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for encode/decode (legacy shared tokenizer)."""
        self._tokenizer = tokenizer

    def set_tokenizer_hub(self, tokenizer_hub) -> None:
        """注册域 tokenizer hub 和 general tokenizer。

        注册后：
        - generate() 用 general tokenizer encode 输入 → shared_embedding
        - neuron lm_head 输出在 domain vocab → hub 的域 tokenizer decode
        - general tokenizer 可热插拔升级

        Args:
            tokenizer_hub: TokenizerHub 实例（含域 tokenizer + general tokenizer）
        """
        from neuroplex.resonance.translator import TokenizerHub
        if not isinstance(tokenizer_hub, TokenizerHub):
            raise TypeError(
                f"[Cortex] set_tokenizer_hub expects TokenizerHub, "
                f"got {type(tokenizer_hub).__name__}"
            )
        self._tokenizer_hub = tokenizer_hub
        # C26 增量七（2026-08-14）：转发到 ensemble——forward_train 跨 vocab 联合
        # 训练路径需要 hub（此前仅训练脚本手动 set，产品 integrate 路径缺 hub 报错
        # "跨 vocab 联合训练需要 tokenizer hub"）。
        try:
            if getattr(self, "ensemble", None) is not None:
                self.ensemble.set_tokenizer_hub(tokenizer_hub)
        except Exception:
            pass
        domains = tokenizer_hub.list_domains()
        print(f"[Cortex] TokenizerHub registered (P7 模式)")
        print(f"  domains: {domains}")
        for d in domains:
            print(f"  {d}: vocab={tokenizer_hub.vocab_size(d)}")

    def set_shared_embedding(self, embedding: torch.nn.Embedding) -> None:
        """Set the shared embedding table (highest precedence source)."""
        self._shared_embedding = embedding

    def set_neuron_shared_embeddings(
        self, neuron_shared_embeddings: Dict[str, torch.nn.Embedding]
    ) -> None:
        """P7-修复：设置 per-neuron shared embeddings（与训练一致）。

        训练时每个神经元用自己 ckpt 的 shared_embedding_state 编码输入；
        _generate_p7 优先用此 dict 构建 neuron_embeddings，回退到 _shared_embedding。
        """
        self._neuron_shared_embeddings = neuron_shared_embeddings
        logger.info(
            "[Cortex] Per-neuron shared embeddings set: %d neurons",
            len(neuron_shared_embeddings),
        )

    def set_general_tokenizer(self, general_sp) -> None:
        """Set the general 256K tokenizer for I/O protocol.

        This tokenizer encodes raw text → general token IDs for shared embedding lookup.
        Can be hot-swapped: upgrading from 16K to 256K tokenizer doesn't require retraining neurons.
        """
        self._general_sp = general_sp

    # ── 状态持久化（经验积累） ──

    def save_state(self, path: str) -> None:
        """保存可学习状态到磁盘（经验积累持久化）。

        保存内容：
        - shared_embedding 权重（感官层，经验驱动学习积累）
        - 每个 neuron 的 lm_head 权重（输出层）
        - 每个 neuron 的 embed_adapter 权重（如果有）

        不保存：frozen backbone（来自已训练 ckpt，不变）、field state（运行时状态）

        Args:
            path: 保存路径（目录或文件路径）
        """
        import os
        if os.path.isdir(path) or path.endswith(os.sep):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "cortex_state.pt")
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        state = {"version": 3, "saved_at": __import__("time").time()}

        # shared_embedding（fp16 压缩：524MB → 262MB）
        if self._shared_embedding is not None:
            sd = self._shared_embedding.state_dict()
            sd_fp16 = {k: v.half() if v.is_floating_point() else v
                       for k, v in sd.items()}
            state["shared_embedding"] = sd_fp16
            state["shared_embedding_dtype"] = "fp16"

        # per-neuron 可学习参数
        neuron_states = {}
        for nid, neuron in self.neurons.items():
            nsd = {}
            if hasattr(neuron, "lm_head") and neuron.lm_head is not None:
                nsd["lm_head"] = neuron.lm_head.state_dict()
            if hasattr(neuron, "embed_adapter") and neuron.embed_adapter is not None:
                nsd["embed_adapter"] = neuron.embed_adapter.state_dict()
            if nsd:
                neuron_states[nid] = nsd
        state["neurons"] = neuron_states

        # neuromodulator state（自主进化调质：多巴胺/血清素/去甲肾上腺素）
        # 使跨会话调质状态连续，自主进化不中断
        if self._neuromodulator is not None:
            state["neuromodulator"] = self._neuromodulator.get_state_dict()
            logger.debug("[Cortex]   neuromodulator 已保存")

        # coaction state（共激活追踪：跨会话部落分组+孤立检测连续性）
        if self.coaction is not None:
            state["coaction"] = self.coaction.get_state_dict()
            logger.debug("[Cortex]   coaction 已保存")

        # sleep_consolidator state（睡眠巩固：跨会话 replay buffer 连续性）
        if self._sleep_consolidator is not None:
            state["sleep_consolidator"] = self._sleep_consolidator.get_state_dict()
            logger.debug("[Cortex]   sleep_consolidator 已保存")

        # C27 增量四（2026-08-14）：可学习振荡器参数（BioOSS 节奏控制器）
        # ——ω/coupling/gaba_amp 训练后随状态持久化，跨会话节奏连续。
        osc_states = []
        for _osc in getattr(self.ensemble, "oscillators", []) or []:
            if hasattr(_osc, "state_dict"):
                osc_states.append({"nid": _osc.nid, "state_dict": _osc.state_dict()})
        if osc_states:
            state["oscillators"] = osc_states
            logger.debug("[Cortex]   %d oscillators 已保存（节奏控制器）", len(osc_states))

        torch.save(state, path)
        logger.info(f"[Cortex] 状态已保存: {path} "
                    f"(shared_emb={'yes' if 'shared_embedding' in state else 'no'}, "
                    f"neurons={len(neuron_states)}, "
                    f"neuromodulator={'yes' if 'neuromodulator' in state else 'no'}, "
                    f"coaction={'yes' if 'coaction' in state else 'no'}, "
                    f"sleep_consolidator={'yes' if 'sleep_consolidator' in state else 'no'})")

    def load_state(self, path: str, strict: bool = False) -> bool:
        """从磁盘加载可学习状态（恢复经验积累）。

        Args:
            path: 状态文件路径（cortex_state.pt）或目录
            strict: True 时要求所有参数必须匹配

        Returns:
            True 如果成功加载，False 如果文件不存在
        """
        import os
        if os.path.isdir(path):
            path = os.path.join(path, "cortex_state.pt")
        if not os.path.exists(path):
            logger.debug(f"[Cortex] 状态文件不存在: {path}")
            return False

        state = torch.load(path, map_location=self.device, weights_only=False)
        logger.info(f"[Cortex] 加载状态: {path} (version={state.get('version', 1)})")

        # shared_embedding（fp16 → fp32 恢复）
        if "shared_embedding" in state and self._shared_embedding is not None:
            sd = state["shared_embedding"]
            target_dtype = self._shared_embedding.weight.dtype
            sd_restored = {k: v.to(target_dtype) if v.is_floating_point() else v
                           for k, v in sd.items()}
            self._shared_embedding.load_state_dict(sd_restored, strict=strict)
            logger.info("[Cortex]   shared_embedding 已恢复 (dtype=%s)",
                        state.get("shared_embedding_dtype", "fp32"))

        # per-neuron
        neuron_states = state.get("neurons", {})
        loaded = 0
        for nid, nsd in neuron_states.items():
            neuron = self.neurons.get(nid)
            if neuron is None:
                continue
            if "lm_head" in nsd and hasattr(neuron, "lm_head"):
                neuron.lm_head.load_state_dict(nsd["lm_head"], strict=False)
            if "embed_adapter" in nsd and hasattr(neuron, "embed_adapter"):
                neuron.embed_adapter.load_state_dict(nsd["embed_adapter"], strict=False)
            loaded += 1
        logger.info(f"[Cortex]   {loaded}/{len(neuron_states)} neurons 恢复")

        # C27 增量四（2026-08-14）：恢复可学习振荡器参数（BioOSS 节奏控制器）
        osc_states = state.get("oscillators", [])
        osc_loaded = 0
        for _osd in osc_states:
            for _osc in getattr(self.ensemble, "oscillators", []) or []:
                if getattr(_osc, "nid", None) == _osd.get("nid"):
                    try:
                        _osc.load_state_dict(_osd["state_dict"], strict=False)
                        osc_loaded += 1
                    except Exception as _e:
                        logger.warning("[Cortex]   oscillator %s 恢复失败: %s",
                                       _osd.get("nid"), _e)
                    break
        if osc_states:
            logger.info("[Cortex]   %d/%d oscillators 恢复（节奏控制器）",
                        osc_loaded, len(osc_states))

        # neuromodulator state 恢复（跨会话调质连续性）
        if "neuromodulator" in state and self._neuromodulator is not None:
            self._neuromodulator.load_state_dict(state["neuromodulator"])
            logger.info("[Cortex]   neuromodulator 已恢复 "
                        "(DA=%.2f, 5HT=%.2f, NE=%.2f)" % (
                            self._neuromodulator.dopamine,
                            self._neuromodulator.serotonin,
                            self._neuromodulator.norepinephrine,
                        ))

        # coaction state 恢复（跨会话共激活追踪连续性）
        if "coaction" in state and self.coaction is not None:
            self.coaction.load_state_dict(state["coaction"])
            logger.info("[Cortex]   coaction 已恢复 "
                        "(pairs=%d, neurons=%d)" % (
                            len(self.coaction._slow_matrix),
                            len(self.coaction._activation_counts),
                        ))

        # sleep_consolidator state 恢复（跨会话 replay buffer 连续性）
        if "sleep_consolidator" in state and self._sleep_consolidator is not None:
            self._sleep_consolidator.load_state_dict(state["sleep_consolidator"])
            logger.info("[Cortex]   sleep_consolidator 已恢复 "
                        "(replay=%d, last_step=%d)" % (
                            len(self._sleep_consolidator._replay_buffer),
                            self._sleep_consolidator._last_consolidation_step,
                        ))
        return True

    def set_gamma_oscillator(self, oscillator) -> None:
        """P6-3: 注册 Gamma 同步振荡器并按 domain 分配 phase。

        注入后，ResonanceField.write/update 会自动用 gamma gate 调制写入强度：
        - 同 domain 的 neuron 同 phase → 写入互相增强（feature binding）
        - 不同 domain 的 neuron 不同 phase → 写入互相衰减（解绑）

        phase 分配：按 neuron_id 的 domain 前缀分组（zh_xxx → phase 0，
        en_xxx → phase π/3，...）。若 neuron_id 就是 domain 本身（如 "zh"），
        直接用 domain 分组。

        Args:
            oscillator: GammaOscillator 实例
        """
        from neuroplex.resonance.gamma_oscillator import apply_gamma_gate

        # 按 domain 分组 neuron
        domain_to_nids: Dict[str, list] = {}
        for nid in self.neurons.keys():
            # nid 可能是 "zh" / "math" / "code_xxx" 等
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids.setdefault(domain, []).append(nid)

        # C23-C5（2026-08-08）：loader 可能已注入训练 phasor_state（PhasorDynamics
        # 已有相位）→ 跳过 assign，避免覆盖训练学到的相位自组织；无相位时
        # （直接创建注入）才按 domain 先验分配。
        if not oscillator.phases:
            oscillator.assign_phase_by_domain(domain_to_nids)
        apply_gamma_gate(self.field, oscillator)
        self.gamma_oscillator = oscillator
        # KoPE/Kuramoto: 注入 ensemble，每轮共振后执行相位耦合
        if hasattr(self, 'ensemble') and self.ensemble is not None:
            self.ensemble.gamma_oscillator = oscillator
        print(f"[Cortex] GammaOscillator enabled "
              f"({len(oscillator.phases)} neurons phased, "
              f"{len(domain_to_nids)} domains)")
    def tick_gamma(self) -> None:
        """P6-3: 推进 Gamma 振荡相位（每轮共振后调用）。"""
        if hasattr(self, 'gamma_oscillator') and self.gamma_oscillator is not None:
            self.gamma_oscillator.tick()

    def set_working_memory(self, memory) -> None:
        """P6-4: 注册工作记忆模块。

        注册后，generate() 会：
        - 把 memory 内容作为前缀拼到 prompt token 前面（维持上下文）
        - 生成完成后把 prompt + generated 追加到 memory

        未注册时（默认）完全无状态，向后兼容。

        Args:
            memory: WorkingMemory 实例
        """
        self.working_memory = memory
        print(f"[Cortex] WorkingMemory enabled "
              f"(max_tokens={memory.max_tokens}, current={len(memory)})")

    def clear_working_memory(self) -> None:
        """P6-4: 清空工作记忆（新会话开始时调用）。"""
        if hasattr(self, 'working_memory') and self.working_memory is not None:
            self.working_memory.reset()
            print(f"[Cortex] WorkingMemory cleared")

    def set_dialogue_state(self, dialogue_state) -> None:
        """S12: 注册多轮对话状态管理器。

        注册后，generate() 会：
        - 每轮开始时加载上一轮的 field_state（隐式记忆上下文）
        - 每轮结束时保存 field_state 快照
        - 在 prompt 前插入轮次标记 token（可选）

        替代前缀拼接方案，让模型通过 field_state 记忆历史，
        而非把所有历史对话文本重新读一遍。

        未注册时（默认）完全无状态，保持原前缀拼接行为（向后兼容）。

        Args:
            dialogue_state: DialogueState 实例
        """
        self._dialogue_state = dialogue_state
        max_rounds = dialogue_state.max_rounds if dialogue_state else 0
        print(f"[Cortex] DialogueState enabled (max_rounds={max_rounds})")

    def clear_dialogue_state(self) -> None:
        """S12: 清空对话状态（新会话开始时调用）。"""
        if self._dialogue_state is not None:
            self._dialogue_state.reset()
            print(f"[Cortex] DialogueState cleared")

    def set_neuromodulator(self, neuromodulator) -> None:
        """P1-2: 注册神经调质状态，注入到 ensemble。

        注册后，ensemble.forward 每轮会：
        - 读取 get_refractory_multiplier() 调整不应期长度（血清素）
        - 读取 get_field_write_scale() 缩放场写入强度（去甲肾上腺素）

        未注册时 ensemble 退化为默认值 1.0（向后兼容）。
        """
        self._neuromodulator = neuromodulator
        self.ensemble.neuromodulator = neuromodulator
        print(f"[Cortex] NeuromodulatorState enabled "
              f"(dopamine={neuromodulator.dopamine:.2f}, "
              f"serotonin={neuromodulator.serotonin:.2f}, "
              f"norepinephrine={neuromodulator.norepinephrine:.2f})")

    def set_maturity(self, maturity) -> None:
        """注册成熟度追踪器，注入到 ensemble。

        注册后，ensemble.forward 写入场时：
        - 幼稚态神经元共振权重 = 0.1（先听后说，不污染集体意识场）
        - 成熟态神经元共振权重 = 1.0（完全贡献）

        未注册时 ensemble 退化为默认值 1.0（向后兼容）。
        """
        self.ensemble.maturity = maturity
        print(f"[Cortex] MaturityTracker enabled (幼稚态 weight=0.1, lr×3.0)")

    def set_sleep_consolidator(self, sleep_consolidator) -> None:
        """注册睡眠巩固器，用于跨会话 replay buffer 持久化。

        注册后，save_state/load_state 会自动持久化 sleep_consolidator 的
        replay_buffer 和 last_consolidation_step，使高共振经验不因重启丢失。
        """
        self._sleep_consolidator = sleep_consolidator
        print(f"[Cortex] SleepConsolidator registered (replay buffer 持久化)")

    def add_neuron(self, domain: str, lifecycle=None, from_split: Optional[str] = None) -> str:
        """运行时创建新神经元并加入 ensemble（neurogenesis 入口）。

        流程：
        1. 生成新 neuron ID（{domain}_{n} 格式，如 zh_1）
        2. 用 get_domain_neuron_config 创建 NeuronConfig（COMPACT 规格）
        3. 实例化 ResonanceNeuron → to(device) → eval
           - domain_prototype 由 sleep_engine contrastive phase EMA 更新
           - LuminaNet splitting 融合：from_split 指定父 neuron ID 时，
             继承父权重 + 微调噪声分化，新 neuron 起点更高
        4. 多模态注册（auto_register_modalities）
        5. 持久化 ckpt 到 neurons_dir/neuron_{nid}.pt
        6. 注入 cortex.neurons + ensemble.add_neuron
        7. lifecycle.maturity.register_new（幼稚态追踪）

        Args:
            domain: 域名（zh/en/code/math/general）
            lifecycle: LifecycleManager 实例（可选，用于 maturity.register_new）
            from_split: 父 neuron ID（LuminaNet splitting 融合）。
                        指定时继承父 neuron 权重 + 微调噪声分化，
                        新 neuron 起点高于随机初始化。None 时从零新建。

        Returns:
            新神经元的 ID（如 "zh_1"）
        """
        from neuroplex.resonance.config import get_domain_neuron_config, DOMAIN_VOCAB_SIZES

        if domain not in DOMAIN_VOCAB_SIZES:
            raise ValueError(
                f"未知 domain: {domain}. 可选: {list(DOMAIN_VOCAB_SIZES.keys())}"
            )

        # 校验 from_split 父 neuron 存在
        if from_split is not None and from_split not in self.neurons:
            raise ValueError(
                f"分裂父 neuron {from_split} 不存在，当前 neurons: {list(self.neurons.keys())}"
            )

        # 1. 生成唯一 neuron ID
        n = 1
        while f"{domain}_{n}" in self.neurons:
            n += 1
        nid = f"{domain}_{n}"

        # 2. 创建 NeuronConfig
        # 断裂 E 修复：接入 SpecSelector，根据错误率自动选择规格
        # - from_split 分裂模式：继承父 neuron 规格（保持同域同规格分化）
        # - 新建模式 + lifecycle：neurogenesis.select_spec(domain) 按错误率选 compact/standard/expert
        # - 新建模式 + 无 lifecycle：默认 compact（向后兼容）
        if from_split is not None:
            parent_spec = self.neurons[from_split].config.spec
            cfg = get_domain_neuron_config(domain, spec=parent_spec)
            logger.info(f"[Cortex] split 模式: {nid} 继承父 spec={parent_spec}")
        elif lifecycle is not None and hasattr(lifecycle, "neurogenesis"):
            selected_spec = lifecycle.neurogenesis.select_spec(domain)
            cfg = get_domain_neuron_config(domain, spec=selected_spec)
            logger.info(f"[Cortex] neurogenesis spec 选择: {domain} → {selected_spec}")
        else:
            cfg = get_domain_neuron_config(domain)
        cfg.neuron_id = nid

        # BioOSS: 按 ~20% 比例生成 inhibitory 神经元（人脑启发：兴奋/抑制分化）
        # 统计当前域内 inhibitory 比例，若 < 20% 则新建 inhibitory，否则 excitatory
        # from_split 模式：继承父 neuron 的 neuron_type（分裂保持同类）
        if from_split is not None:
            parent_neuron = self.neurons[from_split]
            cfg.neuron_type = parent_neuron.neuron_type
            logger.info(
                f"[Cortex] LuminaNet split: {nid} 继承父 neuron {from_split} "
                f"(neuron_type={cfg.neuron_type})"
            )
        else:
            domain_nids = [n for n in self.neurons if n.startswith(f"{domain}_")]
            if domain_nids:
                n_inhibitory = sum(
                    1 for n in domain_nids if self.neurons[n].is_inhibitory
                )
                inhibitory_ratio = n_inhibitory / len(domain_nids)
                if inhibitory_ratio < 0.2:
                    cfg.neuron_type = "inhibitory"
                    logger.info(
                        f"[Cortex] BioOSS: 新神经元 {nid} 设为 inhibitory "
                        f"(域 {domain} 当前 inhibitory 比例 {inhibitory_ratio:.0%} < 20%)"
                    )
                else:
                    cfg.neuron_type = "excitatory"
            else:
                # 域内首 neuron 默认 excitatory（先建立基础能力再分化抑制）
                cfg.neuron_type = "excitatory"

        # 3. 实例化神经元
        neuron = ResonanceNeuron(cfg).to(self.device)

        # LuminaNet splitting: 继承父权重 + 微调噪声分化
        # 子 neuron 初始权重 = 父权重 × (1 + ε)，ε ~ N(0, 0.01)
        # 这让子 neuron 起点接近父 neuron 但不完全相同，
        # 后续 intra_group diversity loss 会推动它们进一步分化
        if from_split is not None:
            parent_sd = self.neurons[from_split].state_dict()
            child_sd = neuron.state_dict()
            for key in child_sd:
                if key in parent_sd and child_sd[key].shape == parent_sd[key].shape:
                    # 只对 float 类型参数继承 + 噪声分化（跳过 int/refractory_counter 等）
                    if child_sd[key].dtype in (torch.float32, torch.float16, torch.float64):
                        noise = torch.randn_like(child_sd[key]) * 0.01
                        child_sd[key] = parent_sd[key].clone().to(dtype=child_sd[key].dtype) + noise
            neuron.load_state_dict(child_sd, strict=False)
            logger.info(f"[Cortex] split: {nid} 已继承 {from_split} 的权重 + 1% 噪声分化")

        neuron.eval()
        # domain_prototype 由 sleep_engine contrastive phase EMA 更新

        # 4. 多模态注册
        if self._tokenizer_hub is not None:
            try:
                neuron.auto_register_modalities(self._tokenizer_hub)
            except Exception as e:
                logger.warning(f"[Cortex] 新神经元 {nid} 多模态注册失败（非致命）: {e}")

        # 5. 持久化 ckpt
        ckpt_path = os.path.join(self.neurons_dir, f"neuron_{nid}.pt")
        os.makedirs(self.neurons_dir, exist_ok=True)
        torch.save(
            {"neuron_config": cfg, "state_dict": neuron.state_dict()},
            ckpt_path,
        )

        # 6. 注入 ensemble（cortex.neurons 和 ensemble.neurons 是同一引用）
        self.ensemble.add_neuron(nid, neuron)

        # P7 热插拔契约：生成路径若已启用 per-neuron shared embedding，
        # 新生 neuron 也必须有对应的感知层。分裂模式沿用父 neuron 的 embedding；
        # 无父 neuron 时回退到目录级 shared embedding，避免 _parallel_forward
        # 收到 None 并在 embed_adapter 处崩溃。
        if self._neuron_shared_embeddings is not None:
            shared_for_new = None
            if from_split is not None:
                shared_for_new = self._neuron_shared_embeddings.get(from_split)
            if shared_for_new is None:
                shared_for_new = self._shared_embedding
            if shared_for_new is not None:
                self._neuron_shared_embeddings[nid] = shared_for_new

        # 6.5 C26 增量七：新 neuron 注册到相位动力学（PhasorDynamics.add_neuron）。
        # 装配时相位表按初始 9 集合固定形状，缺此步 → continuous_forward 的
        # binding_tensor 维度错配（9 vs 10）崩溃。相位用同域先验（0 = 与 zh 同相，
        # 与装配 assign_phase_by_domain 的 zh 域 base 一致）。
        try:
            gamma = getattr(self, "gamma_oscillator", None)
            if gamma is not None and hasattr(gamma, "add_neuron"):
                gamma.add_neuron(nid, phase=0.0)
                logger.info(f"[Cortex] 相位动力学注册: {nid}（相位先验 0）")
        except Exception as e:
            logger.warning(f"[Cortex] 相位注册失败（非致命）: {e}")

        # 7. 注册幼稚态追踪
        if lifecycle is not None:
            try:
                lifecycle.maturity.register_new(nid)
            except Exception as e:
                logger.warning(f"[Cortex] maturity.register_new({nid}) 失败（非致命）: {e}")

        n_params = sum(p.numel() for p in neuron.parameters())
        logger.info(
            f"[Cortex] Neurogenesis: 新神经元 {nid} 已创建 "
            f"({cfg.spec}, {n_params/1e6:.0f}M params, ckpt→{ckpt_path})"
        )
        print(f"[Cortex] 🌱 Neurogenesis: {nid} ({cfg.spec}, {n_params/1e6:.0f}M params)")

        return nid

    def remove_neuron(self, nid: str, delete_ckpt: bool = True) -> bool:
        """运行时移除神经元（apoptosis 清理入口）。

        流程：
        1. 从 cortex.neurons / ensemble.neurons 移除（同一引用）
        2. 清理其他神经元的 excite_channels / inhibit_channels 中的引用
        3. 删除磁盘 ckpt 文件（可选）

        安全检查：不移除最后一个神经元（避免 ensemble 为空）。

        Args:
            nid: 要移除的神经元 ID
            delete_ckpt: 是否删除磁盘 ckpt 文件

        Returns:
            True 如果成功移除
        """
        if nid not in self.neurons:
            logger.warning(f"[Cortex] remove_neuron: {nid} 不存在")
            return False

        if len(self.neurons) <= 1:
            logger.warning(f"[Cortex] remove_neuron: 拒绝移除最后一个神经元 {nid}")
            return False

        # 增删锁：热插拔（推理读走快照隔离不拿锁，增删之间互斥防交错）
        with self._neurons_lock:
            if nid not in self.neurons:
                return False
            # 1. 从 neurons dict 移除（cortex.neurons 和 ensemble.neurons 是同一引用）
            self.neurons.pop(nid)
            if self._neuron_shared_embeddings is not None:
                self._neuron_shared_embeddings.pop(nid, None)

            # 2. 清理其他神经元的 side channel 引用（ModuleDict 按 key 删除）
            for other_neuron in self.neurons.values():
                try:
                    if hasattr(other_neuron, "excite_channels") and nid in other_neuron.excite_channels:
                        del other_neuron.excite_channels[nid]
                    if hasattr(other_neuron, "inhibit_channels") and nid in other_neuron.inhibit_channels:
                        del other_neuron.inhibit_channels[nid]
                except Exception:
                    pass

        # 3. 删除 ckpt 文件（锁外，I/O 不阻塞增删并发）
        if delete_ckpt:
            ckpt_path = os.path.join(self.neurons_dir, f"neuron_{nid}.pt")
            if os.path.exists(ckpt_path):
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    logger.warning(f"[Cortex] remove_neuron: 删除 ckpt 失败: {e}")

        logger.info(f"[Cortex] Apoptosis: 神经元 {nid} 已移除 (剩余 {len(self.neurons)} 个)")
        print(f"[Cortex] 🧹 Apoptosis: {nid} 已移除 (剩余 {len(self.neurons)} 个)")
        return True

    def isolate_neuron(self, nid: str) -> bool:
        """隔离神经元（凋亡级联：启动后摘除路由，保留权重与 ckpt）。

        人脑对应：凋亡级联启动后神经元功能被抑制但仍存活（可复活）。
        与 remove_neuron 的区别：不删除磁盘 ckpt，并记录恢复信息。

        Args:
            nid: 神经元 ID

        Returns:
            True 如果成功隔离
        """
        if nid not in self.neurons:
            logger.warning(f"[Cortex] isolate_neuron: {nid} 不存在")
            return False
        if len(self.neurons) <= 1:
            logger.warning(f"[Cortex] isolate_neuron: 拒绝隔离最后一个神经元 {nid}")
            return False

        domain = nid.split("_")[0] if "_" in nid else nid
        # 增删锁：与 remove/add 互斥（推理读走快照隔离不拿锁）
        with self._neurons_lock:
            if nid not in self.neurons:
                return False
            # 隔离前先把运行时最新权重（包括睡眠/整合写入的读路径与 LoRA）
            # 写回 checkpoint。否则 revive 只能恢复 add_neuron 时的初始副本，
            # 破坏“隔离保留权重、复活继续运行”的生命周期契约。
            ckpt_path = os.path.join(self.neurons_dir, f"neuron_{nid}.pt")
            try:
                neuron = self.neurons[nid]
                torch.save(
                    {"neuron_config": neuron.config,
                     "state_dict": neuron.state_dict()},
                    ckpt_path,
                )
            except Exception as e:
                logger.warning(f"[Cortex] isolate_neuron: 保存最新权重失败: {e}")
            # 从共享 dict 摘除（cortex.neurons 与 ensemble.neurons 同一引用）
            self.neurons.pop(nid)
            # 清理其他神经元的 side channel 引用（ModuleDict 按 key 删除）
            for other_neuron in self.neurons.values():
                try:
                    if hasattr(other_neuron, "excite_channels") and nid in other_neuron.excite_channels:
                        del other_neuron.excite_channels[nid]
                    if hasattr(other_neuron, "inhibit_channels") and nid in other_neuron.inhibit_channels:
                        del other_neuron.inhibit_channels[nid]
                except Exception:
                    pass

            # 记录恢复信息（ckpt 保留，未删除）
            if not hasattr(self, "_isolated"):
                self._isolated: Dict[str, dict] = {}
            self._isolated[nid] = {
                "domain": domain,
                "ckpt": ckpt_path,
                "shared_embedding": (
                    self._neuron_shared_embeddings.pop(nid, None)
                    if self._neuron_shared_embeddings is not None else None
                ),
            }
        logger.info(f"[Cortex] Isolate: 神经元 {nid} 已隔离（保留 ckpt，可复活）")
        print(f"[Cortex] 💤 Isolate: {nid} 已隔离（保留 ckpt，可复活）")
        return True

    def revive_neuron(self, nid: str) -> bool:
        """复活被隔离的神经元（隔离观察期分数回升，或手动干预）。

        从保留的 ckpt 重新加载 neuron 并加回共享 dict。
        注意：side_channels 拓扑需由调用方（sleep_engine）重新建立。

        Args:
            nid: 神经元 ID

        Returns:
            True 如果成功复活
        """
        isolated = getattr(self, "_isolated", {})
        if nid not in isolated:
            logger.warning(f"[Cortex] revive_neuron: {nid} 不在隔离池")
            return False

        info = isolated[nid]
        ckpt_path = info.get("ckpt", os.path.join(self.neurons_dir, f"neuron_{nid}.pt"))
        if not os.path.exists(ckpt_path):
            logger.warning(f"[Cortex] revive_neuron: ckpt 不存在 {ckpt_path}")
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg: NeuronConfig = ckpt["neuron_config"]
            sd = ckpt["state_dict"]
            sd = self._migrate_state_dict(sd, cfg)
            neuron = ResonanceNeuron(cfg).to(self.device)
            has_v2 = {"field_pool_query", "field_read_gate.weight"} <= set(sd.keys())
            neuron.load_state_dict(sd, strict=False)
            neuron.v1_compat = not has_v2
            neuron.eval()
            # C26 增量三补：恢复沉淀 LoRA 增量（与主装配路径同款）
            if neuron.load_lora(sd):
                logger.info("[Cortex] revive %s 恢复沉淀 LoRA 增量", nid)
            with self._neurons_lock:
                if nid in self.neurons:
                    logger.warning(f"[Cortex] revive_neuron: {nid} 已在运行中，跳过")
                    return True
                self.neurons[nid] = neuron
                if self._neuron_shared_embeddings is not None:
                    shared_emb = info.get("shared_embedding") or self._shared_embedding
                    if shared_emb is not None:
                        self._neuron_shared_embeddings[nid] = shared_emb
                del isolated[nid]
        except Exception as e:
            logger.error(f"[Cortex] revive_neuron 加载失败 {nid}: {e}")
            return False
        logger.info(f"[Cortex] Revive: 神经元 {nid} 已复活 (总数 {len(self.neurons)})")
        print(f"[Cortex] 🌱 Revive: {nid} 已复活")
        return True

    def get_isolated_neurons(self) -> list:
        """获取隔离池中的神经元 ID（诊断/复活用）。"""
        return list(getattr(self, "_isolated", {}).keys())

    def think(
        self,
        shared_embeddings: Optional[torch.Tensor] = None,
        active_nids: Optional[List[str]] = None,
        fusion_mode: str = "soft",
        neuron_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        return_judge_logits: bool = False,
        collab_mode: str = "fusion",
        memory_vectors: Optional[List] = None,
    ) -> Dict:
        """Run one round of resonance thinking.

        All neurons receive the same shared_embeddings (from shared embedding table).
        This ensures field vectors are comparable — cosine similarity is meaningful.

        P7-修复：支持 neuron_embeddings dict（每神经元独立编码，与训练一致）。
        2026-08-07 收敛：默认 fusion_mode 从 "per_position"（旧 entropy 启发式）改为
        "soft"（共振分融合，与训练 forward_train 对齐，见 generate/_generate_p7 默认）。

        Args:
            shared_embeddings: [B, L, base_embed_dim] from shared_embedding(general_ids).
            active_nids: 如果指定，只激活这些 neuron（硬件受限路由）。
                        None 表示全部参与（默认行为，向后兼容）。
            fusion_mode: 推理融合模式（主路径 "soft"；实验模式 "residual"/"consensus"/
                        "division"/"per_position" 供诊断对照）
                        - "soft"（默认）：共振分 softmax 融合（训练对齐）
                        - "per_position"：每位置按熵/置信度独立路由（旧，诊断用）
                        - "residual"：族长完整预测 + 其他神经元残差修正（方向③，实验）
            neuron_embeddings: {nid: [B, L, base_embed_dim]} 每神经元预编码 embedding
                              （优先级高于 shared_embeddings，与 ensemble.forward 一致）
            return_judge_logits: C24 双头（2026-08-09）：额外收集各 neuron 的
                judge_lm_head（general 判定头）logits → result["round1_judge_logits"]。
                executive 判定用 judge NLL（C20 原始信号链，可比）。
            collab_mode: C25-E（2026-08-11）："continuous" = 连续时间共振
                （ensemble.continuous_forward：相位绑定驱动的连续动力学替代离散轮次，
                融合权重 = 时间平均激活）。其余值走离散 forward（不变）。

        Returns:
            dict with field_state, neuron_logits, final_scores, n_rounds.
        """
        # C26 增量二：记忆可读进生成——归一化 memory_vectors 为
        # [(vec, weight), ...]（vec [D] 或 [1,D]），透传给 ensemble 写入场
        # （round1 判定信号之后），round2+ 场条件化 forward 读到记忆。
        # C27 增量二（KoPE）：支持 (vec, weight, phase) 3 元组/dict["phase"]
        # ——phase 为该记忆沉淀时的加权均值相角，注入按记忆相位对齐 theta
        # （相位归属记忆；无 phase 回退峰值对齐，增量五零回归）。
        seed_memories = None
        if memory_vectors:
            seed_memories = []
            for item in memory_vectors:
                phase = None
                if isinstance(item, dict):
                    vec, w = item.get("vector"), float(item.get("weight", 1.0))
                    phase = item.get("phase")
                elif isinstance(item, (tuple, list)) and len(item) >= 3:
                    vec, w, phase = item[0], float(item[1]), item[2]
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    vec, w = item[0], float(item[1])
                else:
                    vec, w = item, 1.0
                if vec is None:
                    continue
                seed_memories.append(
                    (vec, w, phase) if phase is not None else (vec, w))
        kwargs = dict(return_logits=True, active_nids=active_nids, fusion_mode=fusion_mode)
        if return_judge_logits:
            kwargs["return_judge_logits"] = True
        if neuron_embeddings is not None:
            kwargs["neuron_embeddings"] = {
                k: v.to(self.device) for k, v in neuron_embeddings.items()
            }
        elif shared_embeddings is not None:
            kwargs["shared_embeddings"] = shared_embeddings.to(self.device)
        if seed_memories:
            kwargs["seed_memories"] = seed_memories
        if collab_mode == "continuous":
            # C25-E：连续时间共振（相位绑定驱动，融合权重 = 时间平均激活）
            result = self.ensemble.continuous_forward(**kwargs)
        else:
            result = self.ensemble.forward(**kwargs)
        return result

    @torch.no_grad()
    def set_field_memory(self, bank) -> None:
        """C26 增量四：注入场记忆库——generate 自动记忆检索的数据源。

        由 sleep_engine.set_brain_interfaces 装配时调用（产品默认接入）；
        也可手动注入（未注入时 generate 自动检索静默跳过，向后兼容）。
        """
        self._memory_bank = bank

    @staticmethod
    def _is_degenerate_text(text: str) -> bool:
        """R9（REMEDIATION_PLAN 2026-08-14）：退化输出检测（与 verify 脚本同规则）。

        三类已知退化（欠训练 dialogue neuron 实测）：
        - 编号列表塌缩：`1.\n` 或字面量 `1.<0x0A>`（裸 prompt 死循环残留）
        - 重复标点/字符：同字符连续 >=4
        - 纯数字长串：剔除数字/空白/运算符/标点后仍 >=2 个非汉字非字母字符
          （>=2 排除 "1 + " 这类短算术 stub——那是截断不是退化循环）
        """
        if not text:
            return False
        if re.search(r"\d+\.\s*\n", text) or "1.<0x0A>" in text:
            return True
        if re.search(r"([。！？，,.！、]{2})\1{1,}", text) or re.search(
                r"(.)\1{3,}", text.replace("……", "。。")):
            return True
        stripped = re.sub(r"[0-9\s,，。.!！?？;；:：、+\-*/=×÷\"\'“”（）()<>]", "", text)
        return bool(stripped) and len(stripped) >= 2 and not re.search(
            r"[\u4e00-\u9fff\w]", stripped)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 60,
        temperature: float = 0.55,
        top_k: int = 15,
        domain: Optional[str] = None,
        repetition_penalty: float = 1.4,
        n_candidates: int = 1,
        routing_level: int = 1,
        active_nids: Optional[Union[str, List[str]]] = None,
        collab_mode: str = "continuous",
        fusion_mode: str = "soft",
        # 口径守卫例外开关（2026-08-12）：base/域 neuron 评估用纯问题 prompt 时传 True。
        _allow_plain_prompt: bool = False,
        # C26 增量二（2026-08-14）：记忆可读进生成——检索到的记忆向量
        # [{"vector": [D], "weight": sim} 或 (vec, weight)]，写入共振场做
        # 生成条件化（round2+ field_state 注入，区别于仅文本标签通道）。
        memory_vectors: Optional[List] = None,
        # C26 增量四（2026-08-14）：自动记忆检索——未显式传 memory_vectors
        # 且注入过记忆库时，用 prompt 场状态自动检索 top-1 记忆注入生成
        # （记忆从"显式 API"走向"模型内在自动调取"）。
        auto_memory: bool = True,
        # C27 增量一（2026-08-14）：实例级路由（SMCS 借鉴）——continuous 生成
        # 中激活子集按 chunk 级混合后验（共振分 + 滚动 NLL）双向域内演化，
        # 让"任务模式激活"在实例内自校正。默认开；显式关回退 C25-E 行为。
        instance_routing: bool = True,
    ) -> str:
        """Generate text using resonance ensemble (P7 only).

        Args:
            prompt: input text.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: top-k sampling.
            domain: P7 域指定（"zh"/"en"/"code"/"math"/"general"），
                    None 时自动推断。
            repetition_penalty: 重复惩罚系数（1.0=无惩罚，1.2=默认）。
            n_candidates: SMCS EPE 候选数。>1 时生成多条候选，用混合后验评分
                         （inter-response 一致性 + intra-response 置验度）选最优。
            routing_level: 硬件受限路由等级。
                           1=域路由（domain+general, 默认），
                           2=指纹 top-k 路由（fingerprint cosine 选最相关 neuron）。
            active_nids: 显式指定激活的神经元列表（实验用，覆盖路由逻辑）。
                         支持字符串模式：'auto_topK'/'auto_all'/'auto_top1'（稀疏激活，方向④）。
            fusion_mode: 推理融合模式（方向③ 残差预测编码）
                         - "per_position"（默认）：每位置按熵/置信度独立路由
                         - "residual"：族长完整预测 + 其他神经元残差修正
            collab_mode: "continuous"（默认，C25-E：同 executive 判定（judge NLL
                          主信号），共振为连续时间动力学——相位绑定驱动的连续
                          激活替代离散轮次，leader 用 round1_scores（t=0 场共振
                          分，质量信号）+ 时间平均激活选。2026-08-11 多次采样
                          统计确认（verify_c25_e_ab_stats 4/4：非空 1.00 持平、
                          重复率 0.011<0.022、质量 9胜2负1平）→ 替换 executive
                          为默认）
                          / "executive"（C19 任务级：回合级执行控制判定 dominant
                          域 → 任务模式激活 + 族长稳定生成；C20 回合级监督 5/5）
                          / "fusion"（token 级融合，C19 前旧范式，实验保留）
                          / "leader"（族长主导，不融合，实验保留）

        Returns:
            generated text string.
        """
        if self._tokenizer_hub is None:
            raise RuntimeError("TokenizerHub not set. Call cortex.set_tokenizer_hub() first.")

        # 训练/推理分离（人脑：学习时正常对话）：推理无锁读
        # 训练侧（sleep_engine）在影子权重（deepcopy）上进行，live 权重训练期间稳定，
        # 推理快照（nmap = dict(self.neurons)）读到稳定权重 → 无需与训练互斥。
        if n_candidates <= 1:
            text = self._generate_p7(
                prompt, max_tokens, temperature, top_k, domain,
                repetition_penalty, routing_level=routing_level,
                active_nids=active_nids, collab_mode=collab_mode,
                fusion_mode=fusion_mode,
                _allow_plain_prompt=_allow_plain_prompt,
                memory_vectors=memory_vectors,
                auto_memory=auto_memory,
                instance_routing=instance_routing,
            )
            # R9（REMEDIATION_PLAN 2026-08-14）：退化重试——已知退化模式
            # （编号塌缩 `1.\n` / 字面量 `1.<0x0A>`、重复标点、纯数字长串）
            # 命中时以更高温度重采样一次（人脑类比：感知异常输出即重置注意力）。
            # 二次仍退化则原样返回并打日志（保留证据，不静默吞掉）。
            if text and self._is_degenerate_text(text):
                retry_temp = min(temperature + 0.15, 1.2)
                logger.warning(
                    "[Cortex] 检测到退化输出 %r，重试（temp %.2f→%.2f）",
                    text[:30], temperature, retry_temp,
                )
                text = self._generate_p7(
                    prompt, max_tokens, retry_temp, top_k, domain,
                    repetition_penalty, routing_level=routing_level,
                    active_nids=active_nids, collab_mode=collab_mode,
                    fusion_mode=fusion_mode,
                    _allow_plain_prompt=_allow_plain_prompt,
                    memory_vectors=memory_vectors,
                    auto_memory=auto_memory,
                    instance_routing=instance_routing,
                )
                if text and self._is_degenerate_text(text):
                    logger.warning(
                        "[Cortex] 二次重试仍退化 %r，返回原文本（保留证据）",
                        text[:30],
                    )
            return text
        # SMCS EPE: 生成多条候选，混合后验评分选最优
        candidates = []
        for _ in range(n_candidates):
            try:
                text = self._generate_p7(
                    prompt, max_tokens, temperature, top_k, domain,
                    repetition_penalty, routing_level=routing_level,
                    active_nids=active_nids, collab_mode=collab_mode,
                    fusion_mode=fusion_mode,
                    _allow_plain_prompt=_allow_plain_prompt,
                    memory_vectors=memory_vectors,
                    auto_memory=auto_memory,
                    instance_routing=instance_routing,
                )
                if text:
                    candidates.append(text)
            except Exception:
                continue
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        return self._select_best_candidate(candidates)

    # ── C26 增量八：多阶段任务模式链 v2（2026-08-14）：TaskSet 序列调度器 ──
    # 对比 C25-F 首步（generate_staged dict 阶段，R17 曾标死代码）：v2 用
    # TaskSet 类型化对象（显式激活子集 = 任务集切换）+ 三重阶段间传递
    # （文本 prev + 场状态 seed_memories + 记忆写入）+ 阶段质量门，接入生产。

    def generate_task_chain(
        self,
        stages: List[TaskSet],
        max_tokens_per_stage: int = 32,
        temperature: float = 0.55,
        top_k: int = 15,
        repetition_penalty: float = 1.4,
        fusion_mode: str = "soft",
    ) -> Dict[str, object]:
        """多阶段任务模式链 v2（task-set 序列，人脑任务集切换）。

        一个任务 = TaskSet 序列，每阶段是一个任务集（激活模式 + 判定约束 +
        质量门）。阶段间三重传递：
        1. 文本：{prev} 模板填充上一阶段输出（C25-F 机制保留）
        2. 场状态：上一阶段 final field_state 作为下阶段 seed_memories
           （记忆注意窗条件化，C26 增量二/五机制）
        3. 记忆写入：阶段结束把场状态沉淀为记忆候选（record_memory=True，
           睡眠固化路径，C26 增量一/三机制）

        阶段质量门（quality_gate=True）：退化检测（_is_degenerate_text）→
        高温重试一次 → 仍退化则空串（异常隔离，阶段间互不污染）。

        Args:
            stages: TaskSet 序列
            max_tokens_per_stage: 默认阶段生成长度
            temperature/top_k/repetition_penalty/fusion_mode: 默认透传参数

        Returns:
            {"outputs": [阶段文本...], "field_states": [场状态...],
             "gates": [{stage_i: "ok"|"retried"|"degenerate"}...]}
        """
        outputs: List[str] = []
        field_states: List[Optional[torch.Tensor]] = []
        gates: List[Dict[str, str]] = []
        prev = ""
        prev_fs: Optional[torch.Tensor] = None
        for i, st in enumerate(stages):
            tpl = st.prompt.strip()
            gate_i: Dict[str, str] = {}
            if not tpl:
                outputs.append("")
                field_states.append(None)
                gates.append({"error": "empty_prompt"})
                continue
            if "{prev}" in tpl:
                stage_prompt = tpl.format(prev=prev)
            elif prev:
                stage_prompt = f"{tpl}\n{prev}"
            else:
                stage_prompt = tpl
            # 三重传递之二：上一阶段场状态 → 记忆注意窗（seed_memories）
            memory_vectors = None
            if prev_fs is not None:
                memory_vectors = [(prev_fs, 0.8)]
            try:
                text = self.generate(
                    prompt=stage_prompt,
                    max_tokens=st.max_tokens or max_tokens_per_stage,
                    temperature=st.temperature or temperature,
                    top_k=top_k,
                    domain=st.domain,
                    repetition_penalty=repetition_penalty,
                    active_nids=st.active_nids,
                    collab_mode=st.mode,
                    fusion_mode=fusion_mode,
                    memory_vectors=memory_vectors,
                )
            except Exception as e:
                text = ""
                gate_i["error"] = str(e)[:80]
                logger.error(f"[Cortex] generate_task_chain 阶段 {i} 失败: {e}")
            # 阶段质量门：退化检测 → 高温重试 → 仍退化隔离
            if st.quality_gate and text and self._is_degenerate_text(text):
                retry_temp = min((st.temperature or temperature) + 0.15, 1.2)
                logger.warning(
                    "[Cortex] task_chain 阶段 %d 退化 %r，重试（temp %.2f）",
                    i, text[:30], retry_temp,
                )
                try:
                    text = self.generate(
                        prompt=stage_prompt,
                        max_tokens=st.max_tokens or max_tokens_per_stage,
                        temperature=retry_temp,
                        top_k=top_k,
                        domain=st.domain,
                        repetition_penalty=repetition_penalty,
                        active_nids=st.active_nids,
                        collab_mode=st.mode,
                        fusion_mode=fusion_mode,
                        memory_vectors=memory_vectors,
                    )
                except Exception:
                    text = ""
                if text and self._is_degenerate_text(text):
                    gate_i["quality"] = "degenerate"
                    logger.warning("[Cortex] task_chain 阶段 %d 重试仍退化，隔离", i)
                else:
                    gate_i["quality"] = "retried"
            else:
                gate_i["quality"] = "ok"
            outputs.append(text)
            # 三重传递之一/二：截获本阶段 final 场状态 → 下阶段 seed_memories
            try:
                fs = self.get_last_field_state()
            except Exception:
                fs = None
            field_states.append(fs)
            # 三重传递之三：阶段场状态沉淀为记忆候选（睡眠固化路径）
            if st.record_memory and fs is not None and text.strip():
                try:
                    from neuroplex.life.sleep_engine import get_sleep_engine
                    engine = get_sleep_engine()
                    label = st.memory_label or tpl.strip()[:40]
                    engine.record_field_memory(
                        fs, label, text=text, phase=self.get_last_phase())
                    gate_i["memory"] = "recorded"
                except Exception as e:
                    gate_i["memory"] = f"skip:{str(e)[:40]}"
            gates.append(gate_i)
            prev = text
            prev_fs = fs
        return {"outputs": outputs, "field_states": field_states, "gates": gates}

    # ── C25-F 兼容层（2026-08-11）：dict 阶段 → TaskSet 转发（生产不再推荐）──
    def generate_staged(
        self,
        stages: List[Dict],
        max_tokens_per_stage: int = 32,
        temperature: float = 0.55,
        top_k: int = 15,
        repetition_penalty: float = 1.4,
        fusion_mode: str = "soft",
    ) -> List[str]:
        """C25-F 首步兼容入口：dict 阶段自动升级为 TaskSet 走 v2 调度器。

        生产路径推荐 generate_task_chain（TaskSet 序列，三重传递 + 质量门）。
        本方法仅保持 C25-F 时代调用点兼容（verify_c25_f 等），返回各阶段文本。
        """
        task_sets = [TaskSet(
            prompt=st.get("prompt", ""),
            mode=st.get("mode", "continuous"),
            domain=st.get("domain"),
            active_nids=st.get("active_nids"),
            max_tokens=st.get("max_tokens"),
            temperature=st.get("temperature"),
            quality_gate=st.get("quality_gate", True),
        ) for st in stages]
        return self.generate_task_chain(
            task_sets,
            max_tokens_per_stage=max_tokens_per_stage,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            fusion_mode=fusion_mode,
        )["outputs"]

    # 注：R17 曾标 DEAD CODE（"仅 generate_staged 调用"），核实有误——本方法
    # 由 generate(n_candidates>1) 的 SMCS EPE 路径调用，是活跃生产代码。
    def _select_best_candidate(self, candidates: List[str]) -> str:
        """SMCS EPE 混合后验评分选最优候选。

        评分维度：
        1. Intra-response 置信度：候选长度（太短=低置信，太长=可能跑偏）
        2. Inter-response 一致性：与其他候选的 n-gram 重叠度（高一致=多采样收敛）
        3. 重复率惩罚：单候选内部 token 重复率（越低越好）

        综合分 = 一致性 + 长度置信 - 重复率
        """
        if not candidates:
            return ""
        n = len(candidates)
        if n == 1:
            return candidates[0]

        # 1. 计算 4-gram 集合（用于 inter-response 一致性）
        def to_ngrams(text: str, n: int = 4) -> set:
            tokens = text.split()
            if len(tokens) < n:
                return set(tokens)
            return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

        ngram_sets = [to_ngrams(c) for c in candidates]

        scores = []
        for i, text in enumerate(candidates):
            # Intra: 长度置信度（对数尺度，中等长度最优）
            length = len(text.split())
            if length == 0:
                scores.append(-1e9)
                continue
            length_score = -abs((length - 30) / max(length, 1)) * 0.3

            # Inter: 与其他候选的平均 n-gram 重叠
            if ngram_sets[i] and n > 1:
                overlaps = []
                for j in range(n):
                    if j != i and ngram_sets[j]:
                        overlap = len(ngram_sets[i] & ngram_sets[j]) / max(
                            len(ngram_sets[i] | ngram_sets[j]), 1
                        )
                        overlaps.append(overlap)
                inter_score = sum(overlaps) / max(len(overlaps), 1)
            else:
                inter_score = 0.0

            # 重复率：单候选内部重复 token 比例
            tokens = text.split()
            if tokens:
                unique_ratio = len(set(tokens)) / len(tokens)
            else:
                unique_ratio = 0.0
            repeat_penalty = (1 - unique_ratio) * 0.5

            total = inter_score + length_score - repeat_penalty
            scores.append(total)

        best_idx = scores.index(max(scores))
        return candidates[best_idx]

    def detect_modality(self, input_data: Union[str, torch.Tensor, dict]) -> str:
        """P8: 检测输入数据的模态。

        路由顺序：
        1. 显式 dict {"modality": "image", "data": ...} → 直接取
        2. torch.Tensor → 根据维度推断（3D float → image/audio 连续特征）
        3. str → "text"

        Args:
            input_data: str / torch.Tensor / dict

        Returns:
            modality name ("text"/"image"/"audio"/"video")
        """
        if isinstance(input_data, dict):
            return input_data.get("modality", "text")
        if isinstance(input_data, torch.Tensor):
            # [B, L, raw_dim] float → 连续特征（图像/音频）
            if input_data.dim() == 3 and input_data.dtype != torch.long:
                # 默认归为 image，具体模态由调用方通过 dict 指定
                return "image"
            # [B, L] long → token id（文本或离散化的多模态）
            return "text"
        return "text"

    def _infer_domain(self, text: str) -> str:
        """P7: 从文本内容启发式推断域。

        检测顺序：code > math > zh > en > general。
        code/math 检测必须在 CJK 之前，防止英文数学/代码被误判为 en。
        仅在对应域 neuron 已加载时返回该域。

        Returns:
            domain name ("zh"/"en"/"code"/"math"/"general")
        """
        neuron_domains = set(self.neurons.keys())
        if not neuron_domains:
            return "general"

        # 从 neuron key 提取纯域前缀（支持同域多神经元：zh_aug0_dialogue → zh）
        def _has_domain(prefix: str) -> bool:
            return any(k == prefix or k.startswith(prefix + "_") for k in neuron_domains)

        def _first_domain() -> str:
            """返回第一个 neuron 的纯域前缀（fallback）。"""
            first_key = next(iter(neuron_domains))
            return first_key.split("_")[0]

        # 1. 代码检测：强信号关键字（1 个即判定）+ 结构特征
        strong_code = [
            'def ', 'class ', 'function ', 'async ',
            'const ', 'let ', 'var ', 'SELECT ', 'CREATE TABLE',
            'docker ', 'git ', 'npm ', 'kubectl ', 'pip ',
            'package ', '#include', 'require(',
        ]
        # import/from 需要排除自然语言用法（"from 0 to 1", "from the"）
        code_keywords = [
            'return ', 'if __name__', 'print(',
            'lambda ', 'try:', 'except ', 'raise ',
        ]
        code_patterns = ['{', '};', '=>', 'self.', 'std::',
                         'def __init__', 'class ']
        if _has_domain('code'):
            strong_score = sum(1 for kw in strong_code if kw in text)
            weak_score = sum(1 for kw in code_keywords if kw in text)
            pattern_score = sum(1 for p in code_patterns if p in text)
            # import/from: 排除 "from 0", "from the", "from a" 等自然语言用法
            import_score = 0
            for m in ['import ', 'from ']:
                idx = text.find(m)
                if idx >= 0:
                    after = text[idx + len(m):].lstrip()
                    # 如果后面是自然语言词（数字、冠词等），不是代码
                    if not (after[:1].isdigit() or after.startswith(('the ', 'a ', 'an '))):
                        import_score += 1
            if '```' in text:
                pattern_score += 5
            # 强信号 1 个即判定，弱信号需 2 个
            if strong_score >= 1 or import_score >= 1 or weak_score + pattern_score >= 2:
                return 'code'

        # 2. 数学检测：多路信号融合（符号 + 关键词 + 公式特征 + 上下标）
        if _has_domain('math'):
            # 2a. 数学符号（Unicode 数学字符 + 基础运算符）
            math_symbols = set('=+-*/^∑∫∏√∞∂∇∈⊂∪∩∀∃≤≥≠≈±→←↑↓⇒⇐')
            math_sym_count = sum(1 for c in text if c in math_symbols)

            # 2b. 数学关键词（英文数学术语，大小写不敏感）
            math_keywords = [
                'derivative', 'integral', 'theorem', 'proof', 'equation',
                'sin', 'cos', 'tan', 'log', 'ln', 'limit',
                'matrix', 'vector', 'tensor', 'eigen', 'calculus',
                'algebra', 'geometry', 'probability', 'distribution',
                'gradient', 'fourier', 'laplace', 'taylor', 'riemann',
                'convergence', 'divergence', 'differential', 'polynomial',
                'hypothesis', 'variable', 'coefficient', 'parameter',
                'pythagorean', 'fibonacci', 'factorial', 'logarithm',
                'bayes', 'gaussian', 'stochastic', 'determinant',
                'chain rule', 'product rule', 'quotient rule',
            ]
            math_kw_count = sum(1 for kw in math_keywords if kw.lower() in text.lower())

            # 2c. 公式特征：上下标数字、希腊字母、函数调用模式
            superscript = sum(1 for c in text if '\u00b2' <= c <= '\u00b9')  # ²³⁴...
            subscript = sum(1 for c in text if '\u2080' <= c <= '\u2089')    # ₀₁₂...
            greek = sum(1 for c in text if '\u0391' <= c <= '\u03c9')       # Α-ω
            # 函数调用模式 f(x), g(x), h(x)
            fn_call = 1 if text.count('(') >= 1 and text.count(')') >= 1 and any(
                p in text for p in ['f(', 'g(', 'h(', 'f(x)', 'g(x)', 'h(x)',
                                    'sin(', 'cos(', 'tan(', 'log(', 'ln(']
            ) else 0

            # 2d. 数字表达式密度（纯数字 + 运算符占比高）
            stripped = text.replace(' ', '').replace('\n', '')
            digit_ops = sum(1 for c in stripped if c.isdigit() or c in '+-*/=()^.,')
            digit_ratio = digit_ops / max(len(stripped), 1)

            # 综合判定（满足任一条件）
            math_total = (math_sym_count + math_kw_count * 2
                          + superscript + subscript + greek + fn_call)
            if math_total >= 1 or digit_ratio > 0.4:
                return 'math'

        # 3. 中文检测（CJK 统一汉字区块）
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cjk_count > len(text.replace(' ', '')) * 0.3:
            return 'zh' if _has_domain('zh') else _first_domain()

        # 4. 默认：en 或 general
        if _has_domain('en'):
            return 'en'
        if _has_domain('general'):
            return 'general'
        return _first_domain()

    @torch.no_grad()
    def _executive_route(
        self,
        text: str,
        quality_weight: float = 0.4,
    ) -> tuple:
        """C19 回合级任务模式判定（执行控制，2026-08-08）。

        人脑参照：前额叶执行控制网络决定"当前任务模式"（task set），模式确定后
        整条通路激活直到任务结束——不做 token 级竞争。取代 C12-C16 的 token 级
        全局 softmax 路由（NLL/cosine/logit 跨 neuron 天然不可比，三次失败）。

        混合信号：
        1. 启发式域判定 _infer_domain（内容类型信号：代码关键字/数学符号/CJK）
        2. quality_head 回合级聚合（learned：各 neuron round1 对回合文本的
           预测质量 logit，按域聚合 mean）——C16 结构升级为回合级粒度

        融合规则：启发式定基础域；quality 域分显著占优（>1.5× 基准）且与
        启发式不同 → 切换（与 hybrid 共振校验同逻辑，回退安全）。

        Returns:
            (dominant_domain, confidence, per_domain_scores)
        """
        domain = self._infer_domain(text)
        per_domain = {}  # {domain: mean quality z-score}
        quality_ready = False
        judge_nll: Optional[Dict[str, float]] = None
        try:
            if self._neuron_shared_embeddings and self._general_sp is not None:
                general_ids = self._general_sp.encode(text)
                if not general_ids:
                    general_ids = [0]
                ids_t = torch.tensor([general_ids], dtype=torch.long, device=self.device)
                neuron_embeddings = {}
                for nid, emb in self._neuron_shared_embeddings.items():
                    neuron_embeddings[nid] = emb(ids_t)
                probe = self.think(
                    active_nids=list(self.neurons.keys()),
                    neuron_embeddings=neuron_embeddings,
                    return_judge_logits=True,  # C24 双头：judge NLL 判定信号
                )

                # ── C24 双头（2026-08-09）：judge NLL 主信号 ──
                # 各 neuron 的 judge_lm_head（general 256K 判定空间）对回合文本的
                # NLL 天然可比（C20 当年 5/5 的原始信号链）。替代 quality_head proxy
                # ——其 logit 会膨胀（zh_aug2 ql 68-102，softmax 饱和 → KL 梯度消失
                # → 自我强化），EMA z-score 压不住。judge NLL 无训练依赖、无膨胀。
                jl = probe.get("round1_judge_logits")
                if jl:
                    judge_nll = {}
                    tgt = torch.tensor(general_ids[1:], dtype=torch.long, device=self.device)
                    for nid, lg in jl.items():
                        # lg: [1, L, V]；logits[:, :-1] 预测 ids[1:]（next-token）
                        logp = torch.log_softmax(lg[:, :-1, :], dim=-1)  # [1, L-1, V]
                        nll_tok = -logp[0].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                        mask = (tgt != 1) & (tgt != 0)  # 忽略 unk/pad
                        if mask.sum() == 0:
                            continue
                        judge_nll[nid] = float((nll_tok * mask).sum() / mask.sum().float())
                if judge_nll is None:
                    judge_nll = {}

                ql = probe.get("quality_logits")
                if ql is not None:
                    nids = list(self.neurons.keys())
                    # 对齐校验：ql 顺序 = round1 active 顺序（active_nids 传全量时 == nids）
                    # 长度不匹配（共振快照异常）→ 放弃 quality，回退纯启发式
                    if len(ql) == len(nids):
                        # per-neuron EMA 标准化（C16b 教训：quality logit 跨 neuron
                        # 不可比，code 未校准 head 恒高 → 不做相对自身水平的 z-score
                        # 会 code 独占）。EMA 纯统计（detach），不参与梯度。
                        quality_ready = True
                        zs = []
                        for i, nid in enumerate(nids):
                            v = float(ql[i].detach())
                            s = self._quality_logit_ema.get(nid)
                            if s is None:
                                s = {"mean": v, "ms": v * v, "count": 1.0}
                                self._quality_logit_ema[nid] = s
                            else:
                                s["count"] += 1.0
                                a = min(self._quality_ema_alpha, 1.0 / s["count"])
                                s["mean"] = (1 - a) * s["mean"] + a * v
                                s["ms"] = (1 - a) * s["ms"] + a * v * v
                            if s["count"] >= self._quality_ema_warmup:
                                var = max(s["ms"] - s["mean"] ** 2, 1e-4)
                                zs.append((float(ql[i]) - s["mean"]) / (var ** 0.5))
                            else:
                                zs.append(None)  # 未成熟，该 neuron 不参与
                        # 只聚合成熟 neuron 的 z；全未成熟 → quality_ready=False
                        for i, nid in enumerate(nids):
                            if zs[i] is None:
                                quality_ready = False
                                break
                        if quality_ready:
                            for i, nid in enumerate(nids):
                                d = nid.split("_")[0]
                                per_domain.setdefault(d, []).append(zs[i])
        except Exception:
            pass  # probe 失败 → 纯启发式（回退安全）

        # ── C24 双头：judge NLL 域判定（主信号）──
        # 判定 = judge NLL 最低域；与启发式不同且显著占优（NLL 差 ≥ 1.0，
        # 诊断最小显著差 ~2）→ 切换（回退安全：judge 不显著时保留启发式）。
        per_domain_nll: Optional[Dict[str, float]] = None
        if judge_nll:
            agg: Dict[str, List[float]] = {}
            for nid, v in judge_nll.items():
                d = nid.split("_")[0]
                agg.setdefault(d, []).append(v)
            per_domain_nll = {d: sum(v) / len(v) for d, v in agg.items()}
            judge_domain = min(per_domain_nll, key=per_domain_nll.get)
            base_nll = per_domain_nll.get(domain, float("inf"))
            if judge_domain != domain and base_nll - per_domain_nll[judge_domain] >= 1.0:
                domain = judge_domain

        per_domain_scores = {}
        if per_domain_nll is None and quality_ready and per_domain:
            # judge 不可用 → quality z-score 回退（原 C20 逻辑）
            per_domain_scores = {d: sum(v) / len(v) for d, v in per_domain.items()}
            best_q_domain = max(per_domain_scores, key=per_domain_scores.get)
            base = per_domain_scores.get(domain, 0.0)
            if (best_q_domain != domain
                    and per_domain_scores[best_q_domain] > base * 1.5 + 1e-6
                    and per_domain_scores[best_q_domain] - base >= 0.7):
                domain = best_q_domain
        elif per_domain_nll is not None:
            # 诊断信息：judge NLL 域分数（供验证输出/监控）
            per_domain_scores = {d: -v for d, v in per_domain_nll.items()}
        # 未成熟（warmup 内）或 probe 失败 → 纯启发式，quality 不主导（回退安全）
        conf = 0.7 + 0.3 * (quality_weight if per_domain_scores else 0.0)
        return domain, conf, per_domain_scores

    def _fingerprint_route(
        self,
        general_ids: List[int],
        top_k: int = 2,
    ) -> List[str]:
        """Level 2 prototype 路由：用 domain_prototype cosine 相似度选 top-k neuron。

        每个 neuron 用自己的 embed_adapter 投影 prompt，再与自己的 domain_prototype
        做 cosine。每个 neuron 用自己的视角"看"prompt，符合神经元独立性。

        Args:
            general_ids: prompt 的 general tokenizer id 列表。
            top_k: 选择的 neuron 数量（不含 general）。

        Returns:
            active neuron id 列表。
        """
        if not self.neurons or self._shared_embedding is None:
            return list(self.neurons.keys())

        try:
            ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=self.device)
            prompt_emb = self._shared_embedding(ids_tensor)  # [1, L, 512]
            prompt_pooled = prompt_emb.mean(dim=1)  # [1, 512]
        except Exception:
            return list(self.neurons.keys())

        # 每个 neuron 用自己的 embed_adapter 投影 prompt，再与 prototype 比较
        # C5: 多原型模式取 max cosine（与最近原型的相似度）
        sims = {}
        for nid, neuron in self.neurons.items():
            try:
                if hasattr(neuron, 'embed_adapter') and neuron.embed_adapter is not None:
                    # 用 neuron 自己的 embed_adapter 投影到 768 维
                    projected = neuron.embed_adapter(prompt_pooled)  # [1, 768]
                    proj_vec = projected.squeeze(0)  # [768]
                    proj_norm = proj_vec / (proj_vec.norm() + 1e-8)
                    # C5: 多原型取 max cosine
                    if getattr(neuron, "num_prototypes", 1) > 1 and neuron.domain_prototypes is not None:
                        # 多原型: [K, 768] → max cosine
                        protos = neuron.domain_prototypes  # [K, 768]
                        proto_norms = protos / (protos.norm(dim=-1, keepdim=True) + 1e-8)
                        sim = float((proj_norm.unsqueeze(0) * proto_norms).sum(dim=-1).max().item())
                    else:
                        # 单原型（向后兼容）
                        proto = neuron.domain_prototype  # [768]
                        proto_norm = proto / (proto.norm() + 1e-8)
                        sim = float((proj_norm * proto_norm).sum().item())
                else:
                    # fallback: 无 embed_adapter 则跳过
                    continue
                sims[nid] = sim
            except Exception:
                continue

        if not sims:
            return list(self.neurons.keys())

        # 按相似度排序，选 top-k（排除 general，单独保证）
        sorted_nids = sorted(sims, key=sims.get, reverse=True)
        non_general = [nid for nid in sorted_nids if nid != "general"]
        selected = non_general[:top_k]

        # general 始终包含
        if "general" in self.neurons and "general" not in selected:
            selected.append("general")

        return selected if selected else list(self.neurons.keys())

    def _auto_topk_route(
        self,
        general_ids: List[int],
        top_k: int = 3,
    ) -> List[str]:
        """自动选共振分 top-K 神经元（稀疏激活 auto_topK）。

        复用 _fingerprint_route 的 domain_prototype cosine 相似度计算，
        但不强制包含 general，支持灵活 top_k。

        三模式：
          auto_top1 → k=1（实时模式：单族长主导）
          auto_top3 → k=3（平衡模式：族长协作）
          auto_all  → 全激活（高质量模式）

        Args:
            general_ids: prompt 的 general tokenizer id 列表。
            top_k: 选择的 neuron 数量。

        Returns:
            active neuron id 列表。
        """
        if top_k <= 0 or top_k >= len(self.neurons):
            return list(self.neurons.keys())

        if not self.neurons or self._shared_embedding is None:
            return list(self.neurons.keys())

        try:
            ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=self.device)
            prompt_emb = self._shared_embedding(ids_tensor)  # [1, L, 512]
            prompt_pooled = prompt_emb.mean(dim=1)  # [1, 512]
        except Exception:
            return list(self.neurons.keys())

        # 每个 neuron 用自己的 embed_adapter 投影 prompt，再与 prototype 比较
        sims = {}
        for nid, neuron in self.neurons.items():
            try:
                if hasattr(neuron, 'embed_adapter') and neuron.embed_adapter is not None:
                    projected = neuron.embed_adapter(prompt_pooled)
                    proj_vec = projected.squeeze(0)
                    proj_norm = proj_vec / (proj_vec.norm() + 1e-8)
                    proto = neuron.domain_prototype
                    proto_norm = proto / (proto.norm() + 1e-8)
                    sim = float((proj_norm * proto_norm).sum().item())
                else:
                    continue
                sims[nid] = sim
            except Exception:
                continue

        if not sims:
            return list(self.neurons.keys())

        # 按相似度排序，选 top-k（不强制包含 general）
        sorted_nids = sorted(sims, key=sims.get, reverse=True)
        selected = sorted_nids[:top_k]

        return selected if selected else list(self.neurons.keys())

    def _get_domain_to_general_alignment(
        self, domain: str, domain_sp
    ) -> Dict[int, list]:
        """S6: 构建 domain token ID → general token IDs 对齐表（带缓存 + 热插拔失效）。

        消除自回归生成时的 domain→text→general re-encode 往返。
        对每个 domain token，预计算其 general token IDs 映射。

        热插拔：缓存项携带 tokenizer 指纹，任一 tokenizer（域/general）被替换后
        自动失效重建；也可用 invalidate_alignment_cache() 手动失效。

        可编辑层：set_alignment_rules() 注入的 AlignmentRules 中匹配的 domain
        piece 跳过自动转译，改用人工指定的 general piece 文本编码
        （新增特殊神经元时补充专业术语映射）。

        Args:
            domain: 域名（如 "zh"）
            domain_sp: 域 tokenizer

        Returns:
            {domain_token_id: [general_token_ids]} 映射表
        """
        # 指纹 = (域 tokenizer 指纹, general tokenizer 指纹, 规则版本)
        rules_ver = self._alignment_rules.version if self._alignment_rules is not None else 0
        fp = (
            tokenizer_fingerprint(domain_sp),
            tokenizer_fingerprint(self._general_sp),
            rules_ver,
        )
        cached = self._domain_to_general_cache.get(domain)
        if cached is not None and cached.get("fp") == fp:
            return cached["alignment"]

        if self._general_sp is None:
            return {}

        alignment: Dict[int, list] = {}
        vocab_size = domain_sp.GetPieceSize() if hasattr(domain_sp, 'GetPieceSize') else 0
        for domain_id in range(vocab_size):
            piece = domain_sp.id_to_piece(domain_id)
            manual = None
            if self._alignment_rules is not None:
                manual = self._alignment_rules.get(domain, piece)
            if manual is not None:
                # 人工规则：general piece 文本 → general ids（可多段，逐段 encode 拼接）
                general_ids = []
                for gp in manual:
                    general_ids.extend(self._general_sp.encode(gp))
                alignment[domain_id] = general_ids if general_ids else [
                    self._general_sp.pad_id() if hasattr(self._general_sp, 'pad_id') else 0
                ]
                continue
            if piece.startswith("<0x") and piece.endswith(">"):
                # byte fallback piece（如 <0x0A>）：必须 decode 成真实字节再 encode，
                # 否则 "<0x0A>" 会被当作 6 个字符编码，换行语义丢失
                text = domain_sp.decode([domain_id])
            else:
                text = piece
            general_ids = self._general_sp.encode(text)
            if general_ids:
                alignment[domain_id] = general_ids
            else:
                # 空映射用 pad_id 兜底
                pad_id = self._general_sp.pad_id() if hasattr(self._general_sp, 'pad_id') else 0
                alignment[domain_id] = [pad_id]

        self._domain_to_general_cache[domain] = {"fp": fp, "alignment": alignment}
        print(f"[S6] 域 '{domain}' 对齐表已构建: {len(alignment)} entries", flush=True)
        return alignment

    def set_alignment_rules(self, rules) -> None:
        """注入可编辑词库规则层（AlignmentRules）。

        规则增删后（version 变化）S6 对齐表缓存自动失效重建。

        Args:
            rules: AlignmentRules 实例（None 时清除规则层）。
        """
        self._alignment_rules = rules

    def invalidate_alignment_cache(self, domain: Optional[str] = None) -> None:
        """词库热插拔手动失效：tokenizer 替换后强制重建对齐表缓存。

        指纹校验已覆盖大多数替换场景（自动失效）；此接口用于强制清空
        （如 TokenizerHub 整体重载、或调试时需要重建）。

        Args:
            domain: 指定域名失效；None 时清空全部。
        """
        if domain is None:
            self._domain_to_general_cache.clear()
        else:
            self._domain_to_general_cache.pop(domain, None)

    # ─── C25-E 遗留：continuous leader 融合质量信号（2026-08-14）─────────────
    # 诊断证实（diag_c25e_leader_quality_gap.py）：aug2 场共振分系统性碾压
    # （0.7-0.93 vs 0.01-0.17）当选 leader 5/7 次，但其生成质量（zh lm_head
    # NLL）常是 5 个 dialogue 中最差——"共振分高但生成差"弱 neuron 独占。
    # 融合：域内归一化共振分 × 质量信号（prompt NLL 的负数）等权求和。
    @staticmethod
    def _fuse_leader_quality(resonance_scores: dict, nll_quality: dict,
                             alpha: float = 0.5) -> dict:
        """连续 leader 融合分数（共振分与 -NLL 域内 min-max 归一化后加权）。

        Args:
            resonance_scores: {nid: round1_scores}（t=0 场共振分，越大越强）
            nll_quality: {nid: -NLL}（生成质量，越大越好）
            alpha: 共振分权重（0.5 = 等权；质量权重 = 1-alpha）

        Returns:
            {nid: fused_score}（仅含两信号都有的 neuron；空 → 调用方回退）
        """
        common = [k for k in nll_quality if k in resonance_scores]
        if not common:
            return {}

        def _minmax(values: dict) -> dict:
            lo, hi = min(values.values()), max(values.values())
            if hi - lo < 1e-9:
                return {k: 0.5 for k in values}
            return {k: (v - lo) / (hi - lo) for k, v in values.items()}

        r_norm = _minmax({k: resonance_scores[k] for k in common})
        q_norm = _minmax({k: nll_quality[k] for k in common})
        return {k: alpha * r_norm[k] + (1 - alpha) * q_norm[k] for k in common}

    def _nll_quality_from_round1_logits(
        self, result: dict, prompt: str, domain: str,
    ) -> dict:
        """从 round1 logits 计算各 neuron 对 prompt 的 next-token NLL 质量。

        continuous leader 融合信号（C25-E 遗留）：质量 = 该 neuron 对 prompt
        的拟合度（域头在 general→domain 位置对齐空间的 next-token NLL，越低
        越贴合该域训练分布）。用 round1 独立 logits（leader 生成同源），零额外
        前向。返回 {nid: -NLL}（越大质量越好）；失败返回 {}（调用方回退）。

        Args:
            result: think() 返回（含 round1_logits: {nid: [1, L, V]}）
            prompt: 生成输入（质量信号针对初始 prompt，不随生成增长）
            domain: 域（用于取对应 tokenizer；zh 50K 与 dialogue lm_head 对齐）
        """
        r1_logits = result.get("round1_logits") or {}
        if not r1_logits:
            return {}
        hub = getattr(self, "_tokenizer_hub", None)
        if hub is None or not hasattr(hub, "get_tokenizer"):
            return {}
        try:
            tok = hub.get_tokenizer(domain) or hub.get_tokenizer("general")
            if tok is None or self._general_sp is None:
                return {}
            # 生成前向的序列位置来自 general tokenizer；域头目标来自 domain
            # tokenizer。两者词元数通常不同，必须按字符 span 对齐后再计算
            # next-token NLL，不能直接把两套 tokenizer 的 id 按位置硬配。
            _, aligned_targets = build_position_alignment(
                prompt, tok, self._general_sp,
            )
            aligned_targets = aligned_targets.to(self.device)
        except Exception:
            return {}
        if aligned_targets.numel() < 2:
            return {}
        vocab = int(tok.GetPieceSize()) if hasattr(tok, "GetPieceSize") else None
        if not vocab:
            return {}
        out: dict = {}
        for nid, lg in r1_logits.items():
            if lg.shape[-1] != vocab:
                continue
            try:
                lg = lg.detach()  # 推理质量信号：仅前向，不携带梯度
                n = min(lg.shape[1] - 1, aligned_targets.numel() - 1)
                if n < 1:
                    continue
                logp = torch.log_softmax(lg[:, :n, :], dim=-1)  # [1, n, V]
                tgt = aligned_targets[1:n + 1]
                mask = (tgt >= 0) & (tgt != 1) & (tgt != 0)
                safe_tgt = tgt.clamp_min(0).unsqueeze(0).unsqueeze(-1)
                nll_tok = -logp.gather(-1, safe_tgt).squeeze(-1)  # [1, n]
                if mask.sum() == 0:
                    continue
                out[nid] = -float((nll_tok * mask).sum() / mask.sum().float())
            except Exception:
                continue
        return out

    # ─── C27 增量一（2026-08-14）：实例级路由 + 混合后验（SMCS 借鉴）──────
    # SMCS 的 contextual selection 在实例内重新选 expert 子集；此处受 C22
    # "回合级任务判定"外层约束，在 continuous 生成中每 instance_chunk token
    # 用滚动后验（共振分 + 已生成文本滚动 NLL）双向域内演化激活子集——
    # 剔除持续劣化（迟滞防抖）、加入后验显著更高的未激活同域 neuron。

    def _rolling_nll_quality(
        self, result: dict, gen_text: str, domain: str, window: int,
    ) -> dict:
        """滚动后验（C27 增量一）：对已生成文本窗口的 next-token NLL 质量。

        与 _nll_quality_from_round1_logits（prompt 一次性，C25-E）不同：本函数
        取 round1_logits 尾部窗口（已生成文本区段），衡量各 neuron 对"最近
        生成内容"的续写拟合度——随生成演化，捕获实例内漂移。零额外前向
        （round1_logits 由生成主循环 think 产出）。返回 {nid: -NLL}；失败 {}。
        """
        r1 = result.get("round1_logits") or {}
        if not r1:
            return {}
        hub = getattr(self, "_tokenizer_hub", None)
        if hub is None or not hasattr(hub, "get_tokenizer"):
            return {}
        try:
            tok = hub.get_tokenizer(domain) or hub.get_tokenizer("general")
            if tok is None:
                return {}
            zids = torch.tensor([tok.encode(gen_text)], dtype=torch.long,
                                device=self.device)
        except Exception:
            return {}
        if zids.numel() < 1:
            return {}
        vocab = int(tok.GetPieceSize()) if hasattr(tok, "GetPieceSize") else None
        if not vocab:
            return {}
        lens = [int(lg.shape[1]) for lg in r1.values() if lg.shape[-1] == vocab]
        if not lens:
            return {}
        # 对齐（与 C25-E 同口径）：round1_logits 位置 t 预测上下文 t+1。
        # 取 logits 倒数 n+1 个位置中的前 n 个，target = 已生成文本最后 n 个
        # token（续写 NLL；软信号，尽力对齐即可，不追求逐 token 严格映射）。
        n = min(int(window), int(zids.numel()), min(lens) - 1)
        if n < 1:
            return {}
        tgt = zids[0][-n:].unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
        out: dict = {}
        for nid, lg in r1.items():
            if lg.shape[-1] != vocab:
                continue
            try:
                lg_win = lg.detach()[:, -(n + 1):-1, :]  # [1, n, V]
                logp = torch.log_softmax(lg_win, dim=-1)
                nll_tok = -logp.gather(-1, tgt).squeeze(-1)  # [1, n]
                mask = (tgt.squeeze(-1) != 1) & (tgt.squeeze(-1) != 0)
                if mask.sum() == 0:
                    continue
                out[nid] = -float((nll_tok * mask).sum() / mask.sum().float())
            except Exception:
                continue
        return out

    def _probe_inactive_fused(
        self, inactive_nids: List[str], general_ids: List[int],
        gen_text: str, domain: str, window: int, fusion_mode: str,
    ) -> dict:
        """chunk 边界轻量 probe：未激活同域 neuron 的后验分（加入评估依据）。

        用 ensemble.forward（thread-local 独立共振场，不写 cortex 场、不污染
        生成主循环；与 R1 resonance probe 同路径）对未激活同域 neuron 前向
        当前上下文，返回 {nid: fused_score}；失败回退 {}。
        """
        if not inactive_nids:
            return {}
        try:
            probe_ids = torch.tensor([general_ids], dtype=torch.long,
                                     device=self.device)
            probe_emb = self._shared_embedding(probe_ids)
            with torch.no_grad():
                probe = self.ensemble.forward(
                    shared_embeddings=probe_emb, return_logits=True,
                    active_nids=inactive_nids, fusion_mode=fusion_mode,
                )
            base = probe.get("round1_scores") or probe.get("final_scores") or {}
            q = self._rolling_nll_quality(probe, gen_text, domain, window)
            if not base or not q:
                return {}
            common = [k for k in q if k in base]
            if not common:
                return {}
            return self._fuse_leader_quality(
                {k: base[k] for k in common}, {k: q[k] for k in common})
        except Exception:
            return {}

    def _instance_route_evolve(
        self,
        active_nids: List[str],
        result: dict,
        gen_text: str,
        domain: str,
        streaks: Dict[str, int],
        window: int,
        evict_ratio: float,
        evict_streak: int,
        add_ratio: float,
        min_active: int,
        general_ids: List[int],
        fusion_mode: str,
    ) -> tuple:
        """C27 增量一：实例级路由——chunk 级混合后验双向域内演化。

        1. 滚动后验 = 共振分（round1_scores）+ 滚动 NLL（_rolling_nll_quality）
           → _fuse_leader_quality 等权融合（C25-E 机制复用）
        2. 剔除：激活集内同域 neuron 后验 < evict_ratio × leader，且连续
           evict_streak 个 chunk 如此 → 移除（迟滞防抖）
        3. 加入：未激活同域 neuron（_probe_inactive_fused 轻量前向）后验 >
           激活集最小 × add_ratio → 加入
        4. 保护：同域激活数 >= min_active；general 恒激活；顺序稳定。

        Returns:
            (new_active_nids, new_streaks)；任何失败回退 (active_nids, streaks)。
        """
        # 1. 滚动后验（对已生成文本；失败回退原激活集）
        nll_q = self._rolling_nll_quality(result, gen_text, domain, window)
        if not nll_q:
            return active_nids, streaks
        base_scores = result.get("round1_scores") or result.get("final_scores") or {}
        base_scores = {k: v for k, v in base_scores.items()}
        common = [k for k in nll_q if k in base_scores]
        if not common:
            return active_nids, streaks
        fused = self._fuse_leader_quality(
            {k: base_scores[k] for k in common},
            {k: nll_q[k] for k in common},
        )
        if not fused:
            return active_nids, streaks
        # 2. 域约束全集（演化边界——C22 回合判定保持外层约束）
        domain_all = [
            k for k in self.neurons
            if k == domain or k.startswith(domain + "_")
        ]
        if not domain_all:
            return active_nids, streaks
        # 3. 未激活同域 neuron 后验（chunk 边界轻量 probe）
        inactive = [k for k in domain_all if k not in active_nids]
        if inactive:
            fused_inactive = self._probe_inactive_fused(
                inactive, general_ids, gen_text, domain, window, fusion_mode)
            if fused_inactive:
                fused = {**fused, **fused_inactive}
        if not fused:
            return active_nids, streaks
        # 4. 双向演化（迟滞）
        leader_id = max(fused, key=fused.get)
        leader_score = fused[leader_id]
        keep = set(active_nids)
        new_streaks = dict(streaks)
        for nid in list(active_nids):
            if nid not in domain_all:
                continue  # general 等恒激活，域外不动
            if nid == leader_id:
                new_streaks[nid] = 0
                continue
            ratio = fused.get(nid, 0.0) / (leader_score + 1e-9)
            if ratio < evict_ratio:
                new_streaks[nid] = new_streaks.get(nid, 0) + 1
                if new_streaks[nid] >= evict_streak:
                    keep.discard(nid)
                    new_streaks[nid] = 0
            else:
                new_streaks[nid] = 0
        # 5. 加入：未激活同域 neuron 后验显著高于激活集最小
        domain_keep = [k for k in keep if k in domain_all]
        if domain_keep:
            active_min = min(fused[k] for k in domain_keep if k in fused)
        else:
            active_min = None
        for nid, sc in fused.items():
            if nid in keep or nid not in domain_all:
                continue
            if active_min is None or sc > active_min * add_ratio:
                keep.add(nid)
        # 6. 保护：同域激活数下限（min_active）
        domain_keep = [k for k in keep if k in domain_all]
        if len(domain_keep) < min_active:
            for nid in sorted(fused, key=fused.get, reverse=True):
                if nid in domain_all and nid not in keep:
                    keep.add(nid)
                    domain_keep.append(nid)
                    if len(domain_keep) >= min_active:
                        break
        # 7. 保持原顺序 + 新加入 append + general 保留
        new_nids = [k for k in active_nids if k in keep]
        new_nids += [k for k in domain_all if k in keep and k not in active_nids]
        new_nids += [k for k in active_nids
                     if k in keep and k not in domain_all and k not in new_nids]
        return new_nids, new_streaks

    def _generate_p7(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        domain: Optional[str] = None,
        repetition_penalty: float = 1.4,
        routing_level: int = 1,
        active_nids: Optional[Union[str, List[str]]] = None,
        collab_mode: str = "fusion",
        fusion_mode: str = "soft",
        # ── R1: 共振分数软路由 ──
        # "hybrid"（默认）= keyword 路由 + 共振校验（50% 阈值硬切换），向后兼容
        # "resonance" = 共振分数软路由（probe forward → final_scores → top-k 激活）
        # "keyword" = 纯关键词路由（无共振校验，最快）
        routing_mode: str = "hybrid",
        resonance_top_k: int = 3,  # R1: resonance 模式下激活的神经元数量
        # 口径守卫例外开关（2026-08-12）：base/域 neuron 评估用纯问题 prompt
        # （无 "问：/答：" 格式）时显式传 True，绕过 dialogue 格式守卫。
        _allow_plain_prompt: bool = False,
        # C26 增量二（2026-08-14）：记忆可读进生成——检索到的记忆向量
        # [{"vector": [D], "weight": sim} 或 (vec, weight)]，经 think 写入
        # 共振场，round2+ 场条件化 forward 让记忆直接参与 token 生成。
        memory_vectors: Optional[List] = None,
        # C26 增量四（2026-08-14）：自动记忆检索（同 generate，见其签名说明）。
        auto_memory: bool = True,
        # C27 增量一（2026-08-14）：实例级路由（SMCS 借鉴）——continuous 生成
        # 中每 instance_chunk token 用混合后验（共振分 + 滚动 NLL）双向域内
        # 演化激活子集。chunk/迟滞/阈值均为生产默认参数，演化失败静默回退。
        instance_routing: bool = True,
        instance_chunk: int = 12,
        instance_evict_ratio: float = 0.35,
        instance_evict_streak: int = 2,
        instance_add_ratio: float = 1.3,
        instance_min_active: int = 2,
    ) -> str:
        """Generate text using shared embedding + domain-specific lm_head.

        Flow:
        1. Encode prompt with general tokenizer → general_ids
        2. shared_embedding(general_ids) → shared_emb
        3. Ensemble resonance → neuron_logits (per-neuron, domain vocab)
        4. Sample in domain vocab
        5. Decode with domain tokenizer
        6. For autoregressive: decode domain token → re-encode with general tokenizer

        Args:
            prompt: input text.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: top-k sampling.
            domain: target domain.
            active_nids: 显式指定激活的神经元列表（实验用）。
                       None 时由 routing_level 自动决定。
            routing_mode: 路由模式 ("hybrid"/"resonance"/"keyword")。
            resonance_top_k: resonance 模式下按共振分数激活的神经元数量。

        Returns:
            generated text string.
        """
        if self._tokenizer_hub is None:
            raise RuntimeError("TokenizerHub not set. Call cortex.set_tokenizer_hub() first.")
        if self._shared_embedding is None:
            raise RuntimeError("Shared embedding not set. Call cortex.set_shared_embedding() first.")
        if self._general_sp is None:
            raise RuntimeError("General tokenizer not set. Call cortex.set_general_tokenizer() first.")

        hub = self._tokenizer_hub

        # 1. Determine domain
        # C19（2026-08-08）：collab_mode="executive" = 回合级执行控制判定
        # （混合信号：启发式 + quality_head 回合级聚合），取代 token 级竞争。
        # C25-E（2026-08-11）："continuous" 复用 executive 判定（judge NLL 主信号），
        # 仅生成路径换为连续时间共振。
        self._executive_confidence = 0.0
        self._executive_domains: dict = {}
        if domain is None and collab_mode in ("executive", "continuous"):
            domain, self._executive_confidence, self._executive_domains = \
                self._executive_route(prompt)
        elif domain is None:
            domain = self._infer_domain(prompt)
        if domain not in hub.list_domains():
            domain = hub.list_domains()[0] if hub.list_domains() else "general"

        # 2. Encode prompt with general tokenizer → shared embedding
        prompt_general_ids = self._general_sp.encode(prompt)
        if not prompt_general_ids:
            prompt_general_ids = [0]
        general_ids = list(prompt_general_ids)

        # 2.1 C26 增量四（2026-08-14）：记忆自动检索注入
        # 未显式传 memory_vectors 且注入过记忆库时，用 prompt 的场状态自动
        # 检索 top-1 记忆注入生成——记忆从"显式 API"走向"模型内在自动调取"
        # （Titans 式内部记忆的产品化落点）。检索本身一次额外共振前向
        # （场状态作 query），命中后走增量二同款向量条件化路径。
        if memory_vectors is None and auto_memory and self._memory_bank is not None:
            try:
                if len(self._memory_bank) > 0:
                    _qids = torch.tensor([general_ids], dtype=torch.long,
                                         device=self.device)
                    _qemb = self._shared_embedding(_qids)
                    _qres = self.think(
                        _qemb, active_nids=None, fusion_mode=fusion_mode,
                        collab_mode=collab_mode,
                    )
                    _qfs = _qres.get("field_state")
                    if _qfs is not None and _qfs.dim() == 2:
                        _qfs = _qfs.mean(dim=0)
                    _top = self._memory_bank.retrieve_with_phase(_qfs, top_k=1)
                    if _top:
                        _lab, _sim, _vec, _ph = _top[0]
                        # C27 增量二（KoPE）：记忆带相位 → 注入按记忆相位对齐
                        # theta（相位归属记忆）；无相位回退 2 元组（峰值对齐）。
                        memory_vectors = [(_vec, _sim, _ph)] if _ph is not None \
                            else [(_vec, _sim)]
            except Exception:
                pass  # 自动检索失败静默跳过（显式向量通道不受影响）

        # S12: 多轮对话状态管理
        # - start_round: 加载上一轮的 field_state（隐式记忆上下文）
        # - prepend_round_token: 在 prompt 前插入轮次标记（第 2 轮及以后）
        if self._dialogue_state is not None:
            self._dialogue_state.start_round(self.field)
            self._dialogue_state.add_dialogue_entry("user", prompt)
            # 在 prompt 前插入轮次标记 token（第 2 轮及以后）
            general_ids = self._dialogue_state.prepend_round_token(general_ids)

        # 2.5 R1: 共振分数路由（三种模式）
        # - "keyword": 纯关键词路由，跳过 probe forward（最快）
        # - "hybrid"（默认）: keyword 路由 + 共振校验（50% 阈值硬切换 domain），向后兼容
        # - "resonance": 共振分数软路由（probe → final_scores → top-k 激活，跨域协作）
        # R1 上限提升：resonance 模式让共振分数直接驱动激活，神经元自发协作决定"谁发言"，
        # 与 C12（可比分数）+ C9（自适应停止）+ C14（动态 shared 权重）形成完整闭环。
        # C22（2026-08-08 收敛）：collab_mode="executive" 时跳过本块——executive 已有
        # 回合级判定（_executive_route），再跑 token 级共振校验会造成双路径打架
        # （共振校验可能覆盖 executive 的 dominant 判定）。C25-E "continuous" 同。
        resonance_active_nids: Optional[List[str]] = None  # resonance 模式填充
        if (len(self.neurons) > 1 and routing_mode != "keyword"
                and collab_mode not in ("executive", "continuous")):
            try:
                probe_ids = torch.tensor([general_ids], dtype=torch.long, device=self.device)
                probe_emb = self._shared_embedding(probe_ids)
                with torch.no_grad():
                    probe_result = self.ensemble.forward(
                        shared_embeddings=probe_emb, return_logits=False,
                    )
                probe_scores = probe_result.get("final_scores", {})
                if probe_scores:
                    if routing_mode == "resonance":
                        # R1: 共振分数软路由 —— 按分数排序选 top-k 神经元
                        # 跨域协作：不限定 domain，让共振分数自发决定激活集合
                        # shared_expert（若存在）始终包含，保证基础语言能力
                        sorted_nids = sorted(
                            probe_scores.items(), key=lambda x: x[1], reverse=True
                        )
                        top_nids = [nid for nid, _ in sorted_nids[:resonance_top_k]]
                        # 确保 shared_expert 在激活集中
                        if self.ensemble.shared_expert_id:
                            se_id = self.ensemble.shared_expert_id
                            if se_id not in top_nids and se_id in self.neurons:
                                top_nids.append(se_id)
                        resonance_active_nids = top_nids
                        # domain 仍用于 tokenizer 选择（取分数最高的 neuron 的 domain）
                        best_nid = sorted_nids[0][0] if sorted_nids else domain
                        best_domain = best_nid.split("_")[0] if "_" in best_nid else best_nid
                        if best_domain in hub.list_domains():
                            domain = best_domain
                    else:  # hybrid 模式：保留现有共振校验逻辑
                        # neurons 的 key 即为 domain（见 _infer_domain L709）
                        best_nid = max(probe_scores, key=probe_scores.get)
                        best_domain = best_nid
                        chosen_score = max(
                            (probe_scores.get(nid, 0.0)
                             for nid in self.neurons
                             if nid == domain),
                            default=0.0,
                        )
                        # 切换条件：最强域分数比选定域高 50% 以上，且最强域已加载
                        if (best_domain != domain
                                and best_domain in hub.list_domains()
                                and chosen_score > 0
                                and probe_scores[best_nid] > chosen_score * 1.5):
                            domain = best_domain
            except Exception:
                pass  # 共振探测失败，保留关键词路由结果

        # 3. Domain EOS
        # C21（2026-08-08）：词库多词表架构——decode 按生成 logits 的词表空间。
        # 默认 general 256K（general 头 neuron）；leader 是 zh 头 neuron（50K）时
        # 切到 zh decode + domain→general 回填（v3 口径，自回归输入保持 general 空间）。
        # C19 曾统一 general decode → zh 空间 id 被 general 词表错位解析 → dialogue 碎片。
        # C24（2026-08-09）：decode 域扩展——按 leader 词表尺寸匹配 hub 域 tokenizer
        # （code 12K/math 10K/zh 50K/en 16K），替代 C21 硬编码 50000→zh。
        eos_id = hub.eos_token_id("general")
        decode_sp = self._general_sp  # 当前生成词表空间（leader 分支可能覆盖为域）
        decode_domain = "general"     # 当前 decode 空间域（P7 CJK 截断仅对中文生效）
        domain_sp = None
        alignment_table = None

        generated_pieces = []
        generated_token_ids = set()
        generated_token_list = []  # 保持顺序，用于 no-repeat-ngram
        generated_ids_ordered = []  # 保持顺序，用于 DecodeIds（正确处理 byte fallback）
        # 域自适应 no-repeat-ngram：中文 n=4（更宽松，避免短句过度抑制），其他 n=3
        # 中文短句字符数少，n=3 会误杀正常重复用字；n=4 给中文更多重复容忍度
        no_repeat_ngram_size = 4 if domain == "zh" else 3

        # 稀疏激活：支持字符串模式自动选择（auto_topK / auto_all / auto_top1）
        if isinstance(active_nids, str):
            if active_nids.startswith("auto_top"):
                k_str = active_nids[len("auto_top"):]
                k = int(k_str) if k_str else 3
                active_nids = self._auto_topk_route(general_ids, top_k=k)
            elif active_nids in ("auto_all", "all"):
                active_nids = list(self.neurons.keys())
            elif active_nids in self.neurons:
                active_nids = [active_nids]
            else:
                active_nids = list(self.neurons.keys())
        elif active_nids is None:
            # R1: resonance 模式优先使用共振分数选的 active_nids
            if resonance_active_nids is not None:
                active_nids = resonance_active_nids
            elif collab_mode in ("executive", "continuous"):
                # C19（2026-08-08）：任务模式激活 = dominant 域 neuron + general
                # （回合内稳定生成，不做 token 级竞争；人脑"模式确定→整通路激活"）
                # C25-E：continuous 同 executive 激活（判定信号链一致）
                domain_neurons = [
                    k for k in self.neurons
                    if k == domain or k.startswith(domain + "_")
                ]
                active_nids = domain_neurons
                general_neurons = [
                    k for k in self.neurons
                    if (k == "general" or k.startswith("general_"))
                ]
                if general_neurons and domain not in general_neurons:
                    active_nids.extend(general_neurons)
            elif routing_level >= 2:
                # Level 2 prototype top-k 路由：用 prompt embedding vs domain_prototype cosine
                active_nids = self._fingerprint_route(general_ids, top_k=2)
            else:
                # Level 1 域路由：激活 domain 前缀的全部神经元（同域多神经元协作）
                # 如 domain="zh" → 激活 zh_aug0_dialogue / zh_std0_dialogue 等
                domain_neurons = [
                    k for k in self.neurons
                    if k == domain or k.startswith(domain + "_")
                ]
                active_nids = domain_neurons
                # general 神经元 always-active（基础语言能力）
                general_neurons = [
                    k for k in self.neurons
                    if (k == "general" or k.startswith("general_"))
                ]
                if general_neurons and domain not in general_neurons:
                    active_nids.extend(general_neurons)

        # 2.5 评估口径守卫（2026-08-12 机制化，硬失败）：
        # dialogue neuron（zh_aug*_dialogue / zh_std0_dialogue）用 "问：...\n答：" 格式
        # 训练（SFT answer masking），裸 prompt 下模型陷入换行/空格死循环 → 假退化。
        # 判断逻辑见 neuroplex/resonance/dialogue_format.dialogue_prompt_requires_guard
        # （核心库单一真相源，配合 tests/test_dialogue_format.py 回归防口径漂移）。
        # 此处 active_nids 已完成归一化（必为 list，非 None）。
        if dialogue_prompt_requires_guard(prompt, domain, active_nids, _allow_plain_prompt):
            raise ValueError(
                f"[口径守卫] domain='zh' 且激活 dialogue neuron 时，prompt 必须用训练格式 "
                f"'问：{{question}}\\n答：'（当前: {prompt[:50]!r}）。"
                f"裸 prompt 会触发换行死循环导致假退化。"
                f"请用 scripts/training/experiment_config.build_dialogue_prompt() 构造；"
                f"base/域 neuron 评估请显式传 _allow_plain_prompt=True。"
            )

        # C24（2026-08-09）：leader 是域目标空间 SFT neuron 时首次迭代补 "\n"
        # 分隔符（见下方 leader 分支）——训练 answer 起点在 prompt+"\n" 之后。
        _c24_prefixed = False
        _c24_domain_nids = getattr(self, "_c24_domain_nids", None) or set()
        # C25-E 遗留（2026-08-14）：质量信号针对固定 prompt 不随生成增长，
        # 首次计算后缓存（避免每 token 重复 log_softmax 50000 词表）。
        _leader_nll_quality_cache: Optional[dict] = None
        # C27 增量一（2026-08-14）：实例级路由演化状态——neuron 连续劣化
        # chunk 计数（迟滞防抖），chunk 边界在 _instance_route_evolve 更新。
        _ir_streaks: Dict[str, int] = {}

        for ir_step in range(max_tokens):
            # Trim context to prevent memory issues and maintain coherence
            if len(general_ids) > 512:
                general_ids = general_ids[-512:]

            # Embed general IDs → [1, L, 512]
            ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=self.device)
            # P7-修复：per-neuron shared embedding（与训练一致，避免单 shared_embedding 错配）
            if self._neuron_shared_embeddings:
                neuron_embeddings = {}
                for nid, emb in self._neuron_shared_embeddings.items():
                    if active_nids is not None and nid not in active_nids:
                        continue
                    neuron_embeddings[nid] = emb(ids_tensor)
                result = self.think(
                    active_nids=active_nids, fusion_mode=fusion_mode,
                    neuron_embeddings=neuron_embeddings,
                    collab_mode=collab_mode,
                    memory_vectors=memory_vectors,
                )
            else:
                shared_emb = self._shared_embedding(ids_tensor)
                result = self.think(shared_emb, active_nids=active_nids, fusion_mode=fusion_mode,
                                    collab_mode=collab_mode,
                                    memory_vectors=memory_vectors)

            # Get logits: 协作模式选择
            neuron_logits = result.get("neuron_logits", {})
            final_scores = result.get("final_scores", {})

            if collab_mode in ("leader", "executive", "continuous") and final_scores and neuron_logits:
                # 族长主导（C19 executive 复用）：选共振分最高的 neuron 的 logits
                # （不融合）——回合内稳定生成，避免异构 logit 融合干扰。
                # 族长在 round 2+ 已读共振场（受其他 neuron 影响），
                # 用自己 logits 干净输出，避免异构 logit 融合干扰（confidence陷阱）
                # C19（2026-08-08）：executive 模式下 leader 限定在 dominant 域内
                # ——任务模式激活后由域内最强 neuron 稳定生成，不用跨域最强
                #（否则回合级判定白做，leader 被其他域 neuron 抢占）。
                # C25-E：continuous 用时间平均激活权重（continuous_weights）选 leader。
                # C25-E 增量四（2026-08-11）：continuous leader 用 round1_scores
                # （t=0 场共振分，质量信号）优先——时间平均激活=参与度不区分强弱，
                # 同相群体权重均分 → leader 选到弱响应 neuron → zh 对话空输出；
                # 场共振分区分强弱（与 executive 同口径）。
                if collab_mode in ("executive", "continuous") and domain:
                    if collab_mode == "continuous":
                        # 质量信号优先，fallback 时间平均激活
                        qual_scores = result.get("round1_scores") or final_scores
                        domain_scores = {
                            k: v for k, v in qual_scores.items()
                            if k == domain or k.startswith(domain + "_")
                        }
                        base_scores = domain_scores if domain_scores else qual_scores
                        # C25-E 遗留（2026-08-14）：融合 NLL 质量信号防弱 neuron
                        # 独占——共振分高但生成质量差的 neuron（如 aug2）不再独占
                        # leader。质量信号 = 各 neuron 对 prompt 的 next-token NLL
                        # （zh lm_head 对齐，越低质量越好），域内归一化后与共振分
                        # 等权融合。质量信号获取失败时回退纯共振分（向后兼容）。
                        fused = None
                        if len(base_scores) >= 2:
                            if _leader_nll_quality_cache is None:
                                _leader_nll_quality_cache = (
                                    self._nll_quality_from_round1_logits(
                                        result, prompt, domain))
                            if _leader_nll_quality_cache:
                                fused = self._fuse_leader_quality(
                                    base_scores, _leader_nll_quality_cache)
                        if fused:
                            leader_nid = max(fused, key=fused.get)
                        else:
                            leader_nid = max(base_scores, key=base_scores.get)
                    else:
                        domain_scores = {
                            k: v for k, v in final_scores.items()
                            if k == domain or k.startswith(domain + "_")
                        }
                        if domain_scores:
                            leader_nid = max(domain_scores, key=domain_scores.get)
                        else:
                            leader_nid = max(final_scores, key=final_scores.get)
                else:
                    leader_nid = max(final_scores, key=final_scores.get)
                # C24（2026-08-09）：leader 是域目标空间 SFT neuron 时，首次迭代
                # 补输入分隔符 "\n" 重新 forward——训练时 answer 起点在 prompt+"\n"
                # 之后（train_domain_target_sft.py build_sample），生成输入无 "\n"
                # 会导致 first-token 概率 EOS > 换行 → 生成空/碎片。
                if (not _c24_prefixed and leader_nid in _c24_domain_nids):
                    _c24_prefixed = True
                    _nl_ids = self._general_sp.encode("\n")
                    if _nl_ids:
                        general_ids = general_ids + _nl_ids
                        continue  # 重新 forward（输入含换行分隔符）
                if leader_nid in neuron_logits:
                    # C21（词库多词表）：leader 用 round1 独立 logits（无场条件化）——
                    # round2 场注入混合域信号会稀释 leader 的域词表能力（dialogue 的
                    # zh 输出被英文 neuron 的场污染 → 中英混合碎片）。协作/共振分只
                    # 用于任务模式判定，生成用 leader 自身干净能力。fallback 到
                    # neuron_logits（round1_logits 未提供时）。
                    r1_logits = result.get("round1_logits") or {}
                    if leader_nid in r1_logits:
                        leader_logits_full = r1_logits[leader_nid]
                    else:
                        leader_logits_full = neuron_logits[leader_nid]
                    logits = leader_logits_full[:, -1, :] / temperature
                else:
                    # 族长 logits 未保留（large_scale top-K 过滤），取任意可用
                    r1_logits = result.get("round1_logits") or {}
                    if leader_nid in r1_logits:
                        leader_logits_full = r1_logits[leader_nid]
                    else:
                        leader_logits_full = next(iter(neuron_logits.values()))
                    logits = leader_logits_full[:, -1, :] / temperature
                # C21（词库多词表）：decode 按 leader 词表空间——general 头（256K）
                # → general decode（identity 回填）；域头（zh 50K/code 12K/math 10K/
                # en 16K 等）→ 域 decode + domain→general 回填（v3 口径）。
                # C24（2026-08-09）：按词表尺寸动态匹配 hub 域 tokenizer，
                # 词库容量不限定，新词表在此自动扩展。
                _lv = leader_logits_full.shape[-1]
                _general_vocab = getattr(self._general_sp, "GetPieceSize", lambda: 256000)()
                if _lv != _general_vocab:
                    _matched_dom = None
                    for _dom in hub.list_domains():
                        _sp = hub.get_tokenizer(_dom)
                        if _sp is not None and getattr(_sp, "GetPieceSize", lambda: 0)() == _lv:
                            _matched_dom = _dom
                            break
                    if _matched_dom is not None:
                        decode_sp = hub.get_tokenizer(_matched_dom)
                        decode_domain = _matched_dom
                        eos_id = decode_sp.eos_id() if hasattr(decode_sp, "eos_id") else eos_id
                    else:
                        decode_sp = self._general_sp
                        decode_domain = "general"
                else:
                    decode_sp = self._general_sp
                    decode_domain = "general"
            elif result.get("weighted_logits") is not None:
                # 优先用 ensemble 的 per-position routing（同 vocab 时由
                # _compute_per_position_weights 算出，基于每位置 entropy/confidence
                # 选最 confident 的 neuron，比共振分简单加权更精细，避免弱模型
                # logits 被均分平均化导致 argmax 落到符号噪声）
                logits = result["weighted_logits"][:, -1, :] / temperature
            elif neuron_logits and final_scores:
                # 跨 vocab 投影失败（ensemble.forward 未产出 weighted_logits）时，
                # 直接取当前域 neuron 的 logits（_dynamic_logit_fusion 已删除——
                # 旧 MoCo 加权与训练口径不一致，且 forward 已能产出 weighted_logits）
                if domain in neuron_logits:
                    logits = neuron_logits[domain][:, -1, :] / temperature
                else:
                    first_logits = next(iter(neuron_logits.values()))
                    logits = first_logits[:, -1, :] / temperature
            elif domain in neuron_logits:
                # Fallback: domain-specific logits only
                logits = neuron_logits[domain][:, -1, :] / temperature
            elif neuron_logits:
                first_logits = next(iter(neuron_logits.values()))
                logits = first_logits[:, -1, :] / temperature
            else:
                break

            # Repetition penalty: penalize tokens that have been generated
            if generated_token_ids and repetition_penalty > 1.0:
                for tid in generated_token_ids:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            # No-repeat-ngram: ban tokens that would complete an existing n-gram
            if no_repeat_ngram_size > 0 and len(generated_token_list) >= no_repeat_ngram_size - 1:
                ngram_prefix = tuple(generated_token_list[-(no_repeat_ngram_size - 1):])
                # 查找已生成文本中所有匹配前缀的 n-gram 的下一个 token
                banned_ids = set()
                for i in range(len(generated_token_list) - no_repeat_ngram_size + 1):
                    if tuple(generated_token_list[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                        banned_ids.add(generated_token_list[i + no_repeat_ngram_size - 1])
                # 将 banned tokens 的 logit 设为 -inf
                for tid in banned_ids:
                    logits[0, tid] = float('-inf')

            # P7-修复（2026-08-04）：EOS logit 增强 + 熵停止 + 跑偏截断
            # 训练数据（alpaca clean）无 EOS 标记，模型从未学会输出 </s>，
            # 生成永不自然停止 → 一直生成到 max_tokens 导致长序列崩坏。
            # 1) 每步给 eos_id 加温和 bias，鼓励在自然结束点终止；
            # 2) 连续 3+ 个非中文字符 token（英文/符号/数字碎片）视为跑偏 → 截断停止。
            if eos_id is not None:
                logits[0, eos_id] += 0.5  # 温和 EOS bias（top-k 后可能仍在候选）
            else:
                # 无 EOS：softmax 熵 > 阈值时视为跑偏，提前停止
                probs_ent = F.softmax(logits / temperature, dim=-1)
                ent = -(probs_ent * probs_ent.clamp_min(1e-9).log()).sum(-1)
                if ent[0].item() > 8.0 and len(generated_ids_ordered) >= 8:
                    break

            # Top-k sampling in domain vocab
            if top_k > 0:
                actual_k = min(top_k, logits.shape[-1])
                top_k_vals, top_k_indices = torch.topk(logits, actual_k)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx_in_topk = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, sampled_idx_in_topk[0]].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated_token_ids.add(next_token)
            generated_token_list.append(next_token)
            generated_ids_ordered.append(next_token)

            if self.gamma_oscillator is not None:
                self.tick_gamma()

            # EOS check
            if eos_id is not None and next_token == eos_id:
                break

            # P7-修复：跑偏截断——连续 3+ 个非中文 token（英文/符号/数字碎片）
            # 视为模型长序列生成跑偏，回退到最后一个中文 token 后停止。
            # （中文标点 .，！？ 等含全角形式，一并视为正常内容）
            # C24（2026-08-09）：仅对 zh 域 decode 生效——code/math/en 域空间
            # 输出天然非中文（代码/数字/公式），CJK 截断会误杀域生成。
            if decode_domain == "zh" and len(generated_ids_ordered) >= 4:
                _cjk_or_punct = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2014\u2018\u2019\u201c\u201d\u2026]')
                last_pieces = [decode_sp.id_to_piece(t) for t in generated_ids_ordered[-4:]]
                non_cjk = sum(1 for p in last_pieces if not _cjk_or_punct.search(p))
                if non_cjk >= 3:
                    while generated_ids_ordered and not _cjk_or_punct.search(
                        decode_sp.id_to_piece(generated_ids_ordered[-1])
                    ):
                        generated_ids_ordered.pop()
                    break

            # C21（2026-08-08）：词库多词表回填——按当前生成词表空间。
            # - general 空间（256K）：恒等回填（neuron logits 原生在此空间）
            # - zh 空间（50K）：domain token → 文本 → general ids 回填（v3 口径，
            #   自回归输入保持 general 空间，避免 C19 的 id 语义错位）
            if decode_sp is self._general_sp:
                generated_pieces.append(self._general_sp.id_to_piece(next_token))
                general_ids.append(next_token)
            else:
                _piece = decode_sp.id_to_piece(next_token)
                generated_pieces.append(_piece)
                _text = decode_sp.decode([next_token])
                _new_general = self._general_sp.encode(_text)
                if _new_general:
                    general_ids.extend(_new_general)

            # C27 增量一（2026-08-14）：实例级路由（SMCS 借鉴）——chunk 级
            # 混合后验双向域内演化。每 instance_chunk token，用滚动 NLL（对
            # 已生成文本，round1_logits 零额外前向）+ 共振分重估同域 neuron
            # 后验：剔除持续劣化（迟滞防抖）、加入后验显著更高的未激活同域
            # neuron。仅在 continuous 模式（C22 收敛路径）；异常静默回退。
            if (collab_mode == "continuous" and instance_routing
                    and domain and ir_step > 0
                    and (ir_step + 1) % instance_chunk == 0):
                try:
                    _ctx = decode_sp.DecodeIds(generated_ids_ordered) if generated_ids_ordered else ""
                    if _ctx.strip():
                        _evolved, _ir_streaks = self._instance_route_evolve(
                            active_nids, result, _ctx, domain, _ir_streaks,
                            window=instance_chunk,
                            evict_ratio=instance_evict_ratio,
                            evict_streak=instance_evict_streak,
                            add_ratio=instance_add_ratio,
                            min_active=instance_min_active,
                            general_ids=general_ids,
                            fusion_mode=fusion_mode,
                        )
                        if _evolved and _evolved != active_nids:
                            logger.info(
                                "[Cortex] 实例级路由 chunk %d：激活 %d→%d",
                                (ir_step + 1) // instance_chunk,
                                len(active_nids), len(_evolved),
                            )
                            active_nids = _evolved
                except Exception as _e:
                    logger.debug("[Cortex] 实例级路由 chunk %d 失败回退: %s",
                                 (ir_step + 1) // instance_chunk, _e)

        # Decode with 当前词表空间 tokenizer（多词表架构）
        # P7-修复（2026-08-04）：用 DecodeIds 替代 "".join(pieces)
        # 旧拼接会把 byte fallback piece（如 <0x0A>）原样输出，DecodeIds 正确处理字节 token。
        if generated_ids_ordered:
            result_text = decode_sp.DecodeIds(generated_ids_ordered)
        elif generated_pieces:
            result_text = "".join(generated_pieces).replace("▁", " ")
        else:
            result_text = ""

        # S12: 结束当前轮次，保存 field_state 快照（用于下一轮隐式记忆）
        if self._dialogue_state is not None:
            self._dialogue_state.add_dialogue_entry("assistant", result_text)
            self._dialogue_state.end_round(self.field)

        # C27 增量二（KoPE）：截获最近一次共振的相位均值（记忆沉淀/任务链
        # 记录带相位——相位归属记忆；无 phase → None 回退）。
        try:
            self._last_phase_mean = (
                result.get("phase_mean") if isinstance(result, dict) else None)
        except Exception:
            self._last_phase_mean = None

        return result_text

    @torch.no_grad()
    def generate_multimodal(
        self,
        input_data: Union[torch.Tensor, dict],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        domain: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> Union[str, torch.Tensor]:
        """P8: 多模态生成入口。

        与 generate() 并列，专用于非文本模态（图像/音频/视频）。
        文本输入仍走 generate()。

        Args:
            input_data: 多模态输入，支持两种格式：
                - torch.Tensor [B, L, raw_dim] float: 连续特征（图像 patch / 音频 frame）
                - dict {"modality": "image", "data": tensor, "domain": "general"}: 显式指定
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_k: top-k 采样。
            domain: 目标域（None 时用 "general"）。
            modality: 模态（None 时从 input_data 推断）。

        Returns:
            生成的 token id 列表（list[int]），由调用方通过 hub.decode(ids, modality=...) 解码。
        """
        if self._tokenizer_hub is None:
            raise RuntimeError("TokenizerHub not set. Call cortex.set_tokenizer_hub() first.")

        # 1. 解析输入
        if isinstance(input_data, dict):
            actual_modality = modality or input_data.get("modality", "image")
            features = input_data.get("data", input_data.get("features"))
            domain = domain or input_data.get("domain", "general")
        else:
            actual_modality = modality or self.detect_modality(input_data)
            features = input_data
            domain = domain or "general"

        if actual_modality == "text":
            # 文本走 generate()，这里不应到达
            raise ValueError("text 模态请用 generate()")

        if features is None:
            raise ValueError("多模态输入缺少 data/features 字段")

        # 训练/推理分离：无锁读（同 generate，训练侧影子权重保证 live 稳定）
        return self._generate_multimodal_p8(
            features, actual_modality, domain,
            max_tokens, temperature, top_k,
        )

    def _generate_multimodal_p8(
        self,
        features: torch.Tensor,
        modality: str,
        domain: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list:
        """P8: 多模态生成的内部实现（走 ensemble 共振路径）。

        2026-08-07 收敛：输出统一走共享 general lm_head（256K vocab）。
        - 输入：codec 索引 → codebook → mm_projections → 共享 embedding
        - 输出：general 256K logits，mask 到 codec 段采样
        - 自回归：采样 general_id → 映射回 codec_index → codebook → 下一步输入

        策略：
        1. 所有注册了该模态投影层的 neuron 都参与共振（小神经元协同）
        2. 每个 neuron 独立预编码多模态 embedding（neuron_embeddings）
        3. ensemble.forward 多轮共振，输出统一走共享 general lm_head（256K vocab）
        4. mask 到 codec 段采样，映射回 codec 索引，自回归生成
        """
        hub = self._tokenizer_hub

        # 2026-08-07 收敛：确定模态在 general 词表的 codec 段
        # image/audio 段在 tokenizer_contract.json 预留；video 暂无预留段，v1 不支持生成
        from neuroplex.config import MULTIMODAL_TOKENS
        if modality == "image":
            mm_token_base = MULTIMODAL_TOKENS["image_token_base"]
            mm_codebook_size = MULTIMODAL_TOKENS["image_codebook_size"]
        elif modality == "audio":
            mm_token_base = MULTIMODAL_TOKENS["audio_token_base"]
            mm_codebook_size = MULTIMODAL_TOKENS["audio_codebook_size"]
        else:
            # video 等未在 general 词表预留段的模态，v1 不支持生成
            logger.warning(
                f"模态 '{modality}' 在 general 词表无预留段，v1 不支持生成，fallback 到 text 路径"
            )
            if features.dim() == 2 and features.dtype == torch.long:
                ids = features.tolist()[0] if features.dim() == 2 else features.tolist()
                return ids[:max_tokens]
            raise RuntimeError(
                f"模态 '{modality}' v1 不支持生成（general 词表无预留段），且输入不是离散 token id"
            )

        # 1. 找出所有支持该模态输入投影的 neuron
        # （输出统一走共享 general lm_head，不再需要 per-neuron mm_lm_heads）
        mm_nids = [
            nid for nid, neuron in self.neurons.items()
            if modality in neuron.mm_projections
        ]
        if not mm_nids:
            logger.warning(
                f"无 neuron 注册了模态 '{modality}' 投影层，fallback 到 text 路径"
            )
            if features.dim() == 2 and features.dtype == torch.long:
                ids = features.tolist()[0] if features.dim() == 2 else features.tolist()
                return ids[:max_tokens]
            raise RuntimeError(
                f"无 neuron 支持模态 '{modality}'，且输入不是离散 token id"
            )

        # 2. 输入维度归一化
        features = features.to(self.device)
        if features.dim() == 2 and features.dtype != torch.long:
            features = features.unsqueeze(0)  # [L, D] → [1, L, D]

        # 3. 为每个 neuron 预编码多模态 embedding
        # 每个 neuron 的 mm_projections 独立，所以 embedding 不同
        neuron_embeddings: Dict[str, torch.Tensor] = {}
        for nid in mm_nids:
            neuron = self.neurons[nid]
            emb = neuron.encode_multimodal_input(features, modality)  # [1, L, base_embed_dim]
            neuron_embeddings[nid] = emb

        # 4. codec（用于自回归时查 codebook）
        codec = hub.modal_encoders.get(modality)
        has_codebook = (
            codec is not None
            and hasattr(codec, "model")
            and hasattr(codec.model, "quantizer")
        )

        # 5. 自回归生成（多 neuron 共振）
        generated = []

        for step in range(max_tokens):
            # 多轮共振：输出统一走共享 general lm_head（256K vocab）
            res = self.ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                active_filter=True,
                active_nids=mm_nids,
            )

            # 取加权 logits（general 256K vocab）
            if "weighted_logits" in res:
                logits = res["weighted_logits"]  # [B, L, general_vocab=256K]
            elif "neuron_logits" in res and res["neuron_logits"]:
                # fallback: 取第一个 neuron 的 logits
                first_nid = next(iter(res["neuron_logits"].keys()))
                logits = res["neuron_logits"][first_nid]
            else:
                raise RuntimeError("共振未返回 logits，无法生成")

            logits = logits[:, -1, :] / temperature  # [B, general_vocab]

            # mask 到 codec 段：非 codec 段 logits 设为 -inf，确保采样落在 codec 段
            mask = torch.full_like(logits, float('-inf'))
            mask[:, mm_token_base:mm_token_base + mm_codebook_size] = 0
            logits = logits + mask

            if top_k > 0:
                actual_k = min(top_k, mm_codebook_size)
                top_k_vals, top_k_indices = torch.topk(logits, actual_k)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx_in_topk = torch.multinomial(probs, 1)
                next_general_id = top_k_indices[0, sampled_idx_in_topk[0]].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_general_id = torch.multinomial(probs, 1).item()

            # 映射回 codec 索引（生成结果用 codec 索引表示，与原接口一致）
            next_codec_index = next_general_id - mm_token_base
            if next_codec_index < 0 or next_codec_index >= mm_codebook_size:
                # 越界（理论上 mask 后不会发生），停止生成
                break

            generated.append(next_codec_index)

            # 自回归：用 codec 索引查 codebook 得到 embedding，拼接到每个 neuron
            next_token_tensor = torch.tensor([[next_codec_index]], dtype=torch.long, device=self.device)
            if has_codebook:
                codebook = codec.model.quantizer.codebook  # Embedding
                next_feat = codebook(next_token_tensor)  # [1, 1, latent_dim]
                for nid in mm_nids:
                    neuron = self.neurons[nid]
                    next_emb = neuron.encode_multimodal_input(next_feat, modality)
                    neuron_embeddings[nid] = torch.cat(
                        [neuron_embeddings[nid], next_emb], dim=1
                    )
            else:
                # codec 不可用时用 zeros 填充（退化）
                first_emb = next(iter(neuron_embeddings.values()))
                next_emb = torch.zeros(1, 1, first_emb.shape[-1], device=self.device)
                for nid in mm_nids:
                    neuron_embeddings[nid] = torch.cat(
                        [neuron_embeddings[nid], next_emb], dim=1
                    )

        return generated

    def get_field_state(self) -> torch.Tensor:
        """Get current resonance field state (consciousness snapshot)."""
        return self.field.get_state()

    def get_last_phase(self) -> Optional[float]:
        """C27 增量二（KoPE）：最近一次共振的加权均值相角（相位归属记忆）。

        记忆沉淀时随场快照记录该相位——注入时按记忆相位对齐 theta
        （不同记忆不同相位唤醒）。无相位（编码失败/旧路径）→ None。
        """
        return getattr(self, "_last_phase_mean", None)

    def get_last_field_state(self) -> Optional[torch.Tensor]:
        """最近一次共振后的任务场状态（推理实际写入的场）。

        get_field_state() 返回默认场（多线程推理下恒为陈旧/零状态）；
        本方法取当前线程任务场（_get_task_field），即最近一次 forward/
        continuous_forward 真实写入的状态。R10（REMEDIATION_PLAN 2026-08-14）
        生产记忆接线使用：对话后记录场快照 → 睡眠固化。
        """
        f = self.ensemble._get_task_field()
        st = f.get_state()
        if st is None:
            return None
        if st.numel() == 0 or float(st.norm()) < 1e-8:
            return None
        return st

    # ── 缺口 L：跨域语义锚点投影（2026-08-11，可选挂载，不影响生成路径）──

    def set_anchor_projector(self, projector) -> None:
        """挂载跨域语义锚点投影（AnchorProjector）。

        只作用在场读出侧（field_state → 锚点空间），不改变场写入/判定/生成；
        未挂载时 project_field_state 原样返回（零影响）。
        """
        self._anchor_projector = projector

    def project_field_state(self, field_state) -> torch.Tensor:
        """把场状态投影到跨域语义锚点空间（未挂载时原样返回）。"""
        proj = getattr(self, "_anchor_projector", None)
        if proj is None or field_state is None:
            return field_state
        fs = field_state
        if fs.dim() == 1:
            fs = fs.unsqueeze(0)
        out = proj(fs)
        return out.squeeze(0) if field_state.dim() == 1 else out

    def get_dominant_domain(self) -> Optional[str]:
        """Identify which domain is dominating the current thought."""
        if not self.field.scores:
            return None
        return max(self.field.scores, key=self.field.scores.get)


# _AdaptiveField removed: field_dim is unified under H9; no padding needed.
