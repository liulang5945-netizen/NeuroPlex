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
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger("Cortex")

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig,
)


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
    ):
        self.device = device
        self.neurons_dir = neurons_dir
        self.max_rounds = max_rounds
        self.is_loaded = False

        # ── Load neurons ──
        self.neurons: Dict[str, ResonanceNeuron] = {}
        self._load_neurons()

        # ── Create field and ensemble ──
        if self.neurons:
            dims = {n.config.field_dim for n in self.neurons.values()}
            if len(dims) > 1:
                raise ValueError(
                    f"[Cortex] neurons disagree on field_dim: {dims}. "
                    f"Re-distill legacy checkpoints under H9 (field_dim=4096) before loading."
                )
            hidden_sizes = {n.config.hidden_size for n in self.neurons.values()}
            if len(hidden_sizes) > 1:
                raise ValueError(
                    f"[Cortex] neurons disagree on hidden_size: {hidden_sizes}. "
                    f"All neurons must share the same hidden_size."
                )
            field_dim = dims.pop()
        else:
            field_dim = 4096
        self.field = ResonanceField(dim=field_dim)
        # P1-1: CoactivationTracker（共激活追踪，供孤立检测+部落分组）
        from taiji.resonance.tribal import CoactivationTracker
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

        # ── General tokenizer (256K, hot-swappable I/O protocol) ──
        self._general_sp = general_tokenizer

        # ── Domain tokenizer hub ──
        # Manages per-domain tokenizers (zh=20K, en=16K, code=12K, math=10K).
        # Used for domain-specific lm_head targets and decoding.
        self._tokenizer_hub = None

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
        self.working_memory = None

        # ── State ──
        self.is_loaded = len(self.neurons) > 0
        print(f"[Cortex] Loaded {len(self.neurons)} neurons, field_dim={field_dim}")
        if self._shared_embedding is not None:
            print(f"[Cortex] Shared embedding: {self._shared_embedding.num_embeddings} × {self._shared_embedding.embedding_dim}")

    def _load_neurons(self):
        """Load all distilled neurons from disk.

        H3 修复：原来硬编码 5 域 ['zh','en','code','math','general']，
        新生 neuron（如 neuron_physics_1.pt）被静默忽略。
        现改为扫描 neurons_dir 下所有 neuron_*.pt 文件动态加载。
        """
        import glob
        # 扫描所有 neuron_*.pt（排除 _fieldcond.pt 等非 neuron 文件）
        ckpt_paths = sorted(glob.glob(os.path.join(self.neurons_dir, "neuron_*.pt")))
        for ckpt_path in ckpt_paths:
            name = os.path.basename(ckpt_path)
            # 跳过 fieldcond 版（下面会优先加载）、W_base 等非 neuron 文件
            if "_fieldcond" in name or name.startswith("_"):
                continue
            # 从文件名提取 domain：neuron_{domain}.pt → {domain}
            domain = name[len("neuron_"):-len(".pt")]

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
            neuron.freeze_fingerprint()
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
        from taiji.resonance.translator import TokenizerHub
        if not isinstance(tokenizer_hub, TokenizerHub):
            raise TypeError(
                f"[Cortex] set_tokenizer_hub expects TokenizerHub, "
                f"got {type(tokenizer_hub).__name__}"
            )
        self._tokenizer_hub = tokenizer_hub
        domains = tokenizer_hub.list_domains()
        print(f"[Cortex] TokenizerHub registered (P7 模式)")
        print(f"  domains: {domains}")
        for d in domains:
            print(f"  {d}: vocab={tokenizer_hub.vocab_size(d)}")

    def set_shared_embedding(self, embedding: torch.nn.Embedding) -> None:
        """Set the shared embedding table (highest precedence source)."""
        self._shared_embedding = embedding

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

        不保存：frozen backbone（来自蒸馏 ckpt，不变）、field state（运行时状态）

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
        from taiji.resonance.gamma_oscillator import apply_gamma_gate

        # 按 domain 分组 neuron
        domain_to_nids: Dict[str, list] = {}
        for nid in self.neurons.keys():
            # nid 可能是 "zh" / "math" / "code_xxx" 等
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids.setdefault(domain, []).append(nid)

        oscillator.assign_phase_by_domain(domain_to_nids)
        apply_gamma_gate(self.field, oscillator)
        self.gamma_oscillator = oscillator
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

    def add_neuron(self, domain: str, lifecycle=None) -> str:
        """运行时创建新神经元并加入 ensemble（neurogenesis 入口）。

        流程：
        1. 生成新 neuron ID（{domain}_{n} 格式，如 zh_1）
        2. 用 get_domain_neuron_config 创建 NeuronConfig（COMPACT 规格）
        3. 实例化 ResonanceNeuron → to(device) → eval → freeze_fingerprint
        4. 多模态注册（auto_register_modalities）
        5. 持久化 ckpt 到 neurons_dir/neuron_{nid}.pt
        6. 注入 cortex.neurons + ensemble.add_neuron
        7. lifecycle.maturity.register_new（幼稚态追踪）

        Args:
            domain: 域名（zh/en/code/math/general）
            lifecycle: LifecycleManager 实例（可选，用于 maturity.register_new）

        Returns:
            新神经元的 ID（如 "zh_1"）
        """
        from taiji.resonance.config import get_domain_neuron_config, DOMAIN_VOCAB_SIZES

        if domain not in DOMAIN_VOCAB_SIZES:
            raise ValueError(
                f"未知 domain: {domain}. 可选: {list(DOMAIN_VOCAB_SIZES.keys())}"
            )

        # 1. 生成唯一 neuron ID
        n = 1
        while f"{domain}_{n}" in self.neurons:
            n += 1
        nid = f"{domain}_{n}"

        # 2. 创建 NeuronConfig（COMPACT 规格，与现有神经元一致）
        cfg = get_domain_neuron_config(domain)
        cfg.neuron_id = nid

        # 3. 实例化神经元
        neuron = ResonanceNeuron(cfg).to(self.device)
        neuron.eval()
        neuron.freeze_fingerprint()

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

        # 1. 从 neurons dict 移除（cortex.neurons 和 ensemble.neurons 是同一引用）
        removed = self.neurons.pop(nid)

        # 2. 清理其他神经元的 side channel 引用
        for other_nid, other_neuron in self.neurons.items():
            try:
                if hasattr(other_neuron, "excite_channels"):
                    other_neuron.excite_channels = [
                        ch for ch in other_neuron.excite_channels
                        if getattr(ch, "target_id", None) != nid
                    ]
                if hasattr(other_neuron, "inhibit_channels"):
                    other_neuron.inhibit_channels = [
                        ch for ch in other_neuron.inhibit_channels
                        if getattr(ch, "target_id", None) != nid
                    ]
            except Exception:
                pass

        # 3. 删除 ckpt 文件
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

    def think(self, shared_embeddings: torch.Tensor) -> Dict:
        """Run one round of resonance thinking.

        All neurons receive the same shared_embeddings (from shared embedding table).
        This ensures field vectors are comparable — cosine similarity is meaningful.

        Args:
            shared_embeddings: [B, L, base_embed_dim] from shared_embedding(general_ids).

        Returns:
            dict with field_state, neuron_logits, final_scores, n_rounds.
        """
        shared_embeddings = shared_embeddings.to(self.device)
        result = self.ensemble.forward(
            shared_embeddings=shared_embeddings,
            return_logits=True,
        )
        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        domain: Optional[str] = None,
        repetition_penalty: float = 1.2,
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

        Returns:
            generated text string.
        """
        if self._tokenizer_hub is None:
            raise RuntimeError("TokenizerHub not set. Call cortex.set_tokenizer_hub() first.")

        # 读侧保护：与 sleep 训练互斥，最多等待 10 秒
        # sleep 训练用 try_start_training() 非阻塞获取，失败跳过；generate 持有锁时 sleep 让步
        from taiji.core.app_state import app_state
        acquired = app_state.train_lock.acquire(timeout=10)
        if not acquired:
            logger.warning("Cortex.generate 等待训练锁超时（10s），可能并发")
        try:
            return self._generate_p7(
                prompt, max_tokens, temperature, top_k, domain,
                repetition_penalty,
            )
        finally:
            if acquired:
                app_state.train_lock.release()

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
        仅在对应域 neuron 已加载时返回该域。

        Returns:
            domain name ("zh"/"en"/"code"/"math"/"general")
        """
        neuron_domains = set(self.neurons.keys())
        if not neuron_domains:
            return "general"

        # 1. 代码检测：编程关键字 + 代码结构特征
        code_keywords = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'function ', 'const ', 'let ', 'var ', 'async ',
            'if __name__', 'print(', 'lambda ',
        ]
        code_patterns = ['{', '};', '=>', 'self.', 'std::']
        if 'code' in neuron_domains:
            code_score = sum(1 for kw in code_keywords if kw in text)
            code_score += sum(1 for p in code_patterns if p in text)
            # 三引号代码块是强信号
            if '```' in text:
                code_score += 5
            if code_score >= 2:
                return 'code'

        # 2. 数学检测：数学符号密度 + 公式特征
        if 'math' in neuron_domains:
            math_chars = set('=+-*/^∑∫∏√∞∂∇∈⊂∪∩∀∃≤≥≠≈±')
            math_count = sum(1 for c in text if c in math_chars)
            # 纯数字表达式（如 "1+1=", "5*3-2"）
            stripped = text.replace(' ', '').replace('\n', '')
            digit_math = sum(1 for c in stripped if c.isdigit() or c in '+-*/=()^.')
            total_len = max(len(stripped), 1)
            if math_count >= 3 or (digit_math / total_len > 0.5 and len(stripped) > 2):
                return 'math'

        # 3. 中文检测（CJK 统一汉字区块）
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cjk_count > len(text.replace(' ', '')) * 0.3:
            return 'zh' if 'zh' in neuron_domains else next(iter(neuron_domains))

        # 4. 默认：en 或 general
        if 'en' in neuron_domains:
            return 'en'
        if 'general' in neuron_domains:
            return 'general'
        return next(iter(neuron_domains))

    def _generate_p7(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        domain: Optional[str] = None,
        repetition_penalty: float = 1.2,
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
        if domain is None:
            domain = self._infer_domain(prompt)
        if domain not in hub.list_domains():
            domain = hub.list_domains()[0] if hub.list_domains() else "general"

        # 2. Encode prompt with general tokenizer → shared embedding
        prompt_general_ids = self._general_sp.encode(prompt)
        if not prompt_general_ids:
            prompt_general_ids = [0]
        general_ids = list(prompt_general_ids)

        # 2.5 SMCS-inspired instance-level routing: resonance-based domain verification
        # 关键词路由可能误判（如中文夹杂代码、数学符号）。用共振分数校验：
        # 如果选定域的共振分数显著低于最强域，切换到最强域。
        if len(self.neurons) > 1:
            try:
                probe_ids = torch.tensor([general_ids], dtype=torch.long, device=self.device)
                probe_emb = self._shared_embedding(probe_ids)
                with torch.no_grad():
                    probe_result = self.ensemble.forward(
                        shared_embeddings=probe_emb, return_logits=False,
                    )
                probe_scores = probe_result.get("final_scores", {})
                if probe_scores:
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
        eos_id = hub.eos_token_id(domain)

        # Get domain tokenizer for decoding
        domain_sp = hub.get_tokenizer(domain)

        generated_pieces = []
        generated_token_ids = set()
        generated_token_list = []  # 保持顺序，用于 no-repeat-ngram
        no_repeat_ngram_size = 3

        for _ in range(max_tokens):
            # Trim context to prevent memory issues and maintain coherence
            if len(general_ids) > 512:
                general_ids = general_ids[-512:]

            # Embed general IDs → [1, L, 512]
            ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=self.device)
            shared_emb = self._shared_embedding(ids_tensor)

            result = self.think(shared_emb)

            # Get logits: MoCo-inspired dynamic fusion
            neuron_logits = result.get("neuron_logits", {})
            final_scores = result.get("final_scores", {})

            if neuron_logits and final_scores:
                # MoCo-style dynamic logit fusion: all neurons participate,
                # dynamically weighted by current resonance scores
                fused_logits = self.ensemble._dynamic_logit_fusion(
                    neuron_logits, final_scores, temperature=0.5,
                )
                logits = fused_logits[:, -1, :] / temperature
                # 融合后 logits vocab = max(all neuron vocabs)，但解码只用 domain_sp
                # 截断到 domain vocab，避免 id_to_piece 越界
                domain_vocab = hub.vocab_size(domain)
                if logits.shape[-1] > domain_vocab:
                    logits = logits[..., :domain_vocab]
            elif domain in neuron_logits:
                # Fallback: domain-specific logits only
                logits = neuron_logits[domain][:, -1, :] / temperature
            elif neuron_logits:
                first_logits = next(iter(neuron_logits.values()))
                logits = first_logits[:, -1, :] / temperature
            elif "weighted_logits" in result:
                logits = result["weighted_logits"][:, -1, :] / temperature
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

            if self.gamma_oscillator is not None:
                self.tick_gamma()

            # EOS check
            if eos_id is not None and next_token == eos_id:
                break

            # Decode domain token → text piece → re-encode with general tokenizer
            if domain_sp is not None:
                piece = domain_sp.id_to_piece(next_token)
                generated_pieces.append(piece)
                # Re-encode with general tokenizer for next iteration
                # Remove "▁" prefix for clean concatenation
                new_general_ids = self._general_sp.encode(piece)
                if new_general_ids:
                    general_ids.extend(new_general_ids)
            else:
                general_ids.append(next_token)

        # Decode with domain tokenizer
        if generated_pieces:
            return "".join(generated_pieces).replace("▁", " ")
        return ""

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

        # 2. 读侧保护（与 generate 共用 train_lock）
        from taiji.core.app_state import app_state
        acquired = app_state.train_lock.acquire(timeout=10)
        if not acquired:
            logger.warning("Cortex.generate_multimodal 等待训练锁超时（10s），可能并发")
        try:
            return self._generate_multimodal_p8(
                features, actual_modality, domain,
                max_tokens, temperature, top_k,
            )
        finally:
            if acquired:
                app_state.train_lock.release()

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

        策略：
        1. 所有注册了该模态投影层的 neuron 都参与共振（小神经元协同）
        2. 每个 neuron 独立预编码多模态 embedding（neuron_embeddings）
        3. ensemble.forward 多轮共振，mm_logits_modality 让所有 neuron 输出同 vocab 的 codebook logits
        4. 取 weighted_logits（共振加权）采样，自回归生成
        """
        hub = self._tokenizer_hub

        # 1. 找出所有支持该模态的 neuron
        mm_nids = [
            nid for nid, neuron in self.neurons.items()
            if modality in neuron.mm_projections and modality in neuron.mm_lm_heads
        ]
        if not mm_nids:
            logger.warning(
                f"无 neuron 注册了模态 '{modality}' 投影层和输出头，fallback 到 text 路径"
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

        # 4. 多模态 EOS
        eos_id = hub.eos_token_id(domain=domain, modality=modality)

        # 5. 自回归生成（多 neuron 共振）
        generated = []
        codec = hub.modal_encoders.get(modality)
        has_codebook = (
            codec is not None
            and hasattr(codec, "model")
            and hasattr(codec.model, "quantizer")
        )

        for step in range(max_tokens):
            # 多轮共振：所有 neuron 参与，mm_logits_modality 统一输出 codebook logits
            res = self.ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                active_filter=True,
                active_nids=mm_nids,
                mm_logits_modality=modality,
            )

            # 取加权 logits（所有 neuron vocab 相同，ensemble 已加权合并）
            if "weighted_logits" in res:
                logits = res["weighted_logits"]  # [B, L, mm_vocab]
            elif "neuron_logits" in res and res["neuron_logits"]:
                # fallback: 取第一个 neuron 的 logits
                first_nid = next(iter(res["neuron_logits"].keys()))
                logits = res["neuron_logits"][first_nid]
            else:
                raise RuntimeError("共振未返回 logits，无法生成")

            logits = logits[:, -1, :] / temperature  # [B, mm_vocab]

            if top_k > 0:
                actual_k = min(top_k, logits.shape[-1])
                top_k_vals, top_k_indices = torch.topk(logits, actual_k)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx_in_topk = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, sampled_idx_in_topk[0]].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            if eos_id is not None and eos_id >= 0 and next_token == eos_id:
                break

            generated.append(next_token)

            # 自回归：把 next_token 编码为新 embedding，拼接到每个 neuron 的 embedding
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
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

    def get_dominant_domain(self) -> Optional[str]:
        """Identify which domain is dominating the current thought."""
        if not self.field.scores:
            return None
        return max(self.field.scores, key=self.field.scores.get)


# _AdaptiveField removed: field_dim is unified under H9; no padding needed.
