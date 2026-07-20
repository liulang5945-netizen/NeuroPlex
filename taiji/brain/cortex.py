"""Cortex — consciousness center via resonance field state.

Phase 4: Replaces the monolithic ModelSelf with ResonanceEnsemble.
The field state IS the consciousness — not a single model's hidden state,
but the collective resonance pattern across domain-specialized neurons.

Architecture:
    Input tokens → Shared Embedding (256K) → ResonanceEnsemble
        ├── zh neuron (standard)
        ├── en neuron (standard)
        ├── code neuron (expert)
        ├── math neuron (expert)
        └── general neuron (standard)
            ↓
    Resonance Field (shared consciousness)
            ↓
    Weighted Logits → Token prediction

Usage:
    cortex = Cortex(neurons_dir="data/neurons")
    output = cortex.generate("今天天气怎么样？", max_tokens=256)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig,
    ConfidenceGate, EarlyStopResonance, QualityFilter, DivisionPath,
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
        confidence_threshold: float = 0.9,
        enable_gating: bool = True,
    ):
        self.device = device
        self.neurons_dir = neurons_dir
        self.max_rounds = max_rounds
        self.enable_gating = enable_gating

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
            field_dim = dims.pop()
        else:
            field_dim = 4096
        self.field = ResonanceField(dim=field_dim)
        self.ensemble = ResonanceEnsemble(
            self.neurons, self.field,
            max_rounds=max_rounds,
            confidence_gate=ConfidenceGate(threshold=confidence_threshold) if enable_gating else None,
            early_stop=EarlyStopResonance() if enable_gating else None,
        )

        # ── Shared embedding (placeholder — in production, use SVD-initialized embedding) ──
        self._shared_embedding: Optional[torch.nn.Embedding] = None
        self._embed_pipeline = None  # H10: callable(input_ids) -> [B, L, base_embed_dim]
        self._tokenizer = None

        # ── Phase 5.1: Thalamic Router ──
        # 丘脑路由器：输入路由到匹配域的 neuron，错误 neuron 不参与 forward
        # None = 关闭丘脑路由（向后兼容，所有 neuron 都 forward）
        self.thalamic_router = None
        # 教师 hidden state 提取器（与 _embed_pipeline 共用教师模型）
        self._teacher_model = None
        self._extract_hidden_fn = None
        # 路由 top-K：hard route 用 1，soft route 用 2
        self._route_top_k = 2
        # 最近一次路由信息（用于诊断日志）
        self._last_routing: Optional[Dict] = None
        # Phase 5.2: 待处理的新生信号（unknown_buffer 满）
        self._pending_neurogenesis: bool = False

        # ── P6-6: SharedContextEncoder（自主进化 encoder，上下文感知 hidden state）──
        # 注册后，think() 和 _route_input() 优先用 encoder.encode() 替代 standalone_embedding lookup
        # 这是 P6-8 训练成果上线的必要条件
        self._context_encoder = None

        # ── P6-3: GammaOscillator（40Hz 同步绑定）──
        # 显式初始化（P0-1 fix: 避免 hasattr 探测的脆弱模式）
        self.gamma_oscillator = None

        # ── P6-4: WorkingMemory（前额叶工作记忆）──
        self.working_memory = None

        # ── State ──
        self.is_loaded = len(self.neurons) > 0
        print(f"[Cortex] Loaded {len(self.neurons)} neurons, field_dim={field_dim}")

    def _load_neurons(self):
        """Load all distilled neurons from disk."""
        for domain in ["zh", "en", "code", "math", "general"]:
            # Prefer Phase 2 field-conditioned checkpoints; fall back to base.
            fc_path = os.path.join(self.neurons_dir, f"neuron_{domain}_fieldcond.pt")
            base_path = os.path.join(self.neurons_dir, f"neuron_{domain}.pt")
            ckpt_path = fc_path if os.path.exists(fc_path) else base_path
            if os.path.exists(ckpt_path):
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
        1. lm_head.weight → 如果 cfg.lm_head_rank>0，禁用低秩（保持传统模式）
        2. side_channels.* → excite_channels.*
        3. 其他新增字段（refractory_counter 等）由 strict=False 跳过
        """
        sd_keys = set(sd.keys())

        # 1. lm_head 兼容：旧 ckpt 有 lm_head.weight，新代码期望 lm_head_delta_u/v
        if "lm_head.weight" in sd_keys and cfg.lm_head_rank > 0:
            # 旧 ckpt + 新 cfg：禁用低秩模式，保持传统 lm_head
            cfg.lm_head_rank = 0
            print(f"  [compat] 检测到旧 lm_head，切换到传统模式 (lm_head_rank=0)")

        # 2. side_channels → excite_channels 重命名
        side_keys = [k for k in sd_keys if k.startswith("side_channels.")]
        if side_keys:
            for k in side_keys:
                new_k = k.replace("side_channels.", "excite_channels.", 1)
                sd[new_k] = sd.pop(k)
            print(f"  [compat] 重命名 {len(side_keys)} 个 side_channels → excite_channels")

        return sd

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for encode/decode."""
        self._tokenizer = tokenizer

    def set_shared_embedding(self, embedding: torch.nn.Embedding) -> None:
        """Set the shared embedding table (highest precedence source)."""
        self._shared_embedding = embedding

    def set_standalone_embedding(self, standalone_embedding) -> None:
        """P6-1: 注册独立 embedding 表（脱教师推理）。

        注册后，think() 会优先用这个 embedding 表 lookup，不再 forward 教师。
        这取代了 set_teacher_pipeline 的依赖路径。

        Args:
            standalone_embedding: StandaloneEmbedding 实例
        """
        self._shared_embedding = standalone_embedding
        # 标记：独立模式，无需教师
        self._teacher_model = None
        self._extract_hidden_fn = None
        self._embed_pipeline = None
        # P0-3 fix: 移到正确 device
        if hasattr(standalone_embedding, 'to'):
            standalone_embedding.to(self.device)
        print(f"[Cortex] StandaloneEmbedding registered (脱教师模式)")
        print(f"  vocab={standalone_embedding.vocab_size}, "
              f"dim={standalone_embedding.embed_dim}")
        print(f"  推理时不再 forward 教师模型")

    def set_context_encoder(self, encoder) -> None:
        """P0-3: 注册 SharedContextEncoder（P6-6 训练产物上线）。

        注册后，think() 和 _route_input() 优先用 encoder.encode() 替代
        standalone_embedding lookup，让 P6-8 训练的上下文感知能力真正生效。

        encoder 必须实现 encode(input_ids) -> [B, L, hidden_dim] 接口。
        通常 hidden_dim 与 standalone_embedding.embed_dim 一致（如 512）。

        Args:
            encoder: SharedContextEncoder 实例
        """
        self._context_encoder = encoder
        if hasattr(encoder, 'to'):
            encoder.to(self.device)
        encoder.eval()  # 推理模式
        print(f"[Cortex] SharedContextEncoder registered (P6-8 训练成果上线)")
        print(f"  推理路径：input_ids → encoder.encode() → shared_emb")
        print(f"  路由路径：input_ids → encoder.encode() → prototypes_embed 匹配")

    def set_teacher_pipeline(self, teacher_model, shared_proj) -> None:
        """H10: register a teacher-model + SharedEmbedProj chain to derive
        shared embeddings from token IDs. Used when no preloaded embedding
        table is available. shared_proj must be an nn.Module mapping
        [..., 2048] -> [..., base_embed_dim].
        """
        import torch as _torch
        from taiji.training.checkpoint_bridge import extract_hidden_states

        # 缓存教师模型，供 ThalamicRouter 路由用
        self._teacher_model = teacher_model
        self._extract_hidden_fn = extract_hidden_states

        def _pipeline(input_ids: _torch.Tensor) -> _torch.Tensor:
            with _torch.no_grad():
                hidden = extract_hidden_states(teacher_model, input_ids)  # [B, L, 2048]
                return shared_proj(hidden)                                  # [B, L, 512]

        self._embed_pipeline = _pipeline

    def set_thalamic_router(self, router, top_k: int = 2) -> None:
        """Phase 5.1: 注册丘脑路由器。

        Args:
            router: ThalamicRouter 实例
            top_k: 路由 top-K neuron 参与推理（hard=1, soft=2）
        """
        self.thalamic_router = router
        self._route_top_k = top_k
        print(f"[Cortex] ThalamicRouter enabled (top_k={top_k}, "
              f"domains={router.list_domains() if router else []})")

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
        self.ensemble.neuromodulator = neuromodulator
        print(f"[Cortex] NeuromodulatorState enabled "
              f"(dopamine={neuromodulator.dopamine:.2f}, "
              f"serotonin={neuromodulator.serotonin:.2f}, "
              f"norepinephrine={neuromodulator.norepinephrine:.2f})")

    def _route_input(self, input_ids: torch.Tensor) -> Optional[List[str]]:
        """Phase 5.1 / P6-2 / P0-3: 用 ThalamicRouter 对输入路由，返回 active_nids。

        P0-3 扩展：当 SharedContextEncoder 已注册时，优先用 route_top_k_by_hidden，
        让 P6-8 训练的上下文感知能力真正生效。

        路由优先级：
        (a) SharedContextEncoder.encode() → route_top_k_by_hidden（P0-3，最佳）
        (b) standalone_embedding → route_top_k_by_embedding（P6-2，fallback）
        (c) teacher hidden state → route_top_k（旧路径，兼容）

        Returns:
            active_nids: List[str] 参与 forward 的 neuron IDs
                         None 表示不做路由（全部参与）
        """
        if self.thalamic_router is None:
            self._last_routing = None
            return None

        # P0-3: 优先用 SharedContextEncoder（P6-8 训练成果上线）
        if self._context_encoder is not None:
            if not getattr(self.thalamic_router, 'prototypes_embed', None):
                # 还没建 embedding prototypes，暂时不路由
                self._last_routing = None
                return None
            with torch.no_grad():
                hidden_states = self._context_encoder.encode(input_ids)
                weights, top_nids = self.thalamic_router.route_top_k_by_hidden(
                    hidden_states, k=self._route_top_k,
                )
                decision = self.thalamic_router.get_routing_decision_by_hidden(
                    hidden_states,
                )
                is_unknown = decision.get('is_unknown', False)
                routing_mode = 'encoder'
        # P6-2: 脱教师路径 — standalone_embedding 已注册时用
        elif self._shared_embedding is not None and not callable(getattr(self, '_extract_hidden_fn', None)):
            if not getattr(self.thalamic_router, 'prototypes_embed', None):
                self._last_routing = None
                return None
            with torch.no_grad():
                weights, top_nids = self.thalamic_router.route_top_k_by_embedding(
                    input_ids, self._shared_embedding, k=self._route_top_k,
                )
                decision = self.thalamic_router.get_routing_decision_by_embedding(
                    input_ids, self._shared_embedding,
                )
                is_unknown = decision.get('is_unknown', False)
                routing_mode = 'embedding'
        elif self._teacher_model is not None:
            with torch.no_grad():
                hidden = self._extract_hidden_fn(self._teacher_model, input_ids)
                weights, top_nids = self.thalamic_router.route_top_k(
                    hidden[0], k=self._route_top_k,
                )
                decision = self.thalamic_router.get_routing_decision(hidden[0])
                is_unknown = decision.get('is_unknown', False)
                routing_mode = 'teacher'
        else:
            self._last_routing = None
            return None

        if not top_nids:
            self._last_routing = None
            return None

        # P0-1 fix: 校验 top_nids 真的存在于 self.neurons（避免路由失败静默 fallback）
        valid_nids = [nid for nid in top_nids if nid in self.neurons]
        if not valid_nids:
            print(f"[Cortex] 警告：路由返回的 top_nids {top_nids} "
                  f"不在 neurons {list(self.neurons.keys())} 中，路由失效")
            self._last_routing = None
            return None
        top_nids = valid_nids

        # 缓存路由信息用于日志
        self._last_routing = {
            'weights': weights,
            'top_nids': top_nids,
            'strategy': 'hard' if self._route_top_k == 1 else 'soft',
            'is_unknown': is_unknown,
            'max_sim': decision.get('max_sim', 0.0),
            'mode': routing_mode,
        }

        # Phase 5.2: 未知域 buffer 检查（不阻塞当前 generate）
        if is_unknown:
            should_trigger, buf_size = self.thalamic_router.check_unknown_buffer()
            if should_trigger:
                self._pending_neurogenesis = True
                print(f"[Cortex] 未知域 buffer 已满 ({buf_size})，"
                      f"待 sleep cycle 触发神经新生")

        return top_nids

    def drain_unknown_buffer(self) -> Optional[torch.Tensor]:
        """Phase 5.2: 清空并返回 unknown_buffer（sleep cycle 时调用）。

        Returns:
            [N, hidden_dim] tensor，N=0 表示无未知样本
        """
        if self.thalamic_router is None:
            return None
        self._pending_neurogenesis = False
        return self.thalamic_router.drain_unknown_buffer()

    def think(self, input_ids: torch.Tensor, route: bool = True) -> Dict:
        """Run one round of resonance thinking.

        Args:
            input_ids: [B, L] token IDs.
            route: Phase 5.1 - 是否用 ThalamicRouter 路由（默认 True）

        Returns:
            dict with field_state, weighted_logits, final_scores, n_rounds.
        """
        # P0-3 fix: shared_emb 优先级
        # (a) SharedContextEncoder.encode() — P6-8 训练成果，上下文感知
        # (b) set_shared_embedding / set_standalone_embedding — 纯 lookup
        # (c) set_teacher_pipeline — 教师 hidden state
        # (d) fail loudly
        input_ids = input_ids.to(self.device)
        if self._context_encoder is not None:
            with torch.no_grad():
                shared_emb = self._context_encoder.encode(input_ids)
        elif self._shared_embedding is not None:
            shared_emb = self._shared_embedding(input_ids)
        elif self._embed_pipeline is not None:
            shared_emb = self._embed_pipeline(input_ids)
        else:
            raise RuntimeError(
                "[Cortex.think] no shared_embedding source. Either call "
                "set_context_encoder(encoder) for P6-8 trained encoder, "
                "set_shared_embedding(embedding_table) with a preloaded table, "
                "or set_teacher_pipeline(teacher_model, shared_proj) to derive "
                "embeddings from teacher hidden states on the fly."
            )

        # Phase 5.1: 丘脑路由 - 只激活匹配域的 neuron
        active_nids = None
        if route:
            active_nids = self._route_input(input_ids)

        result = self.ensemble.forward(
            shared_emb, return_logits=True,
            enable_gating=self.enable_gating,
            active_nids=active_nids,
        )
        # 把路由信息附加到结果（便于诊断）
        if self._last_routing:
            result['routing'] = self._last_routing
        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate text using resonance ensemble.

        Args:
            prompt: input text.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: top-k sampling.

        Returns:
            generated text string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call cortex.set_tokenizer() first.")

        # Encode
        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = list(prompt_ids)

        # P6-4: 若启用 WorkingMemory，把 memory 内容作为前缀拼接
        memory_enabled = (
            self.working_memory is not None
            and not self.working_memory.is_empty()
        )
        if memory_enabled:
            memory_ids = self.working_memory.get_context_ids()
            input_ids = memory_ids + input_ids

        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = []
        for _ in range(max_tokens):
            # Get logits from ensemble
            result = self.think(ids_tensor)

            if "weighted_logits" not in result:
                break

            logits = result["weighted_logits"][:, -1, :] / temperature  # [B, vocab]

            # Top-k sampling
            if top_k > 0:
                top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx_in_topk = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, sampled_idx_in_topk[0]].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # P1-4: 每生成一个 token 推进 Gamma 相位（40Hz 同步绑定）
            if self.gamma_oscillator is not None:
                self.tick_gamma()

            # Check for EOS (P0-1 fix: 处理 tensor 类型的 eos_id)
            eos_id = getattr(self._tokenizer, 'eos_token_id', None)
            if eos_id is None:
                eos_id = getattr(self._tokenizer, 'eos_id', None)
            if callable(eos_id):
                eos_id = eos_id()
            # 归一化 eos_id 为 int（防止 tokenizer 返回 tensor 导致误判）
            if isinstance(eos_id, torch.Tensor):
                eos_id = eos_id.item() if eos_id.numel() == 1 else None
            if eos_id is not None and next_token == eos_id:
                break

            generated.append(next_token)
            ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        # P6-4: 若启用 WorkingMemory，记录本轮对话（P0-1 fix: 只在生成非空时记录）
        if self.working_memory is not None and generated:
            self.working_memory.append_round(
                prompt_ids=prompt_ids,
                generated_ids=generated,
            )

        # Decode
        return self._tokenizer.decode(generated) if generated else ""

    def get_field_state(self) -> torch.Tensor:
        """Get current resonance field state (consciousness snapshot)."""
        return self.field.get_state()

    def get_dominant_domain(self) -> Optional[str]:
        """Identify which domain is dominating the current thought."""
        if not self.field.scores:
            return None
        return max(self.field.scores, key=self.field.scores.get)


# _AdaptiveField removed: field_dim is unified under H9; no padding needed.
