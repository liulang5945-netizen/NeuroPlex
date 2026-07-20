"""神经新生创建器 — 分场景教师选择（P2-7 决策 C）。

人脑启发：海马齿状回新生神经元需要"教师"引导分化。
态极的神经新生教师选择策略：

- **子域**（domain 已有神经元）：用现有神经元作教师
  → copy 权重 + 在新数据上 fine-tune，保留域内共性、学习子域个性
  → 类比：新神经元从成熟的"邻居"学习，快速分化

- **新域**（domain 无神经元）：用 1.5B gen1 ModelSelf 作教师
  → 从 1.5B 教师蒸馏，获得该域的基础语言能力
  → 类比：新脑区从原始皮层学习基础能力

使用方式：
    creator = NeurogenesisCreator(cortex, lifecycle, feed_engine)
    result = creator.create_neuron_for_domain("physics", samples)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("Taiji.Neurogenesis")


class NeurogenesisCreator:
    """神经新生创建器 — 根据域场景选择教师并创建新神经元。

    P2-7 决策 C：子域用神经元作教师，新域用 1.5B 作教师。
    """

    def __init__(
        self,
        cortex: Any,
        lifecycle: Any,
        feed_engine: Any = None,
        teacher_1_5b_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            cortex: Cortex 实例（含 neurons + ensemble）
            lifecycle: LifecycleManager（含 maturity.register_new）
            feed_engine: FeedEngine（获取域训练样本）
            teacher_1_5b_dir: 1.5B gen1 教师模型目录（新域教师）
            device: 计算设备
        """
        self.cortex = cortex
        self.lifecycle = lifecycle
        self.feed_engine = feed_engine
        self.teacher_1_5b_dir = teacher_1_5b_dir
        self.device = device

        # 域名 → 神经元 ID 前缀映射（用于生成新神经元 ID）
        self._domain_counter: Dict[str, int] = {}

    def create_neuron_for_domain(
        self,
        domain: str,
        samples: Optional[List[str]] = None,
        num_finetune_steps: int = 200,
        lr: float = 5e-4,
    ) -> Dict[str, Any]:
        """为指定域创建新神经元（分场景教师选择）。

        Args:
            domain: 域名（如 "math"、"physics"）
            samples: 训练样本（文本列表）。若为 None，从 feed_engine 获取。
            num_finetune_steps: 微调步数（子域场景）
            lr: 学习率

        Returns:
            {
                "success": bool,
                "neuron_id": str,
                "teacher_type": "neuron" | "1.5b" | "none",
                "error": str | None,
            }
        """
        if self.cortex is None:
            return {"success": False, "neuron_id": "", "teacher_type": "none",
                    "error": "cortex 未注入"}

        # 1. 获取训练样本
        if samples is None and self.feed_engine is not None:
            try:
                domain_samples = self.feed_engine.get_pending_samples_by_domain()
                samples = domain_samples.get(domain, [])
            except Exception as e:
                logger.debug(f"从 feed_engine 获取样本失败: {e}")
                samples = []

        if not samples:
            logger.warning(f"域 '{domain}' 无训练样本，新生神经元将随机初始化")
            samples = []

        # 2. 判断场景：子域 vs 新域
        existing_neuron = self.cortex.neurons.get(domain)
        is_subdomain = existing_neuron is not None

        # 3. 选择教师并创建神经元
        if is_subdomain:
            return self._create_from_neuron_teacher(
                domain, existing_neuron, samples, num_finetune_steps, lr
            )
        else:
            return self._create_from_1_5b_teacher(domain, samples, lr)

    def _create_from_neuron_teacher(
        self,
        domain: str,
        teacher_neuron: Any,
        samples: List[str],
        num_steps: int,
        lr: float,
    ) -> Dict[str, Any]:
        """子域场景：用现有神经元作教师（copy + fine-tune）。

        策略：复制现有神经元的权重作为初始化，然后在新样本上微调。
        这样新神经元继承了域内共性，又能学习子域个性。

        Args:
            domain: 域名
            teacher_neuron: 现有神经元（教师）
            samples: 训练样本
            num_steps: 微调步数
            lr: 学习率

        Returns:
            创建结果 dict
        """
        try:
            from taiji.resonance import ResonanceNeuron, NeuronConfig

            # 生成新神经元 ID
            new_id = self._generate_neuron_id(domain)

            # 复制教师配置（子域用相同 spec）
            teacher_cfg = teacher_neuron.config
            new_cfg = NeuronConfig(
                hidden_size=teacher_cfg.hidden_size,
                num_hidden_layers=teacher_cfg.num_hidden_layers,
                num_attention_heads=teacher_cfg.num_attention_heads,
                num_key_value_heads=teacher_cfg.num_key_value_heads,
                intermediate_size=teacher_cfg.intermediate_size,
                spec=f"{domain}-sub-{new_id}",
                field_dim=teacher_cfg.field_dim,
                vocab_size=teacher_cfg.vocab_size,
                base_embed_dim=teacher_cfg.base_embed_dim,
                neuron_type="excitatory",
            )

            # 创建新神经元并复制权重
            new_neuron = ResonanceNeuron(new_cfg).to(self.device)
            new_neuron.load_state_dict(teacher_neuron.state_dict(), strict=False)
            logger.info(
                f"子域神经新生: 域='{domain}', 新ID='{new_id}', "
                f"教师=现有神经元({type(teacher_neuron).__name__})"
            )

            # 微调（若有样本）
            if samples:
                self._finetune_neuron(new_neuron, domain, samples, num_steps, lr)

            new_neuron.eval()
            self._register_neuron(domain, new_id, new_neuron)

            return {
                "success": True,
                "neuron_id": new_id,
                "teacher_type": "neuron",
                "domain": domain,
                "error": None,
            }

        except Exception as e:
            logger.error(f"子域神经新生失败: {e}", exc_info=True)
            return {"success": False, "neuron_id": "", "teacher_type": "neuron",
                    "error": str(e)}

    def _create_from_1_5b_teacher(
        self,
        domain: str,
        samples: List[str],
        lr: float,
    ) -> Dict[str, Any]:
        """新域场景：用 1.5B gen1 ModelSelf 作教师。

        策略：从 1.5B 教师蒸馏新神经元。需要 teacher_1_5b_dir 配置。
        若教师不可用，降级为随机初始化（fallback）。

        Args:
            domain: 域名（新域）
            samples: 训练样本
            lr: 学习率

        Returns:
            创建结果 dict
        """
        try:
            from taiji.resonance import ResonanceNeuron, NeuronConfig

            new_id = self._generate_neuron_id(domain)

            # 创建标准配置的新神经元
            new_cfg = NeuronConfig(
                spec=f"{domain}-new-{new_id}",
                field_dim=4096,
                neuron_type="excitatory",
            )
            new_neuron = ResonanceNeuron(new_cfg).to(self.device)

            # 尝试从 1.5B 教师蒸馏
            teacher_loaded = False
            if self.teacher_1_5b_dir and os.path.exists(self.teacher_1_5b_dir):
                try:
                    teacher_loaded = self._distill_from_1_5b(
                        new_neuron, domain, samples, lr
                    )
                except Exception as e:
                    logger.warning(f"1.5B 教师蒸馏失败，降级为随机初始化: {e}")
                    teacher_loaded = False

            if teacher_loaded:
                logger.info(
                    f"新域神经新生: 域='{domain}', 新ID='{new_id}', "
                    f"教师=1.5B ModelSelf"
                )
            else:
                logger.warning(
                    f"新域神经新生: 域='{domain}', 新ID='{new_id}', "
                    f"教师=无（随机初始化 fallback，1.5B 教师不可用）"
                )

            new_neuron.eval()
            self._register_neuron(domain, new_id, new_neuron)

            return {
                "success": True,
                "neuron_id": new_id,
                "teacher_type": "1.5b" if teacher_loaded else "none",
                "domain": domain,
                "error": None,
            }

        except Exception as e:
            logger.error(f"新域神经新生失败: {e}", exc_info=True)
            return {"success": False, "neuron_id": "", "teacher_type": "1.5b",
                    "error": str(e)}

    def _distill_from_1_5b(
        self,
        student_neuron: Any,
        domain: str,
        samples: List[str],
        lr: float,
        num_steps: int = 500,
    ) -> bool:
        """从 1.5B 教师蒸馏新神经元。

        Args:
            student_neuron: 待训练的学生神经元
            domain: 域名
            samples: 训练样本
            lr: 学习率
            num_steps: 蒸馏步数

        Returns:
            True 如果蒸馏成功
        """
        from taiji.training.checkpoint_bridge import load_teacher_model

        # 加载 1.5B 教师
        teacher_model, shared_embedding = load_teacher_model(
            self.teacher_1_5b_dir, device=self.device
        )

        tokenizer = getattr(self.cortex, '_tokenizer', None)
        if tokenizer is None:
            logger.warning("Cortex 未设置 tokenizer，无法蒸馏")
            return False

        # 设置 shared_embedding 到 cortex（若未设置）
        if getattr(self.cortex, '_shared_embedding', None) is None:
            self.cortex.set_shared_embedding(shared_embedding)

        # 准备训练数据
        if not samples:
            logger.warning(f"域 '{domain}' 无样本，跳过蒸馏（仅随机初始化）")
            return False

        # 编码样本
        encoded = []
        for text in samples[:50]:  # 限制样本数避免过长
            try:
                ids = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer._encode(text)
                if len(ids) > 10:
                    encoded.append(ids[:512])
            except Exception:
                continue

        if not encoded:
            logger.warning(f"域 '{domain}' 样本编码失败")
            return False

        # 简化蒸馏：仅用 LM loss 训练（完整蒸馏见 distill.py）
        optimizer = torch.optim.AdamW(student_neuron.parameters(), lr=lr)
        student_neuron.train()
        teacher_model.eval()

        step = 0
        total_loss = 0.0
        for epoch in range(10):  # 最多 10 个 epoch
            for ids in encoded:
                if step >= num_steps:
                    break
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    shared_emb = shared_embedding(input_ids)

                # 学生 forward
                output = student_neuron.forward(shared_emb, return_logits=True)
                logits = output.get('logits') if isinstance(output, dict) else output
                if logits is None:
                    continue

                # LM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_neuron.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                step += 1

        avg_loss = total_loss / max(step, 1)
        logger.info(f"1.5B 蒸馏完成: 域='{domain}', steps={step}, avg_loss={avg_loss:.4f}")
        return True

    def _finetune_neuron(
        self,
        neuron: Any,
        domain: str,
        samples: List[str],
        num_steps: int,
        lr: float,
    ) -> None:
        """微调神经元（子域场景）。

        Args:
            neuron: 待微调神经元
            domain: 域名
            samples: 训练样本
            num_steps: 微调步数
            lr: 学习率
        """
        tokenizer = getattr(self.cortex, '_tokenizer', None)
        shared_embedding = getattr(self.cortex, '_shared_embedding', None)
        embed_pipeline = getattr(self.cortex, '_embed_pipeline', None)

        if tokenizer is None:
            logger.warning("Cortex 未设置 tokenizer，跳过微调")
            return

        # 编码样本
        encoded = []
        for text in samples[:50]:
            try:
                ids = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer._encode(text)
                if len(ids) > 10:
                    encoded.append(ids[:512])
            except Exception:
                continue

        if not encoded:
            logger.warning(f"域 '{domain}' 样本编码失败，跳过微调")
            return

        optimizer = torch.optim.AdamW(
            [p for p in neuron.parameters() if p.requires_grad], lr=lr
        )
        neuron.train()

        step = 0
        total_loss = 0.0
        for epoch in range(10):
            for ids in encoded:
                if step >= num_steps:
                    break
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

                # 获取 shared embedding
                with torch.no_grad():
                    if shared_embedding is not None:
                        shared_emb = shared_embedding(input_ids)
                    elif embed_pipeline is not None:
                        shared_emb = embed_pipeline(input_ids)
                    else:
                        continue

                output = neuron.forward(shared_emb, return_logits=True)
                logits = output.get('logits') if isinstance(output, dict) else output
                if logits is None:
                    continue

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(neuron.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                step += 1

        avg_loss = total_loss / max(step, 1)
        logger.info(f"子域微调完成: 域='{domain}', steps={step}, avg_loss={avg_loss:.4f}")

    def _generate_neuron_id(self, domain: str) -> str:
        """生成新神经元 ID（格式：{domain}_{counter}）。"""
        self._domain_counter[domain] = self._domain_counter.get(domain, 0) + 1
        return f"{domain}_{self._domain_counter[domain]}"

    def _register_neuron(self, domain: str, neuron_id: str, neuron: Any) -> None:
        """注册新神经元到 cortex 和 lifecycle.maturity。

        Args:
            domain: 域名
            neuron_id: 神经元 ID
            neuron: ResonanceNeuron 实例
        """
        # 注册到 cortex.neurons
        self.cortex.neurons[neuron_id] = neuron

        # 重建 ensemble 以包含新神经元
        try:
            from taiji.resonance import (
                ResonanceField, ResonanceEnsemble,
                ConfidenceGate, EarlyStopResonance,
            )
            field_dim = neuron.config.field_dim
            self.cortex.field = ResonanceField(dim=field_dim)
            self.cortex.ensemble = ResonanceEnsemble(
                self.cortex.neurons, self.cortex.field,
                max_rounds=self.cortex.max_rounds,
                confidence_gate=ConfidenceGate(threshold=0.9)
                    if self.cortex.enable_gating else None,
                early_stop=EarlyStopResonance()
                    if self.cortex.enable_gating else None,
            )
        except Exception as e:
            logger.warning(f"重建 ensemble 失败（非关键）: {e}")

        # 注册到 lifecycle.maturity（幼稚态）
        if self.lifecycle is not None:
            try:
                self.lifecycle.maturity.register_new(neuron_id)
            except Exception as e:
                logger.debug(f"maturity.register_new 失败: {e}")

        # Phase 5.2: 注册到 ThalamicRouter（如有）
        self._register_with_thalamic_router(domain, neuron_id, neuron)

        logger.info(
            f"新神经元已注册: id='{neuron_id}', domain='{domain}', "
            f"params={sum(p.numel() for p in neuron.parameters()) / 1e6:.0f}M"
        )

    def _register_with_thalamic_router(
        self, domain: str, neuron_id: str, neuron: Any,
    ) -> None:
        """Phase 5.2: 注册新 neuron 到 Cortex 的 ThalamicRouter。

        用新 neuron 的训练样本（若有）通过 teacher 计算 prototype。
        若无样本，用 neuron 自己的 field_vector 作为 fallback prototype。

        Args:
            domain: 域名
            neuron_id: 神经元 ID
            neuron: ResonanceNeuron 实例
        """
        try:
            router = getattr(self.cortex, 'thalamic_router', None)
            if router is None:
                return  # 没启用丘脑路由，跳过

            # 优先用 teacher + 训练样本计算 prototype
            teacher = getattr(self.cortex, '_teacher_model', None)
            extract_fn = getattr(self.cortex, '_extract_hidden_fn', None)

            prototype = None
            if teacher is not None and extract_fn is not None:
                # 尝试从 feed_engine 获取本域样本
                samples_tensor = None
                if self.feed_engine is not None:
                    try:
                        domain_samples = self.feed_engine.get_pending_samples_by_domain()
                        texts = domain_samples.get(domain, [])[:50]  # 最多 50 条
                        if texts and hasattr(self.cortex, '_tokenizer') and self.cortex._tokenizer:
                            import torch
                            ids_list = [self.cortex._tokenizer.encode(t)[:256] for t in texts]
                            # pad 到等长
                            max_len = min(256, max(len(ids) for ids in ids_list))
                            padded = [ids[:max_len] + [0] * (max_len - len(ids[:max_len])) for ids in ids_list]
                            samples_tensor = torch.tensor(padded, dtype=torch.long)
                    except Exception as e:
                        logger.debug(f"获取样本失败: {e}")

                if samples_tensor is not None and len(samples_tensor) > 0:
                    import torch
                    with torch.no_grad():
                        # 分批计算避免 OOM
                        hidden_sum = None
                        count = 0
                        bs = 4
                        for i in range(0, len(samples_tensor), bs):
                            batch = samples_tensor[i:i+bs]
                            h = extract_fn(teacher, batch)  # [B, L, H]
                            pooled = h.mean(dim=1)  # [B, H]
                            if hidden_sum is None:
                                hidden_sum = pooled.sum(dim=0)
                            else:
                                hidden_sum = hidden_sum + pooled.sum(dim=0)
                            count += pooled.shape[0]
                        if count > 0:
                            prototype = hidden_sum / count

            # Fallback: 用 neuron 的 field_vector（resize 到 hidden_dim 不行，跳过）
            if prototype is None:
                logger.info(
                    f"[Phase 5.2] 无法为 {neuron_id} 计算 prototype "
                    f"(无 teacher 或无样本)，跳过 router 注册"
                )
                return

            # 学徒期：初始 routing_weight = 0.1（Phase 5.3 会逐步解锁）
            router.register_domain(
                neuron_id=neuron_id,
                prototype=prototype,
                meta={'domain': domain, 'apprentice': True},
                routing_weight=0.1,
            )

            # P1-3 fix: 同时写 prototypes_embed（避免 P6-2 脱教师模式下新 neuron 不可见）
            # 优先级：context_encoder > standalone_embedding
            context_encoder = getattr(self.cortex, '_context_encoder', None)
            shared_embedding = getattr(self.cortex, '_shared_embedding', None)
            embed_source = context_encoder if context_encoder is not None else shared_embedding

            if embed_source is not None and samples_tensor is not None and len(samples_tensor) > 0:
                import torch
                try:
                    with torch.no_grad():
                        if context_encoder is not None:
                            # P0-3: 用 encoder.encode() 算上下文感知 prototype
                            emb_sum = None
                            count = 0
                            bs = 4
                            for i in range(0, len(samples_tensor), bs):
                                batch = samples_tensor[i:i+bs]
                                h = context_encoder.encode(batch)  # [B, L, embed_dim]
                                pooled = h.mean(dim=1)  # [B, embed_dim]
                                if emb_sum is None:
                                    emb_sum = pooled.sum(dim=0)
                                else:
                                    emb_sum = emb_sum + pooled.sum(dim=0)
                                count += pooled.shape[0]
                            if count > 0:
                                proto_embed = emb_sum / count
                                proto_embed = proto_embed / (proto_embed.norm() + 1e-8)
                                router.prototypes_embed[neuron_id] = proto_embed.cpu()
                        else:
                            # P6-2: 用 standalone_embedding 算 prototype
                            emb_sum = None
                            count = 0
                            bs = 4
                            for i in range(0, len(samples_tensor), bs):
                                batch = samples_tensor[i:i+bs]
                                h = shared_embedding(batch)  # [B, L, embed_dim]
                                pooled = h.mean(dim=1)
                                if emb_sum is None:
                                    emb_sum = pooled.sum(dim=0)
                                else:
                                    emb_sum = emb_sum + pooled.sum(dim=0)
                                count += pooled.shape[0]
                            if count > 0:
                                proto_embed = emb_sum / count
                                proto_embed = proto_embed / (proto_embed.norm() + 1e-8)
                                router.prototypes_embed[neuron_id] = proto_embed.cpu()
                    logger.info(
                        f"[Phase 5.2] {neuron_id} prototypes_embed 已注册 "
                        f"(脱教师模式可见)"
                    )
                except Exception as e:
                    logger.warning(f"prototypes_embed 注册失败（非致命）: {e}")

            logger.info(
                f"[Phase 5.2] {neuron_id} 已注册到 ThalamicRouter "
                f"(apprentice weight=0.1)"
            )

        except Exception as e:
            logger.warning(f"_register_with_thalamic_router 失败: {e}", exc_info=True)
