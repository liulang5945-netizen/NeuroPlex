"""IntegrateEngine — 新生神经元无缝衔接（C17，2026-08-08）。

人脑启发 4 阶段（海马体齿状回神经发生）：
① 静默期（maturity 0-0.3）：融合权重被 ensemble 按成熟度压低（ResonanceEnsemble
   ._confidence_routing_fusion 的 maturity 注入，见 C17 改动）——新生 neuron 初期
   不参与输出融合（人脑"沉默突触"），只训练输入侧（side_channels）+ quality_head
   + LoRA，用 FeedEngine 累积样本学习。
② 可塑+蒸馏期（0.3-0.8）：高 lr（MaturityTracker.get_lr_multiplier 幼稚 3×）+
   拓扑邻居 KL 蒸馏——新 neuron 输出对齐其 side_channels 输入源（导师），自然
   衔接共振场（人脑"被邻居回路引导"）。
③ 验证期（0.8-1.0）：ablation——有 vs 无该 neuron 的 ensemble CE 对比，贡献为正
   才 commit。
④ 固化/凋亡：贡献正 → tick 满成为正式成员（未来可当导师）；负 → apoptosis 信号。

由 SleepEngine 在 neurogenesis 创建新 neuron 后调用 integrate(new_nid)。
训练在影子权重 COW 上进行（与 _train_cortex_neurons 同模式，见 sleep_engine.py
的 _clone_module），写回 live，成熟 neuron 全程冻结（只动新 neuron 的协作层）。

喂养闭环：数据全部来自 FeedEngine 累积样本 —— 态极自主演化的"喂→睡→醒"最后一环。
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class IntegrateEngine:
    """新生神经元无缝衔接引擎（静默 → 蒸馏 → 验证 → 固化/凋亡）。"""

    # ── 阶段阈值（maturity_ratio）──
    SILENT_END = 0.3      # 静默期结束（此后逐步参与融合）
    VERIFY_START = 0.8    # 验证期开始（ablation 决策）

    # ── 训练超参（v1，保守值）──
    MAX_STEPS_PER_SESSION = 32   # 单次 sleep 会话最大训练步数
    MAX_TEXT_LEN = 256           # 单条样本最大 token 长度
    DISTILL_WEIGHT = 0.3         # 邻居蒸馏权重（C26 增量七降权：记忆生长为主，蒸馏辅助）
    DISTILL_TEMP = 2.0           # 蒸馏温度
    LORA_RANK = 16               # 新 neuron body 保护（C16）
    ABLATION_SAMPLES = 16        # ablation 评估样本数
    # C26 增量七：自组织新生——记忆条件化预训练超参
    MEMORY_PRETRAIN_LR = 3e-4     # 读路径+LoRA 温和学习率（防破坏）
    MEMORY_PRETRAIN_STEPS = 24    # 记忆条件化预训练步数（每 neuron 预算）
    MEMORY_MIN_ACCESS = 1         # 记忆条目最小访问次数（≥1 有实际被检索过）

    def __init__(self, cortex, lifecycle=None, feed_engine=None,
                 memory_bank=None, device: Optional[str] = None):
        self.cortex = cortex
        self.lifecycle = lifecycle
        self.feed_engine = feed_engine
        self.memory_bank = memory_bank
        self.device = device or (cortex.device if hasattr(cortex, "device") else "cpu")

        # 记录每个新 neuron 的整合进度（跨 sleep 会话）
        self._progress: Dict[str, int] = {}  # nid -> 已完成训练步数

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────
    def integrate(self, new_nid: str) -> Dict[str, object]:
        """整合一个新生 neuron（由 sleep_engine 在 neurogenesis 后调用）。

        Returns:
            {"nid", "status": "training"|"committed"|"apoptosis"|"skipped",
             "steps", "maturity", "ablation_gain"}
        """
        from taiji.life.sleep_engine import _clone_module

        cortex = self.cortex
        if new_nid not in cortex.neurons:
            return {"nid": new_nid, "status": "skipped", "reason": "neuron_missing"}

        domain = new_nid.split("_")[0] if "_" in new_nid else new_nid
        samples = self._collect_samples(domain)
        if not samples:
            return {"nid": new_nid, "status": "skipped", "reason": "no_feed_samples",
                    "maturity": self._maturity_ratio(new_nid)}

        # ── 影子权重 COW（训练在克隆副本上，live 稳定；与 _train_cortex_neurons 同模式）──
        live_modules = dict(cortex.neurons)
        live_emb = cortex._shared_embedding
        shadow_modules = {nid: _clone_module(m) for nid, m in live_modules.items()}
        shadow_emb = _clone_module(live_emb) if live_emb is not None else None
        cortex.neurons.update(shadow_modules)
        if shadow_emb is not None:
            cortex._shared_embedding = shadow_emb
        try:
            # 成熟 neuron 全部冻结；新 neuron 只训协作层（side_channels/quality_head/LoRA）
            self._prepare_trainable(shadow_modules, new_nid)
            # C26 增量七：自组织新生——先做记忆条件化预训练（从经验生长，非 teacher
            # 蒸馏）。用记忆库中积累的高频经验（向量+文本）在记忆注意窗（round2+
            # 场条件化）下预热新 neuron 的读路径 + LoRA，让其"从经验出生"而非
            # 完全依赖 feed 样本或邻居蒸馏。无记忆样本时静默跳过（向后兼容）。
            self._memory_pretrain(shadow_modules, shadow_emb, new_nid, domain)
            status = self._integrate_session(shadow_modules, shadow_emb, new_nid, domain, samples)
        finally:
            # 写回 live（只写回整合后的新 neuron 协作层，成熟 neuron 冻结未动）
            from taiji.life.sleep_engine import SleepEngine
            try:
                SleepEngine._copy_shadow_back(live_modules, live_emb,
                                              shadow_modules, shadow_emb)
            except Exception as e:
                logger.warning(f"[IntegrateEngine] 写回 live 失败: {e}")

        return status

    # ──────────────────────────────────────────────
    # 单次 sleep 会话的训练循环
    # ──────────────────────────────────────────────
    def _integrate_session(self, shadow_modules: Dict[str, object], shadow_emb,
                           new_nid: str, domain: str, samples: list) -> Dict[str, object]:
        ensemble = self.cortex.ensemble
        hub = self.cortex._tokenizer_hub
        general_sp = self.cortex._general_sp

        new_neuron = shadow_modules[new_nid]
        texts = self._extract_texts(samples, self.MAX_STEPS_PER_SESSION * 2)
        if not texts:
            return {"nid": new_nid, "status": "skipped", "reason": "no_valid_texts"}

        # 拓扑邻居 = 新 neuron 的 side_channel 输入源（导师）
        neighbor_ids = self._neighbor_ids(new_neuron)
        logger.info(f"[IntegrateEngine] {new_nid} 整合开始: domain={domain}, "
                    f"导师邻居={neighbor_ids}, 样本={len(texts)}")

        # 可训练参数（只新 neuron 的协作层）
        trainable = [p for p in new_neuron.parameters() if p.requires_grad]
        if not trainable:
            return {"nid": new_nid, "status": "skipped", "reason": "no_trainable_params"}
        lr_mult = self._lr_multiplier(new_nid)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3 * lr_mult, weight_decay=0.01)

        # 共享嵌入查表（所有 neuron 共用）
        nids = list(shadow_modules.keys())
        steps = 0
        total_loss = 0.0
        ce_sum = 0.0
        distill_sum = 0.0

        for text in texts:
            try:
                aligned = self._align(text, hub, general_sp, domain)
                if aligned is None:
                    continue
                ids_t, target_ids = aligned

                optimizer.zero_grad()
                neuron_embeddings = {}
                with torch.no_grad():
                    for nid in nids:
                        neuron_embeddings[nid] = shadow_emb(ids_t)

                result = ensemble.forward_train(
                    neuron_embeddings=neuron_embeddings,
                    n_rounds=2,
                    fusion_mode="soft",
                    targets=target_ids,
                    field_conditioning=True,
                    step=self._progress.get(new_nid, 0) + steps,
                    target_domain=domain,
                    return_individual_logits=True,
                )

                # CE（目标域空间）
                fused = result["fused_logits"]
                sl, st = fused[:, :-1, :].contiguous(), target_ids[:, 1:].contiguous()
                ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                                     ignore_index=-100, reduction="mean")

                # 邻居蒸馏（新 neuron 对齐导师输出）
                ind = result.get("individual_logits", {})
                distill = 0.0
                if neighbor_ids:
                    kls = []
                    new_lg = ind.get(new_nid)
                    if new_lg is not None:
                        for nb in neighbor_ids:
                            nb_lg = ind.get(nb)
                            if nb_lg is None or nb_lg.shape != new_lg.shape:
                                continue
                            kls.append(F.kl_div(
                                F.log_softmax(new_lg / self.DISTILL_TEMP, dim=-1),
                                F.softmax(nb_lg.detach() / self.DISTILL_TEMP, dim=-1),
                                reduction="batchmean",
                            ))
                    if kls:
                        distill = sum(kls) / len(kls)

                loss = ce + self.DISTILL_WEIGHT * distill
                if "contrastive_loss" in result:
                    loss = loss + 0.5 * result["contrastive_loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                optimizer.step()

                steps += 1
                total_loss += loss.item()
                ce_sum += ce.item()
                distill_sum += distill
                self._tick(new_nid)

                # 每 8 步打印一次进度
                if steps % 8 == 0:
                    logger.info(f"[IntegrateEngine] {new_nid} step {steps}: "
                                f"loss={loss.item():.3f} ce={ce.item():.3f} "
                                f"distill={distill:.3f} maturity={self._maturity_ratio(new_nid):.2f}")
            except Exception as e:
                logger.warning(f"[IntegrateEngine] {new_nid} 单条样本训练失败: {e}")
                continue

        if steps == 0:
            return {"nid": new_nid, "status": "skipped", "reason": "no_valid_steps",
                    "maturity": self._maturity_ratio(new_nid)}

        self._progress[new_nid] = self._progress.get(new_nid, 0) + steps
        maturity = self._maturity_ratio(new_nid)
        logger.info(f"[IntegrateEngine] {new_nid} 会话完成: steps={steps}, "
                    f"avg_loss={total_loss/steps:.3f}, maturity={maturity:.2f}")

        # ③ 验证期：成熟度达标后做 ablation 决策
        if maturity >= self.VERIFY_START:
            return self._verify_commit(new_nid, shadow_modules, shadow_emb,
                                       domain, samples)
        return {"nid": new_nid, "status": "training", "steps": steps,
                "maturity": maturity, "avg_loss": total_loss / steps}

    # ──────────────────────────────────────────────
    # C26 增量七：自组织新生——记忆条件化预训练
    # ──────────────────────────────────────────────
    def _memory_pretrain(self, shadow_modules: Dict[str, object], shadow_emb,
                         new_nid: str, domain: str) -> int:
        """记忆注意窗预训练（从经验生长，非 teacher 蒸馏）。

        用记忆库中积累的高频经验（记忆向量 + 文本）在 round2+ 场条件化
        （记忆注意窗，同增量六 _sleep_phase_forward_replay 机制）下预热
        新 neuron 的读路径（field_read_layers/gate）+ LoRA——新 neuron
        从自身经验出生，而非只依赖 feed 样本或邻居蒸馏。

        样本格式（用户决策：三者全加入）：
        1. 问答对："问：{label}是什么？\n答：{text}"
        2. 原文：text
        3. （feed 样本由 _integrate_session 继续处理，此处不重复）

        Returns:
            记忆预训练步数（0 = 无记忆样本或跳过）
        """
        if self.memory_bank is None:
            return 0
        try:
            entries = getattr(self.memory_bank, "entries", None)
            if not entries:
                return 0
            # 候选：访问计数 ≥1（实际被检索过）且含向量+文本的记忆
            cands = [e for e in entries
                     if e.get("vector") is not None
                     and (e.get("text") or e.get("label"))
                     and e.get("access_count", 0) >= self.MEMORY_MIN_ACCESS]
            if not cands:
                return 0
        except Exception as e:
            logger.warning(f"[IntegrateEngine] 记忆候选获取失败: {e}")
            return 0

        # 组装样本（问答对 + 原文混合，用户决策）
        samples = []  # [(vector, text)]
        for e in cands:
            label = e.get("label", "")
            text = e.get("text") or label
            if len(text.strip()) < 8:
                continue
            samples.append((e["vector"], f"问：{label}是什么？\n答：{text}"))
            samples.append((e["vector"], text))
        if not samples:
            return 0
        import random
        random.shuffle(samples)
        samples = samples[: self.MEMORY_PRETRAIN_STEPS]

        cortex = self.cortex
        hub = cortex._tokenizer_hub
        general_sp = cortex._general_sp
        shared_embedding = shadow_emb
        if hub is None or general_sp is None or shared_embedding is None:
            return 0
        domain_sp = hub.get_tokenizer(domain)
        if domain_sp is None:
            return 0
        device = self.device
        ensemble = cortex.ensemble
        back_projectors = (getattr(ensemble, "_cross_spec_back_projectors", {})
                           if ensemble is not None else {})

        neuron = shadow_modules[new_nid]
        # 读路径 + LoRA（field_read 是记忆注意窗的条件化调制层）
        read_params = list(neuron.field_read_layers.parameters())
        read_params += list(neuron.field_read_gate.parameters())
        lora_params = list(neuron.lora_adapters.parameters())
        train_params = [p for p in read_params + lora_params if p.requires_grad]
        if not train_params:
            return 0

        def _fs_for(vec):
            """投影记忆向量到 neuron.field_dim（与推理同一 back-projector）。

            无投影器且维度不匹配（记忆统一场空间 vs neuron field_dim）时返回
            None——调用方跳过该样本（新 neuron 常无专属投影器，靠匹配维度兜底）。
            """
            vec = vec.detach().to(device)
            if vec.dim() > 1:
                vec = vec.squeeze(0)
            proj = back_projectors.get(new_nid)
            if proj is not None:
                try:
                    return proj(vec.unsqueeze(0)).squeeze(0)
                except Exception:
                    return None
            # 无投影器：维度必须匹配 neuron 的 field_read 输入（effective_field_dim）
            try:
                expected = neuron.field_read_layers[0].in_features
            except Exception:
                return None
            if vec.shape[-1] == expected:
                return vec
            return None

        optimizer = torch.optim.AdamW(
            [{"params": read_params}, {"params": lora_params}],
            lr=self.MEMORY_PRETRAIN_LR,
        )
        neuron.train()
        steps = 0
        total_loss = 0.0
        for vec, text in samples:
            try:
                domain_ids = hub.encode(text, domain=domain)
                if not domain_ids or len(domain_ids) < 3:
                    continue
                domain_ids = domain_ids[: self.MAX_TEXT_LEN]
                gids = []
                for did in domain_ids:
                    try:
                        piece = domain_sp.id_to_piece(did)
                    except Exception:
                        piece = None
                    if piece:
                        gen_ids = general_sp.EncodeAsIds(piece)
                        gids.append(gen_ids[0] if gen_ids else 0)
                    else:
                        gids.append(0)
                if len(gids) < 3:
                    continue
                ids_t = torch.tensor([gids], dtype=torch.long, device=device)
                target_ids = torch.tensor([domain_ids], dtype=torch.long, device=device)
                emb = shared_embedding(ids_t)
                fs = _fs_for(vec)
                if fs is None:
                    continue
                optimizer.zero_grad()
                # 记忆注意窗：round2+ 场条件化 forward（field_state=记忆向量）
                result = neuron.forward(emb, field_state=fs, round_num=2,
                                        return_logits=True)
                logits = result["logits"]
                min_len = logits.size(1) - 1
                if min_len < 1:
                    continue
                sl = logits[:, :min_len, :].contiguous()
                st = target_ids[:, 1:1 + min_len].contiguous()
                st = st.clamp(0, logits.size(-1) - 1)
                loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1),
                                       ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=5.0)
                optimizer.step()
                steps += 1
                total_loss += loss.item()
            except Exception as e:
                logger.debug(f"[IntegrateEngine] 记忆预训练单条失败: {e}")
                continue
        neuron.eval()
        if steps > 0:
            logger.info(
                f"[IntegrateEngine] {new_nid} 记忆注意窗预训练完成: "
                f"{steps} 步, avg_loss={total_loss / steps:.3f}"
            )
        return steps

    # ──────────────────────────────────────────────
    # ablation 验证 + 固化/凋亡
    # ──────────────────────────────────────────────
    def _verify_commit(self, new_nid: str, shadow_modules: Dict[str, object],
                       shadow_emb, domain: str, samples: list) -> Dict[str, object]:
        """有 vs 无新 neuron 的 ensemble CE 对比：贡献为正 → commit；负 → 凋零信号。"""
        from taiji.life.sleep_engine import _clone_module

        ensemble = self.cortex.ensemble
        hub = self.cortex._tokenizer_hub
        general_sp = self.cortex._general_sp
        texts = self._extract_texts(samples, self.ABLATION_SAMPLES)

        def eval_ce(with_new: bool) -> float:
            cels = []
            saved = None
            if not with_new:
                # 真正的 ablation：临时从 ensemble 移除新 neuron（影子副本上安全）
                saved = ensemble.neurons.pop(new_nid, None)
            try:
                for text in texts:
                    try:
                        aligned = self._align(text, hub, general_sp, domain)
                        if aligned is None:
                            continue
                        ids_t, t_ids = aligned
                        emb = {}
                        with torch.no_grad():
                            for nid in ensemble.neurons:
                                emb[nid] = shadow_emb(ids_t)
                        with torch.no_grad():
                            r = ensemble.forward_train(
                                neuron_embeddings=emb, n_rounds=2, fusion_mode="soft",
                                targets=t_ids, field_conditioning=True,
                                target_domain=domain,
                            )
                        f = r["fused_logits"]
                        cels.append(F.cross_entropy(
                            f[:, :-1, :].reshape(-1, f.size(-1)),
                            t_ids[:, 1:].reshape(-1), ignore_index=-100,
                            reduction="mean").item())
                    except Exception as e:
                        logger.debug(f"[IntegrateEngine] ablation 单条失败: {e}")
            finally:
                if saved is not None:
                    ensemble.neurons[new_nid] = saved
            return sum(cels) / max(len(cels), 1)

        # 注意：forward_train 内部用 ensemble.neurons（影子引用），传 emb 决定参与 neuron
        loss_with = eval_ce(True)
        loss_without = eval_ce(False)
        gain = loss_without - loss_with  # >0 = 有该 neuron 时 CE 更低（有贡献）

        logger.info(f"[IntegrateEngine] {new_nid} ablation: with={loss_with:.3f} "
                    f"without={loss_without:.3f} gain={gain:+.3f}")

        if gain > 0:
            # ④ 固化：tick 到满，成为正式成员（未来可当导师）
            self._commit(new_nid)
            return {"nid": new_nid, "status": "committed", "maturity": 1.0,
                    "ablation_gain": gain, "loss_with": loss_with,
                    "loss_without": loss_without}
        # 凋零信号（无贡献：移除后 ensemble 更好）
        if self.lifecycle is not None:
            try:
                self.lifecycle.apoptosis.record_ppl(domain, loss_without)
            except Exception:
                pass
        return {"nid": new_nid, "status": "apoptosis", "maturity": self._maturity_ratio(new_nid),
                "ablation_gain": gain, "loss_with": loss_with, "loss_without": loss_without}

    # ──────────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────────
    def _prepare_trainable(self, shadow_modules: Dict[str, object], new_nid: str) -> None:
        """冻结成熟 neuron 全部参数；新 neuron 只解冻协作层（side/quality_head/LoRA）。"""
        for nid, neuron in shadow_modules.items():
            for p in neuron.parameters():
                p.requires_grad = False
            if nid != new_nid:
                continue
            # 新 neuron：body 冻结 + LoRA 保护（C16），只训协作层
            neuron.enable_lora(self.LORA_RANK, layers=None)
            for ch in neuron.excite_channels.values():
                for p in ch.parameters():
                    p.requires_grad = True
            for ch in neuron.inhibit_channels.values():
                for p in ch.parameters():
                    p.requires_grad = True
            for name, p in neuron.named_parameters():
                if "scale_" in name or "bias_" in name:
                    p.requires_grad = True
                if name.startswith("quality_head") or "lora_adapters" in name:
                    p.requires_grad = True
                # C26 增量七：读路径（记忆注意窗条件化调制）解冻——自组织新生的
                # 记忆条件化预训练依赖它（从经验生长，非 teacher 蒸馏）
                if name.startswith("field_read_layers") or name.startswith("field_read_gate"):
                    p.requires_grad = True

    def _neighbor_ids(self, neuron) -> List[str]:
        """拓扑邻居 = side_channel 输入源（excite_channels 的 key = pre neuron id）。"""
        try:
            return [pid for pid in neuron.excite_channels.keys()]
        except Exception:
            return []

    def _collect_samples(self, domain: str) -> list:
        """收集生长样本（C26 增量七：feed + 记忆混合，用户决策"三者全加入"）。

        feed_engine 的按域样本 + 记忆库中积累的经验文本（问答对 + 原文）。
        记忆文本并入样本列表：feed 为空时新生不因"无样本"被跳过——从经验生长
        （自组织新生，非 teacher 蒸馏）。
        """
        samples = []
        if self.feed_engine is not None:
            try:
                by_domain = self.feed_engine.get_pending_samples_by_domain()
                samples.extend(by_domain.get(domain, []))
            except Exception as e:
                logger.warning(f"[IntegrateEngine] 获取 feed 样本失败: {e}")
        # 记忆文本并入（问答对 + 原文）
        if self.memory_bank is not None:
            try:
                entries = getattr(self.memory_bank, "entries", None) or []
                mem_texts = []
                for e in entries:
                    text = e.get("text") or e.get("label", "")
                    if len(str(text).strip()) < 8:
                        continue
                    label = e.get("label", "")
                    mem_texts.append({"text": f"问：{label}是什么？\n答：{text}"})
                    mem_texts.append({"text": text})
                if mem_texts:
                    random.shuffle(mem_texts)
                    samples.extend(mem_texts[: self.MEMORY_PRETRAIN_STEPS])
            except Exception as e:
                logger.warning(f"[IntegrateEngine] 获取记忆样本失败: {e}")
        return samples

    def _align(self, text: str, hub, general_sp, domain: str):
        """输入/目标对齐（与 _train_single_neuron 同模式）。

        目标 = domain tokenizer ids（长度 L）；输入 = 逐 domain token 的 piece 用
        general tokenizer 重编码取首 id（长度 L）——保证自回归 CE 的 shift 对齐。
        Returns (input_ids [1,L], target_ids [1,L]) 或 None（不可用）。
        """
        try:
            domain_ids = hub.encode(text, domain=domain)
            if not domain_ids or len(domain_ids) < 3:
                return None
            domain_ids = domain_ids[:self.MAX_TEXT_LEN]
            domain_sp = hub.get_tokenizer(domain)
            general_ids = []
            for did in domain_ids:
                try:
                    piece = domain_sp.id_to_piece(did)
                except Exception:
                    piece = None
                if piece:
                    gen_ids = general_sp.EncodeAsIds(piece)
                    general_ids.append(gen_ids[0] if gen_ids else 0)
                else:
                    general_ids.append(0)
            if len(general_ids) < 3:
                return None
            ids_t = torch.tensor([general_ids], dtype=torch.long, device=self.device)
            target_ids = torch.tensor([domain_ids], dtype=torch.long, device=self.device)
            return ids_t, target_ids
        except Exception as e:
            logger.debug(f"[IntegrateEngine] 对齐失败: {e}")
            return None

    def _extract_texts(self, samples: list, limit: int) -> List[str]:
        texts = []
        for sample in samples:
            if isinstance(sample, dict):
                text = (sample.get("text", "") or sample.get("content", "") or
                        sample.get("task", "") or sample.get("answer", "") or
                        " ".join(str(v) for v in sample.values() if isinstance(v, str)))
            else:
                text = str(sample)
            if len(text.strip()) > 10:
                texts.append(text)
        if len(texts) > limit:
            random.shuffle(texts)
            texts = texts[:limit]
        return texts

    def _maturity_ratio(self, nid: str) -> float:
        if self.lifecycle is None:
            return 1.0
        try:
            return self.lifecycle.maturity.get_maturity_ratio(nid)
        except Exception:
            return 1.0

    def _lr_multiplier(self, nid: str) -> float:
        if self.lifecycle is None:
            return 1.0
        try:
            return self.lifecycle.maturity.get_lr_multiplier(nid)
        except Exception:
            return 1.0

    def _tick(self, nid: str) -> None:
        if self.lifecycle is None:
            return
        try:
            self.lifecycle.maturity.tick(nid)
        except Exception:
            pass

    def _commit(self, nid: str) -> None:
        """固化：maturity 直接 tick 到满（关键期关闭，成为正式成员）。"""
        if self.lifecycle is None:
            return
        try:
            m = self.lifecycle.maturity
            rounds = getattr(m, "maturity_rounds", 100)
            for _ in range(rounds - m.get_maturity_ratio(nid) * rounds):
                m.tick(nid)
        except Exception as e:
            logger.warning(f"[IntegrateEngine] commit {nid} 失败: {e}")
