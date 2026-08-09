"""Taiji Cortex loader.

Cortex 神经元架构是态极的唯一认知主体。本模块提供 create_cortex/load_cortex
入口，装配 Cortex + TokenizerHub + shared_embedding。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Any

import torch

from .config import ModelConfig
from .tokenizer_native_v2 import TaijiNativeTokenizerV2

# 向后兼容别名
ModelSelfTokenizer = TaijiNativeTokenizerV2

logger = logging.getLogger("Taiji")


def _find_default_sentencepiece() -> Optional[str]:
    """查找默认的 sentencepiece.model 路径。

    按优先级查找：
    1. taiji/tokenizer_native_v2/sentencepiece.model
    2. checkpoint-481000/sentencepiece.model
    3. data/sentencepiece.model
    """
    taiji_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(taiji_dir, "tokenizer_native_v2", "sentencepiece.model"),
        os.path.join(os.path.dirname(taiji_dir), "checkpoint-481000", "sentencepiece.model"),
        os.path.join(os.path.dirname(taiji_dir), "data", "sentencepiece.model"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _check_domain_tokenizers() -> bool:
    """P7: 检查域专用 tokenizer 是否可用。

    检测 taiji/domains/ 下是否有至少一个 SentencePiece 模型。
    """
    taiji_dir = os.path.dirname(os.path.abspath(__file__))
    domains_dir = os.path.join(taiji_dir, "domains")
    if not os.path.isdir(domains_dir):
        return False
    # 检查 zh（必选）是否有模型
    sp_path = os.path.join(domains_dir, "zh", "sp_zh.model")
    return os.path.exists(sp_path)


# ======================== Cortex 加载（运行时认知主体） ========================


def create_cortex(
    neurons_dir: str = "data/neurons",
    device: str = "cpu",
    max_rounds: int = 3,
    sp_model_path: str | None = None,
    neuron_ids: Optional[list] = None,
) -> tuple[Any, Optional[Any]]:
    """创建 Cortex（运行时认知主体）+ tokenizer。

    若 neurons_dir 下无蒸馏好的神经元，进入"单神经元 fallback 模式"——
    创建一个随机初始化的 general 神经元，保证系统可运行（能力有限）。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        sp_model_path: SentencePiece 模型路径（若为 None 自动查找）
        neuron_ids: 只装配指定 ID 集合（如对话综合体）；None = 扫描全部
        collab_name: 协作层权重文件名（默认 cross_spec_dialogue.pt；
                     可指定 C16 训练 ckpt 如 collab_v3_c16.ckpt.pt）

    Returns:
        (cortex, tokenizer)
    """
    from taiji.brain.cortex import Cortex

    # P7: 检测域 tokenizer 是否可用
    domain_tokenizers_available = _check_domain_tokenizers()

    # 自动查找 sp_model_path（P7 模式下非必需，但保留兼容性）
    if sp_model_path is None:
        sp_model_path = _find_default_sentencepiece()

    if sp_model_path is not None:
        tokenizer = TaijiNativeTokenizerV2(sp_model_path=sp_model_path)
    elif domain_tokenizers_available:
        # P7 模式：域 tokenizer 可用，不需要共享 tokenizer
        tokenizer = None  # generate() 走 _tokenizer_hub 路径
        logger.info("[create_cortex] P7 模式：跳过共享 tokenizer，使用域 tokenizer")
    else:
        raise FileNotFoundError(
            "未找到 sentencepiece.model。请通过 sp_model_path 参数指定，"
            "或将其放置于 taiji/tokenizer_native_v2/ 目录。"
        )

    cortex = Cortex(
        neurons_dir=neurons_dir,
        device=device,
        max_rounds=max_rounds,
        neuron_ids=neuron_ids,
    )
    if tokenizer is not None:
        cortex.set_tokenizer(tokenizer)

    # 单神经元 fallback：若无任何神经元加载，创建一个随机 general 神经元
    if not cortex.neurons:
        cortex = _ensure_single_neuron_fallback(cortex, device)

    logger.info(
        "Cortex created: %d neurons, device=%s, fallback=%s",
        len(cortex.neurons), device, not cortex.is_loaded,
    )
    return cortex, tokenizer


def load_cortex(
    neurons_dir: str = "data/neurons",
    device: str = "cpu",
    sp_model_path: str | None = None,
    max_rounds: int = 3,
) -> tuple[Any, Optional[Any]]:
    """加载 Cortex（运行时认知主体）+ tokenizer。

    运行时加载入口。等价于 create_cortex，语义上用于"从已蒸馏神经元加载"。
    若无可用神经元，进入单神经元 fallback 模式。

    Returns:
        (cortex, tokenizer)
    """
    return create_cortex(
        neurons_dir=neurons_dir,
        device=device,
        max_rounds=max_rounds,
        sp_model_path=sp_model_path,
    )


def _ensure_single_neuron_fallback(cortex, device: str):
    """单神经元 fallback：当无蒸馏神经元时，创建一个随机 general 神经元。

    人脑启发：新生婴儿只有未分化的神经结构，通过经验逐步分化。
    fallback 神经元是"幼稚态"，能力有限但保证系统可运行，
    后续通过 feed/sleep/neurogenesis 逐步进化出多神经元。

    Args:
        cortex: Cortex 实例（neurons 为空）
        device: 计算设备

    Returns:
        注入了 fallback 神经元的 cortex
    """
    from taiji.resonance import ResonanceNeuron, NeuronConfig

    logger.warning(
        "未找到蒸馏神经元，进入单神经元 fallback 模式 "
        "（random init general neuron，能力有限）"
    )

    # 创建随机初始化的 general 神经元
    # H1 修复：原来用 NeuronConfig 默认（hidden=768 STANDARD），
    # 与生产 5 域（COMPACT 512）不一致。改用全局 DEFAULT_NEURON_SPEC。
    from taiji.resonance import get_default_neuron_config
    cfg = get_default_neuron_config()
    cfg.spec = "general-fallback"
    cfg.field_dim = 4096
    cfg.neuron_type = "excitatory"
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.eval()
    cortex.neurons["general"] = neuron

    # 重建 ensemble 以包含 fallback 神经元
    from taiji.resonance import ResonanceField, ResonanceEnsemble
    cortex.field = ResonanceField(dim=cfg.field_dim)
    cortex.ensemble = ResonanceEnsemble(
        cortex.neurons, cortex.field,
        max_rounds=cortex.max_rounds,
    )
    cortex.is_loaded = True

    n_params = sum(p.numel() for p in neuron.parameters())
    logger.info(
        "Fallback 神经元已创建: [general] %s, %.0fM params (random init)",
        cfg.spec, n_params / 1e6,
    )
    return cortex


# ======================== Cortex 统一装配（生产入口） ========================


def assemble_cortex(
    neurons_dir: str = "data/neurons",
    device: str = "cpu",
    max_rounds: int = 3,
    sp_model_path: str | None = None,
    wire_bio_modules: bool = True,
    neuron_ids: Optional[list] = None,
    collab_name: str = "cross_spec_dialogue.pt",
    extra_neurons_dir: Optional[str] = None,
) -> tuple[Any, Optional[Any], dict]:
    """统一装配 Cortex，接线所有 bio-inspired 模块。

    生产环境推荐入口。在 create_cortex() 基础上额外完成：

    1. TokenizerHub 注册（P7，域专用 tokenizer）
    2. STDPTracker 注入 ensemble（P1-1，发放时序记录）
    3. NeuromodulatorState 注入 cortex + ensemble（P1-2，调质驱动 lr/不应期/写入强度）
    4. GammaOscillator / WorkingMemory 注入 cortex（P1-4，feature binding + 上下文维持）
    5. LifecycleManager / SleepConsolidator 创建并返回（供 sleep_engine 使用）
    6. 协作层权重加载（side_channels + 跨规格投影层，从 cross_spec_dialogue.pt）

    所有可选模块加载失败时退化为默认行为（向后兼容），并记录 warning。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        sp_model_path: SentencePiece 模型路径
        wire_bio_modules: 是否接线 bio-inspired 模块（False=只创建基础 cortex）
        neuron_ids: 只装配指定 ID 集合（如对话综合体）；None = 扫描全部

    Returns:
        (cortex, tokenizer, modules) —
        modules 是 dict，包含所有创建的模块实例，供 sleep_engine.set_brain_interfaces 使用
    """
    # Step 1: 基础 cortex
    cortex, tokenizer = create_cortex(
        neurons_dir=neurons_dir,
        device=device,
        max_rounds=max_rounds,
        sp_model_path=sp_model_path,
        neuron_ids=neuron_ids,
    )

    modules: dict[str, Any] = {}

    # Step 1.5: P7 域 tokenizer hub 注册（核心功能，无论 wire_bio_modules 都执行）
    # P7 架构：每 neuron 有独立 embedding + 独立 lm_head + 域专用 vocab
    try:
        from taiji.resonance.translator import TokenizerHub
        hub = TokenizerHub.load_default_domains()
        cortex.set_tokenizer_hub(hub)
        modules["tokenizer_hub"] = hub
        logger.info(
            "[assemble_cortex] TokenizerHub registered (P7 mode): %d domains",
            len(hub.list_domains()),
        )
    except Exception as e:
        logger.warning(
            "[assemble_cortex] TokenizerHub 注册失败（非致命，P7 模式不可用）: %s", e,
        )

    # Step 1.7: 额外 neuron 源加载（C19，2026-08-08）
    # extra_neurons_dir（如 data/foundation_v1_general）含 general 基座 neuron
    # （code/math/zh/en，unified general 空间），与 neurons_dir 的对话 neuron
    # 组成完整 9 阵容。必须在 Step 1.6（per-neuron embedding 遍历 cortex.neurons）
    # 之前加载，否则 general 4 的 embedding 无法注入。
    # general 基座 ckpt 已剥离共享 lm_head，需注入 shared_lm_head.pt（general 256K）。
    # cortex.neurons 与 ensemble.neurons 同一引用，直接写入即生效。
    if extra_neurons_dir and os.path.isdir(extra_neurons_dir):
        try:
            _load_extra_neurons(cortex, extra_neurons_dir, device)
        except Exception as e:
            logger.warning("[assemble_cortex] extra neurons 加载失败（非致命）: %s", e)

    # Step 1.6: P7 shared_embedding + general tokenizer
    # generate() 走 _generate_p7 路径需要 _shared_embedding 和 _general_sp：
    #   1. general_sp.encode(prompt) → general_ids
    #   2. shared_embedding(general_ids) → shared_emb [1, L, base_embed_dim]
    #   3. ensemble.forward(shared_emb) → neuron_logits（domain vocab）
    # 维度用 base_embed_dim（512，所有 neuron 的 embed_adapter 输入维度），
    # 不是 hidden_size——不同规格 neuron 通过 embed_adapter 投影到各自 hidden_size。
    # general vocab 从 sp_general.model 获取（256K）。
    try:
        import sentencepiece as spm
        general_sp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "domains", "general", "sp_general.model",
        )
        if os.path.exists(general_sp_path):
            general_sp = spm.SentencePieceProcessor()
            general_sp.Load(general_sp_path)
            cortex.set_general_tokenizer(general_sp)

            # base_embed_dim 必须与 neurons 的 embed_adapter 输入一致（512）
            if cortex.neurons:
                base_embed_dim = next(iter(cortex.neurons.values())).config.base_embed_dim
            else:
                base_embed_dim = 512  # NeuronConfig 默认

            general_vocab = general_sp.GetPieceSize()  # 256000
            shared_emb = torch.nn.Embedding(general_vocab, base_embed_dim)
            # 优先加载训练好的 shared_embedding（data/shared_embedding.pt，
            # 由 train_compact_parallel --shared_emb_mode train 保存）；
            # 否则随机初始化（向后兼容，蒸馏路径）
            shared_emb_path = os.path.join(neurons_dir, "..", "shared_embedding.pt")
            if not os.path.exists(shared_emb_path):
                shared_emb_path = os.path.join("data", "shared_embedding.pt")
            loaded_shared = False
            if os.path.exists(shared_emb_path):
                try:
                    state = torch.load(shared_emb_path, map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "weight" in state:
                        weight = state["weight"]
                    else:
                        weight = state
                    if weight.shape == (general_vocab, base_embed_dim):
                        shared_emb.weight.data.copy_(weight)
                        loaded_shared = True
                        logger.info(
                            "[assemble_cortex] Shared embedding 加载训练权重: %s (%d×%d)",
                            shared_emb_path, general_vocab, base_embed_dim,
                        )
                    else:
                        logger.warning(
                            "[assemble_cortex] shared_embedding 维度不匹配 "
                            "(%s vs %s)，使用随机初始化",
                            tuple(weight.shape), (general_vocab, base_embed_dim),
                        )
                except Exception as e:
                    logger.warning(
                        "[assemble_cortex] shared_embedding 加载失败（回退随机初始化）: %s", e,
                    )
            if not loaded_shared:
                torch.nn.init.normal_(shared_emb.weight, mean=0.0, std=0.02)
                logger.info(
                    "[assemble_cortex] Shared embedding 随机初始化 (vocab=%d, dim=%d)",
                    general_vocab, base_embed_dim,
                )
            shared_emb.to(device)
            cortex.set_shared_embedding(shared_emb)

            # P7-修复（2026-08-04）：per-neuron shared embedding 与训练一致
            # 训练（finetune_cross_spec）时每个神经元从各自 ckpt 的 shared_embedding_state
            # 独立加载 embedding，推理若用单个 data/shared_embedding.pt 则与训练不一致
            # → P7 generate 输出垃圾。这里从各 neuron ckpt 加载并注入 cortex。
            # C19（2026-08-08）：extra_neurons_dir（如 foundation_v1_general）的
            # general neuron 同样按 ckpt shared_embedding_state 加载。
            neuron_shared_embeddings: dict = {}
            extra_dir = extra_neurons_dir or ""
            for nid in cortex.neurons:
                n_ckpt_path = None
                for d in (neurons_dir, extra_dir):
                    p = os.path.join(d, f"neuron_{nid}.pt") if d else ""
                    if p and os.path.exists(p):
                        n_ckpt_path = p
                        break
                if n_ckpt_path is None:
                    continue
                try:
                    n_ckpt = torch.load(
                        n_ckpt_path, map_location="cpu", weights_only=False,
                    )
                    n_emb_state = n_ckpt.get("shared_embedding_state")
                    if n_emb_state is None:
                        # general 基座 neuron（C16 阵容）：ckpt 无 per-neuron
                        # embedding，fallback 到目录级 shared_embedding.pt
                        # （训练时 general 4 共用同一共享 embedding）。
                        for d in (neurons_dir, extra_dir):
                            se_path = os.path.join(d, "shared_embedding.pt") if d else ""
                            if se_path and os.path.exists(se_path):
                                se = torch.load(se_path, map_location="cpu", weights_only=False)
                                w = se["weight"] if isinstance(se, dict) and "weight" in se else se
                                n_emb = torch.nn.Embedding(w.shape[0], w.shape[1])
                                n_emb.weight.data.copy_(w)
                                n_emb.to(device)
                                neuron_shared_embeddings[nid] = n_emb
                                break
                        continue
                    n_emb = torch.nn.Embedding(general_vocab, base_embed_dim)
                    n_emb.load_state_dict(n_emb_state)
                    n_emb.to(device)
                    neuron_shared_embeddings[nid] = n_emb
                except Exception as e:
                    logger.warning(
                        "[assemble_cortex] neuron %s shared_embedding 加载失败: %s", nid, e,
                    )
            if neuron_shared_embeddings:
                cortex.set_neuron_shared_embeddings(neuron_shared_embeddings)
                logger.info(
                    "[assemble_cortex] per-neuron shared embedding 加载: %d 个神经元",
                    len(neuron_shared_embeddings),
                )

            modules["general_tokenizer"] = general_sp
            modules["shared_embedding"] = shared_emb
            logger.info(
                "[assemble_cortex] Shared embedding + general tokenizer wired "
                "(vocab=%d, dim=%d)",
                general_vocab, base_embed_dim,
            )
        else:
            logger.warning(
                "[assemble_cortex] sp_general.model 未找到 (%s)，"
                "P7 generate() 将在 'General tokenizer not set' 处失败",
                general_sp_path,
            )
    except Exception as e:
        logger.warning(
            "[assemble_cortex] shared_embedding/general tokenizer 加载失败（非致命）: %s", e,
        )

    # Step 1.7: 协作层权重加载（side_channels + 跨规格投影层 + head/lora）— 推理核心，非可选 bio 模块
    # 训练产物 cross_spec_dialogue.pt（finetune_cross_spec --data dialogue 保存）
    # 或 C16 训练 ckpt（collab_v3_c16.ckpt.pt，含 head_state/lora_state）。
    # 加载后 ensemble 的协作能力在运行时生效（断裂 B 修复）。
    try:
        _load_collab_weights_into_cortex(cortex, neurons_dir, device, collab_name=collab_name)
    except Exception as e:
        logger.warning("[assemble_cortex] 协作层权重加载失败（非致命）: %s", e)

    if not wire_bio_modules:
        return cortex, tokenizer, modules

    # Step 2: STDPTracker（P1-1，注入 ensemble）
    try:
        from taiji.resonance import STDPTracker
        stdp_tracker = STDPTracker()
        cortex.ensemble.stdp_tracker = stdp_tracker
        modules["stdp_tracker"] = stdp_tracker
        logger.info("[assemble_cortex] STDPTracker wired into ensemble")
    except Exception as e:
        logger.warning("[assemble_cortex] STDPTracker 创建失败（非致命）: %s", e)

    # Step 5: NeuromodulatorState（P1-2，注入 cortex + ensemble）
    try:
        from taiji.resonance import NeuromodulatorState
        neuromodulator = NeuromodulatorState()
        cortex.set_neuromodulator(neuromodulator)
        modules["neuromodulator"] = neuromodulator
        logger.info("[assemble_cortex] NeuromodulatorState wired (dopamine=%.2f)",
                    neuromodulator.dopamine)
    except Exception as e:
        logger.warning("[assemble_cortex] NeuromodulatorState 创建失败（非致命）: %s", e)

    # Step 6: GammaOscillator / PhasorDynamics（P1-4，feature binding）
    # C23-C5（2026-08-08）：默认装配 PhasorDynamics（可微相位动力学）——推理与
    # 训练统一（训练用 --enable-phasor 学 ω/K/相位自组织，推理注入 phasor_state）。
    # 兼容路径：① 协作层含 phasor_state → 按训练 id_order 注册 + 注入权重；
    # ② 无 phasor_state → assign_phase_by_domain 先验（同域同相，标量行为等价）；
    # ③ 装配失败 → 回退标量 GammaOscillator（非致命，向后兼容）。
    try:
        from taiji.resonance.phasor import PhasorDynamics
        gamma = PhasorDynamics()
        # 按已加载 neuron 的 domain 分配相位（同域同相先验；phased state 注入前）
        domain_to_nids: dict[str, list[str]] = {}
        for nid in cortex.neurons.keys():
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids.setdefault(domain, []).append(nid)
        ps = getattr(cortex, "_phasor_state", None)
        if ps is not None:
            train_order = getattr(cortex, "_phasor_id_order", None) or []
            if train_order and all(n in cortex.neurons for n in train_order):
                # 按当前装配顺序重排相位行（训练 phasors 行序 = train_order）：
                # phasor_state 张量行序 = 训练 _id_to_idx 序，与推理 cortex.neurons
                # 顺序无必然对应（推理 dialogue 在前、general 在后），必须重排，
                # 否则相位/ω 错位到错误 neuron。
                gamma.register_neurons(
                    list(cortex.neurons.keys()), phases=[0.0] * len(cortex.neurons)
                )
                idx_map = {n: i for i, n in enumerate(train_order)}
                ps_reordered = dict(ps)
                for key in ("phasors", "omega"):
                    if key in ps_reordered:
                        t = ps_reordered[key]  # [N,2] / [N]
                        rows = [t[idx_map[n]] for n in cortex.neurons]
                        ps_reordered[key] = torch.stack(rows)
                gamma.load_state_dict(
                    {k: v for k, v in ps_reordered.items() if k != "id_order"},
                    strict=False,
                )
                logger.info(
                    "[assemble_cortex] PhasorDynamics 注入训练 phasor_state "
                    "(%d neurons, 顺序重排 %d→当前)",
                    len(train_order), len(train_order),
                )
            else:
                logger.warning(
                    "[assemble_cortex] phasor_state 训练顺序与当前 neuron 集合不匹配"
                    "（训练 %d vs 当前 %d），走同域同相先验",
                    len(train_order), len(cortex.neurons),
                )
                if domain_to_nids:
                    gamma.assign_phase_by_domain(domain_to_nids)
        else:
            if domain_to_nids:
                gamma.assign_phase_by_domain(domain_to_nids)
        cortex.set_gamma_oscillator(gamma)
        modules["gamma_oscillator"] = gamma
        # 推理态：冻结可学相位参数（推理 forward 只走 dict binding/gate 标量路径，
        # 无需梯度；Kuramoto 状态推进仍生效）
        gamma.eval()
        for _p in gamma.parameters():
            _p.requires_grad_(False)
        logger.info("[assemble_cortex] PhasorDynamics wired (%d phases)",
                    len(gamma.phases))
    except Exception as e:
        logger.warning("[assemble_cortex] PhasorDynamics 装配失败，回退标量 GammaOscillator: %s", e)
        try:
            from taiji.resonance import GammaOscillator
            gamma = GammaOscillator()
            domain_to_nids = {}
            for nid in cortex.neurons.keys():
                domain = nid.split("_")[0] if "_" in nid else nid
                domain_to_nids.setdefault(domain, []).append(nid)
            if domain_to_nids:
                gamma.assign_phase_by_domain(domain_to_nids)
            cortex.set_gamma_oscillator(gamma)
            modules["gamma_oscillator"] = gamma
            logger.info("[assemble_cortex] GammaOscillator wired (%d phases)",
                        len(gamma.phases))
        except Exception as e2:
            logger.warning("[assemble_cortex] GammaOscillator 创建失败（非致命）: %s", e2)

    # Step 7: WorkingMemory（P1-4，上下文维持）
    try:
        from taiji.brain.working_memory import WorkingMemory
        wm = WorkingMemory(max_tokens=512)
        cortex.set_working_memory(wm)
        modules["working_memory"] = wm
        logger.info("[assemble_cortex] WorkingMemory wired (max_tokens=512)")
    except Exception as e:
        logger.warning("[assemble_cortex] WorkingMemory 创建失败（非致命）: %s", e)

    # Step 8: LifecycleManager + SleepConsolidator（返回给 sleep_engine 使用）
    try:
        from taiji.resonance import LifecycleManager
        lifecycle = LifecycleManager()
        modules["lifecycle"] = lifecycle
        logger.info("[assemble_cortex] LifecycleManager created")
    except Exception as e:
        logger.warning("[assemble_cortex] LifecycleManager 创建失败（非致命）: %s", e)

    try:
        from taiji.resonance import SleepConsolidator
        sleep_consolidator = SleepConsolidator()
        modules["sleep_consolidator"] = sleep_consolidator
        logger.info("[assemble_cortex] SleepConsolidator created")
    except Exception as e:
        logger.warning("[assemble_cortex] SleepConsolidator 创建失败（非致命）: %s", e)

    # Step 9: 接线生命引擎（SleepEngine 拿到 cortex + modules 引用）
    # 之前 set_brain_interfaces 在全代码库零调用，导致 sleep 训练是死代码。
    # 这里闭环：assemble_cortex 装配完 → 立即注入到全局 SleepEngine。
    try:
        from taiji.life.sleep_engine import get_sleep_engine
        sleep_engine = get_sleep_engine()
        sleep_engine.set_brain_interfaces(
            cortex=cortex,
            lifecycle=modules.get("lifecycle"),
            sleep_consolidator=modules.get("sleep_consolidator"),
            stdp_tracker=modules.get("stdp_tracker"),
            neuromodulator=modules.get("neuromodulator"),
        )
        logger.info("[assemble_cortex] SleepEngine wired to Cortex (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] SleepEngine 接线失败（非致命）: %s", e)

    # Step 9.1: 接线生命调度器（LifeScheduler 拿到 cortex + modules 引用）
    # 修复接线 bug：life_scheduler.set_brain_interfaces 未被调用，
    # 导致 _update_neuron_signals 中 hunger→neurogenesis 分支永远不执行。
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        from taiji.life.feed_engine import get_feed_engine
        life_scheduler = get_life_scheduler()
        life_scheduler.set_brain_interfaces(
            cortex=cortex,
            lifecycle=modules.get("lifecycle"),
            neuromodulator=modules.get("neuromodulator"),
            feed_engine=get_feed_engine(),
        )
        logger.info("[assemble_cortex] LifeScheduler wired to Cortex (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] LifeScheduler 接线失败（非致命）: %s", e)

    # Step 9.2: 接线玩耍引擎（PlayEngine → cortex + coaction）
    # 断裂 G 修复：
    #   1. 使用 cortex.coaction 实例（而非新建），确保 play→coaction 数据共享
    #   2. play_engine 调用 coaction.update(ids)（而非不存在的 record_coactivation）
    try:
        from taiji.life.play_engine import get_play_engine
        coactivation = cortex.coaction  # 使用 cortex 已有的实例
        play_engine = get_play_engine()
        play_engine.set_brain_interfaces(
            cortex=cortex,
            coactivation=coactivation,
            sleep_consolidator=modules.get("sleep_consolidator"),
        )
        modules["coactivation"] = coactivation
        modules["play_engine"] = play_engine
        logger.info("[assemble_cortex] PlayEngine wired to Cortex (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] PlayEngine 接线失败（非致命）: %s", e)

    # Step 9.3: 接线进化引擎（EvolutionEngine → cortex + lifecycle）
    # 修复：evolution_engine.set_brain_interfaces 全库零调用，
    # 导致阶段升级→neurogenesis 信号链路断开。
    try:
        from taiji.life.evolution_engine import get_evolution_engine
        from taiji.life.feed_engine import get_feed_engine
        evolution_engine = get_evolution_engine()
        evolution_engine.set_brain_interfaces(
            cortex=cortex,
            lifecycle=modules.get("lifecycle"),
            feed_engine=get_feed_engine(),
        )
        modules["evolution_engine"] = evolution_engine
        logger.info("[assemble_cortex] EvolutionEngine wired to Cortex (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] EvolutionEngine 接线失败（非致命）: %s", e)

    # Step 9.4: 接线 limbs（代码执行结果 → feed_engine 训练样本）
    # 修复：limbs.set_feed_engine 全库零调用，
    # 导致工具执行结果喂不进训练数据。
    try:
        from taiji.body.limbs import set_feed_engine as limbs_set_feed_engine
        from taiji.life.feed_engine import get_feed_engine
        limbs_set_feed_engine(get_feed_engine())
        logger.info("[assemble_cortex] limbs.set_feed_engine wired (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] limbs 接线失败（非致命）: %s", e)

    # Step 9.5: 实例化 Agent 认知系统 + 接线
    # 修复：PerceptionSystem/PlannerSystem/ReflectorSystem/MemorySystem
    # 从未被实例化，Agent 五元闭环全死代码。
    try:
        from taiji.agent.perception import PerceptionSystem
        from taiji.agent.planner import PlannerSystem
        from taiji.agent.reflector import ReflectorSystem
        from taiji.agent.memory import MemorySystem
        from taiji.life.feed_engine import get_feed_engine

        perception = PerceptionSystem(tokenizer=tokenizer)
        planner = PlannerSystem()
        reflector = ReflectorSystem()
        memory_system = MemorySystem()

        # PlannerSystem 接线（规划结果反馈学习）
        planner.set_brain_interfaces(
            feed_engine=get_feed_engine(),
            neuromodulator=modules.get("neuromodulator"),
            lifecycle=modules.get("lifecycle"),
        )

        modules["perception"] = perception
        modules["planner"] = planner
        modules["reflector"] = reflector
        modules["memory_system"] = memory_system
        logger.info("[assemble_cortex] Agent Systems 实例化 + 接线 (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] Agent Systems 接线失败（非致命）: %s", e)

    # Step 9.6: 接线 ContextManager（记忆系统注入）
    # 修复：ContextManager 的 set_working_memory/set_memory_system/set_semantic_memory
    # 从未被调用，三个记忆子系统全部悬空。
    # 同时设置持久化路径，让长期记忆跨会话保留。
    try:
        from taiji.agent.context_manager import get_context_manager
        from taiji.agent.working_memory import get_working_memory
        cm = get_context_manager()
        wm = get_working_memory()
        cm.set_working_memory(wm)
        if "memory_system" in modules:
            cm.set_memory_system(modules["memory_system"])
        cm.set_persistent_path("data/agent_memory.json")
        logger.info("[assemble_cortex] ContextManager wired with memory systems (闭环)")
    except Exception as e:
        logger.warning("[assemble_cortex] ContextManager 接线失败（非致命）: %s", e)

    # Step 10: P8 多模态默认启用 — 加载图像/音频/视频编解码器
    # checkpoint 不存在时跳过（非致命，保持向后兼容）
    # 注册到 TokenizerHub + 为所有 neuron 注册模态投影层
    try:
        from taiji.multimodal.vqvae import VQVAE, VQVAEImageCodec
        from taiji.multimodal.encodec import EnCodec, EnCodecAudioCodec
        from taiji.multimodal.video import VideoVQVAE, VideoCodec

        hub = modules.get("tokenizer_hub")

        # 10.1 图像 VQ-VAE
        vqvae_ckpt = "data/vqvae/vqvae_latest.pt"
        image_latent_dim = 256  # 默认值，有 checkpoint 时会被覆盖
        if os.path.exists(vqvae_ckpt):
            ckpt = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("config", {})
            image_latent_dim = cfg_dict.get("latent_dim", 256)
            vqvae_model = VQVAE(
                in_channels=3,
                hidden_dim=cfg_dict.get("hidden_dim", 128),
                latent_dim=image_latent_dim,
                num_embeddings=cfg_dict.get("num_embeddings", 8192),
                commitment_cost=cfg_dict.get("commitment_cost", 0.25),
                downsample=cfg_dict.get("downsample", 8),
            )
            vqvae_model.load_state_dict(ckpt["model_state_dict"])
            vqvae_model.to(device)
            image_codec = VQVAEImageCodec(model=vqvae_model, device=torch.device(device))
            if hub is not None:
                hub.register_modality("image", image_codec)
                logger.info("[assemble_cortex] VQ-VAE image codec registered to TokenizerHub")
            modules["vqvae_codec"] = image_codec
        else:
            logger.info("[assemble_cortex] VQ-VAE checkpoint not found (%s), skip image", vqvae_ckpt)

        # 10.2 音频 EnCodec
        encodec_ckpt = "data/encodec/encodec_latest.pt"
        audio_latent_dim = 128  # 默认值
        if os.path.exists(encodec_ckpt):
            ckpt = torch.load(encodec_ckpt, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("config", {})
            audio_latent_dim = cfg_dict.get("latent_dim", 128)
            encodec_model = EnCodec(
                hidden_dim=cfg_dict.get("hidden_dim", 64),
                latent_dim=audio_latent_dim,
                num_embeddings=cfg_dict.get("num_embeddings", 4096),
                commitment_cost=cfg_dict.get("commitment_cost", 0.25),
                sample_rate=cfg_dict.get("sample_rate", 16000),
            )
            encodec_model.load_state_dict(ckpt["model_state_dict"])
            encodec_model.to(device)
            audio_codec = EnCodecAudioCodec(model=encodec_model, device=torch.device(device))
            if hub is not None:
                hub.register_modality("audio", audio_codec)
                logger.info("[assemble_cortex] EnCodec audio codec registered to TokenizerHub")
            modules["encodec_codec"] = audio_codec
        else:
            logger.info("[assemble_cortex] EnCodec checkpoint not found (%s), skip audio", encodec_ckpt)

        # 10.3 视频 VideoVQVAE
        video_ckpt = "data/video/video_latest.pt"
        video_latent_dim = 256  # 默认值
        if os.path.exists(video_ckpt):
            ckpt = torch.load(video_ckpt, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("config", {})
            video_latent_dim = cfg_dict.get("latent_dim", 256)
            video_model = VideoVQVAE(
                hidden_dim=cfg_dict.get("hidden_dim", 64),
                latent_dim=video_latent_dim,
                num_embeddings=cfg_dict.get("num_embeddings", 8192),
                commitment_cost=cfg_dict.get("commitment_cost", 0.25),
            )
            video_model.load_state_dict(ckpt["model_state_dict"])
            video_model.to(device)
            video_codec = VideoCodec(model=video_model, device=torch.device(device))
            if hub is not None:
                hub.register_modality("video", video_codec)
                logger.info("[assemble_cortex] Video codec registered to TokenizerHub")
            modules["video_codec"] = video_codec
        else:
            logger.info("[assemble_cortex] Video checkpoint not found (%s), skip video", video_ckpt)

        # 10.4-10.5 为所有 neuron 自动注册多模态投影层和输出头
        # 从 TokenizerHub 获取所有已注册模态，自动注册到每个 neuron
        # 新增模态或新增 neuron 时无需手动修改代码
        for nid, neuron in cortex.neurons.items():
            neuron.auto_register_modalities(hub)
        n_neurons = len(cortex.neurons)
        n_modalities = len(hub.list_modalities())
        logger.info(
            "[assemble_cortex] multimodal projections + heads auto-registered for %d neurons, %d modalities: %s",
            n_neurons, n_modalities, ", ".join(hub.list_modalities()),
        )

    except Exception as e:
        logger.warning("[assemble_cortex] 多模态编解码器加载失败（非致命）: %s", e)

    # Step 11: 加载已保存的可学习状态（经验积累恢复）
    # 启动时自动从 neurons_dir/cortex_state.pt 恢复 shared_embedding + lm_head 权重，
    # 使 Cortex 从上次训练结束的状态继续，而非每次从随机初始化开始。
    # #19: 时间戳检查防止旧的 cortex_state.pt 覆盖新的 neuron_*.pt
    try:
        state_path = os.path.join(neurons_dir, "cortex_state.pt")
        if os.path.exists(state_path):
            state_mtime = os.path.getmtime(state_path)
            # 找最新的 neuron_*.pt 修改时间
            newest_neuron_mtime = 0
            for fname in os.listdir(neurons_dir):
                if fname.startswith("neuron_") and fname.endswith(".pt"):
                    mtime = os.path.getmtime(os.path.join(neurons_dir, fname))
                    newest_neuron_mtime = max(newest_neuron_mtime, mtime)

            if newest_neuron_mtime > state_mtime:
                logger.warning(
                    "[assemble_cortex] neuron_*.pt 比 cortex_state.pt 新 "
                    "(neuron: %s vs state: %s)，离线训练可能被在线状态覆盖",
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(newest_neuron_mtime)),
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(state_mtime)),
                )

            cortex.load_state(state_path)
            logger.info("[assemble_cortex] 已恢复经验积累状态: %s", state_path)
        else:
            logger.info("[assemble_cortex] 无已保存状态，从随机初始化开始（幼稚态）")
    except Exception as e:
        logger.warning("[assemble_cortex] 加载状态失败（非致命，从随机初始化开始）: %s", e)

    logger.info(
        "[assemble_cortex] Done. Wired modules: %s",
        ", ".join(modules.keys()) if modules else "(none)",
    )
    return cortex, tokenizer, modules


# ======================== 协作层权重加载（训练产物 → 运行时） ========================


def _load_extra_neurons(cortex, extra_dir: str, device: str) -> list:
    """C19：从额外目录加载 neuron 加入 cortex（如 foundation_v1_general 的 4 general）。

    general 基座 ckpt 已剥离共享 lm_head（_strip_shared_head），必须注入
    shared_lm_head.pt（general 256K）才能在统一空间输出 logits。
    cortex.neurons 与 ensemble.neurons 同一引用，直接写入即生效。

    Returns:
        新增 nid 列表。
    """
    import glob

    from taiji.resonance import ResonanceNeuron, get_domain_neuron_config

    shared_lm_head = None
    lm_head_path = os.path.join(extra_dir, "shared_lm_head.pt")
    if os.path.exists(lm_head_path):
        sd = torch.load(lm_head_path, map_location=device, weights_only=False)
        w = sd["weight"] if isinstance(sd, dict) and "weight" in sd else sd
        shared_lm_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        shared_lm_head.weight.data.copy_(w)
        logger.info(
            "[assemble_cortex] shared_lm_head 注入 extra neurons (%d×%d)", w.shape[0], w.shape[1],
        )

    added = []
    for path in sorted(glob.glob(os.path.join(extra_dir, "neuron_*.pt"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        nid = name[len("neuron_"):-len(".pt")]
        if nid in cortex.neurons:
            continue
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            cfg = ckpt.get("neuron_config")
            if cfg is None:
                cfg = get_domain_neuron_config(nid, spec="compact")
            cfg.unified_field_dim = None
            neuron = ResonanceNeuron(cfg, shared_lm_head=shared_lm_head).to(device)
            neuron.load_state_dict(ckpt["state_dict"], strict=False)
            neuron.eval()
            # 用 ensemble.add_neuron（自动补建跨规格投影层 + 几何空间注册）；
            # cortex.neurons 与 ensemble.neurons 同一引用，一次调用两边生效。
            cortex.ensemble.add_neuron(nid, neuron)
            added.append(nid)
        except Exception as e:
            logger.warning("[assemble_cortex] extra neuron %s 加载失败: %s", nid, e)
    if added:
        logger.info("[assemble_cortex] extra neurons loaded: %s", added)
    return added


def _load_collab_weights_into_cortex(
    cortex,
    neurons_dir: str,
    device: str,
    collab_name: str = "cross_spec_dialogue.pt",
) -> bool:
    """把训练好的协作层权重加载进 cortex.ensemble。

    C18（2026-08-08）扩展：支持 train_cross_domain_collab 产物的完整分量——
    key 兼容 final artifact（"side_channels"/"cross_spec"）与训练 ckpt
    （"side_channels_state"/"cross_spec_state"），并新增 C16 分量：
    head_state（quality_head 判别器）+ lora_state（LoRA 尾层增量）。

    训练产物结构（两种格式兼容）：
    {
        "side_channels" / "side_channels_state": {nid: {"excite": {...}, "inhibit": {...}}},
        "cross_spec" / "cross_spec_state": {"forward": {nid: sd}, "backward": {nid: sd}},
        "body_state": {nid: sd},              # C16 模式为空 dict（body 冻结，保持原始）
        "scale_bias_state": {nid: sd},
        "head_state": {nid: sd},              # C16: quality_head 独立分量
        "lora_state": {nid: sd},              # C16: LoRA 尾层增量
    }

    Args:
        cortex: Cortex 实例（含 neurons + ensemble）
        neurons_dir: 神经元 ckpt 目录（跨规格投影层在 ensemble 中已按需创建）
        device: 计算设备
        collab_name: 协作层权重文件名（默认推理用 cross_spec_dialogue.pt；
                     可指定训练 ckpt 如 collab_v3_c16.ckpt.pt）

    Returns:
        是否成功加载（False = 文件不存在或加载失败，调用方按非致命处理）
    """
    collab_path = os.path.join(neurons_dir, collab_name)
    if not os.path.exists(collab_path):
        logger.info("[assemble_cortex] 协作层权重未找到: %s（跳过）", collab_path)
        return False

    ckpt = torch.load(collab_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        logger.warning("[assemble_cortex] 协作层权重格式异常（非 dict），跳过")
        return False

    # C23-C5（2026-08-08）：缓存 phasor_state（含 id_order）供 Step 6 装配
    # PhasorDynamics 时注入——推理复用训练学到的 ω/K/相位自组织。cortex 上
    # 无此键时 Step 6 走 assign_phase_by_domain 先验（同域同相）。
    ps = ckpt.get("phasor_state")
    if ps is not None:
        cortex._phasor_state = ps
        # 旧 ckpt（id_order 添加前）无 id_order：用 head_state 的 key 顺序作为
        # 训练 neuron 顺序 fallback（train 脚本保存顺序 = neurons.items() 顺序，
        # 与 phasor 注册顺序一致）。
        id_order = ps.get("id_order") or list((ckpt.get("head_state") or {}).keys())
        if id_order:
            cortex._phasor_id_order = id_order
        logger.info("[assemble_cortex] 协作层含 phasor_state（%d 分量，%d neurons），待注入",
                    len(ps), len(id_order))

    def _pick(*keys):
        """key 兼容：训练 ckpt（_state 后缀）与 final artifact（无后缀）。"""
        for k in keys:
            if k in ckpt:
                return ckpt[k]
        return None

    n_side = 0
    # 1. side_channels：先确保通道存在（ensemble 创建时不自动建 per-pair 通道），再加载权重
    side_state = _pick("side_channels_state", "side_channels")
    if side_state is not None:
        for nid, neuron in cortex.neurons.items():
            if nid not in side_state:
                continue
            for ch_type, peers in side_state[nid].items():
                if ch_type not in ("excite", "inhibit"):
                    continue
                channels = neuron.excite_channels if ch_type == "excite" else neuron.inhibit_channels
                for pid, ch_state in peers.items():
                    if pid not in cortex.neurons:
                        continue
                    if pid not in channels:
                        neuron.establish_side_channel(
                            pid, cortex.neurons[pid], channel_type=ch_type,
                        )
                    channels[pid].load_state_dict(ch_state)
                    n_side += 1
        # 完全无匹配 → 权重 ID 与当前装配集合不一致（如旧版 base 权重 vs 新版 dialogue 集合）
        if n_side == 0:
            ckpt_ids = set(side_state.keys())
            current_ids = set(cortex.neurons.keys())
            logger.warning(
                "[assemble_cortex] 协作层权重 ID 与当前装配集合不匹配: "
                "ckpt=%s, current=%s（可能是旧版权重，等待新训练产物覆盖）",
                sorted(ckpt_ids)[:5], sorted(current_ids)[:5],
            )

    # 2. 跨规格投影层（forward/backward）
    n_proj = 0
    cross_spec = _pick("cross_spec_state", "cross_spec")
    if cross_spec is not None:
        for nid, sd in (cross_spec.get("forward") or {}).items():
            if nid in cortex.ensemble._cross_spec_projectors:
                cortex.ensemble._cross_spec_projectors[nid].load_state_dict(sd)
                n_proj += 1
        for nid, sd in (cross_spec.get("backward") or {}).items():
            if nid in cortex.ensemble._cross_spec_back_projectors:
                cortex.ensemble._cross_spec_back_projectors[nid].load_state_dict(sd)
                n_proj += 1

    # 3. body_state（S8/T12: 微调后的 body 参数——lm_head + 最后 N 层 + field_write）
    # C16 模式 body_state 为空 dict（body 冻结未动）→ 不注入，保持原始 body（保护个体能力）。
    # 旧格式（C13/C14 直接微调）body_state 非空 → 按旧行为注入（兼容）。
    n_body = 0
    body_state = _pick("body_state")
    if body_state:
        for nid, sd in body_state.items():
            if nid not in cortex.neurons:
                continue
            neuron = cortex.neurons[nid]
            for name, p in neuron.named_parameters():
                if name in sd and sd[name].shape == p.shape:
                    p.data.copy_(sd[name])
                    n_body += 1
        logger.info("[assemble_cortex] body_state 已应用: %d 个参数（微调后的 lm_head/body）", n_body)

    # 4. scale_bias_state（S8: 可学习 scale 标量 + 通道 bias 缓冲）
    n_sb = 0
    sb_state = _pick("scale_bias_state")
    if sb_state:
        for nid, sb in sb_state.items():
            if nid not in cortex.neurons:
                continue
            neuron = cortex.neurons[nid]
            for name, p in neuron.named_parameters():
                if name in sb and "scale_" in name:
                    p.data.copy_(sb[name])
                    n_sb += 1
            for name, buf in neuron.named_buffers():
                if name in sb and "bias_" in name:
                    buf.copy_(sb[name])
                    n_sb += 1
        logger.info("[assemble_cortex] scale_bias_state 已应用: %d 个参数", n_sb)

    # 5. head_state（C16：quality_head 判别器独立分量——路由监督，不耦合 body）
    n_head = 0
    head_state = _pick("head_state")
    if head_state:
        for nid, sd in head_state.items():
            if nid not in cortex.neurons:
                continue
            neuron = cortex.neurons[nid]
            qh = getattr(neuron, "quality_head", None)
            if qh is not None:
                try:
                    qh.load_state_dict(sd)
                    n_head += 1
                except Exception as e:
                    logger.warning(
                        "[assemble_cortex] %s quality_head 加载失败: %s", nid, e)
        logger.info("[assemble_cortex] head_state 已应用: %d 个 quality_head", n_head)

    # 6. lora_state（C16：LoRA 尾层增量——body 冻结时的低秩适配）
    # C21（2026-08-08）：词库多词表架构——LoRA 只注入输出空间与 C16 训练目标
    # 一致的 neuron（general 256K 头）。zh 头（50K 域词表）neuron 的 C16 LoRA
    # 是在 general 目标空间 + 转译投影下训练，对域词表能力是**扭曲**（验证：
    # 注入后 dialogue zh 生成退化，清零后恢复流畅中文）。各 neuron 保域词表。
    n_lora = 0
    n_lora_skip = 0
    lora_state = _pick("lora_state")
    if lora_state:
        for nid, sd in lora_state.items():
            if nid not in cortex.neurons:
                continue
            neuron = cortex.neurons[nid]
            _out_vocab = getattr(neuron.lm_head, "out_features", None) if neuron.lm_head is not None else None
            if _out_vocab != 256000:
                n_lora_skip += 1
                logger.info(
                    "[assemble_cortex] 跳过 %s 的 lora_state（lm_head=%s ≠ general 256K，"
                    "域词表 neuron 保域能力）", nid, _out_vocab,
                )
                continue
            if len(neuron.lora_adapters) == 0:
                # 先启用 LoRA（rank 从已保存 a.weight 推断，层默认最后 2 层）
                rank = 0
                for k, v in sd.items():
                    if k.endswith(".a.weight"):
                        rank = max(rank, v.shape[0])
                neuron.enable_lora(rank if rank > 0 else 16, layers=None)
            try:
                neuron.lora_adapters.load_state_dict(sd)
                n_lora += 1
            except Exception as e:
                logger.warning("[assemble_cortex] %s lora_state 加载失败: %s", nid, e)
        logger.info("[assemble_cortex] lora_state 已应用: %d 个 neuron（跳过域词表 %d）", n_lora, n_lora_skip)

    logger.info(
        "[assemble_cortex] 协作层权重已加载: %s (side_channels=%d, 跨规格投影=%d, "
        "body=%d, scale_bias=%d, head=%d, lora=%d)",
        collab_path, n_side, n_proj, n_body, n_sb, n_head, n_lora,
    )
    return True
