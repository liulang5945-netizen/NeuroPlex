"""Taiji Cortex loader.

Cortex 神经元架构是态极的唯一认知主体。本模块提供 create_cortex/load_cortex
入口，装配 Cortex + TokenizerHub + shared_embedding。
"""

from __future__ import annotations

import logging
import os
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
) -> tuple[Any, Optional[Any]]:
    """创建 Cortex（运行时认知主体）+ tokenizer。

    若 neurons_dir 下无蒸馏好的神经元，进入"单神经元 fallback 模式"——
    创建一个随机初始化的 general 神经元，保证系统可运行（能力有限）。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        sp_model_path: SentencePiece 模型路径（若为 None 自动查找）

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
) -> tuple[Any, Optional[Any], dict]:
    """统一装配 Cortex，接线所有 bio-inspired 模块。

    生产环境推荐入口。在 create_cortex() 基础上额外完成：

    1. TokenizerHub 注册（P7，域专用 tokenizer）
    2. STDPTracker 注入 ensemble（P1-1，发放时序记录）
    3. NeuromodulatorState 注入 cortex + ensemble（P1-2，调质驱动 lr/不应期/写入强度）
    4. GammaOscillator / WorkingMemory 注入 cortex（P1-4，feature binding + 上下文维持）
    5. LifecycleManager / SleepConsolidator 创建并返回（供 sleep_engine 使用）

    所有可选模块加载失败时退化为默认行为（向后兼容），并记录 warning。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        sp_model_path: SentencePiece 模型路径
        wire_bio_modules: 是否接线 bio-inspired 模块（False=只创建基础 cortex）

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

    # Step 1.6: P7 shared_embedding + general tokenizer
    # generate() 走 _generate_p7 路径需要 _shared_embedding 和 _general_sp：
    #   1. general_sp.encode(prompt) → general_ids
    #   2. shared_embedding(general_ids) → shared_emb [1, L, hidden_size]
    #   3. ensemble.forward(shared_emb) → neuron_logits（domain vocab）
    # hidden_size 从已加载的 neurons 获取（fallback 模式下 COMPACT=512），
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

            # hidden_size 必须与 neurons 的 hidden_size 一致，否则
            # ensemble.forward 维度不匹配
            if cortex.neurons:
                hidden_size = next(iter(cortex.neurons.values())).config.hidden_size
            else:
                hidden_size = 512  # COMPACT 默认

            general_vocab = general_sp.GetPieceSize()  # 256000
            shared_emb = torch.nn.Embedding(general_vocab, hidden_size)
            # 随机初始化（后续可通过蒸馏/训练对齐教师 embedding）
            torch.nn.init.normal_(shared_emb.weight, mean=0.0, std=0.02)
            shared_emb.to(device)
            cortex.set_shared_embedding(shared_emb)

            modules["general_tokenizer"] = general_sp
            modules["shared_embedding"] = shared_emb
            logger.info(
                "[assemble_cortex] Shared embedding + general tokenizer wired "
                "(vocab=%d, dim=%d)",
                general_vocab, hidden_size,
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

    # Step 6: GammaOscillator（P1-4，feature binding）
    try:
        from taiji.resonance import GammaOscillator
        gamma = GammaOscillator()
        # 按已加载 neuron 的 domain 分配相位
        domain_to_nids: dict[str, list[str]] = {}
        for nid in cortex.neurons.keys():
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids.setdefault(domain, []).append(nid)
        if domain_to_nids:
            gamma.assign_phase_by_domain(domain_to_nids)
        cortex.set_gamma_oscillator(gamma)
        modules["gamma_oscillator"] = gamma
        logger.info("[assemble_cortex] GammaOscillator wired (%d phases)",
                    len(gamma.phases))
    except Exception as e:
        logger.warning("[assemble_cortex] GammaOscillator 创建失败（非致命）: %s", e)

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

    # Step 9.2: 接线玩耍引擎（PlayEngine → cortex + coactivation）
    # 修复：play_engine.set_brain_interfaces 全库零调用，
    # 导致 play→CoactivationTracker 强化链路完全断开。
    try:
        from taiji.life.play_engine import get_play_engine
        from taiji.resonance.tribal import CoactivationTracker
        coactivation = CoactivationTracker()
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
    try:
        state_path = os.path.join(neurons_dir, "cortex_state.pt")
        if os.path.exists(state_path):
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
