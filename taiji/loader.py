"""Taiji model loader and saver for the native-v2 tokenizer contract.

P2-6: ModelSelf 已作为运行时认知主体被 Cortex 取代。本模块保留 create_model/load_model
仅用于：(1) 初始神经元蒸馏的教师模型; (2) 旧 checkpoint 向后兼容。
运行时推理请使用 create_cortex/load_cortex。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from typing import Callable, Optional, Any

import torch

from .config import ModelConfig, NATIVE_V2_TOKENIZER_CONTRACT
from .tokenizer_native_v2 import TaijiNativeTokenizerV2

# P2-6 向后兼容别名：ModelSelfTokenizer 已重命名为 TaijiNativeTokenizerV2
ModelSelfTokenizer = TaijiNativeTokenizerV2

logger = logging.getLogger("Taiji")

# ModelSelf 类已从项目中移除（P2-6 完全移除）。
# 旧 create_model/load_model 保留为 stub，调用时抛出明确错误。
ModelSelf = None  # type: ignore[assignment]

_SIZE_BUILDERS: dict[str, Callable[[], ModelConfig]] = {
    "125m": ModelConfig.size_125m,
    "350m": ModelConfig.size_350m,
    "1b": ModelConfig.size_1b,
    "3b": ModelConfig.size_3b,
    "7b": ModelConfig.size_7b,
}


def create_model(
    size: str = "125m",
    device: str = "cpu",
    active_heads: list[str] | None = None,
    sp_model_path: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    """Create a fresh Taiji model and tokenizer pair.

    P2-6: 已废弃。ModelSelf 类已从项目中完全移除。
    请改用 create_cortex() 创建运行时认知主体。

    Raises:
        NotImplementedError: 始终抛出。ModelSelf 已移除。
    """
    raise NotImplementedError(
        "create_model() is no longer supported (P2-6). "
        "ModelSelf class has been removed. Use create_cortex() instead."
    )


def load_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load a saved Taiji model directory.

    P2-6: 已废弃。ModelSelf 类已从项目中完全移除。
    请改用 load_cortex() 加载运行时认知主体。

    Raises:
        NotImplementedError: 始终抛出。ModelSelf 已移除。
    """
    raise NotImplementedError(
        "load_model() is no longer supported (P2-6). "
        "ModelSelf class has been removed. Use load_cortex() instead."
    )


def _find_sentencepiece(model_path: str) -> Optional[str]:
    candidates = [
        os.path.join(model_path, "sentencepiece.model"),
        os.path.join(model_path, "tokenizer_native_v2", "sentencepiece.model"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_native_v2", "sentencepiece.model"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


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


def save_model(
    model,
    tokenizer,
    save_path: str,
    training_state: Optional[dict] = None,
) -> None:
    """Save a Taiji model, tokenizer, and optional training state.

    P2-6: ModelSelf 已移除。此函数保留用于保存兼容对象（duck-typing），
    如 gen1 教师模型或迁移过程中的临时模型。
    """
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(model.config), handle, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    if hasattr(tokenizer, "save"):
        tokenizer.save(save_path)

    sp_model_path = getattr(tokenizer, "sp_model_path", None)
    if sp_model_path and os.path.exists(sp_model_path):
        shutil.copyfile(sp_model_path, os.path.join(save_path, "sentencepiece.model"))

    if training_state is not None:
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    logger.info("Model saved to %s", save_path)


def _config_to_dict(config: ModelConfig) -> dict:
    """Serialise ModelConfig using dataclass introspection — never drifts."""
    return config.to_dict()


def _dict_to_config(data: dict) -> ModelConfig:
    """Deserialise ModelConfig from dict."""
    return ModelConfig.from_dict(data)

def _remap_legacy_keys(state_dict: dict) -> dict:
    """Remap old-format checkpoint keys to the current architecture naming.

    Old format (flat):   embed.weight, layers.N.attn.wq.weight, layers.N.wg.weight, norm.weight
    New format (nested): backbone.embedding.weight, backbone.layers.N.attention.wq.weight,
                         backbone.layers.N.feed_forward.w_gate.weight, backbone.norm.weight
    """
    if any(k.startswith("backbone.") for k in state_dict):
        return state_dict  # Already new format
    if "embed.weight" not in state_dict and not any("layers." in k for k in state_dict):
        return state_dict  # Unknown format, pass through

    remapped = {}
    for key, val in state_dict.items():
        if key == "embed.weight":
            remapped["backbone.embedding.weight"] = val
        elif key == "norm.weight":
            remapped["backbone.norm.weight"] = val
        elif key.startswith("layers."):
            parts = key.split(".", 2)
            layer_idx, rest = parts[1], parts[2]
            prefix = f"backbone.layers.{layer_idx}"
            if rest.startswith("attn_norm."):
                remapped[f"{prefix}.attention_norm.{rest[10:]}"] = val
            elif rest.startswith("ffn_norm."):
                remapped[f"{prefix}.ffn_norm.{rest[9:]}"] = val
            elif rest.startswith("attn."):
                remapped[f"{prefix}.attention.{rest[5:]}"] = val
            elif rest.startswith("wg."):
                remapped[f"{prefix}.feed_forward.w_gate.{rest[3:]}"] = val
            elif rest.startswith("w1."):
                remapped[f"{prefix}.feed_forward.w1.{rest[3:]}"] = val
            elif rest.startswith("w2."):
                remapped[f"{prefix}.feed_forward.w2.{rest[3:]}"] = val
        else:
            remapped[key] = val  # lm_head.weight etc.
    return remapped


# ======================== Cortex 加载（P2-6 运行时认知主体） ========================


def create_cortex(
    neurons_dir: str = "data/neurons",
    device: str = "cpu",
    max_rounds: int = 3,
    confidence_threshold: float = 0.9,
    enable_gating: bool = True,
    sp_model_path: str | None = None,
) -> tuple[Any, ModelSelfTokenizer]:
    """创建 Cortex（运行时认知主体）+ tokenizer。

    P2-6: Cortex 取代 ModelSelf 作为态极的认知主体。
    若 neurons_dir 下无蒸馏好的神经元，进入"单神经元 fallback 模式"——
    创建一个随机初始化的 general 神经元，保证系统可运行（能力有限）。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        confidence_threshold: 置信度门限
        enable_gating: 是否启用门控
        sp_model_path: SentencePiece 模型路径（若为 None 自动查找）

    Returns:
        (cortex, tokenizer)
    """
    from taiji.brain.cortex import Cortex

    # 自动查找 sp_model_path
    if sp_model_path is None:
        sp_model_path = _find_default_sentencepiece()
    if sp_model_path is None:
        raise FileNotFoundError(
            "未找到 sentencepiece.model。请通过 sp_model_path 参数指定，"
            "或将其放置于 taiji/tokenizer_native_v2/ 目录。"
        )

    tokenizer = TaijiNativeTokenizerV2(sp_model_path=sp_model_path)

    cortex = Cortex(
        neurons_dir=neurons_dir,
        device=device,
        max_rounds=max_rounds,
        confidence_threshold=confidence_threshold,
        enable_gating=enable_gating,
    )
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
    confidence_threshold: float = 0.9,
    enable_gating: bool = True,
) -> tuple[Any, ModelSelfTokenizer]:
    """加载 Cortex（运行时认知主体）+ tokenizer。

    P2-6: 运行时加载入口。等价于 create_cortex，语义上用于"从已蒸馏神经元加载"。
    若无可用神经元，进入单神经元 fallback 模式。

    Returns:
        (cortex, tokenizer)
    """
    return create_cortex(
        neurons_dir=neurons_dir,
        device=device,
        max_rounds=max_rounds,
        confidence_threshold=confidence_threshold,
        enable_gating=enable_gating,
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
    cfg = NeuronConfig(
        spec="general-fallback",
        field_dim=4096,
        neuron_type="excitatory",
    )
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.eval()
    cortex.neurons["general"] = neuron

    # 重建 ensemble 以包含 fallback 神经元
    from taiji.resonance import ResonanceField, ResonanceEnsemble, ConfidenceGate, EarlyStopResonance
    cortex.field = ResonanceField(dim=cfg.field_dim)
    cortex.ensemble = ResonanceEnsemble(
        cortex.neurons, cortex.field,
        max_rounds=cortex.max_rounds,
        confidence_gate=ConfidenceGate(threshold=0.9) if cortex.enable_gating else None,
        early_stop=EarlyStopResonance() if cortex.enable_gating else None,
    )
    cortex.is_loaded = True

    n_params = sum(p.numel() for p in neuron.parameters())
    logger.info(
        "Fallback 神经元已创建: [general] %s, %.0fM params (random init)",
        cfg.spec, n_params / 1e6,
    )
    return cortex


# ======================== Cortex 统一装配（P0-1 生产入口） ========================


def assemble_cortex(
    neurons_dir: str = "data/neurons",
    device: str = "cpu",
    max_rounds: int = 3,
    confidence_threshold: float = 0.9,
    enable_gating: bool = True,
    sp_model_path: str | None = None,
    encoder_ckpt_path: str | None = None,
    thalamic_prototypes_path: str | None = None,
    wire_bio_modules: bool = True,
) -> tuple[Any, ModelSelfTokenizer, dict]:
    """P0-1: 统一装配 Cortex，接线所有 bio-inspired 模块。

    生产环境推荐入口。在 create_cortex() 基础上额外完成：

    1. SharedContextEncoder 加载与注册（P0-3，使推理路径脱教师）
    2. ThalamicRouter 加载与注册（P6-2 / P0-3，输入路由）
    3. STDPTracker / CoactivationTracker 注入 ensemble（P1-1，发放时序记录）
    4. NeuromodulatorState 注入 cortex + ensemble（P1-2，调质驱动 lr/不应期/写入强度）
    5. GammaOscillator / WorkingMemory 注入 cortex（P1-4，feature binding + 上下文维持）
    6. LifecycleManager / SleepConsolidator 创建并返回（供 sleep_engine 使用）

    所有可选模块加载失败时退化为默认行为（向后兼容），并记录 warning。

    Args:
        neurons_dir: 神经元 ckpt 目录
        device: 计算设备
        max_rounds: 共振最大轮数
        confidence_threshold: 置信度门限
        enable_gating: 是否启用门控
        sp_model_path: SentencePiece 模型路径
        encoder_ckpt_path: SharedContextEncoder ckpt 路径
            默认 'data/distill/shared_context_encoder.pt'
        thalamic_prototypes_path: ThalamicRouter prototypes 路径
            默认 'data/distill/thalamic_prototypes_p6.pt'（P6-8 训练产物），
            若不存在回退到 'data/distill/thalamic_prototypes.pt'
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
        confidence_threshold=confidence_threshold,
        enable_gating=enable_gating,
        sp_model_path=sp_model_path,
    )

    modules: dict[str, Any] = {}

    if not wire_bio_modules:
        return cortex, tokenizer, modules

    # Step 2: SharedContextEncoder（P0-3，使推理路径脱教师）
    if encoder_ckpt_path is None:
        encoder_ckpt_path = "data/distill/shared_context_encoder.pt"
    try:
        if os.path.exists(encoder_ckpt_path):
            from taiji.resonance import SharedContextEncoder
            encoder = SharedContextEncoder()
            state = torch.load(encoder_ckpt_path, map_location=device, weights_only=False)
            # 兼容直接 state_dict 或包装 dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            encoder.load_state_dict(state, strict=False)
            encoder.to(device)
            encoder.eval()
            cortex.set_context_encoder(encoder)
            modules["context_encoder"] = encoder
            logger.info("[assemble_cortex] SharedContextEncoder loaded from %s",
                        encoder_ckpt_path)
        else:
            logger.warning("[assemble_cortex] SharedContextEncoder ckpt not found: %s",
                           encoder_ckpt_path)
    except Exception as e:
        logger.warning("[assemble_cortex] SharedContextEncoder 加载失败（非致命）: %s", e)

    # Step 3: ThalamicRouter（P6-2 / P0-3，输入路由）
    if thalamic_prototypes_path is None:
        # 优先 P6-8 训练产物，回退到原版
        p6_path = "data/distill/thalamic_prototypes_p6.pt"
        legacy_path = "data/distill/thalamic_prototypes.pt"
        thalamic_prototypes_path = p6_path if os.path.exists(p6_path) else legacy_path
    try:
        if os.path.exists(thalamic_prototypes_path):
            from taiji.resonance import ThalamicRouter
            router = ThalamicRouter.load(thalamic_prototypes_path, device=device)
            cortex.set_thalamic_router(router, top_k=2)
            modules["thalamic_router"] = router
            logger.info("[assemble_cortex] ThalamicRouter loaded from %s (%d prototypes)",
                        thalamic_prototypes_path, len(router.prototypes))
        else:
            logger.warning("[assemble_cortex] ThalamicRouter prototypes not found: %s",
                           thalamic_prototypes_path)
    except Exception as e:
        logger.warning("[assemble_cortex] ThalamicRouter 加载失败（非致命）: %s", e)

    # Step 4: STDPTracker + CoactivationTracker（P1-1，注入 ensemble）
    try:
        from taiji.resonance import STDPTracker
        stdp_tracker = STDPTracker()
        cortex.ensemble.stdp_tracker = stdp_tracker
        modules["stdp_tracker"] = stdp_tracker
        logger.info("[assemble_cortex] STDPTracker wired into ensemble")
    except Exception as e:
        logger.warning("[assemble_cortex] STDPTracker 创建失败（非致命）: %s", e)

    try:
        from taiji.resonance import CoactivationTracker
        coaction = CoactivationTracker()
        cortex.ensemble.coaction = coaction
        modules["coaction"] = coaction
        logger.info("[assemble_cortex] CoactivationTracker wired into ensemble")
    except Exception as e:
        logger.warning("[assemble_cortex] CoactivationTracker 创建失败（非致命）: %s", e)

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
        # 按域分配相位（若有 thalamic_router 的 neuron_meta）
        router = modules.get("thalamic_router")
        if router is not None and router.neuron_meta:
            domain_to_nids: dict[str, list[str]] = {}
            for nid, meta in router.neuron_meta.items():
                domain = meta.get("domain", "default")
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

    logger.info(
        "[assemble_cortex] Done. Wired modules: %s",
        ", ".join(modules.keys()) if modules else "(none)",
    )
    return cortex, tokenizer, modules
