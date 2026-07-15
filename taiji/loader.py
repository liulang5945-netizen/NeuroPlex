"""Taiji model loader and saver for the native-v2 tokenizer contract."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Callable, Optional

import torch

from .architecture import ModelSelf
from .config import ModelConfig, NATIVE_V2_TOKENIZER_CONTRACT
from .tokenizer import ModelSelfTokenizer

logger = logging.getLogger("Taiji")

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
) -> tuple[ModelSelf, ModelSelfTokenizer]:
    """Create a fresh Taiji model and tokenizer pair."""
    normalized_size = str(size).strip().lower()
    builder = _SIZE_BUILDERS.get(normalized_size)
    if builder is None:
        raise ValueError(f"Unsupported model size: {size}")

    config = builder()
    if active_heads is not None:
        config.active_heads = list(active_heads)

    tokenizer = ModelSelfTokenizer(sp_model_path=sp_model_path)
    config.vocab_size = tokenizer.vocab_size
    config.base_vocab_size = int(NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"])
    config.num_special_tokens = config.vocab_size - config.base_vocab_size
    model = ModelSelf(config).to(device=device, dtype=dtype)
    return model, tokenizer


def load_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ModelSelf, ModelSelfTokenizer]:
    """Load a saved Taiji model directory."""
    model_path = os.path.abspath(model_path)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    config = _dict_to_config(config_dict)

    model = ModelSelf(config)

    weights_path = os.path.join(model_path, "model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(_remap_legacy_keys(state_dict), strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint (will use random init): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
        # Re-establish weight tying broken by load_state_dict
        model.lm_head.weight = model.backbone.embedding.weight
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No weights found at %s; using random initialization", weights_path)

    sp_path = _find_sentencepiece(model_path)
    if sp_path is None:
        raise FileNotFoundError(f"SentencePiece model not found in {model_path}")

    tokenizer = ModelSelfTokenizer.load(model_path, sp_model_path=sp_path)
    if hasattr(tokenizer, "_tool_name_to_id"):
        model.set_num_tools(len(tokenizer._tool_name_to_id))

    model = model.to(device=device, dtype=dtype)
    logger.info("Loaded model: %s", config.describe())
    logger.info("Loaded tokenizer: vocab=%s tools=%s", tokenizer.vocab_size, len(tokenizer.get_all_tool_ids()))
    return model, tokenizer


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


def save_model(
    model: ModelSelf,
    tokenizer: ModelSelfTokenizer,
    save_path: str,
    training_state: Optional[dict] = None,
) -> None:
    """Save a Taiji model, tokenizer, and optional training state."""
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
