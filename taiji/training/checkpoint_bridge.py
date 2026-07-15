"""Checkpoint bridge: load first-gen 1.5B model for distillation.

This module loads a checkpoint from the first-gen taiji project
and exposes the teacher model's hidden states for distillation
into ResonanceNeuron instances.

Uses importlib to load from the first-gen project path directly,
avoiding conflicts with the taiji-neuron package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn


def load_teacher_model(
    checkpoint_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, nn.Embedding]:
    """Load the 1.5B teacher model from a first-gen checkpoint.

    Args:
        checkpoint_dir: path to checkpoint dir (e.g., 'e:/taiji/checkpoint-400000')
        device: 'cpu' or 'cuda'
        dtype: torch dtype

    Returns:
        (teacher_model, shared_embedding) tuple.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # Read config
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Find first-gen project root
    taiji_root = _find_taiji_root(checkpoint_dir)
    if not taiji_root:
        raise ImportError("Cannot find first-gen taiji project (expected at e:/taiji)")

    # Use importlib to load from specific path (avoids conflict with taiji-neuron)
    ModelSelf, ModelConfig = _import_gen1_modules(taiji_root)

    # Build model config
    hidden_size = config_data.get("hidden_size", 2048)
    num_layers = config_data.get("num_hidden_layers", 24)
    num_heads = config_data.get("num_attention_heads", 16)
    num_kv_heads = config_data.get("num_key_value_heads", 4)
    intermediate_size = config_data.get("intermediate_size", 5632)
    vocab_size = config_data.get("vocab_size", 256000)
    max_position = config_data.get("max_position_embeddings", 4096)

    cfg = ModelConfig()
    cfg.hidden_size = hidden_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv_heads
    cfg.intermediate_size = intermediate_size
    cfg.vocab_size = vocab_size
    cfg.max_position_embeddings = max_position

    model = ModelSelf(cfg).to(device=device, dtype=dtype)

    # Load weights
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pt not found in {checkpoint_dir}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    remapped = _remap_legacy_keys(state_dict)
    model.load_state_dict(remapped, strict=False)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Bridge] Loaded teacher: {hidden_size}d, {num_layers}L, {param_count/1e9:.2f}B params")

    embedding = model.backbone.embedding if hasattr(model, "backbone") else model.embedding
    return model, embedding


def extract_hidden_states(
    teacher_model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract hidden states from the teacher model for distillation.

    Handles different first-gen model structures:
    - backbone.layers (architecture.py v2)
    - direct .layers (older architecture)
    - attention may return (tensor, kv_cache) tuple
    """
    with torch.no_grad():
        # Determine the container (backbone or direct)
        if hasattr(teacher_model, "backbone"):
            container = teacher_model.backbone
        else:
            container = teacher_model

        layers = getattr(container, "layers", None)
        embedding = getattr(container, "embedding", None) or getattr(teacher_model, "embedding", None)
        norm = getattr(container, "norm", None) or getattr(teacher_model, "norm", None)

        if embedding is None or layers is None:
            raise AttributeError("Cannot find embedding/layers in teacher model")

        h = embedding(input_ids)
        for layer in layers:
            # First-gen TransformerBlock attributes: attention, attention_norm, feed_forward, ffn_norm
            attn = getattr(layer, "attention", None)
            attn_norm = getattr(layer, "attention_norm", None)
            ffn = getattr(layer, "feed_forward", None)
            ffn_norm = getattr(layer, "ffn_norm", None)

            # attention() returns (output, kv_cache) tuple in first-gen
            attn_out = attn(attn_norm(h))
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h = h + attn_out

            h = h + ffn(ffn_norm(h))

        if norm is not None:
            h = norm(h)
        return h


def _import_gen1_modules(taiji_root: str):
    """Import ModelSelf and ModelConfig from the first-gen project using importlib.

    The first-gen architecture.py uses relative imports (from .config, from .layers).
    To make this work, we register the first-gen taiji package in sys.modules
    before loading, so relative imports resolve correctly.
    """
    taiji_pkg = os.path.join(taiji_root, "taiji")
    arch_path = os.path.join(taiji_pkg, "architecture.py")
    cfg_path = os.path.join(taiji_pkg, "config.py")
    layers_path = os.path.join(taiji_pkg, "layers.py")

    # Step 1: Load layers (no relative imports) and register as 'taiji.layers'
    spec_layers = importlib.util.spec_from_file_location("taiji.layers", layers_path)
    mod_layers = importlib.util.module_from_spec(spec_layers)
    sys.modules["taiji.layers"] = mod_layers
    spec_layers.loader.exec_module(mod_layers)

    # Step 2: Load config (may have relative imports from .layers) and register as 'taiji.config'
    spec_cfg = importlib.util.spec_from_file_location("taiji.config", cfg_path)
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    sys.modules["taiji.config"] = mod_cfg
    spec_cfg.loader.exec_module(mod_cfg)

    # Step 3: Register empty 'taiji' package so relative imports resolve
    taiji_init = os.path.join(taiji_pkg, "__init__.py")
    if os.path.exists(taiji_init):
        spec_pkg = importlib.util.spec_from_file_location("taiji", taiji_init)
        mod_pkg = importlib.util.module_from_spec(spec_pkg)
        sys.modules["taiji"] = mod_pkg
        spec_pkg.loader.exec_module(mod_pkg)
    else:
        # Create a synthetic package
        mod_pkg = type(sys)("taiji")
        mod_pkg.__path__ = [taiji_pkg]
        mod_pkg.__package__ = "taiji"
        sys.modules["taiji"] = mod_pkg

    # Ensure submodule attribute linkage
    mod_pkg.layers = mod_layers
    mod_pkg.config = mod_cfg
    mod_layers.__package__ = "taiji"
    mod_cfg.__package__ = "taiji"

    # Step 4: Load architecture (now relative imports resolve via sys.modules)
    spec_arch = importlib.util.spec_from_file_location("taiji.architecture", arch_path)
    mod_arch = importlib.util.module_from_spec(spec_arch)
    sys.modules["taiji.architecture"] = mod_arch
    spec_arch.loader.exec_module(mod_arch)

    return mod_arch.ModelSelf, mod_cfg.ModelConfig


def _find_taiji_root(checkpoint_dir: str) -> Optional[str]:
    for c in ["e:/taiji", os.path.dirname(checkpoint_dir)]:
        if os.path.exists(os.path.join(c, "taiji", "architecture.py")):
            return c
    return None


def _remap_legacy_keys(state_dict: dict) -> dict:
    return {k.replace("model.backbone.", "backbone."): v for k, v in state_dict.items()}
