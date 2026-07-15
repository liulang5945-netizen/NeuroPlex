"""Core Taiji configuration and token contracts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_native_v2_contract() -> dict[str, Any]:
    contract_path = Path(__file__).with_name("tokenizer_contract.json")
    with contract_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


NATIVE_V2_TOKENIZER_CONTRACT = _load_native_v2_contract()
_SPECIAL = {
    key: int(value)
    for key, value in NATIVE_V2_TOKENIZER_CONTRACT["special_tokens"].items()
}
_GENERATED = NATIVE_V2_TOKENIZER_CONTRACT.get("generated_special_ranges", {})
_MULTIMODAL = NATIVE_V2_TOKENIZER_CONTRACT["multimodal"]
_MULTIMODAL_CONTROL = _MULTIMODAL["control_tokens"]
_MULTIMODAL_CONTROL_RANGE = NATIVE_V2_TOKENIZER_CONTRACT["ranges"]["multimodal_control"]
_TOOL_TOKENS = NATIVE_V2_TOKENIZER_CONTRACT["tool_tokens"]


SPECIAL_TOKEN_CONTENT = {
    "think_start": "<think>",
    "think_end": "</think>",
    "tool_call": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_result": "<tool_result>",
    "tool_result_end": "</tool_result>",
    "answer": "<final_answer>",
    "observe": "<observe>",
    "observe_end": "</observe>",
    "env_tree": "<tree>",
    "env_tree_end": "</tree>",
    "env_state": "<state>",
    "env_state_end": "</state>",
    "env_result": "<result>",
    "env_result_end": "</result>",
    "mem_read": "<mem_read>",
    "mem_write": "<mem_write>",
    "plan_start": "<plan>",
    "plan_end": "</plan>",
    "plan_step": "<step>",
    "plan_step_end": "</step>",
    "plan_done": "<plan_done>",
    "plan_replan": "<replan>",
    "reflect_start": "<reflect>",
    "reflect_end": "</reflect>",
    "reflect_detect": "<detect>",
    "reflect_cause": "<cause>",
    "reflect_correct": "<correct>",
    "reflect_confirm": "<confirm>",
}


SPECIAL_TOKENS = {
    "pad": _SPECIAL["<pad>"],
    "unk": _SPECIAL["<unk>"],
    "bos": _SPECIAL["<s>"],
    "eos": _SPECIAL["</s>"],
    "think_start": _SPECIAL["<think>"],
    "think_end": _SPECIAL["</think>"],
    "tool_call": _SPECIAL["<tool_call>"],
    "tool_call_end": _SPECIAL["</tool_call>"],
    "tool_result": _SPECIAL["<tool_result>"],
    "tool_result_end": _SPECIAL["</tool_result>"],
    "answer": _SPECIAL["<final_answer>"],
    "observe": _SPECIAL["<observe>"],
    "observe_end": _SPECIAL["</observe>"],
    "env_tree": _SPECIAL["<tree>"],
    "env_tree_end": _SPECIAL["</tree>"],
    "env_state": _SPECIAL["<state>"],
    "env_state_end": _SPECIAL["</state>"],
    "env_result": _SPECIAL["<result>"],
    "env_result_end": _SPECIAL["</result>"],
    "mem_read": _SPECIAL["<mem_read>"],
    "mem_write": _SPECIAL["<mem_write>"],
    "mem_slot_base": int(_GENERATED["memory_slots"]["start"]),
    "mem_long_base": _SPECIAL["<long_term>"],
    "plan_start": _SPECIAL["<plan>"],
    "plan_end": _SPECIAL["</plan>"],
    "plan_step": _SPECIAL["<step>"],
    "plan_step_end": _SPECIAL["</step>"],
    "plan_done": _SPECIAL["<plan_done>"],
    "plan_replan": _SPECIAL["<replan>"],
    "reflect_start": _SPECIAL["<reflect>"],
    "reflect_end": _SPECIAL["</reflect>"],
    "reflect_detect": _SPECIAL["<detect>"],
    "reflect_cause": _SPECIAL["<cause>"],
    "reflect_correct": _SPECIAL["<correct>"],
    "reflect_confirm": _SPECIAL["<confirm>"],
    "tool_name_base": int(_TOOL_TOKENS["start"]),
}


MULTIMODAL_TOKENS = {
    "image_token_base": int(_MULTIMODAL["image"]["base"]),
    "image_codebook_size": int(_MULTIMODAL["image"]["codebook_size"]),
    "audio_token_base": int(_MULTIMODAL["audio"]["base"]),
    "audio_codebook_size": int(_MULTIMODAL["audio"]["codebook_size"]),
    "mm_control_base": int(_MULTIMODAL_CONTROL_RANGE[0]),
    "mm_control_size": int(_MULTIMODAL_CONTROL_RANGE[1]) - int(_MULTIMODAL_CONTROL_RANGE[0]) + 1,
}


MM_CONTROL_TOKENS = {
    "image_start": int(_MULTIMODAL_CONTROL["<mm_image>"]),
    "image_end": int(_MULTIMODAL_CONTROL["</mm_image>"]),
    "audio_start": int(_MULTIMODAL_CONTROL["<mm_audio>"]),
    "audio_end": int(_MULTIMODAL_CONTROL["</mm_audio>"]),
    "video_start": int(_MULTIMODAL_CONTROL["<mm_video>"]),
    "video_end": int(_MULTIMODAL_CONTROL["</mm_video>"]),
    "gen_image": int(_MULTIMODAL_CONTROL["<mm_gen_image>"]),
    "gen_audio": int(_MULTIMODAL_CONTROL["<mm_gen_audio>"]),
    "gen_video": int(_MULTIMODAL_CONTROL["<mm_gen_video>"]),
    "img_row": int(_MULTIMODAL_CONTROL["<mm_row>"]),
    "frame": int(_MULTIMODAL_CONTROL["<mm_frame>"]),
}


MULTIMODAL_VOCAB_SIZE = int(NATIVE_V2_TOKENIZER_CONTRACT["total_vocab_size"])


class SpecialTokenResolver:
    """Resolve Taiji structural token IDs from the active tokenizer."""

    def __init__(self, tokenizer: Any):
        self._cache: dict[str, int] = {}
        self._resolve(tokenizer)

    def _resolve(self, tokenizer: Any) -> None:
        for name, content in SPECIAL_TOKEN_CONTENT.items():
            token_id = self._find_token_id(tokenizer, content)
            if token_id is None:
                token_id = SPECIAL_TOKENS.get(name, 0)
            self._cache[name] = int(token_id)

        self._cache["tool_name_base"] = int(SPECIAL_TOKENS["tool_name_base"])
        self._cache["pad"] = int(getattr(tokenizer, "pad_token_id", SPECIAL_TOKENS["pad"]))
        self._cache["unk"] = int(getattr(tokenizer, "unk_token_id", SPECIAL_TOKENS["unk"]))
        self._cache["bos"] = int(getattr(tokenizer, "bos_token_id", SPECIAL_TOKENS["bos"]))
        self._cache["eos"] = int(getattr(tokenizer, "eos_token_id", SPECIAL_TOKENS["eos"]))

    def _find_token_id(self, tokenizer: Any, content: str) -> int | None:
        try:
            token_id = tokenizer.convert_tokens_to_ids(content)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_id:
                return int(token_id)
        except Exception:
            pass

        try:
            token_ids = tokenizer.encode(content, add_special_tokens=False)
            if len(token_ids) == 1:
                return int(token_ids[0])
        except Exception:
            pass

        try:
            added = getattr(tokenizer, "added_tokens_decoder", {})
            for token_id, info in added.items():
                if getattr(info, "content", None) == content:
                    return int(token_id)
                if isinstance(info, dict) and info.get("content") == content:
                    return int(token_id)
        except Exception:
            pass

        return None

    def __getitem__(self, name: str) -> int:
        return int(self._cache.get(name, SPECIAL_TOKENS.get(name, 0)))

    def get(self, name: str, default: int | None = None) -> int | None:
        return self._cache.get(name, default)

    def resolve_all(self) -> dict[str, int]:
        return dict(self._cache)


def get_taiji_data_path(subdir: str) -> str:
    """Return a writable Taiji data directory, preferring the active model folder."""
    try:
        from taiji.core.app_state import app_state

        model_path = getattr(app_state, "_loaded_model_name", "") or ""
        if model_path and os.path.isdir(model_path):
            model_dir = model_path if os.path.exists(os.path.join(model_path, "config.json")) else os.path.dirname(model_path)
            data_path = os.path.join(model_dir, subdir)
            os.makedirs(data_path, exist_ok=True)
            return data_path
    except Exception:
        pass

    try:
        from taiji.core.config import get_external_path

        base = get_external_path(os.path.join("taiji_data", subdir))
    except Exception:
        base = os.path.join(os.getcwd(), "taiji_data", subdir)

    os.makedirs(base, exist_ok=True)
    return base


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    vocab_size: int = int(NATIVE_V2_TOKENIZER_CONTRACT["total_vocab_size"])
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    active_heads: list[str] | None = None
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    base_vocab_size: int = int(NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"])
    num_special_tokens: int = int(
        NATIVE_V2_TOKENIZER_CONTRACT["total_vocab_size"]
        - NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"]
    )
    num_short_term_slots: int = 20
    num_long_term_slots: int = 10
    memory_dim: int = 64

    def to_dict(self) -> dict:
        """Serialise to dict — auto-generated from dataclass fields, never drifts."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialise from dict with sensible defaults."""
        defaults = {
            "vocab_size": int(NATIVE_V2_TOKENIZER_CONTRACT["total_vocab_size"]),
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "attention_bias": False,
            "base_vocab_size": int(NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"]),
            "num_special_tokens": int(NATIVE_V2_TOKENIZER_CONTRACT["total_vocab_size"] - NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"]),
            "num_short_term_slots": 20,
            "num_long_term_slots": 10,
            "memory_dim": 64,
            "active_heads": None,
        }
        return cls(**{k: type(defaults.get(k, v))(v) if isinstance(defaults.get(k), (int, float, bool)) else v
                      for k, v in {**defaults, **data}.items() if k in defaults})

    @classmethod
    def size_125m(cls) -> "ModelConfig":
        return cls(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
        )

    @classmethod
    def size_350m(cls) -> "ModelConfig":
        return cls(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
        )

    @classmethod
    def size_1b(cls) -> "ModelConfig":
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_3b(cls) -> "ModelConfig":
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_7b(cls) -> "ModelConfig":
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def from_qwen(cls, model_name: str = "Qwen/Qwen2.5-0.5B") -> "ModelConfig":
        qwen_configs = {
            "Qwen/Qwen2.5-0.5B": cls(
                vocab_size=151936,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
            "Qwen/Qwen2.5-1.5B": cls(
                vocab_size=151936,
                hidden_size=1536,
                intermediate_size=8960,
                num_hidden_layers=28,
                num_attention_heads=12,
                num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
            "Qwen/Qwen2.5-3B": cls(
                vocab_size=151936,
                hidden_size=2048,
                intermediate_size=11008,
                num_hidden_layers=36,
                num_attention_heads=16,
                num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
        }
        return qwen_configs.get(model_name, qwen_configs["Qwen/Qwen2.5-0.5B"])

    def validate(self) -> bool:
        """Validate model configuration on construction.
        Raises ValueError if any parameter is invalid.
        """
        errors = []
        if self.vocab_size < 1:
            errors.append("vocab_size must be >= 1")
        if self.hidden_size < 1:
            errors.append("hidden_size must be >= 1")
        if self.intermediate_size < 1:
            errors.append("intermediate_size must be >= 1")
        if self.num_hidden_layers < 1:
            errors.append("num_hidden_layers must be >= 1")
        if self.num_attention_heads < 1:
            errors.append("num_attention_heads must be >= 1")
        if self.num_key_value_heads < 1:
            errors.append("num_key_value_heads must be >= 1")
        if self.max_position_embeddings < 1:
            errors.append("max_position_embeddings must be >= 1")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            errors.append(f"num_attention_heads ({self.num_attention_heads}) "
                         f"must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(f"hidden_size ({self.hidden_size}) "
                         f"must be divisible by num_attention_heads ({self.num_attention_heads})")
        if errors:
            raise ValueError(f"ModelConfig validation failed: {'; '.join(errors)}")
        return True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def count_parameters(self) -> int:
        embed = self.vocab_size * self.hidden_size
        head_dim = self.head_dim
        attn = (
            self.hidden_size * self.num_attention_heads * head_dim
            + self.hidden_size * self.num_key_value_heads * head_dim
            + self.hidden_size * self.num_key_value_heads * head_dim
            + self.num_attention_heads * head_dim * self.hidden_size
        )
        ffn = (
            self.hidden_size * self.intermediate_size
            + self.hidden_size * self.intermediate_size
            + self.intermediate_size * self.hidden_size
        )
        norm = self.hidden_size * 2
        return embed + (attn + ffn + norm) * self.num_hidden_layers

    @property
    def size_label(self) -> str:
        params = self.count_parameters()
        if params >= 3e9:
            return "7B"
        if params >= 1e9:
            return "3B"
        if params >= 5e8:
            return "1B"
        if params >= 2e8:
            return "350M"
        if params >= 8e7:
            return "125M"
        return f"{params / 1e6:.0f}M"

    def describe(self) -> str:
        params = self.count_parameters()
        if params >= 1e9:
            size_str = f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            size_str = f"{params / 1e6:.0f}M"
        else:
            size_str = f"{params / 1e3:.0f}K"
        return (
            f"ModelSelf-{size_str} | "
            f"hidden={self.hidden_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads} kv={self.num_key_value_heads} "
            f"ffn={self.intermediate_size} vocab={self.vocab_size}"
        )
