"""Taiji 核心配置模块（API 集成修复，工作4）。

提供 API 服务/硬件自适应/training 公共模块依赖的配置接口：
- TrainingConfig: 训练/推理配置 dataclass
- get_config / save_config: 配置读写
- get_external_path / get_internal_path: 外部可写路径 / 内部打包路径
- get_writable_base_dir: 可写基目录（防只读安装目录）
- apply_env_overrides: 环境变量覆盖
- MODEL_LOAD_TIMEOUT: 模型加载超时

此模块是对 API 层依赖的补齐实现；taiji-neuron 核心训练（resonance）不依赖本模块。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

# 模型加载超时（秒）
MODEL_LOAD_TIMEOUT = 600

# 项目根目录（非打包时）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 应用标识（用于 LocalAppData 子目录）
_APP_DIR_NAME = "Taiji"


def _is_frozen() -> bool:
    """是否打包为 exe。"""
    return getattr(sys, "frozen", False)


def get_writable_base_dir() -> str:
    """获取可写基目录（防只读安装目录）。

    打包时用 %LOCALAPPDATA%/Taiji，否则用项目根目录。
    """
    if _is_frozen():
        local = os.environ.get("LOCALAPPDATA", "")
        base = os.path.join(local, _APP_DIR_NAME) if local else os.getcwd()
    else:
        base = _PROJECT_ROOT
    os.makedirs(base, exist_ok=True)
    return base


def get_external_path(relative_path: str) -> str:
    """获取外部可写路径（如模型缓存、用户数据目录）。

    打包时基于可写基目录（LocalAppData fallback），否则基于项目根。
    """
    base = get_writable_base_dir()
    path = os.path.join(base, relative_path) if relative_path else base
    if relative_path:
        os.makedirs(path, exist_ok=True)
    return path


def get_internal_path(relative_path: str) -> str:
    """获取内部打包路径（如打包进 exe 的前端页面）。

    打包时基于 sys._MEIPASS，否则基于项目根目录。
    """
    if _is_frozen():
        meipass = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
        base = meipass
    else:
        base = _PROJECT_ROOT
    return os.path.join(base, relative_path) if relative_path else base


@dataclass
class TrainingConfig:
    """训练/推理配置（API 层与硬件自适应共用）。

    字段集合来自 api/routes_settings.py、taiji/core/hardware.py、
    api/training/common.py 的实际引用。
    """

    device: str = "auto"            # auto / cuda / mps / cpu
    model_name: str = ""            # 模型名称或路径
    model_type: str = "self"        # self（原生态极单一模型类型）
    cache_dir: str = ""             # 缓存目录
    resume_from_checkpoint: str = ""  # checkpoint 路径
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_lora: bool = False
    lora_r: int = 16
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_length: int = 2048
    gguf_path: str = ""
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    # 内部附加（hw_diag 等）
    _hw_diag: Optional[dict] = field(default=None, repr=False)

    def resolve_device(self) -> str:
        """自动判断最优运算设备（cuda > mps > cpu）。"""
        from neuroplex.core.hardware import resolve_device
        return resolve_device(self)

    @staticmethod
    def get_total_ram_gb() -> float:
        """获取系统总内存（GB），失败时保守回退 16GB。"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            try:
                import torch
                return max(16.0, 0.0)
            except Exception:
                return 16.0


def get_config(args: Optional[List[str]] = None) -> TrainingConfig:
    """获取配置。

    Args:
        args: 命令行剩余参数（key=value 或 --key value 形式，可空）

    Returns:
        TrainingConfig 实例
    """
    config = TrainingConfig()
    # 合并环境变量覆盖（apply_env_overrides 写入）
    for attr, val in _ENV_OVERRIDES.items():
        if hasattr(config, attr):
            setattr(config, attr, val)
    if not args:
        return config

    # 支持 --key value 与 key=value 两种形式
    pairs: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            pairs.append(f"{a[2:]}={args[i + 1]}")
            i += 2
        elif "=" in a:
            pairs.append(a.lstrip("-"))
            i += 1
        else:
            i += 1

    for pair in pairs:
        key, _, value = pair.partition("=")
        key = key.replace("-", "_")
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        try:
            if isinstance(current, bool):
                setattr(config, key, value.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(config, key, int(value))
            elif isinstance(current, float):
                setattr(config, key, float(value))
            else:
                setattr(config, key, value)
        except (ValueError, TypeError):
            pass
    return config


def save_config(config: TrainingConfig, path: Optional[str] = None) -> str:
    """保存配置到 JSON 文件。

    Args:
        config: TrainingConfig 实例
        path: 保存路径（默认 <writable_base>/config.json）

    Returns:
        保存路径
    """
    import json
    from dataclasses import asdict

    path = path or os.path.join(get_writable_base_dir(), "config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {k: v for k, v in asdict(config).items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def apply_env_overrides() -> None:
    """从环境变量应用配置覆盖（TAIJI_* 前缀）。

    支持：TAIJI_MODEL_NAME、TAIJI_DEVICE、TAIJI_MODEL_TYPE 等。
    """
    overrides = {
        "MODEL_NAME": "model_name",
        "DEVICE": "device",
        "MODEL_TYPE": "model_type",
        "CACHE_DIR": "cache_dir",
        "CHECKPOINT": "resume_from_checkpoint",
    }
    changed = False
    for env_key, attr in overrides.items():
        val = os.environ.get(f"TAIJI_{env_key}")
        if val:
            try:
                # 写入进程级配置缓存（供 get_config 默认值使用）
                _ENV_OVERRIDES[attr] = val
                changed = True
            except Exception:
                pass
    if changed:
        print("[config] 已应用环境变量覆盖", flush=True)


# 环境变量覆盖缓存（get_config 时合并）
_ENV_OVERRIDES: dict = {}
