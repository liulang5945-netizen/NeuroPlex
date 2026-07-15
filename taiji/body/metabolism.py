"""
态极代谢系统 (Metabolism)
========================

态极的代谢器官 — 负责硬件感知、资源管理、设备优化。

态极原生实现，专门为态极服务。
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("Taiji.Metabolism")


class HardwareInfo:
    """
    态极的硬件信息

    属性:
        total_ram_gb: 总内存（GB）
        available_memory_gb: 可用内存（GB）
        vram_gb: 显存（GB）
        gpu_name: GPU 名称
        cpu_physical: 物理核心数
        cpu_logical: 逻辑核心数
        device: 计算设备
    """

    def __init__(self):
        self.total_ram_gb = 0.0
        self.available_memory_gb = 0.0
        self.vram_gb = 0.0
        self.gpu_name = ""
        self.cpu_physical = 0
        self.cpu_logical = 0
        self.device = "cpu"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "total_ram_gb": self.total_ram_gb,
            "available_memory_gb": self.available_memory_gb,
            "vram_gb": self.vram_gb,
            "gpu_name": self.gpu_name,
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "device": self.device,
        }

    def is_gpu_available(self) -> bool:
        """检查是否有可用的 GPU"""
        return self.device in ("cuda", "mps", "directml")

    def get_optimal_device(self) -> str:
        """获取最优计算设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            try:
                import torch_directml
                if torch_directml.is_available():
                    return "directml"
            except ImportError:
                pass
        except ImportError:
            pass
        return "cpu"


def analyze_hardware() -> HardwareInfo:
    """
    扫描当前系统硬件信息

    Returns:
        HardwareInfo 对象
    """
    info = HardwareInfo()

    # 获取内存信息
    try:
        import psutil
        info.total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        info.available_memory_gb = round(psutil.virtual_memory().available / (1024**3), 1)
    except ImportError:
        info.total_ram_gb = 8.0
        info.available_memory_gb = 6.0

    # 获取 GPU 信息
    try:
        import torch
        if torch.cuda.is_available():
            info.device = "cuda"
            info.vram_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            info.gpu_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info.device = "mps"
            info.gpu_name = "Apple Silicon"
        else:
            info.device = "cpu"
    except ImportError:
        info.device = "cpu"

    # 获取 CPU 信息
    info.cpu_logical = os.cpu_count() or 8
    try:
        import psutil
        info.cpu_physical = psutil.cpu_count(logical=False) or max(1, info.cpu_logical // 2)
    except ImportError:
        info.cpu_physical = max(1, info.cpu_logical // 2)

    return info


def get_optimal_dtype(device: str):
    """
    根据设备获取最优计算精度

    Args:
        device: 计算设备

    Returns:
        torch.dtype
    """
    try:
        import torch
        if device == "cuda":
            return torch.float16
        elif device == "mps":
            return torch.float16
        else:
            return torch.float32
    except ImportError:
        return None


def estimate_model_params(model) -> Optional[float]:
    """
    估算模型参数量（单位：十亿）

    Args:
        model: 模型对象

    Returns:
        参数量（十亿），如果无法估算则返回 None
    """
    if model is None:
        return None

    try:
        # 方式1: 从 model.config 读取
        config = getattr(model, 'config', None)
        if config is not None:
            num_params = getattr(config, 'num_parameters', None) or getattr(config, 'n_params', None)
            if num_params:
                return round(num_params / 1e9, 2)

        # 方式2: 计算实际参数量
        actual = sum(p.numel() for p in model.parameters())
        if actual > 1e6:
            return round(actual / 1e9, 2)
    except Exception as e:
        logger.debug("metabolism: non-critical %s", e, exc_info=True)

    return None


def check_resources() -> dict:
    """
    检查系统资源状态

    Returns:
        资源状态字典
    """
    info = analyze_hardware()
    result = info.to_dict()

    # 添加资源状态判断
    result["memory_healthy"] = info.available_memory_gb > 1.0
    result["gpu_healthy"] = info.vram_gb > 0.5 if info.is_gpu_available() else True

    return result


def get_device_recommendation(model_size_gb: float) -> str:
    """
    根据模型大小推荐设备

    Args:
        model_size_gb: 模型大小（GB）

    Returns:
        推荐的设备
    """
    info = analyze_hardware()

    # 如果有 GPU 且显存足够
    if info.is_gpu_available() and info.vram_gb >= model_size_gb * 1.2:
        return info.device

    # 如果内存足够
    if info.available_memory_gb >= model_size_gb * 1.5:
        return "cpu"

    # 内存不足
    return "cpu"  # 仍然返回 CPU，但会触发警告