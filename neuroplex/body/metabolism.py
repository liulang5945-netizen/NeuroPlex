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

# 模块级 NeuromodulatorState 引用（由 BodyCore 注入）
_neuromodulator = None


def set_neuromodulator(nm_state):
    """设置神经调质状态引用，用于根据硬件状态调节神经调质。

    Args:
        nm_state: NeuromodulatorState 实例（或 None 用于清除）
    """
    global _neuromodulator
    _neuromodulator = nm_state
    if nm_state is not None:
        logger.info(f"NeuromodulatorState 已注入 metabolism: {type(nm_state).__name__}")


def update_neuromodulator():
    """根据当前硬件状态更新神经调质水平（去甲肾上腺素专用通道）。

    硬件状态→神经调质映射（人脑启发）：
    - CPU 负载高 → norepinephrine↓（节能模式，降低 field_write 强度，减少计算量）
    - CPU 负载低 → norepinephrine↑（专注模式，可以全力投入计算）
    - 内存紧张 → dopamine↓（资源不足=负面信号，触发保守模式）
    - 内存充裕 → dopamine↑（资源充足=正面信号）
    - GPU 可用 → dopamine 额外奖励（算力充足）
    - 资源健康 → serotonin↑（满足感，稳定运行）

    注意：本方法只设置 NE 目标值（硬件通道），DA 和 5HT 由 sleep_engine
    的训练反馈驱动（loss→DA, 准确率→5HT）。三个调质各司其职，不会互相覆盖。

    此方法由 SleepEngine 训练前调用（评估硬件状态后决定 field_write 强度）。
    """
    if _neuromodulator is None:
        return

    try:
        info = analyze_hardware()
        resources = check_resources()

        # 1. CPU 负载 → norepinephrine（警觉度/节能控制）
        #    设计原则：高负载 → NE↓（节能，降低 field_write）→ 减少计算量
        #    避免正反馈循环（高负载→NE↑→field_write↑→更多计算→更高负载）
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # CPU 负载 0% → NE=0.9（专注），100% → NE=0.2（节能）
            norepinephrine_target = max(0.2, 0.9 - cpu_percent / 100.0 * 0.7)
        except ImportError:
            norepinephrine_target = 0.5

        # 2. 内存状态 → dopamine（仅当内存紧张时覆盖，否则不动 sleep_engine 的 DA）
        #    sleep_engine 的 DA 由 loss 趋势驱动，metabolism 只在内存危急时介入
        mem_percent = resources.get("memory_percent", 50)
        if mem_percent > 90:
            # 内存紧张 → 多巴胺下降（负面信号，保守模式）
            dopamine_target = 0.2
        elif mem_percent > 75:
            dopamine_target = 0.35
        else:
            # 内存充裕时不覆盖 sleep_engine 的 DA（传 None 表示不更新）
            dopamine_target = None

        # 3. GPU 可用 → dopamine 额外奖励（仅在 dopamine_target 非 None 时叠加）
        if dopamine_target is not None and info.is_gpu_available():
            dopamine_target = min(1.0, dopamine_target + 0.15)

        # 4. 资源健康度 → serotonin（仅当资源不健康时覆盖，否则不动 sleep_engine 的 5HT）
        if not resources.get("memory_healthy", True) or not resources.get("gpu_healthy", True):
            serotonin_target = 0.3
        else:
            # 资源健康时不覆盖 sleep_engine 的 5HT
            serotonin_target = None

        # 设置目标值（实际值会通过 EMA 缓慢趋近）
        # NE 总是设置（硬件通道），DA/5HT 仅在资源异常时覆盖
        _neuromodulator.set_targets(
            dopamine=dopamine_target,
            serotonin=serotonin_target,
            norepinephrine=norepinephrine_target,
        )

        logger.debug(
            f"硬件调质更新: NE={norepinephrine_target:.2f} (cpu={cpu_percent:.0f}%), "
            f"DA={'%.2f' % dopamine_target if dopamine_target is not None else 'skip'}, "
            f"5HT={'%.2f' % serotonin_target if serotonin_target is not None else 'skip'}"
        )
    except Exception as e:
        logger.debug(f"神经调质更新失败（非关键）: {e}")


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

    # 内存使用百分比（供 neuromodulator 使用）
    try:
        import psutil
        result["memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        if info.total_ram_gb > 0:
            result["memory_percent"] = round(
                (1 - info.available_memory_gb / info.total_ram_gb) * 100, 1
            )
        else:
            result["memory_percent"] = 50.0

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