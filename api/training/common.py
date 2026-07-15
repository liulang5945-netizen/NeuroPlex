"""
训练路由公共工具函数

提供被多个训练子模块复用的工具:
  - safe_put(): 带超时的队列写入
  - collect_hardware_diag(): 硬件诊断信息采集
"""
import logging
import queue

logger = logging.getLogger("ApiServer.Training")


def safe_put(log_queue: queue.Queue, msg, timeout: float = 5.0):
    """带超时的队列写入，防止训练线程被慢速 SSE 消费者永久阻塞"""
    try:
        log_queue.put(msg, timeout=timeout)
    except queue.Full:
        pass  # 丢弃过时消息，避免阻塞训练循环


def collect_hardware_diag(device_str: str) -> dict:
    """采集训练设备的硬件信息，生成诊断消息和建议"""
    import torch

    device_type = device_str
    device_name = device_str.upper()
    gpu_name = None
    gpu_memory_gb = None
    ram_gb = None

    if device_str == "cuda" and torch.cuda.is_available():
        device_type = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            device_name = gpu_name
        except Exception:
            device_name = "CUDA GPU"

    # 检测系统 RAM
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            from taiji.core.config import TrainingConfig
            ram_gb = round(TrainingConfig.get_total_ram_gb(), 1)
        except Exception:
            ram_gb = None

    # 生成诊断消息
    if device_type == "cuda":
        msg = (f"训练设备: {device_name} ({gpu_memory_gb}GB 显存)"
               f"{' | 系统内存: ' + str(ram_gb) + 'GB' if ram_gb else ''}"
               f" — GPU 训练性能最优 ✓")
    elif device_type == "cpu":
        msg = (f"训练设备: CPU"
               f"{' | 系统内存: ' + str(ram_gb) + 'GB' if ram_gb else ''}"
               f" — ⚠ CPU 训练较慢（约 GPU 的 1/10~1/50），建议使用 GPU 或 GGUF 推理模式")
    elif device_type == "mps":
        msg = (f"训练设备: Apple MPS (Metal)"
               f"{' | 系统内存: ' + str(ram_gb) + 'GB' if ram_gb else ''}"
               f" — MPS 训练性能介于 CPU 和 CUDA 之间")
    else:
        msg = (f"训练设备: {device_type}"
               f"{' | 系统内存: ' + str(ram_gb) + 'GB' if ram_gb else ''}")

    return {
        "device_type": device_type,
        "device_name": device_name,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "message": msg,
    }


