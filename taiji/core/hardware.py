"""
硬件自适应配置模块

从 core/config.py 中提取的硬件感知算法:
  - resolve_device(): 自动判断最优运算设备
  - estimate_params_b(): 四层递进估算模型参数量
  - auto_configure_for_hardware(): 纯公式驱动的硬件自适应配置
  - get_torch_dtype(): 根据设备自动匹配计算精度
"""
import json
import logging
import os
import re

logger = logging.getLogger("Hardware")


def resolve_device(config: "TrainingConfig") -> str:
    """自动判断最优的运算设备（cuda > mps > directml > cpu）"""
    if config.device != "auto":
        return config.device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"


def estimate_params_b(config, loaded_model=None) -> tuple:
    """
    四层递进估算模型参数量（纯函数，无副作用）。

    优先级：
    1. model.config.num_parameters（HuggingFace 标准属性，不受量化影响）
    2. model.config.hidden_size + num_hidden_layers（从配置估算）
    3. 本地 config.json / 模型名称解析
    4. 保守回退 1.0B

    Returns:
        (params_b: float, source: str)
    """
    # ── 方式1: 从 model.config 直接读取（HuggingFace 标准属性，最准确） ──
    if loaded_model is not None:
        try:
            _cfg = getattr(loaded_model, 'config', None)
            if _cfg is not None:
                # 方式1a: 直接从 config 读 num_parameters
                _np = getattr(_cfg, 'num_parameters', None) or _cfg.get('num_parameters', 0)
                if _np:
                    params_b = round(_np / 1e9, 2)
                    logger.info(f"从 model.config.num_parameters 读取参数量: {params_b:.2f}B")
                    return params_b, "model.config.num_parameters"
                # 方式1b: 从 config 的 hidden_size + num_hidden_layers 公式计算
                hidden = getattr(_cfg, 'hidden_size', None) or _cfg.get('hidden_size', 0)
                layers = getattr(_cfg, 'num_hidden_layers', None) or _cfg.get('num_hidden_layers', 0)
                if hidden and layers:
                    params_b = round((12 * layers * hidden * hidden) / 1e9, 2)
                    if params_b > 0.01:
                        logger.info(f"从 model.config 公式计算参数量: {params_b:.2f}B")
                        return params_b, "model.config(公式)"
        except Exception as e:
            logger.warning(f"从 model.config 读取参数量失败: {e}")

        # ── 方式1c: 回退到 model.parameters() 遍历 ──
        try:
            actual = sum(p.numel() for p in loaded_model.parameters())
            if actual > 1e6:
                params_b = round(actual / 1e9, 2)
                logger.info(f"从 model.parameters() 遍历读取参数量: {params_b:.2f}B")
                return params_b, "模型对象(parameters)"
        except Exception as e:
            logger.warning(f"从模型对象遍历 parameters() 失败: {e}")

    # ── 方式2: 从本地 config.json 读取 ──
    if config.model_name and os.path.isdir(config.model_name):
        cfg_json = os.path.join(config.model_name, "config.json")
        if os.path.exists(cfg_json):
            try:
                with open(cfg_json, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                for key in ("num_parameters", "n_params"):
                    v = cdata.get(key, 0)
                    if v:
                        return round(v / 1e9, 2), f"config.json({key})"
                hidden = cdata.get("hidden_size", 0)
                layers = cdata.get("num_hidden_layers", 0)
                if hidden and layers:
                    return round((12 * layers * hidden * hidden) / 1e9, 2), "config.json(公式)"
            except Exception:
                pass

    # ── 方式2.5: 文件体积反推参数量 ──
    if config.model_name and os.path.isdir(config.model_name):
        import glob as _glob
        _model_files = (_glob.glob(os.path.join(config.model_name, "*.safetensors")) +
                        _glob.glob(os.path.join(config.model_name, "*.bin")))
        if _model_files:
            _total_bytes = sum(os.path.getsize(f) for f in _model_files)
            _bytes_per_param = 2.0
            _cfg_json = os.path.join(config.model_name, "config.json")
            if os.path.exists(_cfg_json):
                try:
                    with open(_cfg_json, "r", encoding="utf-8") as _f:
                        _cdata = json.load(_f)
                    _dtype = str(_cdata.get("torch_dtype", "")).lower()
                    if "float32" in _dtype or "fp32" in _dtype:
                        _bytes_per_param = 4.0
                except Exception:
                    pass
            _params_b = round(_total_bytes / (_bytes_per_param * 1e9), 2)
            if 0.01 < _params_b < 200:
                logger.info(f"从文件体积估算参数量: {_params_b:.2f}B"
                            f" ({_total_bytes/1e9:.1f}GB @ {_bytes_per_param:.0f} bytes/param)")
                return _params_b, "文件体积估算"

    # ── 方式3: 从模型名称解析 ──
    if config.model_name:
        match = re.search(r'(\d+\.?\d*)\s*[bB]', config.model_name)
        if match:
            return float(match.group(1)), "模型名称解析"
        name_lower = config.model_name.lower()
        scale_map = [
            (('72b', '70b'), 72.0), (('32b',), 32.0), (('14b', '13b'), 14.0),
            (('12b',), 12.0), (('9b',), 9.0), (('8b',), 8.0), (('7b',), 7.0),
            (('6b',), 6.0), (('3b', '3.8b'), 3.8), (('2b',), 2.0),
            (('1.5b',), 1.5), (('0.5b',), 0.5),
        ]
        for keywords, scale in scale_map:
            if any(kw in name_lower for kw in keywords):
                return scale, "关键词推断"

    return 1.0, "保守回退"


def auto_configure_for_hardware(config, loaded_model=None) -> dict:
    """
    纯公式驱动的硬件自适应配置。

    支持 0.5B~72B 任意模型，4GB~128GB 任意设备。
    所有决策只依赖于：
    - 模型参数量（自动检测）
    - 可用内存（psutil 实时读取）
    - CPU 物理核心数

    没有任何 if 模型名 or if 内存大小的硬编码分支。
    """
    import torch as _t

    # ── 1. 采集硬件参数 ──
    device = resolve_device(config)

    # 总内存
    from taiji.core.config import TrainingConfig
    ram_gb_total = TrainingConfig.get_total_ram_gb()
    # 当前实际可用内存
    try:
        import psutil as _ps
        ram_gb_available = round(_ps.virtual_memory().available / (1024**3), 1)
    except Exception:
        ram_gb_available = max(1.0, ram_gb_total * 0.75)

    # CUDA 显存
    vram_gb = 0.0
    gpu_name = ""
    is_gpu = device == "cuda" and _t.cuda.is_available()
    if is_gpu:
        vram_gb = round(_t.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        gpu_name = _t.cuda.get_device_name(0)
        effective_mem = max(0.5, vram_gb - 1.5)
    else:
        effective_mem = ram_gb_available

    # CPU 核心
    cpu_total = os.cpu_count() or 8
    try:
        import psutil as _ps
        cpu_physical = _ps.cpu_count(logical=False) or max(1, cpu_total // 2)
    except Exception:
        cpu_physical = max(1, cpu_total // 2)

    # ── 2. 读取模型参数量 ──
    params_b, _param_source = estimate_params_b(config, loaded_model)
    logger.info(f"模型参数量: {params_b:.2f}B（来源: {_param_source}）")

    # ── 3. 纯公式计算所有内存项 ──
    # 常量说明（经验值，基于 NVIDIA/Apple 官方推荐）:
    #   dtype_per_B:  FP16=2 bytes/param, FP32=4 bytes/param
    #   quant_ratio: 4-bit≈55%原体积, 8-bit≈110%（含量化开销）
    #   optim_ratio: LoRA 仅优化 1% 参数，全量训练优化 100%
    #   headroom:    LoRA 预留 25% 参数量的临时空间，全量预留 200%
    #   os_reserve:  至少 3GB 或 20% 总内存留给操作系统
    use_fp16 = is_gpu
    dtype_per_B = 2.0 if use_fp16 else 4.0

    quant_ratio = 1.0
    if config.load_in_4bit:
        quant_ratio = 0.55
    elif config.load_in_8bit:
        quant_ratio = 1.1

    model_gb = params_b * dtype_per_B * quant_ratio
    optim_ratio = 0.01 if config.use_lora else 1.0
    optimizer_gb = params_b * 2 * dtype_per_B * optim_ratio
    headroom_gb = max(0.5, params_b * (0.25 if config.use_lora else 2.0))
    os_reserve_gb = max(3.0, ram_gb_total * 0.20)

    # ── 4. 量化自动决策 ──
    decisions = []
    _quant_auto = False
    model_gb_native = params_b * dtype_per_B

    mem_ratio = model_gb_native / max(0.1, effective_mem)
    if not config.load_in_4bit and not config.load_in_8bit:
        if mem_ratio > 0.50:
            config.load_in_4bit = True
            _quant_auto = True
            decisions.append(f"🔄 模型 ~{model_gb_native:.0f}GB → 4-bit 量化（可用内存 ~{effective_mem:.0f}GB）")
            quant_ratio = 0.55
            model_gb = params_b * dtype_per_B * quant_ratio
        elif mem_ratio > 0.30:
            config.load_in_8bit = True
            _quant_auto = True
            decisions.append(f"🔄 模型 ~{model_gb_native:.0f}GB → 8-bit 量化（可用内存 ~{effective_mem:.0f}GB）")
            quant_ratio = 1.1
            model_gb = params_b * dtype_per_B * quant_ratio

    # ── 5. 剩余可用内存计算 ──
    raw_usable = effective_mem - model_gb - optimizer_gb - headroom_gb - os_reserve_gb
    usable_gb = max(0.2, raw_usable * 0.80)
    usable_gb = round(usable_gb, 2)

    # ── 6. batch_size 纯公式计算 ──
    # 常量说明:
    #   0.12: 每 B 参数在 512 token 序列上的经验激活内存（GB）
    #   clamp [0.02, 8.0]: 防止极端值
    #   CPU ×1.8: CPU 推理/训练内存效率比 GPU 差约 80%
    #   LoRA GPU ×0.7: LoRA 反向传播内存约为全量训练的 70%
    per_sample_gb = params_b * 0.12 * (config.max_length / 512)
    per_sample_gb = max(0.02, min(per_sample_gb, 8.0))
    if not is_gpu:
        per_sample_gb *= 1.8
    if is_gpu and config.use_lora:
        per_sample_gb *= 0.7

    auto_batch = max(1, int(usable_gb / per_sample_gb))

    if auto_batch < config.batch_size:
        decisions.append(f"📐 Batch {auto_batch}（每步约需 {per_sample_gb:.1f}GB）")
        config.batch_size = auto_batch

    # ── 7. 梯度累积 ──
    # 目标有效 batch size: clamp(4, 32, 16/params_b) — 小模型用大 batch，大模型用小 batch
    target_effective = max(4, min(32, int(16 / max(0.5, params_b)))) if params_b > 0 else 4
    if config.batch_size < target_effective:
        auto_grad = max(1, target_effective // config.batch_size)
        if auto_grad > config.gradient_accumulation_steps:
            config.gradient_accumulation_steps = auto_grad
            decisions.append(f"🔗 梯度累积 ×{auto_grad}（等效 Batch≈{config.batch_size * auto_grad}）")

    # ── 8. max_length 自动调节 ──
    if raw_usable < 0 and config.max_length > 128:
        config.max_length = max(128, int(config.max_length * 0.6))
        decisions.append(f"📏 序列长度: {config.max_length}")

    # ── 9. CPU 线程优化 ──
    if device == "cpu":
        reserved = max(1, int(cpu_physical * 0.15))
        train_threads = max(1, cpu_physical - reserved)
        _t.set_num_threads(train_threads)
        os.environ["OMP_NUM_THREADS"] = str(train_threads)
        os.environ["MKL_NUM_THREADS"] = str(train_threads)
        decisions.append(f"🧵 CPU: {train_threads} 核（共 {cpu_physical} 核，预留 {reserved} 核）")
    else:
        gpu_cpu_threads = max(2, cpu_physical // 2)
        _t.set_num_threads(gpu_cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(gpu_cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(gpu_cpu_threads)
        decisions.append(f"🧵 CPU 线程: {gpu_cpu_threads}/{cpu_physical} 物理核（GPU 训练，避免争用）")

    # ── 10. LoRA rank 自适应 ──
    if config.use_lora:
        auto_r = max(8, min(32, round(params_b * 4)))
        if auto_r != config.lora_r:
            decisions.append(f"🔬 LoRA rank: {auto_r}（适配 ~{params_b:.1f}B 模型）")
            config.lora_r = auto_r

    # ── 11. 内存哨兵阈值自适应计算 ──
    _ram_factor = min(1.0, 8.0 / max(1.0, ram_gb_total))
    _mw_level0_pct = max(0.15, 0.35 * _ram_factor)
    _mw_level1_pct = max(0.10, 0.25 * _ram_factor)
    _mw_level2_pct = max(0.05, 0.15 * _ram_factor)
    _mw_level3_pct = max(0.03, 0.08 * _ram_factor)
    _mw_resume_pct = min(0.50, _mw_level1_pct + 0.05)

    # ── 12. 生成诊断摘要 ──
    diag = {
        "model_params_b": round(params_b, 2),
        "param_source": _param_source,
        "device": device,
        "device_name": gpu_name or device.upper(),
        "ram_gb_total": round(ram_gb_total, 1),
        "ram_gb_available": round(ram_gb_available, 1),
        "vram_gb": round(vram_gb, 1) if vram_gb else None,
        "usable_gb": round(usable_gb, 1),
        "model_gb": round(model_gb, 1),
        "optimizer_gb": round(optimizer_gb, 1),
        "headroom_gb": round(headroom_gb, 1),
        "os_reserve_gb": round(os_reserve_gb, 1),
        "per_sample_gb": round(per_sample_gb, 3),
        "cpu_physical": cpu_physical,
        "cpu_logical": cpu_total,
        "dtype": "fp16" if use_fp16 else "fp32",
        "batch_size": config.batch_size,
        "grad_accum": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "quant_auto": _quant_auto,
        "lora_r": config.lora_r,
        "decisions": decisions,
        "summary": "; ".join(decisions) if decisions else "✅ 资源充足，使用用户当前配置",
        "memory_watchdog": {
            "level0_pct": round(_mw_level0_pct, 3),
            "level1_pct": round(_mw_level1_pct, 3),
            "level2_pct": round(_mw_level2_pct, 3),
            "level3_pct": round(_mw_level3_pct, 3),
            "resume_pct": round(_mw_resume_pct, 3),
            "trend_window_size": 8,
        },
    }
    config._hw_diag = diag

    logger.info(
        f"[自适应配置] 模型 {params_b:.1f}B ({_param_source}) | {device}"
        f"{' (' + gpu_name + ' ' + str(vram_gb) + 'GB)' if gpu_name else ' (' + str(ram_gb_total) + 'GB)'}"
        f" | 可用={ram_gb_available:.1f}GB | batch={config.batch_size} grad_accum={config.gradient_accumulation_steps}"
        f" | {diag['summary']}"
    )

    return diag


class HardwareInfo:
    """硬件信息容器"""
    def __init__(self):
        self.total_ram_gb = 0.0
        self.available_memory_gb = 0.0
        self.vram_gb = 0.0
        self.gpu_name = ""
        self.cpu_physical = 0
        self.cpu_logical = 0
        self.device = "cpu"


def analyze_hardware() -> HardwareInfo:
    """
    扫描当前系统硬件信息，返回 HardwareInfo 对象。
    供 routes_taiji_model.py 的升级检查等模块调用。
    """
    info = HardwareInfo()

    # 内存
    try:
        from taiji.core.config import TrainingConfig
        info.total_ram_gb = round(TrainingConfig.get_total_ram_gb(), 1)
    except Exception:
        info.total_ram_gb = 8.0

    try:
        import psutil as _ps
        info.available_memory_gb = round(_ps.virtual_memory().available / (1024**3), 1)
    except Exception:
        info.available_memory_gb = round(info.total_ram_gb * 0.75, 1)

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            info.device = "cuda"
            info.vram_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            info.gpu_name = torch.cuda.get_device_name(0)
        else:
            info.device = "cpu"
    except Exception:
        info.device = "cpu"

    # CPU
    info.cpu_logical = os.cpu_count() or 8
    try:
        import psutil as _ps
        info.cpu_physical = _ps.cpu_count(logical=False) or max(1, info.cpu_logical // 2)
    except Exception:
        info.cpu_physical = max(1, info.cpu_logical // 2)

    return info


def get_torch_dtype(config):
    """
    根据实际训练设备自动匹配最优计算精度。

    CPU 上 fp16 是软件模拟，速度极慢（慢 10-50 倍）。
    - CUDA: fp16（原生硬件加速）
    - CPU/MPS/DirectML: fp32（原生速度）
    """
    import torch

    if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()):
        return torch.float16

    return torch.float32
