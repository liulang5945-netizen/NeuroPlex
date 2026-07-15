"""
态极模型专属 API
提供状态查询、能力评估、升级检查、自动升级等功能
"""
import json
import logging
import os
import threading

from fastapi import APIRouter, HTTPException
from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.ModelSelf")
router = APIRouter()

# 全局自动升级器实例（延迟初始化）
_upgrader = None
_upgrade_lock = threading.Lock()

def _get_upgrader():
    """获取自动升级器实例（延迟初始化）"""
    global _upgrader
    if _upgrader is None:
        return _ensure_upgrader()
    return _upgrader

def _ensure_upgrader():
    global _upgrader
    if _upgrader is None:
        try:
            from taiji.infra.auto_upgrade import AutoUpgrader
            save_dir = get_external_path(os.path.join("taiji", "evolution_data"))
            _upgrader = AutoUpgrader(save_dir=save_dir)
            logger.info("自动升级引擎已初始化")
        except Exception as e:
            logger.error(f"初始化自动升级引擎失败: {e}")
            return None
    return _upgrader


@router.get("/api/taiji_model/status")
def get_taiji_model_status():
    """获取态极模型状态（混合引擎统计、能力分数等）"""
    if not app_state.is_taiji():
        return {"status": "inactive", "message": "态极引擎未激活"}

    taiji = app_state.get_taiji_engine()
    result = {
        "status": "active",
        "taiji_engine": True,
    }

    # 如果有混合引擎，返回其状态
    if hasattr(app_state, '_hybrid_engine') and app_state._hybrid_engine:
        result["hybrid_engine"] = app_state._hybrid_engine.get_status()
    else:
        result["hybrid_engine"] = None

    return result


@router.get("/api/taiji_model/upgrade_check")
def check_upgrade():
    """
    检查态极模型是否应该升级
    返回：当前模型大小、瓶颈状态、推荐升级目标、硬件容量
    """
    from taiji.core.hardware import analyze_hardware

    # 获取当前模型配置
    current_config = "125M"  # 默认
    if hasattr(app_state, '_loaded_model_name') and app_state._loaded_model_name:
        # 检查模型目录中是否有 config.json
        model_path = app_state._loaded_model_name
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    hidden = cfg.get("hidden_size", 768)
                    if hidden >= 2048:
                        current_config = "1B"
                    elif hidden >= 1024:
                        current_config = "350M"
                    else:
                        current_config = "125M"
                except Exception:
                    pass

    # 硬件扫描
    try:
        hw = analyze_hardware()
        available_vram = hw.vram_gb or 0
        available_ram = hw.available_memory_gb or hw.total_ram_gb
    except Exception:
        available_vram = 0
        available_ram = 8

    # 升级路线表
    upgrade_table = {
        "125M": {"next": "350M", "min_vram": 1.5, "min_ram": 4, "hidden_size": 1024},
        "350M": {"next": "1B",   "min_vram": 3.0, "min_ram": 8, "hidden_size": 2048},
        "1B":   {"next": "3B",   "min_vram": 6.0, "min_ram": 16, "hidden_size": 3072},
        "3B":   {"next": "7B",   "min_vram": 12.0, "min_ram": 24, "hidden_size": 4096},
        "7B":   {"next": None,   "min_vram": 24.0, "min_ram": 32, "hidden_size": 4096},
    }

    current_info = upgrade_table.get(current_config)
    if not current_info or not current_info["next"]:
        return {
            "can_upgrade": False,
            "current_model": current_config,
            "message": "已达到当前硬件支持的最大模型",
            "hardware": {"vram_gb": round(available_vram, 1), "ram_gb": round(available_ram, 1)},
        }

    next_model = current_info["next"]
    next_info = upgrade_table.get(next_model, {})
    min_vram = next_info.get("min_vram", 999)
    min_ram = next_info.get("min_ram", 999)

    # 检查硬件是否支持（GPU推理 or CPU量化推理）
    can_gpu = available_vram >= min_vram
    can_cpu_quant = available_ram >= min_ram * 0.6  # 量化后内存需求约60%
    can_upgrade = can_gpu or can_cpu_quant

    # 检查是否有瓶颈
    has_bottleneck = False
    bottleneck_reason = ""

    # 方式1：训练器检测到的瓶颈（态极原生训练）
    trainer = getattr(app_state, '_trainer_ref', None)
    if trainer and hasattr(trainer, '_bottleneck_detected') and trainer._bottleneck_detected:
        has_bottleneck = True
        bottleneck_reason = "训练 loss 停滞，当前模型容量已达上限"

    # 方式2：混合引擎的能力分数
    if not has_bottleneck and hasattr(app_state, '_hybrid_engine') and app_state._hybrid_engine:
        hybrid = app_state._hybrid_engine
        if hasattr(hybrid, 'capability_scores') and hybrid.capability_scores:
            avg_cap = sum(hybrid.capability_scores.values()) / len(hybrid.capability_scores)
            if avg_cap > 0.6 and current_config in ("125M", "350M"):
                has_bottleneck = True
                bottleneck_reason = f"能力分数已达 {avg_cap:.1%}，当前模型容量不足"

    # 方式3：模型太小且已有一定训练量
    if not has_bottleneck and current_config == "125M":
        # 检查是否有训练历史
        checkpoints_dir = get_external_path(os.path.join("taiji_checkpoints", "finetune"))
        best_path = os.path.join(checkpoints_dir, "best")
        if os.path.exists(os.path.join(best_path, "model.pt")):
            has_bottleneck = True
            bottleneck_reason = "125M 模型已有训练数据，建议升级到更大模型以获得更好效果"

    return {
        "can_upgrade": can_upgrade,
        "is_taiji": app_state.is_taiji(),
        "current_model": current_config,
        "next_model": next_model if can_upgrade else None,
        "has_bottleneck": has_bottleneck,
        "bottleneck_reason": bottleneck_reason,
        "hardware": {
            "vram_gb": round(available_vram, 1),
            "ram_gb": round(available_ram, 1),
            "can_gpu": can_gpu,
            "can_cpu_quant": can_cpu_quant,
        },
        "upgrade_mode": "gpu" if can_gpu else ("cpu_quantized" if can_cpu_quant else "not_supported"),
        "message": (
            f"建议升级到 {next_model}（{'GPU模式' if can_gpu else 'CPU量化模式'}）"
            if can_upgrade
            else f"硬件不足以运行 {next_model}，需要 {min_vram}GB 显存或 {min_ram}GB 内存"
        ),
    }


@router.get("/api/taiji_model/capability")
def get_capability():
    """获取态极能力评估分数"""
    upgrader = _ensure_upgrader()
    if not upgrader:
        return {"status": "error", "message": "升级引擎未初始化"}
    return {
        "status": "ok",
        "capability": upgrader.capability_evaluator.get_status(),
        "bottleneck": upgrader.bottleneck_detector.detect_bottleneck(
            upgrader.capability_evaluator
        ),
    }


@router.post("/api/taiji_model/upgrade")
def start_upgrade():
    """
    用户确认后启动自动升级
    执行知识蒸馏：旧模型 → 新模型
    """
    upgrader = _ensure_upgrader()
    if not upgrader:
        raise HTTPException(status_code=500, detail="升级引擎未初始化")

    if upgrader.upgrade_state in ("distilling", "creating", "switching"):
        return {
            "status": "in_progress",
            "message": "升级正在进行中",
            "progress": upgrader.upgrade_progress,
        }

    if not app_state.is_taiji():
        raise HTTPException(status_code=400, detail="态极引擎未激活，无法升级")

    # 检查是否有训练数据
    training_data_path = get_external_path(
        os.path.join("taiji", "evolution_data", "hybrid_training.jsonl")
    )
    seed_data_path = get_external_path(
        os.path.join("taiji", "evolution_data", "seed_training.jsonl")
    )

    training_data = []
    for data_path in [training_data_path, seed_data_path]:
        if os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            training_data.append(json.loads(line))
            except Exception:
                pass

    # 也加载种子数据
    try:
        from taiji.data.seed_data import get_seed_conversation_data, get_seed_react_data
        training_data.extend(get_seed_conversation_data())
        training_data.extend(get_seed_react_data())
    except ImportError:
        try:
            from taiji.data.seed_data import get_seed_conversation_data, get_seed_react_data
            training_data.extend(get_seed_conversation_data())
            training_data.extend(get_seed_react_data())
        except Exception:
            pass
    except Exception:
        pass

    # 加载 training_data 目录下的所有 JSONL 文件
    training_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taiji", "training_data")
    if os.path.exists(training_data_dir):
        for fname in sorted(os.listdir(training_data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(training_data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            training_data.append(json.loads(line))
            except Exception:
                pass
        logger.info(f"从 training_data 加载了 {len(training_data)} 条数据用于蒸馏")

    if len(training_data) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"训练数据不足（当前 {len(training_data)} 条，至少需要 10 条）。请先多使用态极对话积累数据。"
        )

    # 获取教师模型（兼容 NativeInferenceEngine 和 TaijiMultimodalEngine）
    taiji = app_state.get_taiji_engine()
    if hasattr(taiji, 'text_engine'):
        # TaijiMultimodalEngine
        teacher_model = taiji.text_engine.inference.model
        teacher_tokenizer = taiji.text_engine.inference.tokenizer
    elif hasattr(taiji, 'model'):
        # NativeInferenceEngine
        teacher_model = taiji.model
        teacher_tokenizer = taiji.tokenizer
    else:
        # 回退：直接从 app_state 获取
        teacher_model = app_state.model
        teacher_tokenizer = app_state.tokenizer

    # 确定设备
    try:
        device = str(next(teacher_model.parameters()).device)
    except Exception:
        device = "cpu"

    # 后台执行升级
    def _do_upgrade():
        try:
            result = upgrader.start_upgrade(
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                training_data=training_data,
                device=device,
                progress_callback=lambda pct, msg: logger.info(f"升级进度: {pct}% - {msg}"),
            )
            logger.info(f"升级结果: {result.get('status')} - {result.get('message')}")
        except Exception as e:
            logger.error(f"升级线程异常: {e}")
            upgrader.upgrade_state = "error"
            upgrader.upgrade_error = str(e)

    thread = threading.Thread(target=_do_upgrade, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "升级已在后台启动",
        "training_data_count": len(training_data),
    }


@router.get("/api/taiji_model/upgrade_progress")
def get_upgrade_progress():
    """获取升级进度（前端轮询用）"""
    upgrader = _ensure_upgrader()
    if not upgrader:
        return {"state": "error", "message": "升级引擎未初始化"}
    return {
        "state": upgrader.upgrade_state,
        "progress": upgrader.upgrade_progress,
        "message": upgrader.upgrade_message,
        "error": upgrader.upgrade_error,
    }
