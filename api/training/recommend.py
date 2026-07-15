"""
训练配置推荐 API
================
提供硬件自适应训练配置推荐和数据集质量检查端点
"""
import logging
import os
import psutil
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("ApiServer.Training.Recommend")
router = APIRouter()


class RecommendRequest(BaseModel):
    dataset_size: int = 100
    preset: str = "mid"


class DatasetCheckRequest(BaseModel):
    file_path: str = ""


def _detect_hardware() -> dict:
    """检测本地硬件，为原生训练提供配置参考"""
    cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cuda_available = torch.cuda.is_available()
    gpu_name = ""
    vram_gb = 0
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    return {
        "cpu_cores": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
    }


_NATIVE_PRESETS = {
    "tiny": {
        "hidden_size": 384,
        "num_layers": 8,
        "batch_size": 16,
        "grad_accum": 1,
        "description": "极轻量（适合 CPU 快速实验）",
    },
    "small": {
        "hidden_size": 512,
        "num_layers": 12,
        "batch_size": 8,
        "grad_accum": 2,
        "description": "小模型（适合消费级 GPU）",
    },
    "mid": {
        "hidden_size": 768,
        "num_layers": 16,
        "batch_size": 4,
        "grad_accum": 4,
        "description": "中等模型（适合单卡训练）",
    },
    "large": {
        "hidden_size": 1024,
        "num_layers": 24,
        "batch_size": 2,
        "grad_accum": 8,
        "description": "大模型（需多卡或大显存）",
    },
}


@router.post("/api/training/recommend")
async def get_training_recommendation(req: RecommendRequest):
    """获取基于本地硬件的原生训练推荐配置"""
    try:
        hw = _detect_hardware()
        return {
            "status": "success",
            "hardware": hw,
            "presets": {k: {**v, "hardware": hw} for k, v in _NATIVE_PRESETS.items()},
        }
    except Exception as e:
        logger.error(f"获取训练推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/check_dataset")
async def check_dataset_quality(req: DatasetCheckRequest):
    """检查数据集质量"""
    try:
        from taiji.train.dataset_checker import DatasetQualityChecker
        checker = DatasetQualityChecker()
        result = checker.check(req.file_path)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"数据集检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
