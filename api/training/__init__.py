"""
训练 API 路由子包
================
Cortex 模式下训练走 sleep_engine，这里保留数据集管理、训练控制、发布查询等辅助端点。

模块:
  - common.py   → 公共工具函数
  - control.py  → 训练控制 (暂停/恢复/停止/重置)
  - datasets.py → 数据集管理 (上传/列表/删除/预览)
  - publish.py  → 模型发布查询 & GGUF 不支持消息
  - recommend.py → 硬件检测 & 数据集质量检查
"""
from fastapi import APIRouter

router = APIRouter()

from .control import router as control_router
from .datasets import router as datasets_router
from .publish import router as publish_router
from .recommend import router as recommend_router

router.include_router(recommend_router)
router.include_router(control_router)
router.include_router(datasets_router)
router.include_router(publish_router)
