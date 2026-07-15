"""
微调训练 API 路由子包

模块拆分:
  - common.py   → 公共工具函数 (_safe_put, 心跳循环, 硬件诊断)
  - control.py  → 训练控制 (暂停/恢复/停止/重置)
  - datasets.py → 数据集管理 (上传/列表/删除/预览)
  - stream.py   → 流式训练接口
  - resume.py   → 断点续训接口
  - publish.py  → 模型发布 & GGUF 导出
"""
from fastapi import APIRouter

router = APIRouter()

# 延迟导入各子模块的 router 并合并
from .control import router as control_router
from .datasets import router as datasets_router
from .stream import router as stream_router
from .resume import router as resume_router
from .publish import router as publish_router
from .recommend import router as recommend_router

router.include_router(recommend_router)
router.include_router(control_router)
router.include_router(datasets_router)
router.include_router(stream_router)
router.include_router(resume_router)
router.include_router(publish_router)
