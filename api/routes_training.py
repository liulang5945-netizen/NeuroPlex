"""
⚠️ 此文件是旧版单文件训练路由的向后兼容入口。
所有路由已经模块化拆分到 api/training/ 子包中：
  - api/training/common.py   → 公共工具函数
  - api/training/stream.py   → 流式训练接口
  - api/training/resume.py   → 断点续训 & 检查点管理
  - api/training/control.py  → 训练控制（暂停/恢复/停止/重置）
  - api/training/datasets.py → 数据集管理（上传/列表/删除/预览）
  - api/training/publish.py  → 模型发布 & GGUF 导出

请使用 `from api.training import router` 导入路由。
此文件仅保留用于兼容旧版导入（如 api/app.py 中的 `from .routes_training import router`）。
"""
from api.training import router
