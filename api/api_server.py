"""
[DEPRECATED] 兼容垫片 — 新代码请直接 `from api.app import app`
================================================================
此文件是旧版单文件 API 路由的向后兼容入口。
所有路由已经模块化拆分到 api/ 目录下的独立文件中：
  - api/app.py          → FastAPI 应用实例、中间件、模型加载
  - api/models.py       → Pydantic 数据模型
  - api/routes_chat.py  → 聊天 & 健康检查
  - api/routes_training.py → 微调训练
  - api/routes_rag.py   → RAG 知识库
  - api/routes_models.py → 模型市场与下载
  - api/routes_agent.py → Agent 工具 & 工作台
  - api/routes_system.py → 系统设置 & 硬件检测

请使用 `from api.app import app` 导入应用实例。
此文件仅保留用于兼容旧版导入（如 run_app.py 中的 `from api.api_server import app`）。
"""
from api.app import app
from api.app import get_startup_download_progress