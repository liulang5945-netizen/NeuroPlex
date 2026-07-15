"""
通用工具模块
提供 JSON 读写、HTML Toast、历史记录管理、路径解析等辅助功能
"""
import json
import logging
import os
import sys
import traceback

# 日志级别映射
_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# 日志格式单例（避免重复创建 Formatter）
_LOG_FORMAT = logging.Formatter(
    "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# 已注册的 logger 缓存
_loggers: dict = {}

# 默认日志级别
_DEFAULT_LEVEL = logging.INFO


def setup_logging(level: str = "INFO"):
    """
    全局日志配置（应用启动时调用一次）
    设置根 logger 的控制台 Handler
    """
    global _DEFAULT_LEVEL
    _DEFAULT_LEVEL = _LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(_DEFAULT_LEVEL)
    
    # 避免重复添加 Handler
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_LOG_FORMAT)
        root_logger.addHandler(console_handler)


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    获取（或创建）统一配置的 Logger。
    所有模块应使用此函数代替 logging.getLogger()

    用法:
        logger = get_logger("ApiServer")
        logger = get_logger("Agent", "DEBUG")
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(_LOG_LEVEL_MAP.get(level.upper(), _DEFAULT_LEVEL))
    else:
        logger.setLevel(_DEFAULT_LEVEL)
    
    # 确保有 Handler（如果没有父级 Handler）
    if not logger.handlers and not logger.parent.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_LOG_FORMAT)
        logger.addHandler(console_handler)
    
    _loggers[name] = logger
    return logger


logger = get_logger("Utils")


# ======================== 路径工具（统一管理，减少重复代码） ========================

def get_external_path(relative_path: str) -> str:
    """
    获取外部路径（如外部的 model_cache 模型文件夹）
    委托给 core.config.get_external_path（含可写目录检测 + LocalAppData fallback）
    """
    from taiji.core.config import get_external_path as _get
    return _get(relative_path)


def get_internal_path(relative_path: str) -> str:
    """
    获取内部打包路径（如打包进 exe 里的前端页面）
    委托给 core.config.get_internal_path
    """
    from taiji.core.config import get_internal_path as _get
    return _get(relative_path)


def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径"""
    os.makedirs(path, exist_ok=True)
    return path


# ======================== JSON 工具 ========================

def safe_json_load(file_path: str, default=None) -> dict:
    """安全加载 JSON 文件"""
    if default is None:
        default = {}
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 JSON 失败 ({file_path}): {e}")
        return default


def safe_json_save(file_path: str, data: dict):
    """安全保存 JSON 文件"""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 JSON 失败 ({file_path}): {e}")


def safe_remove(file_path: str):
    """安全删除文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"删除文件失败 ({file_path}): {e}")


# ======================== UI 工具 ========================

def build_toast_html(msg: str, type_: str = "info") -> str:
    """
    生成优雅的 Toast 通知 HTML
    type_: info / success / error / warning
    """
    colors = {
        "info": {"bg": "#e0f2fe", "border": "#7dd3fc", "icon": "ℹ️", "text": "#0369a1"},
        "success": {"bg": "#dcfce7", "border": "#86efac", "icon": "✅", "text": "#15803d"},
        "error": {"bg": "#fee2e2", "border": "#fca5a5", "icon": "❌", "text": "#b91c1c"},
        "warning": {"bg": "#fef9c3", "border": "#fde047", "icon": "⚠️", "text": "#a16207"},
    }
    c = colors.get(type_, colors["info"])
    return f"""
    <div style="padding: 10px 16px; background: {c['bg']}; border: 1px solid {c['border']};
                border-radius: 10px; color: {c['text']}; font-size: 14px;
                margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                animation: toastIn 0.3s ease;">
        <span style="margin-right: 6px;">{c['icon']}</span>{msg}
    </div>
    <style>
    @keyframes toastIn {{ 0% {{ opacity: 0; transform: translateY(-10px); }} 
    100% {{ opacity: 1; transform: translateY(0); }} }}
    </style>
    """


# ======================== 历史管理 ========================

def normalize_history(history: list) -> list:
    """规范化聊天历史为 list[list] 格式"""
    norm_history = []
    if not history or not isinstance(history, list):
        return norm_history

    # 检查是否为 dict 格式（新版本 Gradio）
    if history and isinstance(history[0], dict):
        user_tmp = None
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_tmp = content
            elif role == "assistant" and user_tmp is not None:
                norm_history.append([user_tmp, content])
                user_tmp = None
    else:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                norm_history.append([item[0], item[1]])
    return norm_history


# ======================== 杂项 ========================

def filter_by_query(items: list, query: str) -> list:
    """根据查询文本过滤列表"""
    if not query:
        return items
    q = query.lower()
    return [d for d in items if q in d.lower()]


def generate_bg_css(img_path: str, base_dir: str = None) -> str:
    """
    生成自定义背景 CSS
    使用 file:/// 引用避免 base64 内联膨胀
    """
    if not img_path or not os.path.exists(img_path):
        return ""
    abs_path = os.path.abspath(img_path).replace("\\", "/")
    return f"""
    <style>
    .gradio-container {{
        background-image: url('file:///{abs_path}') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        transition: background-image 0.5s ease-in-out;
    }}
    .form, .panel, .box, .wrap, .message {{
        background-color: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    }}
    textarea, input {{
        background-color: rgba(255, 255, 255, 0.8) !important;
    }}
    </style>"""
