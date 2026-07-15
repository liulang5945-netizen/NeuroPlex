"""
聊天 & 健康检查 API 路由
提供：
- POST /api/chat/stream     → 流式聊天（支持本地/云端/Agent/态极引擎）
- POST /api/chat/history/{session_id} → 保存会话历史
- POST /api/chat/upload     → 聊天文件上传
- GET  /api/health          → 健康检查
"""
import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse

from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path
from api.models import ChatRequest
from api.chat_strategies import create_event_generator

logger = logging.getLogger("ApiServer.Chat")
router = APIRouter()


# ======================== 流式聊天 ========================

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天端点，支持 SSE 推送"""
    # 触发用户指令，中断当前生命活动
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        get_life_scheduler().handle_user_directive()
    except Exception as e:
        logger.warning(f"Failed to trigger user directive: {e}")

    # 根据引擎类型选择数据收集器
    def collector_factory():
        try:
            from taiji.agent_ext.data_collector import DataCollector
            return DataCollector(
                save_path=get_external_path(
                    os.path.join("agent", "conversations", f"{int(time.time())}.jsonl")
                )
            )
        except Exception:
            return None

    event_generator = create_event_generator(request, app_state, collector_factory)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

@router.get("/api/chat/stream")
async def chat_stream_get():
    """明确处理对 stream 的 GET 请求，返回 405（方法不允许）以满足客户端预期"""
    raise HTTPException(status_code=405, detail="Method Not Allowed")


# ======================== 会话历史管理 ========================

_history_dir = get_external_path(os.path.join("user_data", "chat_history"))
os.makedirs(_history_dir, exist_ok=True)


def _safe_session_id(session_id: str) -> str:
    """验证 session_id，防止路径穿越"""
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', session_id):
        raise HTTPException(status_code=400, detail="无效的会话 ID")
    return session_id


@router.post("/api/chat/history/{session_id}")
async def save_chat_history(session_id: str, request: Request):
    """保存或更新指定会话的名称和消息历史"""
    session_id = _safe_session_id(session_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体")

    session_file = os.path.join(_history_dir, f"{session_id}.json")

    # 读取已有数据（如存在）
    existing = {}
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    # 合并更新
    if "name" in body:
        existing["name"] = body["name"]
    if "messages" in body:
        existing["messages"] = body["messages"]
    existing["session_id"] = session_id
    existing["updated_at"] = time.time()

    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存会话历史失败: {e}")
        raise HTTPException(status_code=500, detail="保存会话历史失败")

    return {"status": "ok", "session_id": session_id}


@router.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """获取指定会话的历史记录"""
    session_id = _safe_session_id(session_id)
    session_file = os.path.join(_history_dir, f"{session_id}.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"读取会话历史失败: {e}")
        raise HTTPException(status_code=500, detail="读取会话历史失败")


@router.post("/api/chat/sessions")
async def create_chat_session(request: Request):
    """创建会话（兼容测试中使用的简单 JSON body）"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体")
    sid = str(body.get("id") or int(time.time()))
    name = body.get("name", "")
    session_file = os.path.join(_history_dir, f"{sid}.json")
    data = {
        "session_id": sid,
        "name": name,
        "messages": body.get("messages", []),
        "updated_at": time.time(),
    }
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail="创建会话失败")
    return {"status": "ok", "session_id": sid}

@router.get("/api/chat/sessions")
async def list_chat_sessions():
    """列出所有已保存的会话"""
    sessions = []
    if os.path.exists(_history_dir):
        for fname in sorted(os.listdir(_history_dir), reverse=True):
            if fname.endswith(".json"):
                fpath = os.path.join(_history_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", fname.replace(".json", "")),
                        "name": data.get("name", ""),
                        "updated_at": data.get("updated_at", 0),
                    })
                except Exception:
                    continue
    return sessions


@router.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    """删除指定会话的历史记录"""
    session_id = _safe_session_id(session_id)
    session_file = os.path.join(_history_dir, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            os.remove(session_file)
        except Exception as e:
            logger.error(f"删除会话历史失败: {e}")
            raise HTTPException(status_code=500, detail="删除会话历史失败")
    return {"status": "ok"}


# ======================== 文件上传 ========================

# 支持的文本文件扩展名
_TEXT_EXTENSIONS = {
    "txt", "md", "py", "js", "ts", "jsx", "tsx", "vue", "html", "htm", "css",
    "json", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "sh", "bash",
    "bat", "cmd", "ps1", "sql", "java", "c", "cpp", "h", "hpp", "go", "rs",
    "rb", "php", "swift", "kt", "scala", "r", "m", "lua", "pl", "ex", "exs",
    "hs", "ml", "clj", "lisp", "el", "vim", "tex", "csv", "log", "gitignore",
    "dockerfile", "makefile", "cmake", "gradle",
}

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "tif", "svg"}

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/api/chat/upload")
async def upload_chat_file(file: UploadFile = File(...)):
    """上传文件并解析内容（用于聊天上下文）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    # 读取文件内容
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取文件失败: {e}")

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="文件过大，最大支持 20MB")

    # 图片文件：仅返回元信息
    if ext in _IMAGE_EXTENSIONS:
        return {
            "status": "ok",
            "filename": file.filename,
            "type": "image",
            "size": len(content),
            "parsed_text": f"[图片: {file.filename} ({len(content)} bytes)]",
        }

    # 文本文件：尝试解析内容
    if ext in _TEXT_EXTENSIONS or ext == "":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("gbk")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

        # 截断过长内容
        max_chars = 50000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [截断，共 {len(content)} 字节]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "text",
            "size": len(content),
            "parsed_text": text,
        }

    # PDF 文件
    if ext == "pdf":
        try:
            import io
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text = "\n\n".join(text_parts)
            except ImportError:
                text = f"[PDF 文件: {file.filename} - 需要安装 PyPDF2 才能解析 PDF 内容]"
        except Exception as e:
            text = f"[PDF 解析失败: {e}]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "pdf",
            "size": len(content),
            "parsed_text": text,
        }

    # DOCX 文件
    if ext == "docx":
        try:
            import io
            try:
                from docx import Document
                doc = Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                text = f"[DOCX 文件: {file.filename} - 需要安装 python-docx 才能解析 DOCX 内容]"
        except Exception as e:
            text = f"[DOCX 解析失败: {e}]"

        return {
            "status": "ok",
            "filename": file.filename,
            "type": "docx",
            "size": len(content),
            "parsed_text": text,
        }

    # 其他未知类型
    return {
        "status": "ok",
        "filename": file.filename,
        "type": "unknown",
        "size": len(content),
        "parsed_text": f"[不支持的文件类型: .{ext}]",
    }


# ======================== 健康检查 ========================

@router.get("/api/health")
async def health_check():
    """健康检查端点"""
    # 启动未完成时返回 loading / downloading 状态
    if not app_state.startup_complete:
        from taiji.core.model_loader import startup_download_progress
        dl = startup_download_progress
        if dl["active"]:
            return {
                "status": "downloading",
                "message": dl["message"],
                "percent": dl["percent"],
                "total_mb": dl["total_mb"],
                "downloaded_mb": dl["downloaded_mb"],
            }
        return {"status": "loading", "message": "模型正在加载中..."}

    if app_state.startup_error:
        return {"status": "error", "message": app_state.startup_error}

    return {
        "status": "ok",
        "service": "Taiji API",
        "timestamp": time.time(),
        "model_loaded": app_state.model is not None,
        "taiji_available": app_state.is_taiji(),
    }