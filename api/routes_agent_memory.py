"""
记忆系统 API 路由
从 routes_agent.py 拆分：短期记忆、工作记忆、长期记忆
"""
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("ApiServer.Agent.Memory")
router = APIRouter()


@router.get("/api/agent/memory/status")
def memory_status():
    """获取记忆系统状态"""
    try:
        from taiji.agent_ext.memory_manager import memory
        return {"status": "ok", **memory.get_status()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/agent/memory/context")
def memory_context(last_n: int = 20):
    """获取短期记忆中的对话上下文"""
    try:
        from taiji.agent_ext.memory_manager import memory
        context = memory.get_context(last_n)
        return {"status": "ok", "context": context}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/memory/remember")
async def memory_remember(req: dict):
    """存储信息到长期记忆"""
    text = req.get("text", "").strip()
    category = req.get("category", "general")
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        from taiji.agent_ext.memory_manager import memory
        memory.remember(text, category=category)
        return {"status": "ok", "message": "已存储到长期记忆"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/memory/recall")
async def memory_recall(req: dict):
    """从长期记忆中语义检索"""
    query = req.get("query", "").strip()
    top_k = req.get("top_k", 5)
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")

    try:
        from taiji.agent_ext.memory_manager import memory
        results = memory.recall(query, top_k=top_k)
        return {"status": "ok", "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/agent/memory/working")
def memory_working():
    """获取工作记忆中的所有键值对"""
    try:
        from taiji.agent_ext.memory_manager import memory
        return {"status": "ok", "data": memory.working.get_all(), "keys": memory.working.list_keys()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/memory/working/set")
async def memory_working_set(req: dict):
    """设置工作记忆"""
    key = req.get("key", "").strip()
    value = req.get("value", "")
    if not key:
        raise HTTPException(status_code=400, detail="键名不能为空")
    try:
        from taiji.agent_ext.memory_manager import memory
        memory.set_working(key, value)
        return {"status": "ok", "message": f"已设置 {key}"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/agent/memory/longterm")
def memory_longterm_list(category: str = "", limit: int = 50):
    """列出长期记忆条目"""
    try:
        from taiji.agent_ext.memory_manager import memory
        entries = memory.long_term.list_entries(category=category or None, limit=limit)
        return {"status": "ok", "entries": entries, "count": memory.long_term.count()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/memory/clear")
async def memory_clear(req: dict = {}):
    """清除记忆"""
    scope = req.get("scope", "all")  # all / short_term / working / longterm
    try:
        from taiji.agent_ext.memory_manager import memory
        if scope == "all":
            memory.clear_all()
        elif scope == "short_term":
            memory.short_term.clear()
        elif scope == "working":
            memory.working.clear()
        elif scope == "longterm":
            memory.long_term.clear()
        else:
            return {"status": "error", "message": f"未知范围: {scope}"}
        return {"status": "ok", "message": f"已清除 {scope} 记忆"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}