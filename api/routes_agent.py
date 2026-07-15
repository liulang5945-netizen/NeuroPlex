"""
Agent 核心功能 API 路由（精简版）
保留：工具列表、ReAct 推理引擎、工具注册表、多 Agent 协作

已拆分到独立文件的功能：
- routes_agent_workspace.py → 工作台文件操作、代码执行、项目脚手架、插件上传
- routes_agent_mcp.py      → MCP 服务器市场、安装/卸载、启动/停止
- routes_agent_memory.py   → 记忆系统（短期/工作/长期记忆）
"""
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from taiji.services.tool_service import list_tools, get_registry_schemas, execute_tool
from taiji.services import agent_service

logger = logging.getLogger("ApiServer.Agent")
router = APIRouter()


# ======================== Agent 工具列表 ========================

@router.get("/api/agent/tools")
def list_agent_tools():
    """列出已加载的 Agent 工具。"""
    return list_tools()


# ======================== ReAct 推理引擎 ========================

@router.post("/api/agent/react")
async def react_task(req: dict):
    """使用 ReAct 引擎执行自主推理任务"""
    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    max_steps = req.get("max_steps", 15)

    try:
        return agent_service.run_react_task(task, max_steps)
    except Exception as e:
        logger.error(f"ReAct 任务执行失败: {e}")
        return {"status": "error", "message": "Agent 任务执行失败，请查看服务器日志获取详情"}


@router.post("/api/agent/react/stream")
async def react_task_stream(req: dict):
    """ReAct 流式推理（SSE）"""
    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    max_steps = req.get("max_steps", 15)

    async def event_generator():
        try:
            for event in agent_service.run_react_stream(task, max_steps):
                event_type = event.get("type", "unknown")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/agent/cancel")
async def cancel_agent():
    """取消当前正在执行的 ReAct 任务"""
    try:
        message = agent_service.cancel_active_task()
        return {"status": "ok", "message": message}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


# ======================== 工具注册表 ========================

@router.get("/api/agent/tools/registry")
def list_tool_registry():
    """列出工具注册表中所有已注册的工具（JSON Schema 格式）"""
    try:
        schemas = get_registry_schemas()
        return {"status": "ok", "tools": schemas, "count": len(schemas)}
    except Exception as e:
        logger.error(f"获取工具注册表失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/tools/execute")
async def run_tool(req: dict):
    """直接执行一个注册的工具"""
    tool_name = req.get("tool", "").strip()
    tool_args = req.get("args", {})
    if not tool_name:
        raise HTTPException(status_code=400, detail="工具名不能为空")

    try:
        result = execute_tool(tool_name, tool_args)
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


# ======================== 多 Agent 协作 ========================

@router.get("/api/agent/roles")
def list_roles():
    """列出所有可用的 Agent 角色"""
    try:
        return {"status": "ok", "roles": agent_service.list_roles(), "count": len(agent_service.list_roles())}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/agent/collaborate")
async def collaborate(req: dict):
    """多 Agent 协作执行任务"""
    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    try:
        result = agent_service.collaborate(task)
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"多 Agent 协作失败: {e}")
        return {"status": "error", "message": "多 Agent 协作失败，请查看服务器日志获取详情"}


@router.get("/api/agent/collaborate/tasks")
def list_collab_tasks():
    """列出所有协作任务"""
    try:
        return {"status": "ok", "tasks": agent_service.list_collab_tasks()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/agent/collaborate/messages")
def collab_messages(topic: str = "", limit: int = 50):
    """获取 Agent 间通信消息"""
    try:
        messages = agent_service.get_collab_messages(topic=topic, limit=limit)
        return {"status": "ok", "messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}
