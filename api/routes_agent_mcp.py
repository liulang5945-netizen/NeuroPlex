"""
MCP 服务器管理 API 路由
从 routes_agent.py 拆分：MCP 市场、安装/卸载、启动/停止、工具列表
"""
import json
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("ApiServer.Agent.MCP")
router = APIRouter()


@router.get("/api/mcp/marketplace")
def mcp_marketplace(category: str = "", keyword: str = ""):
    """浏览 MCP 服务器市场"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        return {"status": "ok", **mcp_manager.get_marketplace(category=category, keyword=keyword)}
    except Exception as e:
        logger.error(f"获取 MCP 市场数据失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/marketplace/refresh")
def mcp_marketplace_refresh():
    """从远程源刷新 MCP 市场数据"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.refresh_marketplace()
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"刷新 MCP 市场失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/mcp/marketplace/{server_id}")
def mcp_server_detail(server_id: str):
    """获取 MCP 服务器详情"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        detail = mcp_manager.get_server_detail(server_id)
        if detail:
            return {"status": "ok", **detail}
        return {"status": "error", "message": f"服务器 '{server_id}' 不存在"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/install")
async def mcp_install(req: dict):
    """安装 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.install_server(server_id)
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/uninstall")
async def mcp_uninstall(req: dict):
    """卸载 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.uninstall_server(server_id)
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/start")
async def mcp_start(req: dict):
    """启动 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    workspace = req.get("workspace", "")
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.start_server(server_id, workspace_path=workspace)
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/stop")
async def mcp_stop(req: dict):
    """停止 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.stop_server(server_id)
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/restart")
async def mcp_restart(req: dict):
    """重启 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    workspace = req.get("workspace", "")
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.restart_server(server_id, workspace_path=workspace)
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/mcp/installed")
def mcp_installed():
    """获取已安装的 MCP 服务器列表"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        return {"status": "ok", "servers": mcp_manager.get_installed_servers()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/mcp/status")
def mcp_status():
    """获取 MCP 管理器状态"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        return {"status": "ok", **mcp_manager.get_status()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/mcp/tools")
def mcp_tools():
    """获取所有 MCP 工具"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        return {"status": "ok", "tools": mcp_manager.get_all_mcp_tools()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/plugins/marketplace")
def plugin_marketplace(category: str = "", keyword: str = ""):
    """浏览 Agent 插件市场"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        return {"status": "ok", **mcp_manager.get_plugin_marketplace(category=category, keyword=keyword)}
    except Exception as e:
        logger.error(f"获取插件市场数据失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/plugins/marketplace/refresh")
def plugin_marketplace_refresh():
    """从远程源刷新 Agent 插件市场"""
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.refresh_plugin_marketplace()
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"刷新插件市场失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/mcp/add_custom")
async def mcp_add_custom(req: dict):
    """添加自定义 MCP 服务器"""
    server_id = req.get("server_id", "").strip()
    name = req.get("name", server_id)
    command = req.get("command", "cmd")
    args = req.get("args", [])
    npm_package = req.get("npm_package", "")
    description = req.get("description", "")
    env = req.get("env", {})
    if not server_id:
        raise HTTPException(status_code=400, detail="server_id 不能为空")
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager
        result = mcp_manager.add_custom_server(
            server_id, name, command, args,
            npm_package=npm_package, description=description, env=env,
        )
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}