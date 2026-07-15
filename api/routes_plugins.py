"""
插件 API 路由
"""
import logging
from fastapi import APIRouter, HTTPException

from taiji.core.plugin_manager import PluginManager

logger = logging.getLogger("ApiServer.Plugins")
router = APIRouter()
pm = PluginManager()


@router.get("/api/plugins")
async def list_plugins():
    """列出所有插件"""
    return {"status": "success", "plugins": pm.list_plugins()}


@router.post("/api/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str):
    """启用插件"""
    try:
        pm.load_plugin(plugin_id)
        return {"status": "success", "message": f"插件 {plugin_id} 已启用"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/plugins/{plugin_id}/disable")
async def disable_plugin(plugin_id: str):
    """禁用插件"""
    pm.unload_plugin(plugin_id)
    return {"status": "success", "message": f"插件 {plugin_id} 已禁用"}


@router.delete("/api/plugins/{plugin_id}")
async def uninstall_plugin(plugin_id: str):
    """卸载插件"""
    pm.uninstall_plugin(plugin_id)
    return {"status": "success", "message": f"插件 {plugin_id} 已卸载"}


@router.post("/api/plugins/install")
async def install_plugin(data: dict):
    """安装插件"""
    source = data.get("source_path", "")
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source_path")
    try:
        plugin_id = pm.install_plugin(source)
        return {"status": "success", "plugin_id": plugin_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))