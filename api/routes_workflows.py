"""
工作流 API 路由
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from taiji.agent_ext.workflow_engine import WorkflowDefinition, WorkflowEngine, WorkflowStore

logger = logging.getLogger("ApiServer.Workflows")
router = APIRouter()
store = WorkflowStore()
engine = WorkflowEngine()


@router.get("/api/workflows")
async def list_workflows():
    """列出所有工作流"""
    return {"status": "success", "workflows": store.list_all()}


@router.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """获取工作流详情"""
    wf = store.load(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="工作流不存在")
    from dataclasses import asdict
    return {"status": "success", "workflow": asdict(wf)}


@router.post("/api/workflows")
async def create_workflow(data: dict):
    """创建工作流"""
    try:
        wf = WorkflowDefinition(**{k: v for k, v in data.items() if k in WorkflowDefinition.__dataclass_fields__})
        store.save(wf)
        return {"status": "success", "id": wf.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """删除工作流"""
    store.delete(workflow_id)
    return {"status": "success"}


@router.post("/api/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str):
    """执行工作流"""
    wf = store.load(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="工作流不存在")
    try:
        result = engine.execute(wf)
        from dataclasses import asdict
        return {"status": "success", "result": asdict(result)}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")