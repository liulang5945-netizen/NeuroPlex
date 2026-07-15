"""
认证 API 路由
=============
提供登录、登出、密码管理、Token 刷新等认证相关端点
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from taiji.services import auth_service

logger = logging.getLogger("ApiServer.Auth")
router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class EnableAuthRequest(BaseModel):
    username: str
    password: str


@router.post("/api/auth/login")
async def login(req: LoginRequest):
    """用户登录，返回 JWT Token"""
    token = auth_service.login(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"status": "success", "token": token, "username": req.username}


@router.post("/api/auth/change_password")
async def change_password(req: ChangePasswordRequest):
    """修改密码"""
    ok = auth_service.change_password(req.old_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="原密码错误")
    return {"status": "success", "message": "密码已修改"}


@router.get("/api/auth/status")
async def auth_status():
    """获取认证状态"""
    return {"status": "success", **auth_service.get_status()}


@router.post("/api/auth/enable")
async def enable_auth(req: EnableAuthRequest, request: Request):
    """
    启用认证（设置用户名和密码）
    安全策略：
    - 首次运行（认证未启用）时可直接调用
    - 认证已启用后需要提供有效的管理员 Token
    """
    from taiji.core.security import AuthManager
    auth = AuthManager()

    # 如果认证已启用，需要验证现有 Token
    if auth.enabled:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="认证已启用，请提供有效的管理员 Token")
        token = auth_header[7:]
        payload = auth.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Token 无效或已过期")

    auth_service.enable_auth(req.username, req.password)
    return {"status": "success", "message": "认证已启用"}


@router.post("/api/auth/disable")
async def disable_auth(request: Request):
    """禁用认证 — 需要有效的管理员 Token"""
    from taiji.core.security import AuthManager
    auth = AuthManager()

    # 认证未启用时，禁止通过此端点操作（防止竞态条件）
    if not auth.enabled:
        raise HTTPException(status_code=400, detail="认证未启用")

    # 验证现有 Token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证 Token")
    token = auth_header[7:]
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    auth_service.disable_auth()
    return {"status": "success", "message": "认证已禁用"}


@router.get("/api/auth/audit")
async def get_audit_logs(limit: int = 50, days: int = 7):
    """获取审计日志"""
    events = auth_service.get_audit_logs(limit=limit, days=days)
    return {"status": "success", "events": events, "count": len(events)}


@router.post("/api/auth/refresh")
async def refresh_token(request: Request):
    """刷新 Token"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Token")
    token = auth_header[7:]
    new_token = auth_service.refresh_token(token)
    if not new_token:
        raise HTTPException(status_code=400, detail="Token 不需要刷新或已失效")
    return {"status": "success", "token": new_token}
