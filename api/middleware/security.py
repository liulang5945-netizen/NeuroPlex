"""
安全中间件与输入校验
提供：JWT/API Key 鉴权、速率限制、输入参数校验
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import os
import time
from collections import defaultdict
from typing import Callable
import re

# ===== 速率限制 =====
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        # 清理过期请求记录
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] 
            if now - ts < self.window_seconds
        ]
        # 检查是否超限
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

# 全局速率限制器
limiter = RateLimiter(max_requests=100, window_seconds=60)

# ===== 鉴权 =====
class AuthValidator:
    """API 鉴权验证器"""
    
    _env_keys = os.environ.get("TAIJI_API_KEYS", "")
    VALID_API_KEYS = {
        k.strip() for k in _env_keys.split(",") if k.strip()
    } if _env_keys else set()
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """验证 API Key"""
        return api_key in AuthValidator.VALID_API_KEYS
    
    @staticmethod
    def get_client_id(request: Request) -> str:
        """获取客户端 ID（IP 或 API Key）"""
        if "X-API-Key" in request.headers:
            return f"api-{request.headers['X-API-Key']}"
        return f"ip-{request.client.host}"

# ===== 输入校验 =====
class InputValidator:
    """输入参数校验器"""
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """验证会话 ID（防止路径穿越）"""
        return bool(re.match(r'^[a-zA-Z0-9_\-]+$', session_id))
    
    @staticmethod
    def validate_prompt(prompt: str, max_length: int = 10000) -> bool:
        """验证提示文本"""
        if not prompt or not isinstance(prompt, str):
            return False
        return len(prompt) <= max_length
    
    @staticmethod
    def validate_file_size(size: int, max_size: int = 20 * 1024 * 1024) -> bool:
        """验证文件大小"""
        return 0 < size <= max_size

# ===== 中间件工厂 =====
def create_rate_limit_middleware(limiter_instance: RateLimiter):
    """创建速率限制中间件"""
    async def rate_limit_middleware(request: Request, call_next: Callable):
        client_id = AuthValidator.get_client_id(request)
        
        # 某些路径不限制
        if request.url.path.startswith(('/api/health', '/docs', '/openapi.json')):
            return await call_next(request)
        
        if not limiter_instance.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests, please retry later"}
            )
        
        return await call_next(request)
    
    return rate_limit_middleware

def create_auth_middleware():
    """创建鉴权中间件"""
    async def auth_middleware(request: Request, call_next: Callable):
        # 公开路由
        public_paths = {'/api/health', '/api/chat/stream', '/docs', '/openapi.json'}
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
        
        # 检查 API Key（可选，开发模式下）
        if "X-API-Key" in request.headers:
            api_key = request.headers["X-API-Key"]
            if not AuthValidator.validate_api_key(api_key):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid API Key"}
                )
        
        return await call_next(request)
    
    return auth_middleware