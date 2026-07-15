"""
Runtime status payload schemas.

These models define the contract between the backend runtime service
and the frontend runtimeStore. Keep them stable — the client shell
depends on this shape.
"""
from pydantic import BaseModel
from typing import Optional


class HealthPayload(BaseModel):
    state: str = "loading"  # connected | loading | downloading | error
    message: str = ""
    model_loaded: bool = False
    model_name: str = ""
    is_taiji: bool = False
    startup_complete: bool = False
    startup_error: str = ""


class MemoryPayload(BaseModel):
    status: str = "unknown"
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_pct: float = 0.0


class AuthPayload(BaseModel):
    enabled: bool = False
    authenticated: bool = True
    token_valid: bool = False
    username: str = ""
    has_password: bool = False


class LifeNeedsPayload(BaseModel):
    hunger: float = 50.0
    fatigue: float = 50.0
    curiosity: float = 50.0
    social: float = 50.0


class LifePayload(BaseModel):
    status: str = "ok"
    is_running: bool = False
    needs: LifeNeedsPayload = LifeNeedsPayload()
    total_interactions: int = 0
    uptime_seconds: int = 0


class ToolInfo(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}
    source: str = ""
    source_id: str = ""
    category: str = ""
    enabled: bool = True


class ToolsPayload(BaseModel):
    status: str = "ok"  # ok | partial | error
    tools: list[ToolInfo] = []
    count: int = 0
    error: str = ""


class TrainingPayload(BaseModel):
    is_training: bool = False
    publishing: bool = False
    pause_requested: bool = False
    stop_requested: bool = False


class RuntimeStatusPayload(BaseModel):
    """The single trusted status payload for the client shell."""
    status: str = "ok"
    timestamp: int = 0
    health: HealthPayload = HealthPayload()
    memory: MemoryPayload = MemoryPayload()
    auth: AuthPayload = AuthPayload()
    life: LifePayload = LifePayload()
    tools: ToolsPayload = ToolsPayload()
    training: TrainingPayload = TrainingPayload()


class BootstrapPayload(BaseModel):
    """Public endpoint — no auth required. Tells the client what to do next."""
    alive: bool = True
    auth_enabled: bool = False
    need_login: bool = False
    startup_complete: bool = False
    startup_error: str = ""
