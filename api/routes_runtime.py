"""Runtime status API.

Endpoints:
- GET /api/runtime/bootstrap  — public, minimal info for unauthenticated clients
- GET /api/runtime/status     — full status, requires auth if auth is enabled
"""
from fastapi import APIRouter, Request

from taiji.services.runtime_service import get_runtime_status, get_bootstrap_status
from api.models_runtime import RuntimeStatusPayload, BootstrapPayload

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("/bootstrap", response_model=BootstrapPayload)
async def runtime_bootstrap():
    """Public endpoint — no auth required.

    Returns minimal info so the client shell can decide whether to
    show a login screen or proceed to load the full runtime status.
    """
    return get_bootstrap_status()


@router.get("/status", response_model=RuntimeStatusPayload)
async def runtime_status(request: Request):
    """Full runtime status payload.

    If auth is enabled, the client must send a valid Bearer token.
    The auth field in the response reflects the token validity.
    """
    return get_runtime_status(request.headers.get("Authorization", ""))
