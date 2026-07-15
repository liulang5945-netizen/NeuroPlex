"""Workspace management routes."""

import logging
import os
import shutil

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from taiji.core.utils import get_external_path
from taiji.services.settings_service import get_setting, update_settings

from .models import CodeRunRequest, CreateProjectRequest, FileSaveRequest

logger = logging.getLogger("ApiServer.Agent.Workspace")
router = APIRouter()


def _require_admin_auth(request: Request):
    """Validate admin auth for sensitive operations (e.g. pip install)."""
    from taiji.core.security import AuthManager
    auth = AuthManager()

    if not auth.enabled:
        raise HTTPException(
            status_code=403,
            detail="此操作需要启用认证。请先启用认证后再执行此操作。",
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证 Token")

    token = auth_header[7:]
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    return payload


def _get_workspace_dir() -> str:
    """Return the current workspace directory."""
    custom_path = get_setting("workspace_path", "")
    if custom_path and os.path.isdir(custom_path):
        return os.path.abspath(custom_path)
    return get_external_path("agent_workspace")


def _resolve_workspace_path(name: str) -> tuple[str, str]:
    """Resolve a workspace-relative path and validate it stays inside the workspace."""
    ws_dir = os.path.abspath(_get_workspace_dir())
    file_path = os.path.abspath(os.path.join(ws_dir, name))
    return ws_dir, file_path


@router.get("/api/workspace/path")
def get_workspace_path():
    """Get the active workspace path."""
    return {"status": "ok", "path": os.path.abspath(_get_workspace_dir())}


@router.post("/api/workspace/path")
def set_workspace_path(req: dict, request: Request):
    """
    Update the active workspace path.

    安全策略：
    - 路径必须是已存在的目录
    - 路径不能是系统敏感目录（如根目录、/etc、/bin 等）
    - 建议限制在用户主目录或项目目录内
    """
    new_path = req.get("path", "").strip()
    if not new_path:
        raise HTTPException(status_code=400, detail="路径不能为空")
    if not os.path.isdir(new_path):
        raise HTTPException(status_code=400, detail=f"路径不存在或不是目录: {new_path}")

    normalized_path = os.path.abspath(new_path)

    # 安全检查：禁止设置为系统敏感目录
    forbidden_paths = [
        os.path.abspath("/"),
        os.path.abspath("C:\\"),
        os.path.abspath("/etc"),
        os.path.abspath("/bin"),
        os.path.abspath("/usr"),
        os.path.abspath("/var"),
        os.path.abspath("/root"),
        os.path.abspath("/home"),
        os.path.abspath("C:\\Windows"),
        os.path.abspath("C:\\Program Files"),
    ]

    if normalized_path in forbidden_paths:
        raise HTTPException(
            status_code=403,
            detail="不允许将工作区设置为系统敏感目录"
        )

    # 安全检查：路径不能是根目录的直接子目录
    parent_dir = os.path.dirname(normalized_path)
    if parent_dir in [os.path.abspath("/"), os.path.abspath("C:\\")]:
        raise HTTPException(
            status_code=403,
            detail="不允许将工作区设置为根目录的直接子目录"
        )

    # 需要认证才能更改工作区路径
    from taiji.core.security import AuthManager

    auth = AuthManager()
    if auth.enabled:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="缺少认证 Token")
        token = auth_header[7:]
        payload = auth.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Token 无效或已过期")

    update_settings({"workspace_path": normalized_path})
    logger.info(f"工作区路径已更新为: {normalized_path}")
    return {"status": "ok", "path": normalized_path}


@router.get("/api/workspace/files")
def list_workspace_files():
    """List files in the workspace root."""
    ws_dir = _get_workspace_dir()
    os.makedirs(ws_dir, exist_ok=True)
    files = [f for f in os.listdir(ws_dir) if os.path.isfile(os.path.join(ws_dir, f))]
    return {"files": files}


@router.get("/api/workspace/tree")
def list_workspace_tree():
    """Return the recursive workspace tree."""
    ws_dir = _get_workspace_dir()
    os.makedirs(ws_dir, exist_ok=True)

    def build_tree(dir_path: str) -> list:
        entries = []
        try:
            items = sorted(os.listdir(dir_path))
            for name in items:
                item_path = os.path.join(dir_path, name)
                rel_path = os.path.relpath(item_path, ws_dir)
                if os.path.isdir(item_path):
                    entries.append(
                        {
                            "name": name,
                            "path": rel_path,
                            "type": "directory",
                            "children": build_tree(item_path),
                        }
                    )
                else:
                    try:
                        size = os.path.getsize(item_path)
                    except Exception:
                        size = 0
                    entries.append(
                        {
                            "name": name,
                            "path": rel_path,
                            "type": "file",
                            "size": size,
                        }
                    )
        except Exception:
            pass
        return entries

    return {"tree": build_tree(ws_dir)}


@router.get("/api/workspace/file")
def get_workspace_file(name: str):
    """Read a file from the workspace."""
    ws_dir, file_path = _resolve_workspace_path(name)
    if not (file_path == ws_dir or file_path.startswith(ws_dir + os.sep)):
        return {"content": "", "error": "路径不安全"}
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return {"content": handle.read(), "size": os.path.getsize(file_path)}
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as handle:
                    return {"content": handle.read(), "size": os.path.getsize(file_path)}
            except Exception:
                return {"content": "(二进制文件)", "size": os.path.getsize(file_path)}
        except Exception as exc:
            return {"content": "", "error": str(exc)}
    return {"content": "", "error": "文件不存在"}


@router.post("/api/workspace/file")
def save_workspace_file(req: FileSaveRequest):
    """Write a file into the workspace."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=422, detail="文件名不能为空")

    ws_dir = os.path.abspath(_get_workspace_dir())
    os.makedirs(ws_dir, exist_ok=True)
    safe_path = os.path.abspath(os.path.join(ws_dir, req.name))
    if not (safe_path == ws_dir or safe_path.startswith(ws_dir + os.sep)):
        return {"status": "error", "message": "路径不安全"}

    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    try:
        with open(safe_path, "w", encoding="utf-8") as handle:
            handle.write(req.content)
    except IsADirectoryError:
        raise HTTPException(status_code=422, detail=f"'{req.name}' 是目录不是文件")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"无权限写入: {exc}")

    return {"status": "ok", "path": os.path.relpath(safe_path, ws_dir)}


@router.post("/api/workspace/run")
def run_workspace_code(req: CodeRunRequest):
    """Run Python code in the workspace sandbox."""
    try:
        from taiji.agent_ext.sandbox_executor import execute_python_with_files

        result = execute_python_with_files(req.code)
        return {
            "output": result.get("output", ""),
            "files_created": result.get("files_created", []),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }
    except Exception as exc:
        logger.error(f"Code execution failed: {exc}")
        return {"output": "", "success": False, "error": "内部错误，请查看日志"}


@router.post("/api/workspace/create_project")
async def create_project(req: CreateProjectRequest):
    """Create a project scaffold."""
    from taiji.agent_ext.agent import create_project as agent_create_project

    try:
        os.makedirs(_get_workspace_dir(), exist_ok=True)
        result = agent_create_project(f"{req.type} | project_{req.type}")
        return {"status": "ok", "message": result}
    except Exception as exc:
        logger.error(f"Request failed: {exc}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.delete("/api/workspace/delete/{name:path}")
def delete_workspace_item(name: str):
    """Delete a file or directory inside the workspace."""
    try:
        ws_dir, item_path = _resolve_workspace_path(name)
        if item_path == ws_dir:
            raise HTTPException(status_code=403, detail="禁止删除工作区根目录")
        if not item_path.startswith(ws_dir + os.sep):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Request failed: {exc}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.post("/api/agent/analyze_code")
async def analyze_code_api(req: CodeRunRequest):
    """Analyze code through the agent helper."""
    from taiji.agent_ext.agent import analyze_code

    try:
        return {"result": analyze_code(req.code)}
    except Exception as exc:
        logger.error(f"Analyze failed: {exc}")
        return {"result": "分析失败"}


@router.post("/api/agent/install_dependency")
async def install_dependency_api(req: CodeRunRequest, request: Request):
    """Install a dependency through the agent helper (admin only)."""
    _require_admin_auth(request)
    from taiji.agent_ext.agent import install_dependency

    try:
        return {"result": install_dependency(req.code)}
    except Exception as exc:
        logger.error(f"install_dependency failed: {exc}")
        return {"result": "安装失败，请查看日志"}


@router.get("/api/agent/plans")
def list_plans_api():
    """List saved agent plans."""
    from taiji.agent_ext.agent import list_plans

    try:
        return {"plans": list_plans("")}
    except Exception as exc:
        logger.error(f"List plans failed: {exc}")
        return {"plans": "获取失败"}


@router.get("/api/agent/context")
def load_context_api(key: str = ""):
    """Load saved agent context."""
    from taiji.agent_ext.agent import load_context

    try:
        return {"context": load_context(key)}
    except Exception as exc:
        logger.error(f"Load context failed: {exc}")
        return {"context": "读取失败"}


@router.post("/api/agent/save_context")
async def save_context_api(req: CodeRunRequest):
    """Save agent context."""
    from taiji.agent_ext.agent import save_context

    try:
        return {"result": save_context(req.code)}
    except Exception as exc:
        logger.error(f"Save context failed: {exc}")
        return {"result": "保存失败"}


@router.post("/api/plugins/upload")
async def upload_plugin(file: UploadFile = File(...)):
    """Upload a plugin archive into the plugins directory."""
    try:
        plugins_dir = get_external_path("plugins")
        os.makedirs(plugins_dir, exist_ok=True)
        file_path = os.path.join(plugins_dir, os.path.basename(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "message": f"插件 `{file.filename}` 已成功热安装，下一条对话将直接生效。",
        }
    except Exception as exc:
        logger.error(f"插件安装失败: {exc}")
        logger.error(f"Request failed: {exc}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.get("/api/workspace/quick_paths")
def get_quick_paths():
    """Return common filesystem locations for quick selection."""
    import platform

    home = os.path.expanduser("~")
    paths = []

    desktop = os.path.join(home, "Desktop")
    documents = os.path.join(home, "Documents")
    downloads = os.path.join(home, "Downloads")

    if os.path.isdir(desktop):
        paths.append({"label": "桌面", "path": desktop})
    if os.path.isdir(documents):
        paths.append({"label": "文档", "path": documents})
    if os.path.isdir(downloads):
        paths.append({"label": "下载", "path": downloads})

    if platform.system() == "Windows":
        for letter in "CDEFG":
            drive = f"{letter}:\\"
            if os.path.isdir(drive):
                paths.append({"label": f"{letter}: 盘", "path": drive})

    paths.append({"label": "用户主目录", "path": home})
    return {"paths": paths}


@router.get("/api/network/diagnose")
def network_diagnose():
    """Network diagnostics (native Taiji — no remote model downloads needed)."""
    return {"status": "ok", "diagnosis": {"message": "原生态极运行于本地，无需远程模型下载诊断"}}
