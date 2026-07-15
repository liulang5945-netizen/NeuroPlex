"""
系统操作 API 路由（精简版）
保留：硬件检测、重启系统、路径选择对话框、打开文件夹

已拆分到独立文件的功能：
- routes_settings.py  → 设置管理、内存状态、模型信息
- routes_update.py    → 版本检查、更新安装、热更新补丁
- routes_model_switch.py → 模型热切换、发布状态重置
"""
import logging
import os
import sys
import threading

from fastapi import APIRouter, HTTPException, Request

from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.System")
router = APIRouter()


# ======================== 硬件检测 ========================

@router.get("/api/system/hardware")
def get_system_hardware():
    """检测系统硬件配置（原生态极）"""
    try:
        import psutil
        import torch

        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024 ** 3)
        avail_ram_gb = ram.available / (1024 ** 3)

        cpu_info = f"{cpu_count} 核"
        gpu_info = "无"
        vram_info = "N/A"
        gpu_backends = ["cpu"]

        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_info = f"{vram_bytes / (1024**3):.1f} GB"
            gpu_backends = ["cuda"]
        elif hasattr(torch, 'directml') and torch.directml.is_available():
            gpu_info = "AMD GPU (DirectML)"
            vram_info = f"{ram_gb * 0.5:.0f} GB (共享)"
            gpu_backends = ["directml"]

        return {
            "status": "ok",
            "cpu": cpu_info,
            "ram": f"{ram_gb:.0f} GB",
            "gpu": gpu_info,
            "vram": vram_info,
            "recommend": "原生 Taiji 模型（基于 ModelSelf）",
            "gpu_backends": gpu_backends,
            "available_memory_gb": round(avail_ram_gb, 1),
        }
    except Exception as e:
        logger.error(f"硬件检测失败: {e}")
        return {
            "status": "error",
            "message": f"硬件检测失败: {str(e)}",
            "cpu": "", "ram": "", "gpu": "", "vram": "",
            "recommend": "原生 Taiji 模型",
        }


# ======================== 系统操作 ========================

@router.post("/api/system/restart")
def restart_system(request: Request):
    """接收前端发来的重启指令 — 需要认证（认证启用时）"""
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

    import subprocess
    import time

    def _restart():
        try:
            time.sleep(2)
            env = os.environ.copy()
            env.pop("_MEIPASS2", None)
            if getattr(sys, 'frozen', False):
                path_list = env.get("PATH", "").split(os.pathsep)
                env["PATH"] = os.pathsep.join([p for p in path_list if p != getattr(sys, '_MEIPASS', '')])
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
            new_process = subprocess.Popen([sys.executable] + sys.argv, env=env, creationflags=creationflags)
            logger.info(f"新进程已创建: PID={new_process.pid}")
        except Exception as e:
            logger.error(f"无法创建新进程: {e}")
        finally:
            os._exit(0)

    t = threading.Thread(target=_restart, daemon=True)
    t.start()
    return {"status": "ok", "message": "正在重启..."}


# ======================== 路径与文件选择 ========================

@router.post("/api/system/validate_path")
def validate_path(req: dict):
    """验证路径是否存在且类型正确"""
    path = req.get("path", "")
    path_type = req.get("type", "folder")
    if not os.path.exists(path):
        return {"status": "error", "message": "路径不存在"}
    if path_type == "folder" and not os.path.isdir(path):
        return {"status": "error", "message": "所选路径不是文件夹"}
    if path_type == "file" and not os.path.isfile(path):
        return {"status": "error", "message": "所选路径不是文件"}
    return {"status": "ok"}


@router.get("/api/system/select_folder")
def select_folder():
    """打开原生文件夹选择框"""
    if sys.platform != "win32":
        return {"status": "error", "message": "目前仅支持 Windows 系统的原生存档对话框"}

    import ctypes
    from ctypes import wintypes

    try:
        thread_id = threading.current_thread().ident

        class BROWSEINFO(ctypes.Structure):
            _fields_ = [
                ("hwndOwner", wintypes.HWND),
                ("pidlRoot", ctypes.c_void_p),
                ("pszDisplayName", wintypes.LPWSTR),
                ("lpszTitle", wintypes.LPCWSTR),
                ("ulFlags", wintypes.UINT),
                ("lpfn", ctypes.c_void_p),
                ("lParam", wintypes.LPARAM),
                ("iImage", ctypes.c_int)
            ]

        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32

        shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
        shell32.SHGetPathFromIDListW.restype = wintypes.BOOL
        shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR]
        ole32.CoTaskMemFree.restype = None
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]

        hr = ole32.CoInitialize(None)
        need_cleanup = (hr == 0)
        if hr == 0x80010106:
            logger.warning(f"COM 线程模式冲突 (thread {thread_id})")

        display_name = ctypes.create_unicode_buffer(260)
        bi = BROWSEINFO()
        bi.hwndOwner = None
        bi.pidlRoot = None
        bi.pszDisplayName = ctypes.cast(display_name, wintypes.LPWSTR)
        bi.lpszTitle = "请选择模型文件夹（需包含 config.json，支持 HuggingFace 缓存目录）"
        bi.ulFlags = 0x00000040 | 0x00000010

        pidl = shell32.SHBrowseForFolderW(ctypes.byref(bi))
        path = ""

        if pidl:
            path_buffer = ctypes.create_unicode_buffer(260)
            if shell32.SHGetPathFromIDListW(pidl, path_buffer):
                path = path_buffer.value
            ole32.CoTaskMemFree(pidl)

        if need_cleanup:
            ole32.CoUninitialize()

        if path:
            return {"status": "ok", "path": path}
        else:
            return {"status": "cancel"}
    except Exception as e:
        logger.error(f"选择文件夹失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/system/select_file")
def select_file():
    """打开原生文件选择框"""
    if sys.platform != "win32":
        return {"status": "error", "message": "目前仅支持 Windows 系统的原生对话框"}

    import ctypes
    from ctypes import wintypes

    try:
        class OPENFILENAME(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE), ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD), ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wintypes.LPCWSTR)
            ]

        comdlg32 = ctypes.windll.comdlg32
        file_buffer = ctypes.create_unicode_buffer(1024)

        ofn = OPENFILENAME()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
        ofn.hwndOwner = None
        ofn.lpstrFilter = "所有文件 (*.*)\0*.*\0JSONL 数据集 (*.jsonl)\0*.jsonl\0"
        ofn.lpstrFile = ctypes.cast(file_buffer, wintypes.LPWSTR)
        ofn.nMaxFile = 1024
        ofn.lpstrTitle = "请选择本地文件"
        ofn.Flags = 0x00080000 | 0x00001000 | 0x00000008

        if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return {"status": "ok", "path": file_buffer.value}
        else:
            return {"status": "cancel"}
    except Exception as e:
        logger.error(f"选择文件失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/open_folder")
def open_folder(req: dict):
    """在系统资源管理器中打开指定目标文件夹"""
    target = req.get("target", "workspace")
    path = get_external_path("agent_workspace" if target == "workspace" else "data")
    os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}