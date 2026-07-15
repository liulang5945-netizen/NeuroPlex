"""
系统更新、补丁管理 API 路由
从 routes_system.py 拆分：版本检查、更新安装、热更新补丁
"""
import json
import logging
import os
import shutil
import sys
import threading

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Update")
router = APIRouter()


def _require_admin_auth(request: Request):
    """
    验证管理员权限（用于敏感的更新/补丁操作）

    安全策略：
    - 认证启用时：必须提供有效的管理员 Token
    - 认证未启用时：拒绝访问（更新操作必须在认证保护下进行）
    """
    from taiji.core.security import AuthManager
    auth = AuthManager()

    if not auth.enabled:
        raise HTTPException(
            status_code=403,
            detail="更新操作需要启用认证。请先启用认证后再执行此操作。"
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证 Token")

    token = auth_header[7:]
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    return payload


@router.get("/api/system/version")
def get_system_version():
    """获取当前版本信息"""
    try:
        from build_scripts.updater import VersionManager
        vm = VersionManager()
        v = vm.current
        return {
            "status": "ok",
            "version": v.version,
            "build_date": v.build_date,
            "update_url": v.update_url,
            "changelog": v.changelog,
            "notes": v.notes,
        }
    except Exception as e:
        logger.warning(f"获取版本信息失败: {e}")
        return {
            "status": "ok",
            "version": "0.0.0",
            "build_date": "",
            "update_url": "",
            "changelog": "",
            "notes": "开发模式（无版本信息）",
        }


@router.post("/api/system/check_update")
def check_update(req: dict = {}):
    """检查远程更新"""
    try:
        from build_scripts.updater import VersionManager, UpdateChecker
        repo = req.get("repo", "")
        vm = VersionManager()
        checker = UpdateChecker(vm)
        if repo:
            if repo.startswith("http"):
                latest = checker.check_custom_url(repo)
            else:
                latest = checker.check_github_release(repo)
        else:
            url = vm.current.update_url
            if url.startswith("http"):
                latest = checker.check_custom_url(url)
        if latest and checker.has_update(latest):
            return {
                "status": "ok",
                "has_update": True,
                "version": latest.version,
                "build_date": latest.build_date,
                "update_url": latest.update_url,
                "changelog": latest.changelog,
                "notes": latest.notes,
            }
        else:
            return {
                "status": "ok",
                "has_update": False,
                "version": latest.version if latest else vm.current.version,
                "message": "当前已是最新版本",
            }
    except Exception as e:
        logger.error(f"检查更新失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/apply_update")
def apply_update(request: Request, req: dict = {}):
    """应用更新（从 URL 下载并安装更新包）- 需要管理员认证"""
    _require_admin_auth(request)
    try:
        from build_scripts.updater import UpdatePackageInstaller
        url = req.get("url", "")
        if not url:
            return {"status": "error", "message": "未提供更新包 URL"}

        def _install():
            installer = UpdatePackageInstaller()
            installer.install_from_url(url)

        t = threading.Thread(target=_install, daemon=True)
        t.start()
        return {"status": "ok", "message": "更新已开始下载安装，请稍候..."}
    except Exception as e:
        logger.error(f"应用更新失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/upload_update")
async def upload_update(request: Request, file: UploadFile = File(...)):
    """上传并安装更新包（手动模式）- 需要管理员认证"""
    _require_admin_auth(request)
    import zipfile
    try:
        from build_scripts.updater import UpdatePackageInstaller
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                names = zf.namelist()
                has_version = "version.json" in names
                has_code = any(n.startswith("update_code/") for n in names)
                has_ui = any(n.startswith("update_frontend/") for n in names)
                if not has_version and not has_code and not has_ui:
                    os.remove(tmp_path)
                    return {"status": "error", "message": "无效的更新包"}
        except zipfile.BadZipFile:
            os.remove(tmp_path)
            return {"status": "error", "message": "无效的 ZIP 文件"}

        def _install():
            try:
                installer = UpdatePackageInstaller()
                installer.install_from_zip(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        t = threading.Thread(target=_install, daemon=True)
        t.start()
        return {"status": "success", "message": "更新包已上传，正在安装..."}
    except Exception as e:
        logger.error(f"上传更新包失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/reload_modules")
def reload_modules(req: dict, request: Request):
    """热重载 Python 模块 - 需要管理员认证"""
    _require_admin_auth(request)
    try:
        from build_scripts.updater import ModuleHotReloader
        reloader = ModuleHotReloader()
        modules = req.get("modules", [])
        if modules:
            results = {}
            for mod in modules:
                results[mod] = reloader.reload_module(mod)
        else:
            results = reloader.reload_all_patches()
        success = sum(1 for v in results.values() if v)
        fail = sum(1 for v in results.values() if not v)
        return {
            "status": "ok",
            "results": {k: "✅" if v else "❌" for k, v in results.items()},
            "summary": f"{success} 个成功, {fail} 个失败",
        }
    except Exception as e:
        logger.error(f"重载模块失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/set_update_url")
def set_update_url(req: dict, request: Request):
    """设置更新检查 URL - 需要管理员认证"""
    _require_admin_auth(request)
    try:
        from build_scripts.updater import VersionManager
        url = req.get("url", "")
        if not url:
            return {"status": "error", "message": "URL 不能为空"}
        vm = VersionManager()
        vm.set_update_url(url)
        return {"status": "ok", "message": f"更新地址已设置为: {url}"}
    except Exception as e:
        logger.error(f"设置更新地址失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/system/patches")
def list_patches():
    """列出当前可用的 Python 补丁模块（支持子目录结构）"""
    try:
        from build_scripts.updater import ModuleHotReloader
        reloader = ModuleHotReloader()
        patches = reloader.get_available_patches()
        patch_details = []
        for mod_name in patches:
            patch_path = reloader._find_patch_path(mod_name)
            detail = {
                "module": mod_name,
                "path": os.path.relpath(patch_path, reloader.update_dir) if patch_path else "",
                "size": os.path.getsize(patch_path) if patch_path and os.path.exists(patch_path) else 0,
            }
            patch_details.append(detail)
        return {"status": "ok", "patches": patches, "patch_details": patch_details}
    except ImportError:
        logger.info("Patch manager unavailable; returning empty patch list")
        return {"status": "ok", "patches": [], "patch_details": []}
    except Exception as e:
        logger.error(f"列出补丁失败: {e}")
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/system/upload_patch")
async def upload_patch(request: Request, file: UploadFile = File(...), target_path: str = Form("")):
    """上传单个 Python 文件作为热更新补丁 - 需要管理员认证"""
    _require_admin_auth(request)
    try:
        filename = file.filename or "patch.py"
        if not filename.endswith(".py"):
            return {"status": "error", "message": "仅支持 .py 文件"}

        if target_path:
            rel_path = target_path.replace("\\", "/")
            # 安全检查：拒绝路径遍历
            if ".." in rel_path or rel_path.startswith("/"):
                return {"status": "error", "message": "路径不安全，拒绝写入"}
            if not rel_path.endswith(".py"):
                rel_path = rel_path + ".py"
        else:
            rel_path = filename

        update_dir = get_external_path("update_code")
        dst_file = os.path.abspath(os.path.join(update_dir, rel_path))
        # 安全检查：确保目标路径在 update_code 目录内
        abs_update_dir = os.path.abspath(update_dir)
        if not dst_file.startswith(abs_update_dir + os.sep):
            return {"status": "error", "message": "路径不安全，拒绝写入"}
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)

        with open(dst_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(dst_file)
        logger.info(f"补丁已上传: {rel_path} ({file_size} bytes)")

        module_name = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]
        reload_result = ""
        try:
            from build_scripts.updater import ModuleHotReloader
            reloader = ModuleHotReloader()
            if reloader.reload_module(module_name):
                reload_result = f"，模块 {module_name} 已热重载"
            else:
                reload_result = f"，模块 {module_name} 将在重启后生效"
        except Exception as e:
            reload_result = f"，热重载失败: {e}"

        return {
            "status": "success",
            "message": f"补丁 {rel_path} 已部署{reload_result}",
            "module": module_name,
            "size": file_size,
        }
    except Exception as e:
        logger.error(f"上传补丁失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.post("/api/system/upload_ui")
async def upload_ui_zip(request: Request, file: UploadFile = File(...)):
    """接收拖拽的 frontend.zip 并自动覆盖 update_frontend - 需要管理员认证"""
    _require_admin_auth(request)
    import zipfile
    try:
        ui_dir = get_external_path("update_frontend")
        temp_zip = get_external_path("temp_ui.zip")
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if os.path.exists(ui_dir):
            shutil.rmtree(ui_dir, ignore_errors=True)
        os.makedirs(ui_dir, exist_ok=True)
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # ZipSlip 防护：验证所有解压路径在目标目录内
            for info in zip_ref.infolist():
                extract_path = os.path.join(ui_dir, info.filename)
                abs_extract = os.path.abspath(extract_path)
                abs_ui = os.path.abspath(ui_dir)
                if not (abs_extract.startswith(abs_ui + os.sep) or abs_extract == abs_ui):
                    raise HTTPException(status_code=400, detail=f"ZIP 包含非法路径: {info.filename}")
            zip_ref.extractall(ui_dir)
        os.remove(temp_zip)
        return {"status": "success", "message": "界面补丁包部署完毕！2秒后将自动刷新界面立刻生效！"}
    except Exception as e:
        logger.error(f"UI更新失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")
