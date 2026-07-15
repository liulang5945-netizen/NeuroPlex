"""
[打包入口] PyInstaller 桌面客户端 — 进程内嵌入 FastAPI
======================================================

面向 PyInstaller 打包后的生产场景。功能：
- 依赖自检与自动安装（约 30 个核心包）
- 热更新系统（HotUpdateImporter，从 update_code/ 加载补丁）
- PyQt6 桌面 GUI（QWebEngineView + 系统托盘）
- 进程内 QThread 启动 uvicorn（不走子进程）

开发环境请用 desktop/main.py。详见 docs/ENTRYPOINTS.md

安全与架构改进：
1. 镜像源配置移至 config.py 集中管理，支持环境变量覆盖
2. 路径函数复用 config.py，消除重复定义
3. 后台任务使用超时控制，避免启动无限等待
4. 托盘右键菜单中文翻译优化
5. 首次启动自动检测并安装缺失的 PyQt6 + 后端核心依赖
"""
import os
import sys
import traceback
import threading
import subprocess

# 修复 PyQt6 QWebEngine GPU 渲染导致的 segfault（必须在导入 PyQt6 之前设置）
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
os.environ.setdefault('QT_OPENGL', 'software')

# ==========================================
# 将项目根目录加入 sys.path
# ==========================================
base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# ==========================================
# 镜像源配置
# ==========================================
PIP_MIRROR = os.environ.get(
    "TAIJI_PIP_INDEX",
    "https://mirrors.aliyun.com/pypi/simple/"
)

# ==========================================
# 全部依赖定义（名称 -> pip 包名）
# ==========================================
CORE_DEPENDENCIES = {
    # GUI 层
    "PyQt6": "PyQt6",
    "PyQt6.QtWebEngineWidgets": "PyQt6-WebEngine",
    # 后端 API 核心
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "starlette": "starlette",
    "pydantic": "pydantic",
    "multipart": "python-multipart",
    # 深度学习框架
    "torch": "torch>=2.0.0",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    # Agent / LLM
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langchain_experimental": "langchain-experimental",
    # RAG
    "sentence_transformers": "sentence-transformers",
    # 数据处理
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    # 工具
    "requests": "requests",
    "tqdm": "tqdm",
    "PyPDF2": "PyPDF2",
    "docx": "python-docx",
    "pdfminer": "pdfminer.six",
    "jieba": "jieba",
    "bs4": "beautifulsoup4",
    "duckduckgo_search": "duckduckgo-search",
}


def _check_module(import_name: str) -> bool:
    """检测单个 Python 模块是否可导入"""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _check_all_dependencies() -> tuple[list[str], list[str]]:
    """
    全面检测所有核心依赖。
    返回 (已安装列表, 缺失列表)。
    """
    missing_modules = []
    missing_packages = []
    for module_name, pip_name in CORE_DEPENDENCIES.items():
        if not _check_module(module_name):
            missing_modules.append(module_name)
            missing_packages.append(pip_name)
    return missing_modules, missing_packages


def _install_missing_packages(packages: list[str]) -> tuple[bool, str]:
    """通过 pip 安装缺失的包，返回 (是否成功, 错误消息)"""
    if not packages:
        return True, ""

    print(f"[DependencyCheck] 检测到 {len(packages)} 个缺失依赖，正在自动安装...")
    print(f"[DependencyCheck] 镜像: {PIP_MIRROR}")
    print(f"[DependencyCheck] 待安装: {', '.join(packages)}")

    # 分批安装（每次最多 5 个），避免单次命令行过长或网络中断
    batch_size = 5
    failed_packages = []

    for i in range(0, len(packages), batch_size):
        batch = packages[i : i + batch_size]
        print(f"[DependencyCheck] 安装批次 {i // batch_size + 1}: {', '.join(batch)}")
        try:
            # 构建 trusted-host
            mirror_host = PIP_MIRROR.split("//")[-1].split("/")[0].rstrip("/")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "-i", PIP_MIRROR,
                "--trusted-host", mirror_host,
            ] + batch
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            if result.returncode != 0:
                print(f"[DependencyCheck] FAIL 批次安装失败:\n{result.stderr[-500:]}")
                failed_packages.extend(batch)
            else:
                print(f"[DependencyCheck] OK 批次安装成功")
        except subprocess.TimeoutExpired:
            print("[DependencyCheck] FAIL 下载超时")
            failed_packages.extend(batch)
        except FileNotFoundError:
            return False, "未找到 pip 命令，请确保 Python 环境正常。"

    if failed_packages:
        return False, (
            f"以下依赖安装失败: {', '.join(failed_packages)}\n"
            f"请手动运行:\n"
            f"  {sys.executable} -m pip install {' '.join(failed_packages)}"
        )

    print("[DependencyCheck] OK 所有依赖安装成功！")
    return True, ""


def _show_error_dialog(title: str, message: str):
    """弹窗/控制台显示错误信息"""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
    except Exception:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{message}")
        print(f"{'='*60}\n")
        input("按 Enter 键退出...")


# ==========================================
# 启动前全面依赖自检
# ==========================================
print("[DependencyCheck] 正在检查核心依赖...")
missing_modules, missing_packages = _check_all_dependencies()

if missing_modules:
    print(f"[DependencyCheck] WARN 缺失 {len(missing_modules)} 个依赖")
    # 仅在未打包的开发环境中尝试自动安装
    # 打包后的 exe 依赖已内置，此时缺失说明 PyInstaller 漏打了 hiddenimports
    if not getattr(sys, 'frozen', False):
        ok, err = _install_missing_packages(missing_packages)
        if not ok:
            _show_error_dialog(
                "Taiji 依赖缺失",
                f"核心依赖安装失败。\n\n{err}\n\n"
                f"请手动运行:\n"
                f"  {sys.executable} -m pip install -r requirements.txt"
            )
            sys.exit(1)
        # 安装后重新检查
        still_missing, _ = _check_all_dependencies()
        if still_missing:
            _show_error_dialog(
                "Taiji 依赖缺失",
                f"以下依赖仍无法安装: {', '.join(still_missing)}\n\n"
                f"请手动运行:\n"
                f"  {sys.executable} -m pip install -r requirements.txt"
            )
            sys.exit(1)
    else:
        # 打包模式下缺失依赖，可能是 PyInstaller 漏打了
        # 尝试从 external_libs 目录加载补丁
        print(f"[DependencyCheck] WARN 打包模式下缺失: {', '.join(missing_modules)}")
        print(f"[DependencyCheck] 将尝试从 external_libs 目录加载...")
        # 不直接退出，让热更新系统尝试补救
else:
    print("[DependencyCheck] OK 所有核心依赖已就绪")

from PyQt6.QtCore import QThread, QObject, pyqtSignal, QUrl
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QMessageBox, QMainWindow,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

# 应用配置（通过 config.py 集中管理镜像源等设置）
from taiji.core.config import apply_env_overrides, get_external_path, get_internal_path, get_writable_base_dir

apply_env_overrides()

# 日志重定向（打包后）
_log_file_lock = threading.Lock()

if getattr(sys, 'frozen', False):
    # 使用可写基目录存储日志，防止安装到只读目录时写失败
    writable_base = get_writable_base_dir()
    log_path = os.path.join(writable_base, "app.log")
    try:
        sys.stdout = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stderr = sys.stdout
    except Exception:
        class DummyWriter:
            encoding = 'utf-8'
            def write(self, *args, **kwargs): pass
            def flush(self, *args, **kwargs): pass
            def isatty(self): return False
            def fileno(self): return 1
        sys.stdout = DummyWriter()
        sys.stderr = DummyWriter()

# ==========================================
# 共享状态锁（保护 app_state 的并发访问）
# ==========================================
app_state_lock = threading.Lock()

# ==========================================
# 外部依赖挂载系统
# ==========================================
# 使用可写基目录，防止安装到只读目录时无法创建 external_libs 等文件夹
writable_base = get_writable_base_dir()

# 将 external_libs 创建在可写目录下
ext_libs_dir = os.path.join(writable_base, "external_libs")
os.makedirs(ext_libs_dir, exist_ok=True)
sys.path.insert(0, ext_libs_dir)

# ==========================================
# 热更新/补丁系统（增强版）
# ==========================================
import importlib.util
import importlib

class HotUpdateImporter:
    """
    增强版热更新导入器（支持子目录包结构）

    功能：
    1. 首次导入：自动从 update_code/ 加载补丁模块（支持子目录）
    2. 模块重载：调用 reload_module() 可热重载已导入的模块
    3. 安全降级：重载失败自动恢复旧模块
    4. 支持子目录结构：update_code/api/routes_chat.py -> api.routes_chat

    路径映射规则：
    - update_code/xxx.py            -> xxx（扁平模块，兼容旧版）
    - update_code/api/routes_chat.py -> api.routes_chat（子目录包结构）
    """

    def __init__(self, update_dir):
        self.update_dir = update_dir
        self._loaded_from_patch = set()  # 记录从补丁加载的模块
        if self.update_dir not in sys.path:
            sys.path.insert(0, self.update_dir)

    def find_spec(self, fullname, path, target=None):
        """自定义模块查找：支持扁平和子目录两种结构"""
        # 路径 1：扁平模式（旧版兼容）- update_code/xxx.py
        flat_path = os.path.join(self.update_dir, fullname + ".py")
        if os.path.exists(flat_path):
            print(f"[HotUpdate] 热加载补丁(扁平): {fullname}.py")
            self._loaded_from_patch.add(fullname)
            return importlib.util.spec_from_file_location(fullname, flat_path)
        # 路径 2：子目录模式 - update_code/api/routes_chat.py
        rel_path = fullname.replace(".", os.sep) + ".py"
        file_path = os.path.join(self.update_dir, rel_path)
        if os.path.exists(file_path):
            print(f"[HotUpdate] 热加载补丁(子目录): {rel_path}")
            self._loaded_from_patch.add(fullname)
            return importlib.util.spec_from_file_location(fullname, file_path)
        return None

    def _find_patch_path(self, module_name: str) -> str:
        """定位补丁文件路径（支持扁平和子目录两种结构）"""
        flat_path = os.path.join(self.update_dir, f"{module_name}.py")
        if os.path.exists(flat_path):
            return flat_path
        rel_path = module_name.replace(".", os.sep) + ".py"
        subdir_path = os.path.join(self.update_dir, rel_path)
        if os.path.exists(subdir_path):
            return subdir_path
        return ""

    def reload_module(self, module_name: str) -> bool:
        """
        热重载指定模块（支持子目录结构）

        流程：
        1. 定位补丁文件（扁平或子目录）
        2. 如果模块已导入，执行完整重载
        3. 如果重载失败，恢复旧模块
        """
        patch_path = self._find_patch_path(module_name)
        if not patch_path:
            print(f"[HotUpdate] 补丁文件不存在: {module_name}")
            return False

        old_module = None
        try:
            if module_name in sys.modules:
                old_module = sys.modules[module_name]
                spec = importlib.util.spec_from_file_location(module_name, patch_path)
                if spec is None:
                    print(f"[HotUpdate] X 无法创建模块 spec: {module_name}")
                    return False
                new_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = new_module
                spec.loader.exec_module(new_module)
                self._loaded_from_patch.add(module_name)
                print(f"[HotUpdate] OK 模块热重载成功: {module_name}")
                return True
            else:
                print(f"[HotUpdate] 模块 {module_name} 尚未导入，跳过重载")
                return False
        except Exception as e:
            if old_module is not None and module_name in sys.modules:
                sys.modules[module_name] = old_module
            print(f"[HotUpdate] X 模块重载失败 {module_name}: {e}")
            return False

    def reload_all_patches(self) -> dict:
        """重载所有已安装的补丁（递归扫描子目录）"""
        results = {}
        if not os.path.exists(self.update_dir):
            return results
        for root, dirs, files in os.walk(self.update_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if fname.endswith(".py") and fname != "__init__.py":
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, self.update_dir)
                    mod_name = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]
                    results[mod_name] = self.reload_module(mod_name)
        return results

    def get_available_patches(self) -> list:
        """列出可用补丁（递归扫描子目录）"""
        patches = []
        if not os.path.exists(self.update_dir):
            return patches
        for root, dirs, files in os.walk(self.update_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    fpath = os.path.join(root, f)
                    rel_path = os.path.relpath(fpath, self.update_dir)
                    mod_name = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]
                    patches.append(mod_name)
        return sorted(patches)


# 热更新目录创建在可写基目录下（而非可能的只读安装目录）
update_dir = os.path.join(writable_base, "update_code")
os.makedirs(update_dir, exist_ok=True)
update_frontend_dir = os.path.join(writable_base, "update_frontend")
os.makedirs(update_frontend_dir, exist_ok=True)
_hot_importer = HotUpdateImporter(update_dir)
sys.meta_path.insert(0, _hot_importer)
# ⚠️ 不要把 update_dir 放入 sys.path，否则空 __init__.py 会覆盖打包的包模块

# 启动时自动加载已有补丁
print("[HotUpdate] 扫描已有补丁...")
patches = _hot_importer.get_available_patches()
if patches:
    print(f"[HotUpdate] 发现 {len(patches)} 个补丁: {', '.join(patches)}")
    for mod_name in patches:
        _hot_importer.reload_module(mod_name)
else:
    print("[HotUpdate] 暂无补丁")


class Worker(QObject):
    """后台工作线程：加载模型 + 启动 FastAPI"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit("正在启动后端服务...")
            print("[Worker] 后台线程启动，启动 FastAPI 服务...")
            from api.app import app
            import uvicorn
            import threading as py_threading
            import time
            import urllib.request
            import json
            from taiji.core.config import MODEL_LOAD_TIMEOUT

            def start_server():
                uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

            server_thread = py_threading.Thread(target=start_server, daemon=True)
            server_thread.start()

            local_url = "http://127.0.0.1:8000"

            # 快速检测 uvicorn 是否就绪（只检测服务器启动，不等待模型加载）
            # 使用更激进的快速轮询策略：前5次间隔0.3s，后续1s间隔
            is_ready = False
            max_attempts = max(60, MODEL_LOAD_TIMEOUT)
            
            # 禁用系统代理以加速本地连接，防止经过翻墙软件导致响应极慢
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)

            for attempt in range(max_attempts):
                if not server_thread.is_alive():
                    raise Exception("后端服务线程已意外终止！常见原因：8000 端口被占用，或内部依赖严重缺失。")
                    
                try:
                    # 刚开始时更频繁地重试
                    timeout = 0.5 if attempt < 10 else 2
                    req = urllib.request.Request(f"{local_url}/api/health")
                    with opener.open(req, timeout=timeout) as response:
                        if response.status == 200:
                            data = json.loads(response.read().decode("utf-8"))
                            if data.get("status") in ("ok", "loading"):
                                is_ready = True
                                break
                            elif data.get("status") == "error":
                                # 服务已启动，只是模型加载报错（如没下载或显存不足），允许显示前端UI让用户处理
                                print(f"[HealthCheck] 基础服务就绪，但模型异常: {data.get('message')}")
                                is_ready = True
                                break
                except RuntimeError as e:
                    # 捕获到明确的后端报错，直接中断轮询向外抛出
                    raise e
                except Exception as e:
                    print(f"[HealthCheck] 正在等待后端就绪 ({attempt+1}/{max_attempts})... {e}")
                    sleep_time = 0.3 if attempt < 10 else 1
                    time.sleep(sleep_time)

            if not is_ready:
                raise Exception("FastAPI 后端服务启动超时或模型加载失败。请检查网络连接和显存。")

            # 后端已就绪（可能还在加载模型），立即通知前端显示 UI
            server_start_msg = (
                "服务已就绪，模型仍在加载中...\n"
                "前端界面可正常操作，聊天功能将在模型加载完成后可用。"
            )
            print(f"[Worker] 服务就绪: {local_url}")
            self.finished.emit(local_url)
        except Exception as e:
            err_msg = f"加载失败: {e}\n\n{traceback.format_exc()}"
            print(err_msg)
            self.error.emit(err_msg)


class CustomWebView(QWebEngineView):
    """自定义 Web 视图，用于拦截并汉化右键菜单"""
    def contextMenuEvent(self, event):
        try:
            menu = self.createStandardContextMenu()
            if menu:
                translations = {
                    "Back": "返回", "Forward": "前进", "Reload": "重新加载",
                    "Cut": "剪切", "Copy": "复制", "Paste": "粘贴",
                    "Undo": "撤销", "Redo": "重做", "Select all": "全选",
                    "Paste and match style": "粘贴并匹配样式", "Inspect": "检查元素",
                    "Save image": "另存图片", "Copy image": "复制图片",
                    "Copy image address": "复制图片地址", "Save link": "另存链接",
                    "Copy link address": "复制链接地址", "Print...": "打印",
                    "View page source": "查看网页源代码",
                    "Save Media": "另存媒体", "Copy Media Address": "复制媒体地址",
                    "Open link in new tab": "在新标签页中打开",
                    "Open link in new window": "在新窗口中打开",
                }
                for action in menu.actions():
                    text = action.text().replace('&', '')
                    if text in translations:
                        action.setText(translations[text])
                    else:
                        lower_text = text.lower()
                        if any(kw in lower_text for kw in ["translate", "screen studio", "progressive web app", "pwa"]):
                            action.setVisible(False)
                        elif "inspect" in lower_text:
                            action.setText("检查元素")
                        elif "search web for" in lower_text:
                            action.setText("在网页中搜索")
                pos = event.globalPos() if hasattr(event, 'globalPos') else event.globalPosition().toPoint()
                menu.exec(pos)
        except Exception as e:
            print(f"[UI] 右键菜单兼容性问题: {e}")
            super().contextMenuEvent(event)


class MainWindow(QMainWindow):
    """主窗口 - QMainWindow 内含 QWebEngineView，正确支持托盘行为"""

    def __init__(self):
        super().__init__()
        self.tray_icon = None
        try:
            self.menuBar().hide()
            self.setMenuWidget(None)
        except Exception:
            pass
        self._web_view = CustomWebView()
        
        # 🔧 启用跨域访问，允许前端 JS 通过 fetch 调用同源 API
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        self._web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        
        self.setCentralWidget(self._web_view)
        
        # 快捷键：F5 刷新前端页面
        from PyQt6.QtGui import QKeySequence, QShortcut
        self.shortcut_f5 = QShortcut(QKeySequence("F5"), self)
        self.shortcut_f5.activated.connect(self._web_view.reload)
        
        # 快捷键：F12 打开原生开发者工具 (DevTools)
        self.shortcut_f12 = QShortcut(QKeySequence("F12"), self)
        self.shortcut_f12.activated.connect(self.toggle_devtools)

    def toggle_devtools(self):
        if not hasattr(self, 'dev_window'):
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self.dev_window = QWebEngineView()
            self.dev_window.setWindowTitle("Taiji 开发者工具 (F12)")
            self.dev_window.resize(900, 600)
            self._web_view.page().setDevToolsPage(self.dev_window.page())
        
        if self.dev_window.isVisible():
            self.dev_window.hide()
        else:
            self.dev_window.show()

    def get_web_view(self):
        return self._web_view

    def load(self, url):
        self._web_view.load(url)

    def closeEvent(self, event):
        """正确覆盖 QMainWindow.closeEvent：关闭时隐藏到托盘"""
        event.ignore()
        self.hide()
        if self.tray_icon and self.tray_icon.isVisible():
            self.tray_icon.showMessage(
                "Taiji 态极",
                "应用已最小化到托盘，模型仍在后台运行。",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )


def _real_main():
    global app  # noqa — 使 app 在 _real_main 作用域内可见给嵌套函数使用
    app = QApplication(sys.argv)
    app.setApplicationName("Taiji")
    app.setOrganizationName("Taiji")
    app.setOrganizationDomain("taiji.local")
    app.setQuitOnLastWindowClosed(False)

    # 确保浏览器数据（如 localStorage、历史记录）持久化保存，防止打包后临时目录导致数据丢失
    try:
        from PyQt6.QtWebEngineCore import QWebEngineProfile
        profile = QWebEngineProfile.defaultProfile()
        profile.setPersistentStoragePath(get_external_path("user_data"))
        profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
    except Exception as e:
        print(f"配置浏览器持久化失败: {e}")

    # 启动闪屏
    splash = None
    try:
        from build_scripts.splash import LoadingWindow
        splash = LoadingWindow("首次启动需下载模型，请耐心等待")
        splash.show()
        app.processEvents()
    except Exception as e:
        print(f"启动闪屏失败: {e}")

    # 后台线程
    thread = QThread()
    worker = Worker()
    worker.moveToThread(thread)

    def on_loading_progress(msg):
        print(f"[Progress] {msg}")
        if splash:
            splash.status_text = msg
            splash._update_dots()

    def on_loading_complete(local_url):
        print(f"[Main] 服务就绪: {local_url}")
        window = MainWindow()
        window.setWindowTitle("Taiji 态极")
        window.resize(1200, 800)

        base_path = get_external_path("")
        icon_path = os.path.join(base_path, "icon.ico")
        internal_icon = get_internal_path("icon.ico")
        
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
        elif os.path.exists(internal_icon):
            app_icon = QIcon(internal_icon)
        else:
            app_icon = QIcon(sys.executable)
            
        window.setWindowIcon(app_icon)

        tray_icon = QSystemTrayIcon(app_icon, app)
        window.tray_icon = tray_icon
        tray_icon.setToolTip("Taiji 态极 (后台运行)")

        tray_menu = QMenu()
        show_action = QAction("显示主界面", window)
        show_action.triggered.connect(window.showNormal)

        return_home_action = QAction("返回内置界面", window)
        client_url = f"{local_url}/#/?taiji_client=desktop"

        return_home_action.triggered.connect(lambda: window.load(QUrl(client_url)))

        clear_cache_action = QAction("清除界面缓存", window)
        def clear_web_cache():
            from PyQt6.QtWebEngineCore import QWebEngineProfile
            QWebEngineProfile.defaultProfile().clearHttpCache()
            window.get_web_view().reload()
        clear_cache_action.triggered.connect(clear_web_cache)

        restart_action = QAction("重启工作站", window)
        def restart_app():
            # 隐藏托盘图标并关闭闪屏
            tray_icon.hide()
            if splash:
                try:
                    splash.close()
                except Exception:
                    pass
            import subprocess
            import time
            env = os.environ.copy()
            if getattr(sys, 'frozen', False):
                env.pop("_MEIPASS2", None)
                path_list = env.get("PATH", "").split(os.pathsep)
                env["PATH"] = os.pathsep.join([p for p in path_list if p != getattr(sys, '_MEIPASS', '')])
                
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
            subprocess.Popen([sys.executable] + sys.argv[1:], env=env, creationflags=creationflags)
            
            # 必须立刻强制退出以释放 8000 端口，否则新进程会因端口占用而启动失败
            os._exit(0)
        restart_action.triggered.connect(restart_app)

        quit_action = QAction("彻底退出", window)
        def force_quit():
            tray_icon.hide()
            if splash:
                try:
                    splash.close()
                except Exception:
                    pass
            app.quit()
            # 给 Qt 一点时间释放资源
            import time
            time.sleep(0.3)
            os._exit(0)
        quit_action.triggered.connect(force_quit)
        tray_menu.addAction(show_action)
        tray_menu.addAction(return_home_action)
        tray_menu.addSeparator()
        tray_menu.addAction(clear_cache_action)
        tray_menu.addAction(restart_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        tray_icon.setContextMenu(tray_menu)
        tray_icon.activated.connect(
            lambda r: window.showNormal()
            if r == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        tray_icon.show()

        window.load(QUrl(client_url))
        if splash:
            splash.finish(window)
        window.show()
        thread.quit()

    def on_loading_error(error_message):
        if splash:
            splash.close()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("启动失败")
        msg.setText("界面渲染失败！请查看下方详细报错：")
        msg.setDetailedText(error_message)
        msg.exec()
        app.quit()
        thread.quit()

    worker.finished.connect(on_loading_complete)
    worker.error.connect(on_loading_error)
    worker.progress.connect(on_loading_progress)
    thread.started.connect(worker.run)
    thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    # ── 最外层崩溃保护：将任何未捕获异常写入 crash.log ──
    try:
        _real_main()
    except Exception as _e:
        import traceback as _tb
        _writable_base = get_writable_base_dir() if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _crash_path = os.path.join(_writable_base, "crash.log")
        with open(_crash_path, "w", encoding="utf-8") as _f:
            _f.write(f"Taiji 启动崩溃\n{'='*60}\n{_tb.format_exc()}")
        # 尝试弹框提示
        try:
            from PyQt6.QtWidgets import QApplication as _QA, QMessageBox as _QMB
            _tmp_app = _QA(sys.argv)
            _msg = _QMB()
            _msg.setIcon(_QMB.Icon.Critical)
            _msg.setWindowTitle("Taiji 启动失败")
            _msg.setText(f"程序启动时发生致命错误，详情已写入:\n{_crash_path}\n\n{str(_e)}")
            _msg.exec()
        except Exception:
            pass
        sys.exit(1)
