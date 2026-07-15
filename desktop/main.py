"""
[产品入口] 态极桌面客户端 — 开发环境版本
==========================================

原生桌面应用，嵌入 Web 前端，通过子进程管理后端生命周期。

功能：
1. 嵌入 Vue 前端（QWebEngineView）
2. 系统托盘（最小化到托盘、通知）
3. 窗口管理（记住大小、位置）
4. subprocess 启动 uvicorn（端口 8000）和 WebSocket 服务器（端口 8765）
5. 子进程崩溃自动启动

启动方式：python desktop/main.py

注意：此文件与 api/run_app.py 功能重叠。
- 此文件：开发环境，子进程模式，管理 WebSocket 服务器
- api/run_app.py：打包环境，进程内 QThread，有依赖自检和热更新
- 未来计划：合并为一个入口，以 api/run_app.py 为基础，补充 WebSocket 管理
详见 docs/ENTRYPOINTS.md
"""
import os
import sys
import json
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger("TaijiDesktop")

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
SETTINGS_FILE = ROOT_DIR / "desktop" / "settings.json"


def load_settings() -> dict:
    """加载窗口设置"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(settings: dict):
    """保存窗口设置"""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


class BackendManager:
    """后端进程管理器"""

    def __init__(self):
        self.process = None
        self.port = 8000
        self._running = False

    def start(self):
        """启动后端"""
        if self._running:
            return

        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.app:app",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--log-level", "info",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
            self._running = True
            logger.info(f"Backend started on port {self.port} (PID: {self.process.pid})")

            # 等待后端就绪
            self._wait_for_ready()

        except Exception as e:
            logger.error(f"Failed to start backend: {e}")

    def _wait_for_ready(self, timeout: int = 30):
        """等待后端就绪"""
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=2)
                logger.info("Backend is ready")
                return True
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        logger.warning("Backend startup timeout")
        return False

    def stop(self):
        """停止后端"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Backend stopped")
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self._running = False

    def is_running(self) -> bool:
        return self._running and self.process and self.process.poll() is None


class WebSocketManager:
    """WebSocket 核心服务器管理器（端口 8765）"""

    def __init__(self):
        self.process = None
        self.port = 8765
        self._running = False

    def start(self):
        """启动 WebSocket 服务器"""
        if self._running:
            return

        cmd = [sys.executable, str(ROOT_DIR / "start_taiji.py")]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
            self._running = True
            logger.info(f"WebSocket server started on port {self.port} (PID: {self.process.pid})")

            self._wait_for_ready()

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    def _wait_for_ready(self, timeout: int = 15):
        """等待 WebSocket 服务器就绪"""
        import socket

        start = time.time()
        while time.time() - start < timeout:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(('localhost', self.port))
                s.close()
                if result == 0:
                    logger.info(f"WebSocket server ready on port {self.port}")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        logger.warning("WebSocket server startup timeout")
        return False

    def stop(self):
        """停止 WebSocket 服务器"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("WebSocket server stopped")
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self._running = False

    def is_running(self) -> bool:
        return self._running and self.process and self.process.poll() is None


def main():
    """启动态极桌面客户端"""
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QSystemTrayIcon, QMenu,
            QVBoxLayout, QWidget, QMessageBox, QSplashScreen, QLabel
        )
        from PyQt6.QtCore import QUrl, Qt, QTimer, QSize, QPoint
        from PyQt6.QtGui import QIcon, QAction, QPixmap, QColor, QPainter, QFont
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile
    except ImportError:
        logger.error("PyQt6 not installed. Run: pip install PyQt6 PyQt6-WebEngine")
        sys.exit(1)

    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("态极")
    app.setApplicationVersion("1.6.0")
    app.setOrganizationName("Taiji")
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)

    # 设置应用图标
    icon_path = ROOT_DIR / "frontend" / "public" / "favicon.ico"
    if not icon_path.exists():
        icon_path = ROOT_DIR / "icon.ico"

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # 加载设置
    settings = load_settings()

    # 启动后端
    backend = BackendManager()
    backend.start()

    # 启动 WebSocket 核心服务器
    ws_server = WebSocketManager()
    ws_server.start()

    # 创建主窗口
    class TaijiWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("态极 - AI 生命体")
            self.setMinimumSize(QSize(1024, 700))
            self.menuBar().hide()

            # 恢复窗口大小和位置
            geo = settings.get("geometry", {})
            if geo:
                self.setGeometry(
                    geo.get("x", 100), geo.get("y", 100),
                    geo.get("width", 1280), geo.get("height", 800)
                )
            else:
                self.resize(1280, 800)
                # 居中显示
                screen = app.primaryScreen().geometry()
                x = (screen.width() - 1280) // 2
                y = (screen.height() - 800) // 2
                self.move(x, y)

            # 创建 Web 视图
            self.web_view = QWebEngineView()
            self._frontend_loaded = False

            # 配置 Web 设置
            web_settings = self.web_view.settings()
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
            web_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, False)

            # 设置用户代理
            profile = self.web_view.page().profile()
            profile.setHttpUserAgent("TaijiDesktop/1.6.0")

            # 监听加载完成
            self.web_view.loadFinished.connect(self._on_load_finished)

            # 先加载载入动画页面（注入后端端口）
            self.setCentralWidget(self.web_view)
            QTimer.singleShot(300, self._show_loading)

            # 创建系统托盘
            self._create_tray()

            # 状态检查定时器
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self._check_backend)
            self.status_timer.start(10000)

            # 显示载入动画并加载前端
            QTimer.singleShot(300, self._show_loading)

        def _show_loading(self):
            """显示载入动画页面"""
            loading_path = ROOT_DIR / "desktop" / "loading.html"
            if loading_path.exists():
                self.web_view.setUrl(QUrl.fromLocalFile(str(loading_path)))
            else:
                self._load_frontend()

        def _load_frontend(self):
            """加载前端（由后端静态文件服务提供）"""
            if self._frontend_loaded:
                return
            if not backend.is_running():
                QTimer.singleShot(1000, self._load_frontend)
                return

            frontend_url = f"http://127.0.0.1:{backend.port}"
            logger.info(f"Loading frontend: {frontend_url}")
            self.web_view.load(QUrl(frontend_url))

        def _on_load_finished(self, ok):
            """前端加载完成"""
            current_url = self.web_view.url().toString()
            logger.info(f"Page loaded: {current_url} (ok={ok})")

            if ok:
                # 如果是主界面加载完成
                if f"127.0.0.1:{backend.port}" in current_url:
                    if self._frontend_loaded:
                        return
                    self._frontend_loaded = True
                    logger.info("Frontend loaded successfully")
                    # 注入错误捕获
                    self.web_view.page().runJavaScript("""
                        window.onerror = function(msg, url, line, col, error) {
                            console.error('JS Error:', msg, 'at', url, ':', line);
                            return false;
                        };
                        // 检查 Vue app 是否挂载
                        setTimeout(() => {
                            const app = document.getElementById('app');
                            if (app && app.children.length === 0) {
                                document.body.innerHTML = '<div style=\"display:flex;align-items:center;justify-content:center;height:100vh;background:#0d1117;color:#e2e8f0;font-family:sans-serif;\"><div style=\"text-align:center;\"><h1 style=\"font-size:48px;margin-bottom:16px;\">🧠</h1><h2>态极</h2><p style=\"color:#94a3b8;margin-top:8px;\">界面加载中，请稍候...</p><p style=\"color:#64748b;font-size:12px;margin-top:16px;\">如果长时间无响应，请刷新页面</p></div></div>';
                            }
                        }, 3000);
                    """)
                else:
                    # 载入动画页面加载完成
                    QTimer.singleShot(1000, self._load_frontend)
            else:
                logger.warning(f"Page load failed: {current_url}")
                # 显示错误页面
                self.web_view.setHtml(f'''
                    <html><body style="background:#0d1117;color:#e2e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;">
                    <div style="text-align:center;">
                        <h1 style="font-size:48px;">⚠️</h1>
                        <h2>加载失败</h2>
                        <p style="color:#94a3b8;">无法加载页面: {current_url}</p>
                        <p style="color:#64748b;font-size:12px;margin-top:16px;">请检查后端服务是否运行</p>
                    </div></body></html>
                ''')
                QTimer.singleShot(5000, self._load_frontend)

        def _create_tray(self):
            """创建系统托盘"""
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return

            self.tray = QSystemTrayIcon(self)

            # 使用应用图标
            icon_path = ROOT_DIR / "icon.ico"
            if icon_path.exists():
                self.tray.setIcon(QIcon(str(icon_path)))
            else:
                # 创建默认图标
                pixmap = QPixmap(64, 64)
                pixmap.fill(QColor("#6366f1"))
                painter = QPainter(pixmap)
                painter.setPen(QColor("white"))
                painter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
                painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "极")
                painter.end()
                self.tray.setIcon(QIcon(pixmap))

            # 托盘菜单
            tray_menu = QMenu()

            show_action = tray_menu.addAction("显示窗口")
            show_action.triggered.connect(self._show_window)

            tray_menu.addSeparator()

            life_action = tray_menu.addAction("生命状态")
            life_action.triggered.connect(lambda: self._run_js("location.hash='/life'"))

            tray_menu.addSeparator()

            quit_action = tray_menu.addAction("退出")
            quit_action.triggered.connect(self._quit)

            self.tray.setContextMenu(tray_menu)
            self.tray.activated.connect(self._tray_activated)
            self.tray.show()

        def _tray_activated(self, reason):
            """托盘图标被点击"""
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                self._show_window()

        def _show_window(self):
            """显示窗口"""
            self.show()
            self.raise_()
            self.activateWindow()

        def _check_backend(self):
            """检查后端和 WebSocket 服务状态"""
            if not backend.is_running():
                logger.warning("Backend stopped, restarting...")
                backend.start()
            if not ws_server.is_running():
                logger.warning("WebSocket server stopped, restarting...")
                ws_server.start()

        def _run_js(self, code):
            """执行 JavaScript"""
            self.web_view.page().runJavaScript(code)

        def closeEvent(self, event):
            """窗口关闭事件"""
            # 保存窗口位置
            geo = self.geometry()
            settings["geometry"] = {
                "x": geo.x(), "y": geo.y(),
                "width": geo.width(), "height": geo.height(),
            }
            save_settings(settings)

            # 最小化到托盘而不是退出
            if hasattr(self, 'tray') and self.tray.isVisible():
                self.hide()
                self.tray.showMessage(
                    "态极",
                    "已最小化到系统托盘，双击图标恢复窗口",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
                event.ignore()
            else:
                self._quit()
                event.accept()

        def _quit(self):
            """真正退出"""
            backend.stop()
            ws_server.stop()
            app.quit()

    # 创建并显示窗口
    window = TaijiWindow()
    window.show()

    # 启动事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
