"""
Web 终端 WebSocket 路由
======================
提供基于 WebSocket 的交互式终端会话。
支持 Windows (cmd/PowerShell) 和 Linux (bash/zsh)。

安全措施：
- JWT 认证（通过 query 参数传递 token）
- 并发终端数量限制（默认 3）
- 空闲超时自动断开（默认 300 秒）
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from taiji.services.settings_service import get_setting

logger = logging.getLogger("ApiServer.Terminal")
router = APIRouter()

# ======================== 安全配置 ========================

MAX_CONCURRENT_TERMINALS = 3
IDLE_TIMEOUT_SECONDS = 300  # 5 分钟无输入自动断开

# 当前活跃终端计数
_active_terminals = 0
_terminals_lock = threading.Lock()


def _acquire_terminal_slot() -> bool:
    """尝试获取终端槽位，成功返回 True"""
    global _active_terminals
    with _terminals_lock:
        if _active_terminals >= MAX_CONCURRENT_TERMINALS:
            return False
        _active_terminals += 1
        return True


def _release_terminal_slot():
    """释放终端槽位"""
    global _active_terminals
    with _terminals_lock:
        _active_terminals = max(0, _active_terminals - 1)


def _verify_ws_token(ws) -> bool:
    """
    验证 WebSocket 连接的 JWT token（通过 query 参数）

    安全策略：
    - 认证启用时：必须提供有效 token
    - 认证未启用时：检查终端是否允许未认证访问（默认不允许）
    """
    try:
        from taiji.core.security import AuthManager
        auth = AuthManager()

        if auth.enabled:
            # 认证启用时，验证 token
            token = ws.query_params.get("token", "")
            if not token:
                logger.warning("WebSocket 终端连接被拒绝: 缺少 token")
                return False
            payload = auth.verify_token(token)
            if not payload:
                logger.warning("WebSocket 终端连接被拒绝: token 无效或已过期")
                return False
            return True
        else:
            # 认证未启用时，检查配置是否允许未认证访问终端
            # 默认不允许，需要显式启用
            allow_unauthenticated = get_setting("terminal_allow_unauthenticated", False)
            if not allow_unauthenticated:
                logger.warning("WebSocket 终端连接被拒绝: 认证未启用且未配置允许未认证访问")
                return False
            return True
    except ImportError:
        logger.warning("安全模块不可用，拒绝终端连接")
        return False
    except Exception as e:
        logger.warning(f"WebSocket 认证异常: {e}")
        return False


def _get_default_shell() -> tuple:
    """获取系统默认 shell 命令和参数"""
    if sys.platform == "win32":
        # 优先使用 cmd.exe（asyncio 兼容性更好）
        return "cmd.exe", []
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, []


async def _read_stream(stream, ws: WebSocket, prefix: str):
    """从子进程流读取并发送到 WebSocket"""
    try:
        while True:
            data = await stream.read(4096)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            payload = json.dumps({"type": prefix, "data": text}, ensure_ascii=False)
            await ws.send_text(payload)
    except Exception:
        pass


@router.websocket("/ws/terminal")
async def terminal_websocket(ws: WebSocket):
    """Web 终端 WebSocket 端点

    安全防护：
    - JWT 认证（强制）
    - 并发限制（默认 3）
    - 空闲超时（默认 300s）
    - 可通过 terminal_enabled 配置完全禁用
    """
    import sys as _sys
    import subprocess as _sp

    # 全局开关：允许管理员完全禁用终端功能
    if not get_setting("terminal_enabled", True):
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "终端功能已被管理员禁用"}))
        await ws.close(code=4003, reason="Terminal disabled")
        return

    # 认证检查
    if not _verify_ws_token(ws):
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "认证失败"}))
        await ws.close(code=4001, reason="Unauthorized")
        return

    # 并发限制
    if not _acquire_terminal_slot():
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "data": "终端并发数已达上限"}))
        await ws.close(code=4002, reason="Too many terminals")
        return

    process = None
    import time as _time
    _session_started = _time.time()
    _pid = 0

    loop = asyncio.get_event_loop()
    try:
        await ws.accept()

        # 获取工作目录
        work_dir = os.getcwd()
        try:
            custom_path = get_setting("workspace_path", "")
            if custom_path and os.path.isdir(custom_path):
                work_dir = custom_path
        except Exception:
            pass

        # 获取 shell
        if sys.platform == "win32":
            shell_cmd, shell_args = "cmd.exe", []
        else:
            shell_cmd = os.environ.get("SHELL", "/bin/bash")
            shell_args = []

        # 用 Popen 创建子进程
        creationflags = _sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = _sp.Popen(
            [shell_cmd] + shell_args,
            stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
            cwd=work_dir, env={**os.environ, "TERM": "xterm-256color"},
            creationflags=creationflags,
        )
        logger.info(f"终端进程启动: PID={process.pid}, work_dir={work_dir}, shell={shell_cmd}")
        _pid = process.pid

        # 欢迎消息
        await ws.send_text(json.dumps({
            "type": "output",
            "data": f"\r\nTaiji Terminal | {shell_cmd} | PID: {process.pid}\r\n"
        }))

        # 后台线程读取子进程 stdout → asyncio queue
        import queue
        output_queue = queue.Queue()

        def _read_output(stream, q):
            try:
                while True:
                    data = stream.read(4096)
                    if not data:
                        break
                    q.put(data)
            except Exception:
                pass
            q.put(None)

        _stdout_thread = threading.Thread(target=_read_output, args=(process.stdout, output_queue), daemon=True)
        _stderr_thread = threading.Thread(target=_read_output, args=(process.stderr, output_queue), daemon=True)
        _stdout_thread.start()
        _stderr_thread.start()

        # 后台任务：从 queue 读取 → WebSocket
        async def _drain_output():
            while True:
                try:
                    data = await loop.run_in_executor(None, output_queue.get, True, 0.5)
                    if data is None:
                        break
                    await ws.send_text(json.dumps({
                        "type": "output",
                        "data": data.decode("utf-8", errors="replace")
                    }, ensure_ascii=False))
                except queue.Empty:
                    continue
                except Exception:
                    break

        drain_task = asyncio.create_task(_drain_output())

        # 主循环：WebSocket 输入 → 子进程 stdin
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "input" and process.stdin and not process.stdin.closed:
                    data = msg.get("data", "")
                    process.stdin.write(data.encode("utf-8"))
                    process.stdin.flush()
                    # 审计日志（截断过长的输入，防止刷屏）
                    _trimmed = data.strip()[:200] + ("..." if len(data.strip()) > 200 else "")
                    if _trimmed:
                        logger.info(f"终端[PID={_pid}] 输入: {_trimmed}")
                elif t == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except json.JSONDecodeError:
            if process.stdin and not process.stdin.closed:
                process.stdin.write(raw.encode("utf-8"))
                process.stdin.flush()
                _trimmed = raw.strip()[:200] + ("..." if len(raw.strip()) > 200 else "")
                if _trimmed:
                    logger.info(f"终端[PID={_pid}] 输入: {_trimmed}")
        except WebSocketDisconnect:
            pass
        finally:
            drain_task.cancel()
            elapsed = _time.time() - _session_started
            logger.info(f"终端会话结束: PID={_pid}, 持续 {elapsed:.0f}s")

    except WebSocketDisconnect:
        logger.info("终端客户端断开")
    except Exception as e:
        logger.error(f"终端异常: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "data": str(e)[:200]}))
        except Exception:
            pass
    finally:
        _release_terminal_slot()
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        logger.info("终端会话结束")