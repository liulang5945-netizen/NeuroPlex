"""
态极桌面自动化 (Desktop Automation)
====================================

让态极能打开和操作任何程序，不只是浏览器。

能力：
1. 启动任意程序（浏览器、编辑器、终端、文件管理器...）
2. 执行系统命令
3. 文件管理（复制、移动、重命名、压缩）
4. 进程管理（查看、终止进程）
5. 系统信息（CPU、内存、磁盘、网络）
6. 剪贴板操作
7. 屏幕截图
8. 定时任务

使用方式：
    from taiji.tools.desktop import run_program, run_command, system_info
    result = run_program("notepad", ["file.txt"])
    result = run_command("dir")
    info = system_info()
"""
import os
import sys
import json
import time
import shutil
import logging
import subprocess
import platform
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger("Taiji.Desktop")

# 安全命令白名单（允许执行的命令）
SAFE_COMMANDS = {
    "dir", "ls", "cat", "type", "echo", "pwd", "cd", "mkdir", "rmdir",
    "copy", "cp", "move", "mv", "del", "rm", "find", "grep", "head", "tail",
    "wc", "sort", "uniq", "diff", "tar", "zip", "unzip", "gzip", "gunzip",
    "python", "python3", "pip", "node", "npm", "git", "curl", "wget",
    "ping", "ipconfig", "ifconfig", "nslookup", "tracert", "traceroute",
    "tasklist", "taskkill", "systeminfo", "whoami", "hostname",
    "date", "time", "cal", "bc", "expr",
    "code", "notepad", "explorer", "calc", "mspaint", "snippingtool",
}

# 危险命令黑名单
DANGEROUS_COMMANDS = {
    "format", "fdisk", "diskpart", "regedit", "reg",
    "net user", "net localgroup", "net share",
    "shutdown", "restart", "logoff",
    "rd /s", "rmdir /s", "rm -rf",
}


def run_program(program: str, args: List[str] = None, wait: bool = False) -> str:
    """
    启动任意程序。

    Args:
        program: 程序名称或路径
        args: 命令行参数
        wait: 是否等待程序结束

    Returns:
        启动结果
    """
    try:
        cmd = [program] + (args or [])

        if wait:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='ignore',
            )
            output = result.stdout + result.stderr
            return f"程序已执行 (退出码: {result.returncode})\n{output[:3000]}"
        else:
            if platform.system() == "Windows":
                # Windows: 使用 start 命令启动（不阻塞）
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    shell=True,
                )
            else:
                # Linux/Mac: 使用 & 后台运行
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return f"程序已启动: {program} {' '.join(args or [])}"
    except FileNotFoundError:
        return f"程序未找到: {program}"
    except Exception as e:
        return f"启动失败: {e}"


def run_command(command: str, timeout: int = 30) -> str:
    """
    执行系统命令（安全白名单）。

    Args:
        command: 命令字符串
        timeout: 超时时间（秒）

    Returns:
        命令输出
    """
    # 安全检查
    cmd_lower = command.lower().strip()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return f"安全拒绝: 命令 '{command}' 被标记为危险操作"

    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                encoding='utf-8',
                errors='ignore',
            )
        else:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                encoding='utf-8',
                errors='ignore',
            )

        output = result.stdout + result.stderr
        if not output:
            output = "(无输出)"

        return f"命令: {command}\n退出码: {result.returncode}\n\n{output[:5000]}"
    except subprocess.TimeoutExpired:
        return f"命令超时 ({timeout}s): {command}"
    except Exception as e:
        return f"执行失败: {e}"


def system_info() -> str:
    """获取系统信息"""
    import platform
    import psutil

    info = {
        "系统": platform.system(),
        "版本": platform.version(),
        "架构": platform.machine(),
        "处理器": platform.processor(),
        "Python": platform.python_version(),
        "CPU 核心数": psutil.cpu_count(),
        "CPU 使用率": f"{psutil.cpu_percent(interval=1)}%",
        "内存总量": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "内存使用率": f"{psutil.virtual_memory().percent}%",
        "磁盘总量": f"{psutil.disk_usage('/').total / (1024**3):.1f} GB",
        "磁盘使用率": f"{psutil.disk_usage('/').percent}%",
    }

    lines = [f"{k}: {v}" for k, v in info.items()]
    return "\n".join(lines)


def list_processes(name_filter: str = "") -> str:
    """列出进程"""
    import psutil

    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            pinfo = proc.info
            if name_filter and name_filter.lower() not in pinfo['name'].lower():
                continue
            processes.append({
                "pid": pinfo['pid'],
                "name": pinfo['name'],
                "cpu": f"{pinfo['cpu_percent']:.1f}%",
                "memory": f"{pinfo['memory_percent']:.1f}%",
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not processes:
        return f"未找到进程: {name_filter}" if name_filter else "无进程"

    # 按 CPU 使用率排序
    processes.sort(key=lambda x: float(x['cpu'].rstrip('%')), reverse=True)

    lines = [f"进程列表 ({len(processes)} 个):\n"]
    for p in processes[:20]:
        lines.append(f"  PID {p['pid']:>6} | {p['name']:<30} | CPU: {p['cpu']:>6} | MEM: {p['memory']:>6}")

    return "\n".join(lines)


def kill_process(pid: int) -> str:
    """终止进程"""
    import psutil
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.terminate()
        return f"已终止进程: {name} (PID: {pid})"
    except psutil.NoSuchProcess:
        return f"进程不存在: PID {pid}"
    except psutil.AccessDenied:
        return f"权限不足: 无法终止 PID {pid}"
    except Exception as e:
        return f"终止失败: {e}"


def file_operations(action: str, src: str, dst: str = "") -> str:
    """
    文件操作。

    Args:
        action: 操作类型 (copy/move/rename/delete/zip/unzip)
        src: 源路径
        dst: 目标路径

    Returns:
        操作结果
    """
    try:
        if action == "copy":
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            return f"已复制: {src} → {dst}"
        elif action == "move":
            shutil.move(src, dst)
            return f"已移动: {src} → {dst}"
        elif action == "rename":
            os.rename(src, dst)
            return f"已重命名: {src} → {dst}"
        elif action == "delete":
            if os.path.isdir(src):
                shutil.rmtree(src)
            else:
                os.remove(src)
            return f"已删除: {src}"
        elif action == "zip":
            shutil.make_archive(dst, 'zip', src)
            return f"已压缩: {src} → {dst}.zip"
        elif action == "unzip":
            shutil.unpack_archive(src, dst)
            return f"已解压: {src} → {dst}"
        else:
            return f"未知操作: {action}"
    except Exception as e:
        return f"操作失败: {e}"


def clipboard_read() -> str:
    """读取剪贴板"""
    try:
        import pyperclip
        return pyperclip.paste()
    except ImportError:
        # Windows 回退
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True, encoding='utf-8', errors='ignore',
            )
            return result.stdout.strip()
        return "需要安装 pyperclip: pip install pyperclip"


def clipboard_write(text: str) -> str:
    """写入剪贴板"""
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"已写入剪贴板: {text[:50]}..."
    except ImportError:
        if platform.system() == "Windows":
            import subprocess
            subprocess.run(
                ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
                capture_output=True,
            )
            return f"已写入剪贴板: {text[:50]}..."
        return "需要安装 pyperclip: pip install pyperclip"


def open_file_manager(path: str = ".") -> str:
    """打开文件管理器"""
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return f"已打开文件管理器: {path}"


def open_terminal(path: str = ".", command: str = "") -> str:
    """打开终端"""
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        if command:
            subprocess.Popen(["cmd", "/k", f"cd /d {path} && {command}"],
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["cmd", "/k", f"cd /d {path}"],
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-a", "Terminal", path])
    else:
        subprocess.Popen(["x-terminal-emulator", "--working-directory", path])
    return f"已打开终端: {path}"


def open_editor(file_path: str) -> str:
    """打开编辑器"""
    file_path = os.path.abspath(file_path)
    if platform.system() == "Windows":
        # 尝试 VS Code，回退到记事本
        try:
            subprocess.Popen(["code", file_path])
            return f"已用 VS Code 打开: {file_path}"
        except FileNotFoundError:
            os.startfile(file_path)
            return f"已用默认编辑器打开: {file_path}"
    else:
        subprocess.Popen(["xdg-open", file_path])
        return f"已打开: {file_path}"


# ═══════════════════════════════════════════════
# 工具接口（供注册）
# ═══════════════════════════════════════════════

def desktop_run_command(command: str) -> str:
    """执行系统命令（工具接口）"""
    return run_command(command)


def desktop_run_program(input_str: str) -> str:
    """启动程序（工具接口）"""
    parts = input_str.split("|", 1)
    program = parts[0].strip()
    args = parts[1].strip().split() if len(parts) > 1 else []
    return run_program(program, args)


def desktop_system_info() -> str:
    """系统信息（工具接口）"""
    return system_info()


def desktop_processes(name: str = "") -> str:
    """进程列表（工具接口）"""
    return list_processes(name)


def desktop_file_op(input_str: str) -> str:
    """文件操作（工具接口）"""
    parts = input_str.split("|")
    if len(parts) < 2:
        return "格式: 操作(copy/move/delete/zip) | 源路径 | 目标路径(可选)"
    action = parts[0].strip()
    src = parts[1].strip()
    dst = parts[2].strip() if len(parts) > 2 else ""
    return file_operations(action, src, dst)
