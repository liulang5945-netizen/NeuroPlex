"""
沙箱安全配置
============
路径白名单、命令黑名单、审计日志。
"""
import logging
import os
import json
import time
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("SandboxSecurity")

# ======================== 路径安全 ========================

# 禁止访问的系统关键路径
BLOCKED_PATHS = [
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
]

# 用户敏感目录
SENSITIVE_PATHS = [
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.config",
    "~/.env",
]

# 允许的工作目录（相对于项目根）
ALLOWED_WORK_DIRS = [
    "agent_workspace",
    "taiji_data",
    "user_data",
    "knowledge_store",
    "tests",
]


def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_path_allowed(file_path: str) -> bool:
    """
    检查路径是否在安全白名单内。

    规则:
    1. 禁止访问 BLOCKED_PATHS 中的系统路径
    2. 禁止访问 SENSITIVE_PATHS 中的用户敏感目录
    3. 允许访问项目根目录下的所有路径
    4. 允许访问 ALLOWED_WORK_DIRS 中的工作目录
    5. 禁止路径遍历攻击 (../)
    """
    try:
        # 解析路径
        resolved = os.path.realpath(os.path.expanduser(file_path))
        project_root = get_project_root()

        # 检查路径遍历
        if ".." in file_path:
            logger.warning(f"路径遍历尝试被拒绝: {file_path}")
            return False

        # 检查系统路径
        for blocked in BLOCKED_PATHS:
            if resolved.startswith(os.path.realpath(blocked)):
                logger.warning(f"系统路径访问被拒绝: {file_path}")
                return False

        # 检查敏感目录
        for sensitive in SENSITIVE_PATHS:
            expanded = os.path.realpath(os.path.expanduser(sensitive))
            if resolved.startswith(expanded):
                logger.warning(f"敏感目录访问被拒绝: {file_path}")
                return False

        # 检查是否在项目目录内
        if resolved.startswith(os.path.realpath(project_root)):
            return True

        # 检查是否在允许的工作目录内
        for work_dir in ALLOWED_WORK_DIRS:
            work_path = os.path.realpath(os.path.join(project_root, work_dir))
            if resolved.startswith(work_path):
                return True

        # 默认拒绝
        logger.warning(f"路径不在白名单内: {file_path}")
        return False

    except Exception as e:
        logger.error(f"路径检查异常: {e}")
        return False


# ======================== 命令安全 ========================

# 危险命令黑名单
DANGEROUS_COMMANDS = [
    # 文件删除
    "rm -rf",
    "rm -rf /",
    "rm -rf /*",
    "del /f /s /q",
    "rmdir /s /q",
    "format ",
    "mkfs",
    # 系统修改
    "chmod 777",
    "chown",
    "passwd",
    "useradd",
    "userdel",
    "groupadd",
    # 网络
    "curl.*|.*sh",
    "wget.*|.*sh",
    "nc -l",
    "ncat",
    # 进程
    "kill -9",
    "pkill",
    "killall",
    # 权限提升
    "sudo",
    "su ",
    "doas",
    # 编码绕过
    "base64 -d",
    "eval(",
    "exec(",
    "__import__",
]


def is_command_safe(command: str) -> bool:
    """
    检查命令是否安全。

    规则:
    1. 检查 DANGEROUS_COMMANDS 黑名单
    2. 检查是否有管道到 shell 的操作
    3. 检查是否有编码绕过尝试
    """
    cmd_lower = command.lower().strip()

    # 检查黑名单
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous.lower() in cmd_lower:
            logger.warning(f"危险命令被拒绝: {command[:100]}")
            return False

    # 检查管道到 shell
    if "| sh" in cmd_lower or "| bash" in cmd_lower or "| cmd" in cmd_lower:
        logger.warning(f"管道到 shell 被拒绝: {command[:100]}")
        return False

    # 检查反引号执行
    if "`" in command and ("rm" in cmd_lower or "del" in cmd_lower):
        logger.warning(f"反引号执行危险命令被拒绝: {command[:100]}")
        return False

    return True


# ======================== 审计日志 ========================

class AuditLogger:
    """沙箱操作审计日志"""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            log_dir = os.path.join(get_project_root(), "taiji_data", "audit_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "sandbox_audit.jsonl")

    def log(self, action: str, detail: str, allowed: bool, user: str = "system"):
        """记录审计事件"""
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "detail": detail[:500],  # 截断长内容
            "allowed": allowed,
            "user": user,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"审计日志写入失败: {e}")

    def get_recent(self, count: int = 100) -> list:
        """获取最近的审计事件"""
        events = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    events.append(json.loads(line))
        except FileNotFoundError:
            pass
        return events


# 全局审计日志实例
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
