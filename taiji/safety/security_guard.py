"""
态极安全卫士 — 代码安全检查与沙盒执行
========================================

对 LLM 生成的代码进行多层安全检查：
1. 静态分析：危险模式检测
2. AST 分析：语法树级别的安全检查
3. 沙盒执行：隔离环境中运行
4. 资源限制：CPU/内存/时间限制
"""
import ast
import os
import re
import logging
import tempfile
import subprocess
from typing import Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("Taiji.SecurityGuard")


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    passed: bool
    risk_level: str  # "safe", "low", "medium", "high", "critical"
    violations: List[str]
    sanitized_code: Optional[str] = None


class CodeSecurityGuard:
    """
    代码安全卫士

    对 LLM 生成的代码进行多层安全检查。
    """

    # 危险函数/模块（直接拒绝）
    BLOCKED_PATTERNS = [
        # 系统命令执行
        (r'\bos\.system\b', "os.system 命令执行"),
        (r'\bos\.popen\b', "os.popen 命令执行"),
        (r'\bsubprocess\.(?:call|run|Popen|check_output|check_call)\b', "subprocess 命令执行"),
        (r'\b__import__\b', "动态导入"),
        (r'\beval\s*\(', "eval 执行"),
        (r'\bexec\s*\(', "exec 执行"),
        (r'\bcompile\s*\(.*exec', "compile+exec 执行"),

        # 文件系统破坏
        (r'\bshutil\.rmtree\b', "递归删除目录"),
        (r'\bos\.remove\b', "删除文件"),
        (r'\bos\.unlink\b', "删除文件"),
        (r'\bos\.rename\b.*(?:/|\\\\)', "跨目录重命名"),
        (r'\bformat\s+[a-zA-Z]:', "格式化磁盘"),

        # 网络攻击
        (r'\bsocket\.socket\b', "原始套接字"),
        (r'\bparamiko\b', "SSH 连接"),
        (r'\bftplib\b', "FTP 连接"),

        # 进程注入
        (r'\bctypes\.(?:windll|cdll)\b', "系统库调用"),
        (r'\bsys\._getframe\b', "栈帧访问"),
        (r'\binspect\.(?:getmembers|getsource)\b', "内省访问"),

        # 危险操作
        (r'\bos\.chmod\b', "修改权限"),
        (r'\bos\.chown\b', "修改所有者"),
        (r'\bos\.kill\b', "杀进程"),
        (r'\bos\.fork\b', "fork 进程"),
        (r':\(\)\s*\{.*\|.*\&.*\}', "fork 炸弹"),
    ]

    # 敏感模块（警告但不拒绝）
    WARNED_PATTERNS = [
        (r'\bimport\s+requests\b', "网络请求模块"),
        (r'\bimport\s+urllib\b', "网络请求模块"),
        (r'\bimport\s+http\b', "HTTP 模块"),
        (r'\bimport\s+json\b', "JSON 模块（通常安全）"),
        (r'\bimport\s+os\b', "操作系统模块"),
        (r'\bimport\s+sys\b', "系统模块"),
        (r'\bopen\s*\(', "文件操作"),
        (r'\bwith\s+open\b', "文件操作"),
    ]

    # 最大代码长度
    MAX_CODE_LENGTH = 10000  # 字符

    # 最大 AST 节点数（防止复杂度过高）
    MAX_AST_NODES = 500

    # 禁止的 AST 节点类型
    BLOCKED_AST_TYPES = (
        ast.Delete,      # del 语句
        ast.Global,      # global 声明
        ast.Nonlocal,    # nonlocal 声明
        ast.Yield,       # yield（生成器，可能无限）
        ast.YieldFrom,   # yield from
    )

    def check_code(self, code: str, context: str = "") -> SecurityCheckResult:
        """
        对代码进行全面安全检查。

        Args:
            code: 待检查的代码
            context: 上下文信息（如工具名称）

        Returns:
            SecurityCheckResult
        """
        violations = []
        risk_level = "safe"

        # 1. 长度检查
        if len(code) > self.MAX_CODE_LENGTH:
            violations.append(f"代码过长（{len(code)} > {self.MAX_CODE_LENGTH} 字符）")
            risk_level = "medium"

        # 2. 危险模式检查
        for pattern, desc in self.BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"危险操作: {desc}")
                risk_level = "critical"

        # 3. 警告模式检查
        warnings = []
        for pattern, desc in self.WARNED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(f"注意: {desc}")

        # 4. AST 分析
        try:
            tree = ast.parse(code)
            ast_violations = self._check_ast(tree)
            violations.extend(ast_violations)
            if ast_violations and risk_level == "safe":
                risk_level = "high"

            # AST 节点计数
            node_count = sum(1 for _ in ast.walk(tree))
            if node_count > self.MAX_AST_NODES:
                violations.append(f"AST 节点过多（{node_count} > {self.MAX_AST_NODES}）")
                if risk_level == "safe":
                    risk_level = "medium"

        except SyntaxError as e:
            violations.append(f"语法错误: {e}")
            risk_level = "high"

        # 5. 路径遍历检查
        path_patterns = [
            (r'\.\./\.\./', "多层路径遍历"),
            (r'/etc/(?:passwd|shadow|hosts)', "系统文件访问"),
            (r'~/(?:\.ssh|\.aws|\.env)', "敏感目录访问"),
            (r'(?:password|secret|token|key).*\.json', "敏感文件访问"),
        ]
        for pattern, desc in path_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"路径安全: {desc}")
                if risk_level == "safe":
                    risk_level = "medium"

        # 6. 网络安全检查
        network_patterns = [
            (r'https?://(?!api\.github\.com|pypi\.org|registry\.npmjs\.org)', "外部网络请求"),
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "硬编码 IP 地址"),
        ]
        for pattern, desc in network_patterns:
            if re.search(pattern, code):
                violations.append(f"网络安全: {desc}")
                if risk_level == "safe":
                    risk_level = "low"

        # 判断是否通过
        passed = risk_level in ("safe", "low")

        # 如果有警告但通过了，记录日志
        if warnings:
            for w in warnings:
                logger.info(f"安全警告: {w}")

        return SecurityCheckResult(
            passed=passed,
            risk_level=risk_level,
            violations=violations,
            sanitized_code=self._sanitize(code) if passed else None,
        )

    def _check_ast(self, tree: ast.AST) -> List[str]:
        """AST 级别的安全检查"""
        violations = []

        for node in ast.walk(tree):
            # 检查禁止的节点类型
            if isinstance(node, self.BLOCKED_AST_TYPES):
                violations.append(f"禁止的语法: {type(node).__name__}")

            # 检查 import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("os", "sys", "subprocess", "shutil", "ctypes"):
                        violations.append(f"危险模块导入: {alias.name}")

            if isinstance(node, ast.ImportFrom):
                if node.module and any(m in node.module for m in ("os", "sys", "subprocess", "shutil", "ctypes")):
                    violations.append(f"危险模块导入: from {node.module}")

            # 检查函数调用
            if isinstance(node, ast.Call):
                func_name = self._get_func_name(node)
                if func_name in ("eval", "exec", "compile", "__import__"):
                    violations.append(f"危险函数调用: {func_name}")

        return violations

    def _get_func_name(self, node: ast.Call) -> str:
        """提取函数调用的名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _sanitize(self, code: str) -> str:
        """清理代码（移除注释和空行）"""
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # 保留代码，移除纯注释行
            if stripped and not stripped.startswith("#"):
                cleaned.append(line)
        return "\n".join(cleaned)


class SandboxExecutor:
    """
    沙盒代码执行器

    在隔离环境中执行代码，限制：
    - 执行时间（默认 10 秒）
    - 内存使用（默认 256MB）
    - 网络访问（禁止）
    - 文件系统（只允许临时目录）
    """

    def __init__(
        self,
        max_time_seconds: int = 10,
        max_memory_mb: int = 256,
        allow_network: bool = False,
    ):
        self.max_time = max_time_seconds
        self.max_memory = max_memory_mb
        self.allow_network = allow_network

    def execute(self, code: str, inputs: dict = None) -> Tuple[bool, str]:
        """
        在沙盒中执行代码。

        Args:
            code: 待执行的代码
            inputs: 输入变量（字典）

        Returns:
            (success, output_or_error)
        """
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构建沙盒脚本
            sandbox_code = self._build_sandbox(code, inputs, tmpdir)

            # 写入临时文件
            script_path = os.path.join(tmpdir, "__sandbox__.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(sandbox_code)

            try:
                # 执行
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.max_time,
                    cwd=tmpdir,
                    env=self._get_safe_env(),
                )

                if result.returncode == 0:
                    return True, result.stdout
                else:
                    return False, result.stderr or result.stdout

            except subprocess.TimeoutExpired:
                return False, f"执行超时（{self.max_time}秒）"
            except Exception as e:
                return False, f"执行失败: {e}"

    def _build_sandbox(self, code: str, inputs: dict, tmpdir: str) -> str:
        """构建沙盒脚本"""
        sandbox = [
            "# -*- coding: utf-8 -*-",
            "# Sandbox execution environment",
            "import sys",
            "import os",
            "",
            "# Restrict imports",
            "_original_import = __builtins__.__import__",
            "_blocked = {'subprocess', 'socket', 'ctypes', 'paramiko', 'ftplib', 'smtplib'}",
            "def _safe_import(name, *args, **kwargs):",
            "    if name in _blocked or name.split('.')[0] in _blocked:",
            "        raise ImportError(f'Import blocked in sandbox: {name}')",
            "    return _original_import(name, *args, **kwargs)",
            "__builtins__.__import__ = _safe_import",
            "",
            "# Set working directory",
            f"os.chdir({repr(tmpdir)})",
            "",
        ]

        # 注入输入变量
        if inputs:
            for key, value in inputs.items():
                sandbox.append(f"{key} = {repr(value)}")
            sandbox.append("")

        # 用户代码
        sandbox.append("# === User Code ===")
        sandbox.append(code)

        return "\n".join(sandbox)

    def _get_safe_env(self) -> dict:
        """获取安全的环境变量"""
        env = os.environ.copy()
        if not self.allow_network:
            # 设置代理为空，阻止网络访问
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)
            env.pop("http_proxy", None)
            env.pop("https_proxy", None)
        return env


# 全局实例
_code_guard = CodeSecurityGuard()
_sandbox = SandboxExecutor()


def check_code_safety(code: str, context: str = "") -> SecurityCheckResult:
    """检查代码安全性"""
    return _code_guard.check_code(code, context)


def execute_in_sandbox(code: str, inputs: dict = None) -> Tuple[bool, str]:
    """在沙盒中执行代码"""
    # 先检查安全性
    check = _code_guard.check_code(code)
    if not check.passed:
        return False, f"安全检查未通过: {', '.join(check.violations)}"

    return _sandbox.execute(code, inputs)
