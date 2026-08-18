"""
态极行动系统 (Limbs)
====================

态极的手脚 — 负责执行动作：工具调用、代码执行、文件操作。

态极原生实现，专门为态极服务。
"""
import json
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("Taiji.Limbs")

_WORKSPACE_DIR = None

# 模块级 FeedEngine 引用（由 BodyCore 注入，用于实践反馈）
_feed_engine = None


def set_feed_engine(engine):
    """设置 FeedEngine 引用，用于将执行结果反馈为训练样本。

    Args:
        engine: FeedEngine 实例（或 None 用于清除）
    """
    global _feed_engine
    _feed_engine = engine
    if engine is not None:
        logger.info(f"FeedEngine 已注入 limbs: {type(engine).__name__}")


def _feedback_to_feed(code: str, output: str, success: bool, domain: str = "code"):
    """将代码执行结果反馈到 FeedEngine（如果已注入）。

    闭合"实践→喂养→睡眠训练"回路：limbs 执行代码后，
    将结果喂给 feed_engine，供睡眠时训练对应域的神经元。

    Args:
        code: 执行的代码
        output: 执行输出
        success: 是否成功
        domain: 知识域（默认 code）
    """
    if _feed_engine is None:
        return
    try:
        _feed_engine.feed_from_practice(
            code=code, output=output, success=success, domain=domain
        )
    except Exception as e:
        logger.debug(f"实践反馈失败（非关键）: {e}")


def _get_workspace() -> str:
    """获取态极的工作台目录"""
    global _WORKSPACE_DIR
    if _WORKSPACE_DIR is None:
        _WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "agent_workspace")
        os.makedirs(_WORKSPACE_DIR, exist_ok=True)
    return _WORKSPACE_DIR


def _resolve_safe_path(target: str) -> Optional[str]:
    """安全解析路径，防止目录穿越"""
    ws = os.path.abspath(_get_workspace())
    result = os.path.abspath(os.path.join(ws, target))
    if result == ws or result.startswith(ws + os.sep):
        return result
    return None


def _check_path_security(file_path: str) -> bool:
    """额外的路径安全检查（系统路径白名单）"""
    try:
        from neuroplex.safety.sandbox_security import is_path_allowed
        return is_path_allowed(file_path)
    except ImportError:
        return True  # 安全模块不可用时放行


# ======================== 文件系统操作 ========================

def read_file(file_path: str, page: int = 1) -> str:
    """
    读取文件内容（分页支持）

    Args:
        file_path: 文件路径
        page: 页码（从1开始）

    Returns:
        文件内容
    """
    try:
        if not os.path.exists(file_path):
            ws_path = os.path.join(_get_workspace(), file_path)
            if os.path.exists(ws_path):
                file_path = ws_path
            else:
                return f"错误: 找不到文件 '{file_path}'"

        if not _check_path_security(file_path):
            return f"⚠️ 安全限制: 不允许访问系统关键路径 '{file_path}'"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text:
            return "文件为空"

        page_size = 4000
        total_pages = max(1, (len(text) + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        content = text[start_idx:end_idx]

        return f"--- 第 {page}/{total_pages} 页 ---\n{content}\n--- (若需阅读下一页，请传入 page={page + 1}) ---"
    except Exception as e:
        return f"读取文件失败: {e}"


def write_file(file_path: str, content: str) -> str:
    """
    将内容写入文件

    Args:
        file_path: 文件路径
        content: 文件内容

    Returns:
        操作结果
    """
    try:
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 文件已写入: {rel_path} (共 {len(content)} 字符)"
    except Exception as e:
        return f"❌ 写入文件失败: {e}"


def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件中的特定内容

    Args:
        file_path: 文件路径
        old_text: 要搜索的旧文本
        new_text: 替换后的新文本

    Returns:
        操作结果
    """
    try:
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        if not os.path.exists(resolved):
            return f"错误: 文件不存在 '{resolved}'"
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            stripped_old = old_text.strip()
            if stripped_old in content:
                old_text = stripped_old
            else:
                return "错误: 在文件中未找到匹配的文本"
        new_content = content.replace(old_text, new_text, 1)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 文件已编辑: {rel_path}"
    except Exception as e:
        return f"❌ 编辑文件失败: {e}"


def delete_file(file_path: str) -> str:
    """
    删除工作台中的文件或空目录

    Args:
        file_path: 文件路径

    Returns:
        操作结果
    """
    try:
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        if not os.path.exists(resolved):
            return f"错误: 路径不存在 '{resolved}'"
        if os.path.isfile(resolved):
            os.remove(resolved)
            rel_path = os.path.relpath(resolved, _get_workspace())
            return f"✅ 文件已删除: {rel_path}"
        elif os.path.isdir(resolved):
            try:
                os.rmdir(resolved)
                rel_path = os.path.relpath(resolved, _get_workspace())
                return f"✅ 空目录已删除: {rel_path}"
            except OSError:
                return "错误: 目录非空，无法删除"
        return f"错误: 未知路径类型 '{resolved}'"
    except Exception as e:
        return f"❌ 删除失败: {e}"


def list_directory(dir_path: str = "") -> dict:
    """
    列出工作台目录内容

    Args:
        dir_path: 目录路径（相对于工作台）

    Returns:
        目录内容字典
    """
    try:
        if dir_path:
            resolved = _resolve_safe_path(dir_path)
            if resolved is None:
                return {"error": f"路径不安全或超出工作台目录: {dir_path}"}
        else:
            resolved = _get_workspace()
        if not os.path.exists(resolved):
            return {"error": f"目录不存在 '{resolved}'"}
        if not os.path.isdir(resolved):
            return {"error": f"'{resolved}' 不是目录"}
        entries = os.listdir(resolved)
        dirs = []
        files = []
        for entry in sorted(entries):
            entry_path = os.path.join(resolved, entry)
            if os.path.isdir(entry_path):
                dirs.append(entry)
            else:
                try:
                    size = os.path.getsize(entry_path)
                    files.append({"name": entry, "size": size})
                except Exception:
                    files.append({"name": entry, "size": 0})
        rel = os.path.relpath(resolved, _get_workspace())
        return {
            "path": rel if rel != '.' else '.',
            "dirs": dirs,
            "files": files,
            "total_dirs": len(dirs),
            "total_files": len(files),
        }
    except Exception as e:
        return {"error": f"列出目录失败: {e}"}


def create_directory(dir_path: str) -> str:
    """
    在工作台中创建目录（支持多级）

    Args:
        dir_path: 目录路径

    Returns:
        操作结果
    """
    try:
        resolved = _resolve_safe_path(dir_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {dir_path}"
        os.makedirs(resolved, exist_ok=True)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 目录已创建: {rel_path}/"
    except Exception as e:
        return f"❌ 创建目录失败: {e}"


def analyze_code(file_path: str) -> dict:
    """
    分析工作台中的代码文件

    Args:
        file_path: 文件路径

    Returns:
        分析结果字典
    """
    try:
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return {"error": f"路径不安全或超出工作台目录: {file_path}"}
        if not os.path.exists(resolved):
            return {"error": f"文件不存在 '{resolved}'"}

        ext = os.path.splitext(resolved)[1].lower()
        with open(resolved, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = {
            "file": file_path,
            "lines": len(lines),
            "extension": ext,
        }

        if ext == ".py":
            source = "".join(lines)
            try:
                compile(source, resolved, "exec")
                code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
                imports = [l for l in lines if l.strip().startswith(("import ", "from "))]
                funcs = [l for l in lines if l.strip().startswith(("def ", "async def "))]
                classes = [l for l in lines if l.strip().startswith("class ")]
                result.update({
                    "valid": True,
                    "code_lines": len(code_lines),
                    "imports": len(imports),
                    "functions": len(funcs),
                    "classes": len(classes),
                })
            except SyntaxError as e:
                result.update({
                    "valid": False,
                    "error": f"行 {e.lineno}: {e.msg}",
                })
        elif ext == ".json":
            try:
                json.loads("".join(lines))
                result["valid"] = True
            except json.JSONDecodeError as e:
                result.update({
                    "valid": False,
                    "error": str(e),
                })

        return result
    except Exception as e:
        return {"error": f"分析失败: {e}"}


def run_python(code: str) -> dict:
    """
    在工作台中执行 Python 代码

    Args:
        code: Python 代码

    Returns:
        执行结果字典
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_get_workspace(),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        success = result.returncode == 0
        output = result.stdout[:2000] if success else result.stderr[:1000]
        # 将执行结果反馈到 FeedEngine（闭合实践→喂养→训练回路）
        _feedback_to_feed(code=code, output=output, success=success, domain="code")
        return {
            "success": success,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        _feedback_to_feed(code=code, output="执行超时", success=False, domain="code")
        return {"success": False, "error": "执行超时（超过 30 秒）"}
    except Exception as e:
        _feedback_to_feed(code=code, output=str(e), success=False, domain="code")
        return {"success": False, "error": str(e)}


# ======================== 工具注册 ========================

# 态极原生工具列表
TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "list_directory": list_directory,
    "create_directory": create_directory,
    "analyze_code": analyze_code,
    "run_python": run_python,
}


def get_tool(name: str):
    """获取工具函数"""
    return TOOLS.get(name)


def list_tools() -> list:
    """列出所有可用工具"""
    return list(TOOLS.keys())