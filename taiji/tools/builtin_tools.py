"""
态极内置工具注册
================
在系统启动时注册所有内置工具。
"""
import logging
from taiji.agent_ext.tool_registry import registry, ToolDef

logger = logging.getLogger("Taiji.BuiltinTools")


def register_all_tools():
    """注册所有内置工具"""
    _register_web_tools()
    _register_file_tools()
    _register_code_tools()
    logger.info(f"内置工具注册完成: {len(registry.list_tools())} 个")


def _register_web_tools():
    """注册网络工具"""

    # 搜索网页
    def web_search(query: str, num_results: int = 5) -> str:
        """搜索网页，返回搜索结果"""
        try:
            from taiji.tools.web import search
            results = search(query, max_results=num_results)
            if not results:
                return "未找到相关结果"
            output = []
            for i, r in enumerate(results, 1):
                output.append(f"{i}. {r.title}\n   {r.snippet}\n   来源: {r.url}")
            return "\n\n".join(output)
        except Exception as e:
            return f"搜索失败: {e}"

    registry.register(ToolDef(
        name="web_search",
        description="搜索网页获取信息。用于查找最新资讯、技术文档、事实查询等。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "num_results": {"type": "integer", "description": "返回结果数量", "default": 5},
            },
            "required": ["query"],
        },
        func=web_search,
        category="网络",
    ))

    # 读取网页
    def web_fetch(url: str) -> str:
        """读取网页内容，返回正文"""
        try:
            from taiji.tools.web import fetch
            page = fetch(url)
            if not page:
                return "无法读取网页"
            return f"# {page.title}\n\n{page.content[:5000]}"
        except Exception as e:
            return f"读取网页失败: {e}"

    registry.register(ToolDef(
        name="web_fetch",
        description="读取指定URL的网页内容。用于获取网页正文、文章内容等。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "网页URL"},
            },
            "required": ["url"],
        },
        func=web_fetch,
        category="网络",
    ))


def _register_file_tools():
    """注册文件工具"""

    # 读取文件
    def read_file(file_path: str, page: int = 1) -> str:
        """读取文件内容"""
        try:
            from taiji.body.limbs import read_file as _read_file
            return _read_file(file_path, page)
        except Exception as e:
            return f"读取文件失败: {e}"

    registry.register(ToolDef(
        name="read_file",
        description="读取本地文件内容。支持文本文件、代码文件等。",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
                "page": {"type": "integer", "description": "页码（从1开始）", "default": 1},
            },
            "required": ["file_path"],
        },
        func=read_file,
        category="文件",
    ))

    # 写入文件
    def write_file(file_path: str, content: str) -> str:
        """将内容写入文件"""
        try:
            from taiji.body.limbs import write_file as _write_file
            return _write_file(file_path, content)
        except Exception as e:
            return f"写入文件失败: {e}"

    registry.register(ToolDef(
        name="write_file",
        description="将内容写入文件。用于创建或修改文件。",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["file_path", "content"],
        },
        func=write_file,
        category="文件",
    ))

    # 列出目录
    def list_directory(dir_path: str = ".") -> str:
        """列出目录内容"""
        try:
            import os
            if not os.path.exists(dir_path):
                return f"目录不存在: {dir_path}"
            items = []
            for item in sorted(os.listdir(dir_path)):
                full_path = os.path.join(dir_path, item)
                if os.path.isdir(full_path):
                    items.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(full_path)
                    if size > 1024 * 1024:
                        size_str = f"{size / 1024 / 1024:.1f}MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    items.append(f"  📄 {item} ({size_str})")
            return f"目录: {dir_path}\n" + "\n".join(items[:50])
        except Exception as e:
            return f"列出目录失败: {e}"

    registry.register(ToolDef(
        name="list_directory",
        description="列出目录中的文件和子目录。",
        parameters={
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "目录路径", "default": "."},
            },
        },
        func=list_directory,
        category="文件",
    ))


def _register_code_tools():
    """注册代码工具"""

    # 执行代码
    def execute_code(code: str, language: str = "python") -> str:
        """执行代码并返回结果"""
        try:
            from taiji.body.limbs import execute_code as _execute_code
            return _execute_code(code, language)
        except Exception as e:
            return f"执行代码失败: {e}"

    registry.register(ToolDef(
        name="execute_code",
        description="执行代码并返回结果。支持Python代码。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的代码"},
                "language": {"type": "string", "description": "编程语言", "default": "python"},
            },
            "required": ["code"],
        },
        func=execute_code,
        category="代码",
    ))
