"""
ModelSelf 感知系统
眼睛 — 让模型原生感知环境状态

将 Taiji 的工作台状态、文件树、系统信息编码为模型可理解的 token 序列。
"""
import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger("ModelSelf.Perception")


class PerceptionSystem:
    """
    感知系统 — 编码环境状态为 token 序列
    
    职责:
    1. 编码工作台文件树
    2. 编码系统状态（内存、设备等）
    3. 编码工具执行结果
    4. 编码当前任务上下文
    """
    
    def __init__(self, tokenizer, max_tree_depth: int = 3, max_files: int = 50):
        self.tokenizer = tokenizer
        self.max_tree_depth = max_tree_depth
        self.max_files = max_files
    
    def encode_workspace(self, workspace_path: str) -> list:
        """
        编码工作台文件树为 token 序列
        
        输出格式:
        <observe><tree>main.py
        src/
          app.py
          utils.py
        requirements.txt</tree></observe>
        """
        from taiji.config import SPECIAL_TOKENS
        
        tree_text = self._build_file_tree(workspace_path)
        encoded = self.tokenizer._encode(f"<observe><tree>{tree_text}</tree></observe>")
        return encoded
    
    def encode_system_state(self, state: Dict) -> list:
        """
        编码系统状态
        
        Args:
            state: {"memory_free": "8GB", "device": "cpu", "model": "125M", ...}
        
        输出: <observe><state>device=cpu\nmemory=8GB\nmodel=125M</state></observe>
        """
        state_text = "\n".join(f"{k}={v}" for k, v in state.items())
        return self.tokenizer._encode(f"<observe><state>{state_text}</state></observe>")
    
    def encode_tool_result(self, tool_name: str, result: str, max_len: int = 400) -> list:
        """
        编码工具执行结果
        
        输出: <tool_result>result text</tool_result>
        """
        if len(result) > max_len:
            result = result[:max_len] + "...(截断)"
        return self.tokenizer._encode(f"<tool_result>{result}</tool_result>")
    
    def encode_observe(self, content: str) -> list:
        """通用观察编码"""
        return self.tokenizer._encode(f"<observe>{content}</observe>")
    
    def _build_file_tree(self, path: str) -> str:
        """构建文件树文本"""
        if not os.path.exists(path):
            return "(工作台不存在)"
        
        lines = []
        count = 0
        
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth >= self.max_tree_depth:
                dirs.clear()
                continue
            
            indent = "  " * depth
            basename = os.path.basename(root)
            if depth > 0:
                lines.append(f"{indent}{basename}/")
            
            for f in sorted(files):
                if count >= self.max_files:
                    lines.append(f"{indent}  ... ({self.max_files}+ 文件)")
                    return "\n".join(lines)
                lines.append(f"{indent}  {f}")
                count += 1
        
        return "\n".join(lines) if lines else "(空工作台)"