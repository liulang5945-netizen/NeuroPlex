"""
态极短期工作记忆 (Working Memory)
==================================

态极的新能力 #1：读一次，记住全文。

人类的短期记忆可以暂时记住 7±2 个信息块。
态极的工作记忆可以记住整个文件的内容，
并在记忆中直接搜索和修改，最后一次性写回。

这样态极不需要每一步都重新读取文件，
大幅减少工具调用次数，提升任务完成速度。

使用方式：
    memory = WorkingMemory()
    memory.remember("main.py", file_content)
    # ... 做其他操作 ...
    content = memory.recall("main.py")  # 立即返回，无需重新读取
    memory.modify("main.py", "old_text", "new_text")
    final = memory.export("main.py")  # 导出修改后的内容
"""
import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger("WorkingMemory")


@dataclass
class MemoryEntry:
    """一条工作记忆"""
    key: str                    # 记忆的标识符（通常是文件路径）
    content: str                # 记住的内容
    timestamp: str              # 记住的时间
    source: str                 # 来源（"file_read", "tool_result", "user_input"）
    modifications: List[dict] = field(default_factory=list)  # 已做的修改记录
    original_hash: str = ""     # 原始内容的哈希（用于检测变更）
    access_count: int = 0       # 访问次数


class WorkingMemory:
    """
    态极的短期工作记忆
    
    核心特性：
    1. 快速存取：O(1) 时间复杂度的读写
    2. 自动淘汰：超过容量时淘汰最久未访问的记忆
    3. 修改追踪：记录所有修改操作，支持回滚
    4. 智能检索：支持按关键词搜索记忆内容
    
    容量限制：
    - 最多记住 50 个条目（文件/变量/结果）
    - 单个条目最大 1MB
    - 总容量最大 10MB
    """
    
    MAX_ENTRIES = 50
    MAX_ENTRY_SIZE = 1_000_000  # 1MB
    MAX_TOTAL_SIZE = 10_000_000  # 10MB
    
    def __init__(self):
        # 使用 OrderedDict 实现 LRU 缓存
        self._memory: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._total_size = 0
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"WorkingMemory initialized (session: {self._session_id})")
    
    # ─── 存取操作 ───────────────────────────────────
    
    def remember(self, key: str, content: str, source: str = "file_read") -> bool:
        """
        记住一段内容。
        
        Args:
            key: 记忆标识符（如文件路径、变量名）
            content: 要记住的内容
            source: 来源（"file_read", "tool_result", "user_input"）
            
        Returns:
            是否成功记住
        """
        # 检查大小限制
        content_size = len(content.encode("utf-8"))
        if content_size > self.MAX_ENTRY_SIZE:
            logger.warning(f"Content too large ({content_size} bytes), truncating")
            content = content[:self.MAX_ENTRY_SIZE]
            content_size = self.MAX_ENTRY_SIZE
        
        # 如果已存在，更新
        if key in self._memory:
            old_size = len(self._memory[key].content.encode("utf-8"))
            self._total_size -= old_size
        
        # 淘汰旧记忆直到有空间
        while self._total_size + content_size > self.MAX_TOTAL_SIZE and self._memory:
            self._evict_oldest()
        
        # 淘汰多余条目
        while len(self._memory) >= self.MAX_ENTRIES:
            self._evict_oldest()
        
        # 存入记忆
        entry = MemoryEntry(
            key=key,
            content=content,
            timestamp=datetime.now().isoformat(),
            source=source,
            original_hash=hashlib.md5(content.encode("utf-8")).hexdigest(),
        )
        
        self._memory[key] = entry
        self._memory.move_to_end(key)  # 移到最新位置
        self._total_size += content_size
        
        logger.debug(f"Remembered: {key} ({content_size} bytes, source: {source})")
        return True
    
    def recall(self, key: str) -> Optional[str]:
        """
        回忆一段记忆。
        
        Args:
            key: 记忆标识符
            
        Returns:
            记住的内容，如果不存在返回 None
        """
        if key not in self._memory:
            return None
        
        entry = self._memory[key]
        entry.access_count += 1
        self._memory.move_to_end(key)  # 更新 LRU 位置
        
        logger.debug(f"Recalled: {key} (access #{entry.access_count})")
        return entry.content
    
    def forget(self, key: str) -> bool:
        """忘记一段记忆"""
        if key not in self._memory:
            return False
        
        entry = self._memory.pop(key)
        self._total_size -= len(entry.content.encode("utf-8"))
        logger.debug(f"Forgot: {key}")
        return True
    
    def has(self, key: str) -> bool:
        """检查是否记得某段内容"""
        return key in self._memory
    
    # ─── 修改操作 ───────────────────────────────────
    
    def modify(self, key: str, old_text: str, new_text: str) -> Tuple[bool, str]:
        """
        在记忆中直接修改内容（不需要重新读取文件）。
        
        Args:
            key: 记忆标识符
            old_text: 要替换的文本
            new_text: 替换后的文本
            
        Returns:
            (成功与否, 说明信息)
        """
        if key not in self._memory:
            return False, f"记忆中没有找到: {key}"
        
        content = self._memory[key].content
        
        if old_text not in content:
            return False, f"在 {key} 中未找到要替换的文本"
        
        # 统计出现次数
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text, 1)
        
        # 更新记忆
        old_size = len(content.encode("utf-8"))
        self._memory[key].content = new_content
        self._total_size += len(new_content.encode("utf-8")) - old_size
        
        # 记录修改
        self._memory[key].modifications.append({
            "timestamp": datetime.now().isoformat(),
            "old_text": old_text[:200],
            "new_text": new_text[:200],
            "occurrences": count,
        })
        
        return True, f"已在 {key} 中替换 1 处（共 {count} 处匹配）"
    
    def modify_all(self, key: str, old_text: str, new_text: str) -> Tuple[bool, str]:
        """
        在记忆中全局替换所有匹配的文本。
        
        Args:
            key: 记忆标识符
            old_text: 要替换的文本
            new_text: 替换后的文本
            
        Returns:
            (成功与否, 说明信息)
        """
        if key not in self._memory:
            return False, f"记忆中没有找到: {key}"
        
        content = self._memory[key].content
        
        if old_text not in content:
            return False, f"在 {key} 中未找到要替换的文本"
        
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        
        old_size = len(content.encode("utf-8"))
        self._memory[key].content = new_content
        self._total_size += len(new_content.encode("utf-8")) - old_size
        
        self._memory[key].modifications.append({
            "timestamp": datetime.now().isoformat(),
            "old_text": old_text[:200],
            "new_text": new_text[:200],
            "occurrences": count,
            "replace_all": True,
        })
        
        return True, f"已在 {key} 中全局替换 {count} 处"
    
    def insert_at_line(self, key: str, line_num: int, text: str) -> Tuple[bool, str]:
        """在指定行插入文本"""
        if key not in self._memory:
            return False, f"记忆中没有找到: {key}"
        
        lines = self._memory[key].content.split("\n")
        if line_num < 0 or line_num > len(lines):
            return False, f"行号 {line_num} 超出范围 (0-{len(lines)})"
        
        lines.insert(line_num, text)
        self._memory[key].content = "\n".join(lines)
        
        return True, f"已在第 {line_num} 行插入内容"
    
    # ─── 搜索操作 ───────────────────────────────────
    
    def search(self, pattern: str, keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        在记忆中搜索文本。
        
        Args:
            pattern: 搜索模式（支持正则表达式）
            keys: 限定搜索范围的记忆键（None = 搜索所有）
            
        Returns:
            搜索结果列表
        """
        results = []
        search_keys = keys or list(self._memory.keys())
        
        for key in search_keys:
            if key not in self._memory:
                continue
            
            content = self._memory[key].content
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            
            for match in matches:
                # 提取上下文
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                # 计算行号
                line_num = content[:match.start()].count("\n") + 1
                
                results.append({
                    "key": key,
                    "line": line_num,
                    "match": match.group(),
                    "context": context,
                })
        
        return results
    
    def search_lines(self, key: str, keyword: str) -> List[Tuple[int, str]]:
        """
        在指定记忆中搜索包含关键词的行。
        
        Args:
            key: 记忆标识符
            keyword: 关键词
            
        Returns:
            [(行号, 行内容), ...]
        """
        if key not in self._memory:
            return []
        
        results = []
        lines = self._memory[key].content.split("\n")
        
        for i, line in enumerate(lines, 1):
            if keyword.lower() in line.lower():
                results.append((i, line.strip()))
        
        return results
    
    # ─── 导出操作 ───────────────────────────────────
    
    def export(self, key: str) -> Optional[str]:
        """
        导出记忆的当前内容（可用于写回文件）。
        
        Args:
            key: 记忆标识符
            
        Returns:
            当前内容，如果不存在返回 None
        """
        return self.recall(key)
    
    def export_all(self) -> Dict[str, str]:
        """导出所有记忆的内容"""
        return {key: entry.content for key, entry in self._memory.items()}
    
    def get_modified_keys(self) -> List[str]:
        """获取所有被修改过的记忆键"""
        return [
            key for key, entry in self._memory.items()
            if entry.modifications
        ]
    
    # ─── 内部操作 ───────────────────────────────────
    
    def _evict_oldest(self):
        """淘汰最久未访问的记忆"""
        if self._memory:
            key, entry = self._memory.popitem(last=False)
            self._total_size -= len(entry.content.encode("utf-8"))
            logger.debug(f"Evicted: {key}")
    
    # ─── 状态查询 ───────────────────────────────────
    
    def get_status(self) -> dict:
        """获取工作记忆状态"""
        return {
            "entries": len(self._memory),
            "max_entries": self.MAX_ENTRIES,
            "total_size_kb": self._total_size / 1024,
            "max_size_kb": self.MAX_TOTAL_SIZE / 1024,
            "usage_percent": round(self._total_size / self.MAX_TOTAL_SIZE * 100, 1),
            "modified_count": len(self.get_modified_keys()),
        }
    
    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        
        lines = [
            "🧠 工作记忆状态",
            "━━━━━━━━━━━━━━━━",
            f"条目数: {status['entries']}/{status['max_entries']}",
            f"总大小: {status['total_size_kb']:.1f} KB / {status['max_size_kb']:.0f} KB ({status['usage_percent']}%)",
            f"已修改: {status['modified_count']} 个条目",
        ]
        
        if self._memory:
            lines.append("\n📋 记忆列表:")
            for key, entry in self._memory.items():
                mod_flag = " ✏️" if entry.modifications else ""
                size_kb = len(entry.content.encode("utf-8")) / 1024
                lines.append(f"  • {key} ({size_kb:.1f}KB, 访问{entry.access_count}次){mod_flag}")
        
        return "\n".join(lines)
    
    def clear(self):
        """清空所有记忆"""
        self._memory.clear()
        self._total_size = 0
        logger.info("WorkingMemory cleared")


# ─── 全局实例 ─────────────────────────────────────

_global_memory: Optional[WorkingMemory] = None


def get_working_memory() -> WorkingMemory:
    """获取全局工作记忆实例"""
    global _global_memory
    if _global_memory is None:
        _global_memory = WorkingMemory()
    return _global_memory