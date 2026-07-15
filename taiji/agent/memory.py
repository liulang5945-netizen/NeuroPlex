"""
ModelSelf 记忆系统
海马体 — 让模型拥有原生记忆能力

提供短期记忆（工作记忆）和长期记忆（持久化存储），
通过特殊 token 实现模型原生的读写操作。
"""
import os
import json
import logging
import time
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger("ModelSelf.Memory")


class MemorySlot:
    """单个记忆槽"""
    def __init__(self, slot_id: int, is_long_term: bool = False):
        self.slot_id = slot_id
        self.is_long_term = is_long_term
        self.content: str = ""
        self.timestamp: float = 0
        self.access_count: int = 0
        self.importance: float = 0.0  # 0-1，重要性评分

    def write(self, content: str, importance: float = 0.5):
        self.content = content
        self.timestamp = time.time()
        self.importance = importance
        self.access_count += 1

    def read(self) -> str:
        self.access_count += 1
        return self.content

    def is_empty(self) -> bool:
        return not self.content

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "is_long_term": self.is_long_term,
            "content": self.content,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemorySlot":
        s = cls(d["slot_id"], d.get("is_long_term", False))
        s.content = d.get("content", "")
        s.timestamp = d.get("timestamp", 0)
        s.access_count = d.get("access_count", 0)
        s.importance = d.get("importance", 0)
        return s


class MemorySystem:
    """
    记忆系统 — 管理短期和长期记忆

    短期记忆 (20 槽): 工作记忆，当前任务相关，会话结束清空
    长期记忆 (10 槽): 持久化存储，跨会话保留

    与模型的交互通过特殊 token:
    - <mem_write slot="N">content</mem_write> → 写入记忆
    - <mem_read slot="N"/> → 读取记忆
    - 巩固机制: 短期记忆 → 评估重要性 → 高重要性自动转为长期
    """

    def __init__(self, num_short_term: int = 20, num_long_term: int = 10):
        # 短期记忆
        self.short_term = [MemorySlot(i, is_long_term=False) for i in range(num_short_term)]
        # 长期记忆
        self.long_term = [MemorySlot(i, is_long_term=True) for i in range(num_long_term)]
        # 下一个可用的短期记忆槽（轮转）
        self._next_short_slot = 0

    def write(self, slot_id: int, content: str, is_long_term: bool = False, importance: float = 0.5):
        """写入记忆槽"""
        slots = self.long_term if is_long_term else self.short_term
        if 0 <= slot_id < len(slots):
            slots[slot_id].write(content, importance)
            logger.debug(f"Memory write: slot={slot_id} long={is_long_term} len={len(content)}")

    def read(self, slot_id: int, is_long_term: bool = False) -> str:
        """读取记忆槽"""
        slots = self.long_term if is_long_term else self.short_term
        if 0 <= slot_id < len(slots):
            return slots[slot_id].read()
        return ""

    def auto_write(self, content: str, importance: float = 0.5) -> int:
        """自动选择一个短期记忆槽写入（轮转策略）"""
        # 优先写入空槽
        for i, slot in enumerate(self.short_term):
            if slot.is_empty():
                slot.write(content, importance)
                return i

        # 没有空槽，覆盖最不重要且最久未访问的
        target = min(range(len(self.short_term)),
                     key=lambda i: self.short_term[i].importance * 0.5 +
                                   (1.0 / (self.short_term[i].access_count + 1)) * 0.5)
        self.short_term[target].write(content, importance)
        return target

    def consolidate(self) -> List[int]:
        """
        巩固机制: 将高重要性的短期记忆转为长期记忆

        Returns:
            被巩固的短期记忆槽 ID 列表
        """
        consolidated = []
        # 找到空的长期记忆槽
        empty_long_slots = [i for i, s in enumerate(self.long_term) if s.is_empty()]

        for i, short_slot in enumerate(self.short_term):
            if short_slot.is_empty() or short_slot.importance < 0.7:
                continue

            if empty_long_slots:
                # 写入空的长期槽
                long_id = empty_long_slots.pop(0)
                self.long_term[long_id].write(short_slot.content, short_slot.importance)
                short_slot.content = ""  # 清空短期槽
                consolidated.append(i)
                logger.info(f"Consolidated short[{i}] -> long[{long_id}]")

        return consolidated

    def get_context_tokens(self, tokenizer) -> list:
        """
        获取当前记忆的上下文 token 序列

        输出格式:
        <mem_read>0: 用户想创建 Python 项目
        1: 已安装 Flask
        L0: Python 是一种编程语言</mem_read>
        """
        lines = []
        for i, slot in enumerate(self.short_term):
            if not slot.is_empty():
                lines.append(f"{i}: {slot.content[:200]}")

        for i, slot in enumerate(self.long_term):
            if not slot.is_empty():
                lines.append(f"L{i}: {slot.content[:200]}")

        if not lines:
            return []

        text = "<mem_read>" + "\n".join(lines) + "</mem_read>"
        return tokenizer._encode(text)

    def clear_short_term(self):
        """清空所有短期记忆"""
        for slot in self.short_term:
            slot.content = ""
            slot.access_count = 0
            slot.importance = 0

    def save(self, path: str):
        """持久化长期记忆"""
        os.makedirs(path, exist_ok=True)
        data = {
            "long_term": [s.to_dict() for s in self.long_term],
        }
        with open(os.path.join(path, "memory.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载持久化的长期记忆"""
        fpath = os.path.join(path, "memory.json")
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for i, d in enumerate(data.get("long_term", [])):
                if i < len(self.long_term):
                    self.long_term[i] = MemorySlot.from_dict(d)
            logger.info(f"Loaded {len(data.get('long_term', []))} long-term memories")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

    def get_stats(self) -> dict:
        """获取记忆统计"""
        st_used = sum(1 for s in self.short_term if not s.is_empty())
        lt_used = sum(1 for s in self.long_term if not s.is_empty())
        return {
            "short_term_used": st_used,
            "short_term_total": len(self.short_term),
            "long_term_used": lt_used,
            "long_term_total": len(self.long_term),
        }

    def parse_write_command(self, token_ids: list, tokenizer) -> Optional[Dict]:
        """
        从 token 序列中解析记忆写入命令

        检测 <mem_write slot="N">content</mem_write> 模式
        """
        from taiji.config import SPECIAL_TOKENS

        ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()
        mem_write_id = SPECIAL_TOKENS["mem_write"]

        for i, tid in enumerate(ids):
            if tid == mem_write_id:
                # 解析后续 token 作为记忆内容
                content_parts = []
                for j in range(i + 1, len(ids)):
                    if ids[j] == SPECIAL_TOKENS["mem_write"]:
                        break
                    text = tokenizer.decode([ids[j]], skip_special_tokens=True)
                    if text:
                        content_parts.append(text)

                if content_parts:
                    content = "".join(content_parts).strip()
                    # 默认写入短期记忆，自动选择槽位
                    slot_id = self.auto_write(content, importance=0.5)
                    return {"slot_id": slot_id, "content": content, "is_long_term": False}

        return None