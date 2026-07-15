"""
态极统一上下文管理器 (Context Manager) v2
==========================================

核心改进：
1. 相关性检索：按任务相关性排序记忆，而非按时间
2. 记忆衰减：旧记忆自然降低重要度
3. 自动重要度：根据内容类型自动评分
4. 上下文压缩：旧对话摘要而非截断
5. 动态窗口：根据模型容量自适应
6. 睡眠整合：与 SleepEngine 联动巩固记忆
"""
import os
import re
import json
import time
import math
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("ContextManager")


@dataclass
class ContextItem:
    """上下文条目"""
    content: str
    source: str       # "memory" / "history" / "tool" / "system" / "interaction"
    importance: float  # 0-1（基础重要度）
    timestamp: float
    token_count: int = 0
    key: str = ""
    access_count: int = 0
    keywords: List[str] = field(default_factory=list)  # 关键词索引
    decay_rate: float = 0.01  # 衰减速率


class ContextManager:
    """
    统一上下文管理器 v2

    能力：
    1. 统一记忆：WorkingMemory + MemorySystem + 持久化
    2. 相关性检索：基于关键词匹配 + TF-IDF 排序
    3. 记忆衰减：时间衰减 + 访问频率加权
    4. 自动重要度：内容类型 + 用户交互 + 访问频率
    5. 上下文压缩：旧对话自动摘要
    6. 动态窗口：自适应 token 预算
    """

    def __init__(self, max_context_tokens: int = 2048, tokenizer=None):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer

        # 记忆系统引用
        self._working_memory = None
        self._memory_system = None

        # 对话历史
        self._conversation_history: List[Dict] = []
        self._max_history_turns = 20
        self._conversation_summaries: List[str] = []  # 压缩的旧对话摘要

        # 上下文缓存
        self._context_cache: Dict[str, ContextItem] = {}

        # 跨会话记忆
        self._persistent_memories: Dict[str, ContextItem] = {}
        self._persistent_path = None

        # 语义记忆（向量检索）
        self._semantic_memory = None

        # 关键词索引（用于相关性检索）
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> [item_keys]

        logger.info(f"ContextManager v2 initialized (max_tokens={max_context_tokens})")

    # ─── 记忆系统注入 ─────────────────────────────────

    def set_working_memory(self, wm):
        self._working_memory = wm

    def set_memory_system(self, ms):
        self._memory_system = ms

    def set_persistent_path(self, path: str):
        self._persistent_path = path
        self._load_persistent()

    def set_semantic_memory(self, sm):
        """注入语义记忆（向量检索）"""
        self._semantic_memory = sm

    # ─── 对话历史管理 ─────────────────────────────────

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加对话消息，自动管理历史长度。

        支持的 metadata 字段：
        - tool_calls: 工具调用记录
        - thought: 思考过程
        - reasoning: 推理链
        - context_used: 使用的上下文信息
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            msg.update(metadata)

        self._conversation_history.append(msg)

        # 自动截断 + 摘要压缩
        if len(self._conversation_history) > self._max_history_turns * 2:
            old_messages = self._conversation_history[:-self._max_history_turns * 2]
            self._conversation_history = self._conversation_history[-self._max_history_turns * 2:]

            # 压缩旧对话为摘要
            summary = self._compress_messages(old_messages)
            if summary:
                self._conversation_summaries.append(summary)
                # 只保留最近 5 个摘要
                if len(self._conversation_summaries) > 5:
                    self._conversation_summaries = self._conversation_summaries[-5:]

    def get_history(self, max_turns: int = None) -> List[Dict]:
        if max_turns:
            return self._conversation_history[-max_turns * 2:]
        return self._conversation_history.copy()

    def clear_history(self):
        self._conversation_history.clear()

    # ─── 记忆操作（带自动重要度 + 关键词索引）────────

    def remember(self, key: str, content: str, source: str = "user",
                 importance: float = None, persistent: bool = False):
        """
        记住信息，自动计算重要度和关键词。

        Args:
            key: 记忆标识
            content: 内容
            source: 来源
            importance: 重要度 (None=自动计算)
            persistent: 是否跨会话持久化
        """
        # 自动计算重要度
        if importance is None:
            importance = self._auto_importance(content, source)

        # 提取关键词
        keywords = self._extract_keywords(content)

        item = ContextItem(
            content=content,
            source=source,
            importance=importance,
            timestamp=time.time(),
            token_count=self._estimate_tokens(content),
            key=key,
            keywords=keywords,
        )

        # 存入缓存
        self._context_cache[key] = item

        # 建立关键词索引
        for kw in keywords:
            if kw not in self._keyword_index:
                self._keyword_index[kw] = []
            if key not in self._keyword_index[kw]:
                self._keyword_index[kw].append(key)

        # 同步到 WorkingMemory
        if self._working_memory:
            self._working_memory.remember(key, content, source)

        # 同步到 MemorySystem
        if self._memory_system:
            self._memory_system.auto_write(content, importance)

        # 跨会话持久化
        if persistent:
            self._persistent_memories[key] = item
            self._save_persistent()

        # 语义索引
        if self._semantic_memory:
            try:
                self._semantic_memory.add(key, content, {"source": source, "importance": importance})
            except Exception:
                pass

    def recall(self, key: str) -> Optional[str]:
        """回忆信息"""
        if key in self._context_cache:
            item = self._context_cache[key]
            item.access_count += 1
            return item.content

        if self._working_memory:
            content = self._working_memory.recall(key)
            if content:
                return content

        if key in self._persistent_memories:
            return self._persistent_memories[key].content

        return None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索记忆（语义检索优先，关键词回退）"""
        # 优先使用语义检索
        if self._semantic_memory:
            try:
                semantic_results = self._semantic_memory.search(query, top_k=top_k)
                if semantic_results:
                    return [{
                        "key": r["key"],
                        "content": r["content"],
                        "source": "semantic",
                        "importance": r.get("metadata", {}).get("importance", 0.5),
                        "relevance": r["score"],
                    } for r in semantic_results]
            except Exception:
                pass

        # 回退到关键词匹配
        results = []
        query_keywords = self._extract_keywords(query)

        all_items = {}
        all_items.update(self._context_cache)
        all_items.update(self._persistent_memories)

        if self._working_memory:
            for key, content in self._working_memory.export_all().items():
                if key not in all_items:
                    all_items[key] = ContextItem(
                        content=content, source="working_memory",
                        importance=0.5, timestamp=time.time(), key=key,
                        keywords=self._extract_keywords(content),
                    )

        for key, item in all_items.items():
            score = self._relevance_score(query_keywords, item)
            if score > 0:
                results.append({
                    "key": key,
                    "content": item.content[:200],
                    "source": item.source,
                    "importance": item.importance,
                    "relevance": score,
                })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    # ─── 上下文构建（相关性驱动 + 任务自适应）────────

    def detect_task_type(self, task: str) -> str:
        """
        自动检测任务类型。

        Returns:
            "code" | "search" | "conversation" | "creative" | "factual"
        """
        task_lower = task.lower()

        code_keywords = ["代码", "python", "function", "def ", "class ", "bug", "error",
                         "debug", "编写", "实现", "code", "script", "程序"]
        if any(kw in task_lower for kw in code_keywords):
            return "code"

        search_keywords = ["搜索", "查找", "search", "find", "查询", "搜", "query"]
        if any(kw in task_lower for kw in search_keywords):
            return "search"

        creative_keywords = ["写", "创作", "故事", "诗", "write", "story", "poem", "创意"]
        if any(kw in task_lower for kw in creative_keywords):
            return "creative"

        factual_keywords = ["什么是", "解释", "定义", "what is", "explain", "define", "原理"]
        if any(kw in task_lower for kw in factual_keywords):
            return "factual"

        return "conversation"

    def build_context(self, task: str, system_prompt: str = "",
                      include_history: bool = True,
                      include_memory: bool = True,
                      max_tokens: int = None,
                      task_type: str = None) -> str:
        """
        构建完整的上下文字符串。

        支持任务类型自适应：
        - code: 高比例相关记忆，低比例对话历史
        - conversation: 高比例对话历史
        - search: 高比例知识库记忆
        - creative: 低比例约束，高比例联想
        """
        if task_type is None:
            task_type = self.detect_task_type(task)

        budget = max_tokens or self.max_context_tokens
        parts = []

        # 根据任务类型调整预算分配
        history_ratio = 0.5
        memory_ratio = 0.6
        if task_type == "code":
            history_ratio = 0.3
            memory_ratio = 0.8  # 代码任务需要更多相关记忆
        elif task_type == "conversation":
            history_ratio = 0.7  # 对话需要更多历史
            memory_ratio = 0.4
        elif task_type == "search":
            history_ratio = 0.2
            memory_ratio = 0.9  # 搜索任务需要最多记忆
        elif task_type == "creative":
            history_ratio = 0.4
            memory_ratio = 0.3

        # 1. 系统提示
        if system_prompt:
            sys_tokens = self._estimate_tokens(system_prompt)
            parts.append(("system", system_prompt, sys_tokens))
            budget -= sys_tokens

        # 2. 长期记忆（高优先级）
        if include_memory and self._memory_system:
            long_term_text = self._get_long_term_context()
            if long_term_text:
                lt_tokens = self._estimate_tokens(long_term_text)
                lt_budget = min(lt_tokens, 300 if task_type != "search" else 500)
                parts.append(("long_term_memory", long_term_text, lt_budget))
                budget -= lt_budget

        # 3. 持久化记忆
        if include_memory and self._persistent_memories:
            persist_text = self._get_persistent_context()
            if persist_text:
                p_tokens = self._estimate_tokens(persist_text)
                parts.append(("persistent_memory", persist_text, min(p_tokens, 200)))
                budget -= min(p_tokens, 200)

        # 4. 对话历史（含摘要 + CoT 上下文）
        if include_history and self._conversation_history:
            history_budget = int(budget * history_ratio)
            history_text = self._get_history_context(history_budget)

            # 代码任务不注入 CoT 上下文（减少噪音）
            if task_type != "code":
                cot_text = self.get_cot_context(max_turns=2)
                if cot_text:
                    history_text = f"{history_text}\n\n【推理上下文】\n{cot_text}"
            if history_text:
                h_tokens = self._estimate_tokens(history_text)
                parts.append(("history", history_text, h_tokens))
                budget -= h_tokens

        # 5. 相关记忆（按任务相关性检索）
        if include_memory:
            relevant_budget = int(budget * memory_ratio)
            relevant_text = self._get_relevant_context(task, relevant_budget)
            if relevant_text:
                r_tokens = self._estimate_tokens(relevant_text)
                parts.append(("relevant_memory", relevant_text, r_tokens))
                budget -= r_tokens

        # 6. 当前任务
        task_tokens = self._estimate_tokens(task)
        parts.append(("task", task, task_tokens))

        # 组装
        context = ""
        for part_type, content, tokens in parts:
            if part_type == "system":
                context += f"[系统] {content}\n\n"
            elif part_type == "long_term_memory":
                context += f"【长期记忆】\n{content}\n\n"
            elif part_type == "persistent_memory":
                context += f"【持久记忆】\n{content}\n\n"
            elif part_type == "history":
                context += f"【对话历史】\n{content}\n\n"
            elif part_type == "relevant_memory":
                context += f"【相关记忆】\n{content}\n\n"
            elif part_type == "task":
                context += f"[用户] {content}\n[助手] "

        return context

    def build_context_adaptive(self, task: str, system_prompt: str = "",
                                include_history: bool = True,
                                include_memory: bool = True,
                                max_tokens: int = None) -> tuple:
        """
        自适应上下文构建 — 自动检测任务类型并调整策略。

        Returns:
            (context: str, task_type: str) 元组
        """
        task_type = self.detect_task_type(task)
        context = self.build_context(
            task, system_prompt, include_history, include_memory,
            max_tokens, task_type=task_type,
        )
        return context, task_type

    def build_messages(self, task: str, system_prompt: str = "",
                       include_history: bool = True,
                       include_memory: bool = True) -> List[Dict]:
        """构建消息列表（用于 ReAct 引擎）"""
        messages = []

        # 系统提示 + 记忆
        sys_parts = []
        if system_prompt:
            sys_parts.append(system_prompt)

        if include_memory:
            lt = self._get_long_term_context()
            if lt:
                sys_parts.append(f"【长期记忆】\n{lt}")

            p = self._get_persistent_context()
            if p:
                sys_parts.append(f"【持久记忆】\n{p}")

            # 相关记忆
            relevant = self._get_relevant_context(task, 500)
            if relevant:
                sys_parts.append(f"【相关记忆】\n{relevant}")

            wm = self._get_working_memory_context(300)
            if wm:
                sys_parts.append(f"【近期记忆】\n{wm}")

        if sys_parts:
            messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

        # 对话历史
        if include_history:
            history = self._get_recent_history_messages()
            messages.extend(history)

        # 当前任务
        messages.append({"role": "user", "content": task})

        return messages

    # ─── 相关性检索核心 ─────────────────────────────

    def _get_relevant_context(self, task: str, token_budget: int) -> str:
        """根据任务内容检索相关记忆（语义检索优先，关键词回退）"""
        # 优先使用语义检索
        if self._semantic_memory:
            try:
                semantic_results = self._semantic_memory.search(task, top_k=5, min_score=0.3)
                if semantic_results:
                    lines = []
                    used_tokens = 0
                    for r in semantic_results:
                        truncated = r["content"][:200]
                        tokens = self._estimate_tokens(truncated)
                        if used_tokens + tokens > token_budget:
                            break
                        lines.append(f"- [{r['key']}] {truncated}")
                        used_tokens += tokens
                    return "\n".join(lines)
            except Exception:
                pass

        # 回退到关键词匹配
        if not self._context_cache and not self._persistent_memories:
            return ""

        task_keywords = self._extract_keywords(task)
        if not task_keywords:
            return ""

        all_items = {}
        all_items.update(self._context_cache)
        all_items.update(self._persistent_memories)

        scored = []
        for key, item in all_items.items():
            score = self._relevance_score(task_keywords, item)
            if score > 0.1:
                scored.append((score, key, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        lines = []
        used_tokens = 0
        for score, key, item in scored[:10]:
            truncated = item.content[:200]
            tokens = self._estimate_tokens(truncated)
            if used_tokens + tokens > token_budget:
                break
            lines.append(f"- [{key}] {truncated}")
            used_tokens += tokens

        return "\n".join(lines)

    def _relevance_score(self, query_keywords: List[str], item: ContextItem) -> float:
        """计算查询与记忆条目的相关性分数"""
        if not query_keywords:
            return 0.0

        # 关键词匹配
        item_keywords = set(item.keywords)
        query_set = set(query_keywords)

        # Jaccard 相似度
        intersection = item_keywords & query_set
        union = item_keywords | query_set

        if not union:
            return 0.0

        keyword_score = len(intersection) / len(union)

        # 内容匹配（包含查询关键词）
        content_lower = item.content.lower()
        content_matches = sum(1 for kw in query_keywords if kw in content_lower)
        content_score = content_matches / len(query_keywords) if query_keywords else 0

        # 综合分数：关键词匹配 40% + 内容匹配 40% + 重要度 20%
        total = keyword_score * 0.4 + content_score * 0.4 + item.importance * 0.2

        # 时间衰减
        age_hours = (time.time() - item.timestamp) / 3600
        decay = math.exp(-item.decay_rate * age_hours)

        # 访问频率加权
        access_boost = min(1.0, item.access_count * 0.1)

        return total * decay * (1 + access_boost)

    def _auto_importance(self, content: str, source: str) -> float:
        """自动计算记忆重要度"""
        base = 0.5

        # 来源加权
        source_weights = {
            "user_input": 0.7,
            "interaction": 0.6,
            "tool_result": 0.5,
            "file_read": 0.4,
            "history": 0.3,
            "system": 0.8,
        }
        base = source_weights.get(source, 0.5)

        # 内容特征加权
        content_lower = content.lower()

        # 包含代码 → 更重要
        if any(kw in content_lower for kw in ["def ", "class ", "import ", "function "]):
            base += 0.1

        # 包含错误 → 更重要（需要记住教训）
        if any(kw in content_lower for kw in ["error", "错误", "失败", "exception"]):
            base += 0.15

        # 包含用户偏好 → 更重要
        if any(kw in content_lower for kw in ["喜欢", "偏好", "prefer", "always", "总是"]):
            base += 0.1

        # 内容很短 → 不太重要
        if len(content) < 20:
            base -= 0.1

        # 内容很长 → 可能是重要文档
        if len(content) > 1000:
            base += 0.05

        return max(0.1, min(1.0, base))

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单的分词 + 停用词过滤）"""
        if not text:
            return []

        # 中文分词（简单按字/词分割）
        # 提取中文词（2-4字）
        chinese_words = re.findall(r'[一-鿿]{2,4}', text)

        # 提取英文词
        english_words = re.findall(r'[a-zA-Z_]{3,}', text.lower())

        # 停用词
        stop_words = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
            'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some', 'them', 'than',
            'its', 'over', 'only', 'into', 'such', 'that', 'this', 'with', 'will',
        }

        keywords = []
        for w in chinese_words + english_words:
            if w not in stop_words and len(w) >= 2:
                keywords.append(w)

        # 去重并限制数量
        return list(dict.fromkeys(keywords))[:20]

    # ─── 上下文获取方法 ─────────────────────────────

    def _get_long_term_context(self) -> str:
        if not self._memory_system:
            return ""
        lines = []
        for i, slot in enumerate(self._memory_system.long_term):
            if not slot.is_empty():
                lines.append(f"L{i}: {slot.content[:200]}")
        return "\n".join(lines)

    def _get_persistent_context(self) -> str:
        if not self._persistent_memories:
            return ""
        # 按重要度排序
        sorted_items = sorted(
            self._persistent_memories.values(),
            key=lambda x: x.importance,
            reverse=True
        )[:5]
        return "\n".join(f"- {item.content[:150]}" for item in sorted_items)

    def _get_working_memory_context(self, token_budget: int) -> str:
        if not self._working_memory:
            return ""
        all_memories = self._working_memory.export_all()
        if not all_memories:
            return ""
        # 按访问次数排序
        sorted_memories = sorted(
            all_memories.items(),
            key=lambda x: self._working_memory._memory.get(x[0], None).access_count if x[0] in self._working_memory._memory else 0,
            reverse=True
        )
        lines = []
        used_tokens = 0
        for key, content in sorted_memories:
            truncated = content[:200]
            tokens = self._estimate_tokens(truncated)
            if used_tokens + tokens > token_budget:
                break
            lines.append(f"- [{key}] {truncated}")
            used_tokens += tokens
        return "\n".join(lines)

    def _get_history_context(self, token_budget: int) -> str:
        """获取对话历史（含旧对话摘要）"""
        parts = []

        # 旧对话摘要
        if self._conversation_summaries:
            summary_text = "【历史摘要】\n" + "\n".join(self._conversation_summaries[-3:])
            parts.append(summary_text)

        # 近期对话
        if self._conversation_history:
            lines = []
            used_tokens = 0
            for msg in reversed(self._conversation_history):
                role = msg["role"]
                content = msg["content"][:500]
                line = f"{'用户' if role == 'user' else '助手'}: {content}"
                tokens = self._estimate_tokens(line)
                if used_tokens + tokens > token_budget:
                    break
                lines.insert(0, line)
                used_tokens += tokens
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _get_recent_history_messages(self, max_turns: int = 10) -> List[Dict]:
        recent = self._conversation_history[-max_turns * 2:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_cot_context(self, max_turns: int = 3) -> str:
        """
        获取最近几轮的 chain-of-thought 上下文。

        包含思考过程、工具调用、推理链等信息，
        帮助模型在多轮推理中保持连贯性。
        """
        recent = self._conversation_history[-max_turns * 2:]
        if not recent:
            return ""

        parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:300]

            # 包含思考过程
            thought = msg.get("thought", "")
            reasoning = msg.get("reasoning", "")
            tool_calls = msg.get("tool_calls", [])

            entry = f"{role}: {content}"
            if thought:
                entry += f"\n  [思考] {thought[:100]}"
            if reasoning:
                entry += f"\n  [推理] {reasoning[:100]}"
            if tool_calls:
                tools = ", ".join([t.get("tool", "") for t in tool_calls[:3]])
                entry += f"\n  [工具] {tools}"

            parts.append(entry)

        return "\n".join(parts)

    def _compress_messages(self, messages: List[Dict]) -> str:
        """压缩多条消息为摘要（尝试用模型，回退到简单提取）"""
        if not messages:
            return ""

        # 尝试用模型生成摘要
        try:
            from taiji.core.app_state import app_state
            taiji = app_state.get_taiji_engine()
            tokenizer = app_state.get_tokenizer()
            if taiji and tokenizer:
                # 构建摘要 prompt
                conversation = ""
                for m in messages:
                    role = "用户" if m["role"] == "user" else "助手"
                    conversation += f"{role}: {m['content'][:100]}\n"

                prompt = f"请用一句话总结以下对话的要点：\n{conversation}\n摘要："
                summary = taiji.generate(prompt, tokenizer, max_new_tokens=100, temperature=0.3)
                if summary and len(summary) > 10:
                    return summary.strip()[:200]
        except Exception:
            pass

        # 回退到简单提取
        user_msgs = [m["content"][:100] for m in messages if m["role"] == "user"]
        ai_msgs = [m["content"][:100] for m in messages if m["role"] == "assistant"]

        parts = []
        if user_msgs:
            parts.append(f"用户讨论了: {'; '.join(user_msgs[:3])}")
        if ai_msgs:
            parts.append(f"助手回复了: {'; '.join(ai_msgs[:3])}")

        return " | ".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    # ─── 记忆衰减 ─────────────────────────────────────

    def decay_memories(self):
        """执行记忆衰减（定期调用）"""
        now = time.time()
        expired = []

        for key, item in self._context_cache.items():
            age_hours = (now - item.timestamp) / 3600
            decay = math.exp(-item.decay_rate * age_hours)

            # 重要度低于阈值，标记为过期
            effective_importance = item.importance * decay
            if effective_importance < 0.05 and item.access_count == 0:
                expired.append(key)

        # 清理过期记忆
        for key in expired:
            del self._context_cache[key]
            # 清理关键词索引
            for kw_keys in self._keyword_index.values():
                if key in kw_keys:
                    kw_keys.remove(key)

        if expired:
            logger.info(f"Decayed {len(expired)} expired memories")

    # ─── 睡眠整合 ─────────────────────────────────────

    def consolidate_for_sleep(self):
        """
        睡眠时整合记忆。

        1. 将高频访问的短期记忆提升为长期记忆
        2. 压缩低重要度记忆
        3. 保存持久化记忆
        """
        # 提升高频记忆到 MemorySystem
        if self._memory_system:
            for key, item in self._context_cache.items():
                if item.access_count >= 3 and item.importance >= 0.5:
                    self._memory_system.auto_write(item.content, item.importance)

            # 执行巩固
            consolidated = self._memory_system.consolidate()
            if consolidated:
                logger.info(f"Consolidated {len(consolidated)} memories during sleep")

        # 保存持久化记忆
        self._save_persistent()

    # ─── 持久化 ─────────────────────────────────────

    def _save_persistent(self):
        if not self._persistent_path:
            return
        os.makedirs(os.path.dirname(self._persistent_path), exist_ok=True)
        data = {
            "memories": {
                k: {
                    "content": v.content,
                    "source": v.source,
                    "importance": v.importance,
                    "timestamp": v.timestamp,
                    "key": v.key,
                    "access_count": v.access_count,
                    "keywords": v.keywords,
                }
                for k, v in self._persistent_memories.items()
            },
            "conversation_summaries": self._conversation_summaries,
            "saved_at": datetime.now().isoformat(),
        }
        try:
            with open(self._persistent_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save persistent memory: {e}")

    def _load_persistent(self):
        if not self._persistent_path or not os.path.exists(self._persistent_path):
            return
        try:
            with open(self._persistent_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.get("memories", {}).items():
                self._persistent_memories[k] = ContextItem(
                    content=v["content"],
                    source=v.get("source", "persistent"),
                    importance=v.get("importance", 0.5),
                    timestamp=v.get("timestamp", 0),
                    key=v.get("key", k),
                    access_count=v.get("access_count", 0),
                    keywords=v.get("keywords", []),
                )
            self._conversation_summaries = data.get("conversation_summaries", [])
            logger.info(f"Loaded {len(self._persistent_memories)} persistent memories")
        except Exception as e:
            logger.warning(f"Failed to load persistent memory: {e}")

    # ─── 状态查询 ─────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "context_cache_size": len(self._context_cache),
            "history_length": len(self._conversation_history),
            "conversation_summaries": len(self._conversation_summaries),
            "persistent_memories": len(self._persistent_memories),
            "keyword_index_size": len(self._keyword_index),
            "max_context_tokens": self.max_context_tokens,
            "working_memory": self._working_memory.get_status() if self._working_memory else None,
            "memory_system": self._memory_system.get_stats() if self._memory_system else None,
        }


# ─── 全局实例 ─────────────────────────────────────

_global_context: Optional[ContextManager] = None


def get_context_manager(max_context_tokens: int = 2048, tokenizer=None) -> ContextManager:
    global _global_context
    if _global_context is None:
        _global_context = ContextManager(max_context_tokens, tokenizer)
    return _global_context
