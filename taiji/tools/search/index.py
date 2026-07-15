"""
索引层 (Index Layer)
====================

自建倒排索引 + BM25 排序。

核心改进：
- 词级分词（中文双字 + 英文单词），不再用 BPE token ID
- 内存索引 + JSON 持久化（sandbox 兼容）
- 增量更新：add_page() 实时加入索引
"""

import os
import re
import json
import time
import math
import logging
import threading
import atexit
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("Taiji.Search.Index")


@dataclass
class IndexedPage:
    """索引中的一页"""
    url: str = ""
    title: str = ""
    text: str = ""
    links: List[str] = field(default_factory=list)
    crawled_at: float = 0.0
    word_count: int = 0
    source: str = ""          # discovery / crawl / browse


@dataclass
class SearchHit:
    """搜索命中"""
    url: str = ""
    title: str = ""
    snippet: str = ""
    score: float = 0.0


# ═══════════════════════════════════════════════
# 词级分词器
# ═══════════════════════════════════════════════

class Tokenizer:
    """
    词级分词，与搜索引擎需求匹配。

    中文：单字 + 双字滑动窗口（bigram）
    英文：按空格和标点分词，转小写
    数字：保留连续数字串

    这样 BM25 的 tf/df 统计才有语义意义。
    """

    CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
    WORD_PATTERN = re.compile(r"[a-zA-Z]{2,}|\d+|[\u4e00-\u9fff]")

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
        text = text.lower()
        tokens: List[str] = []
        i = 0
        while i < len(text):
            char = text[i]
            if self.CJK_PATTERN.match(char):
                # 中文字符：单字 + 与下一个字的 bigram
                tokens.append(char)
                if i + 1 < len(text) and self.CJK_PATTERN.match(text[i + 1]):
                    tokens.append(char + text[i + 1])
                i += 1
            elif char.isalpha():
                # 英文：提取完整单词
                j = i
                while j < len(text) and text[j].isalpha():
                    j += 1
                word = text[i:j]
                if len(word) >= 2:
                    tokens.append(word)
                i = j
            elif char.isdigit():
                # 数字串
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j])
                i = j
            else:
                i += 1
        return tokens

    def tokenize_query(self, query: str) -> List[str]:
        """查询分词（与文档分词相同）"""
        return self.tokenize(query)


# ═══════════════════════════════════════════════
# 倒排索引 + BM25
# ═══════════════════════════════════════════════

class InvertedIndex:
    """
    内存倒排索引 + BM25 排序。

    数据结构：
    - _pages: List[IndexedPage]           所有页面
    - _postings: Dict[token, Set[int]]    倒排表（token → 页面 ID 集合）
    - _doc_tokens: Dict[int, List[str]]   每页的 token 列表（用于 tf 计算）
    - _url_to_idx: Dict[url, int]         URL → 页面 ID 映射
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join("taiji_data", "search_index")
        os.makedirs(self.data_dir, exist_ok=True)
        self._tokenizer = Tokenizer()
        self._pages: List[IndexedPage] = []
        self._postings: Dict[str, Set[int]] = defaultdict(set)
        self._doc_tokens: Dict[int, List[str]] = {}
        self._url_to_idx: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._load()
        atexit.register(self.save)

    # ─── 增量更新 ───

    def add_page(self, page: IndexedPage) -> bool:
        """添加一页到索引（已存在则更新）"""
        with self._lock:
            # 已存在则先删除旧版本
            if page.url in self._url_to_idx:
                self._remove_page_locked(self._url_to_idx[page.url])

            idx = len(self._pages)
            self._pages.append(page)
            self._url_to_idx[page.url] = idx

            tokens = self._tokenizer.tokenize(f"{page.title} {page.text}")
            self._doc_tokens[idx] = tokens
            for token in set(tokens):
                self._postings[token].add(idx)

            logger.debug(f"  索引 +1: {page.title[:40]} ({len(tokens)} tokens)")
            return True

    def add_pages(self, pages: List[IndexedPage]) -> int:
        """批量添加"""
        count = 0
        for p in pages:
            if self.add_page(p):
                count += 1
        return count

    def _remove_page_locked(self, idx: int):
        """删除一页（调用者需持锁）"""
        if idx >= len(self._pages):
            return
        url = self._pages[idx].url
        tokens = self._doc_tokens.pop(idx, [])
        for token in set(tokens):
            self._postings[token].discard(idx)
            if not self._postings[token]:
                del self._postings[token]
        self._url_to_idx.pop(url, None)

    # ─── 搜索 ───

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """BM25 搜索"""
        terms = self._tokenizer.tokenize_query(query)
        if not terms or not self._pages:
            return []

        with self._lock:
            N = len(self._pages)
            avgdl = sum(len(toks) for toks in self._doc_tokens.values()) / max(N, 1)
            k1, b = 1.5, 0.75
            scored: List[Tuple[float, int]] = []

            # 找候选文档（至少包含一个查询词）
            candidates: Set[int] = set()
            for term in terms:
                candidates.update(self._postings.get(term, set()))
            if not candidates:
                return []

            for idx in candidates:
                tokens = self._doc_tokens.get(idx, [])
                doc_len = len(tokens)
                if doc_len == 0:
                    continue
                score = 0.0
                for term in terms:
                    tf = tokens.count(term)
                    if tf == 0:
                        continue
                    df = len(self._postings.get(term, set()))
                    idf = max(0, math.log((N - df + 0.5) / (df + 0.5) + 1))
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1))
                    score += idf * numerator / max(denominator, 0.1)
                if score > 0:
                    scored.append((score, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            hits: List[SearchHit] = []
            for score, idx in scored[:top_k]:
                page = self._pages[idx]
                snippet = self._make_snippet(page.text, terms)
                hits.append(SearchHit(
                    url=page.url, title=page.title,
                    snippet=snippet, score=round(score, 2),
                ))
            return hits

    def _make_snippet(self, text: str, terms: List[str], window: int = 100) -> str:
        """生成摘要"""
        low = text.lower()
        for t in terms:
            idx = low.find(t.lower())
            if idx >= 0:
                start = max(0, idx - window // 2)
                end = min(len(text), idx + window)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                return prefix + text[start:end] + suffix
        return text[:window * 2]

    # ─── 持久化 ───

    def _json_path(self) -> str:
        return os.path.join(self.data_dir, "index.json")

    def _load(self):
        """从 JSON 加载索引"""
        try:
            path = self._json_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._pages = [IndexedPage(**p) for p in data.get("pages", [])]
            # 重建倒排表
            self._postings = defaultdict(set)
            self._doc_tokens = {}
            self._url_to_idx = {}
            for idx, page in enumerate(self._pages):
                self._url_to_idx[page.url] = idx
                tokens = self._tokenizer.tokenize(f"{page.title} {page.text}")
                self._doc_tokens[idx] = tokens
                for token in set(tokens):
                    self._postings[token].add(idx)
            logger.info(f"索引已加载: {len(self._pages)} 页, {len(self._postings)} 词项")
        except Exception as e:
            logger.debug(f"索引加载失败: {e}")
            self._pages = []

    def save(self):
        """保存到 JSON"""
        try:
            with self._lock:
                data = {
                    "pages": [
                        {
                            "url": p.url, "title": p.title[:200],
                            "text": p.text[:8000], "links": p.links[:30],
                            "crawled_at": p.crawled_at, "word_count": p.word_count,
                            "source": p.source,
                        }
                        for p in self._pages
                    ],
                    "saved_at": time.time(),
                }
                with open(self._json_path(), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logger.info(f"索引已保存: {len(self._pages)} 页")
        except Exception as e:
            logger.debug(f"索引保存失败: {e}")

    # ─── 管理 ───

    def stats(self) -> dict:
        return {
            "pages": len(self._pages),
            "terms": len(self._postings),
            "data_dir": self.data_dir,
        }

    def clear(self):
        with self._lock:
            self._pages = []
            self._postings = defaultdict(set)
            self._doc_tokens = {}
            self._url_to_idx = {}
            self.save()

    def has_url(self, url: str) -> bool:
        return url in self._url_to_idx


# ═══════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════

_default_index: Optional[InvertedIndex] = None


def get_index() -> InvertedIndex:
    global _default_index
    if _default_index is None:
        _default_index = InvertedIndex()
    return _default_index
