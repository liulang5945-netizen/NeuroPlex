"""
态极语义记忆 (Semantic Memory)
================================

使用 sentence-transformers 实现向量检索，
让记忆按语义相关性而非关键词匹配来检索。

核心能力：
1. 自动向量化所有记忆内容
2. 余弦相似度检索
3. 增量更新（新记忆自动索引）
4. 持久化向量索引

依赖：
    pip install sentence-transformers numpy
"""
import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("SemanticMemory")


class SemanticMemory:
    """
    语义记忆系统

    使用 sentence-transformers 将记忆向量化，
    通过余弦相似度实现语义检索。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 persist_dir: str = None):
        """
        Args:
            model_name: sentence-transformers 模型名
            persist_dir: 向量索引持久化目录
        """
        self.model_name = model_name
        self.persist_dir = persist_dir or os.path.join("taiji_data", "semantic_memory")

        # 记忆存储
        self._memories: Dict[str, dict] = {}  # key -> {content, embedding, metadata}
        self._embedder = None
        self._embedding_dim = 384  # MiniLM 默认维度

        os.makedirs(self.persist_dir, exist_ok=True)
        self._load_index()

        logger.info(f"SemanticMemory initialized (model={model_name}, {len(self._memories)} memories)")

    def _get_embedder(self):
        """延迟加载 embedder"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed: pip install sentence-transformers")
                return None
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None
        return self._embedder

    def add(self, key: str, content: str, metadata: dict = None):
        """
        添加记忆并生成向量。

        Args:
            key: 记忆标识
            content: 记忆内容
            metadata: 附加元数据
        """
        embedder = self._get_embedder()
        if embedder is None:
            return

        try:
            # 生成向量
            embedding = embedder.encode(content, convert_to_numpy=True)

            self._memories[key] = {
                "content": content,
                "embedding": embedding.tolist(),
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            logger.debug(f"Added semantic memory: {key}")
        except Exception as e:
            logger.warning(f"Failed to add semantic memory: {e}")

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.3) -> List[dict]:
        """
        语义检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值

        Returns:
            [{key, content, score, metadata}, ...]
        """
        if not self._memories:
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return []

        try:
            # 查询向量化
            query_embedding = embedder.encode(query, convert_to_numpy=True)

            # 计算余弦相似度
            results = []
            for key, mem in self._memories.items():
                mem_embedding = np.array(mem["embedding"])
                score = self._cosine_similarity(query_embedding, mem_embedding)

                if score >= min_score:
                    results.append({
                        "key": key,
                        "content": mem["content"][:200],
                        "score": float(score),
                        "metadata": mem.get("metadata", {}),
                    })

            # 按相似度排序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def remove(self, key: str):
        """删除记忆"""
        if key in self._memories:
            del self._memories[key]

    def clear(self):
        """清空所有记忆"""
        self._memories.clear()

    def size(self) -> int:
        """返回记忆数量"""
        return len(self._memories)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ─── 持久化 ─────────────────────────────────────

    def save(self):
        """保存向量索引"""
        path = os.path.join(self.persist_dir, "semantic_index.json")
        try:
            data = {
                "model_name": self.model_name,
                "embedding_dim": self._embedding_dim,
                "memories": {
                    k: {
                        "content": v["content"],
                        "embedding": v["embedding"],
                        "metadata": v.get("metadata", {}),
                        "timestamp": v.get("timestamp", 0),
                    }
                    for k, v in self._memories.items()
                },
                "saved_at": time.time(),
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info(f"Saved {len(self._memories)} semantic memories")
        except Exception as e:
            logger.warning(f"Failed to save semantic memory: {e}")

    def _load_index(self):
        """加载向量索引"""
        path = os.path.join(self.persist_dir, "semantic_index.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for k, v in data.get("memories", {}).items():
                self._memories[k] = {
                    "content": v["content"],
                    "embedding": v["embedding"],
                    "metadata": v.get("metadata", {}),
                    "timestamp": v.get("timestamp", 0),
                }

            logger.info(f"Loaded {len(self._memories)} semantic memories")
        except Exception as e:
            logger.warning(f"Failed to load semantic memory: {e}")

    # ─── 批量操作 ─────────────────────────────────────

    def batch_add(self, items: List[Tuple[str, str, dict]]):
        """
        批量添加记忆（更高效）。

        Args:
            items: [(key, content, metadata), ...]
        """
        embedder = self._get_embedder()
        if embedder is None:
            return

        try:
            contents = [item[1] for item in items]
            embeddings = embedder.encode(contents, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

            for (key, content, metadata), embedding in zip(items, embeddings):
                self._memories[key] = {
                    "content": content,
                    "embedding": embedding.tolist(),
                    "metadata": metadata or {},
                    "timestamp": time.time(),
                }

            logger.info(f"Batch added {len(items)} semantic memories")
        except Exception as e:
            logger.warning(f"Batch add failed: {e}")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_memories": len(self._memories),
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "persist_dir": self.persist_dir,
        }


# ─── 全局实例 ─────────────────────────────────────

_global_semantic: Optional[SemanticMemory] = None


def get_semantic_memory() -> SemanticMemory:
    """获取全局语义记忆实例"""
    global _global_semantic
    if _global_semantic is None:
        _global_semantic = SemanticMemory()
    return _global_semantic
