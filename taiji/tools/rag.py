"""RAG knowledge base with dense and BM25 retrieval."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import re
import threading
import time
from collections import Counter
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Tuple

from taiji.core.memory_watchdog import MemoryWatchdog, memory_guarded
from taiji.core.utils import safe_json_load, safe_json_save
from taiji.services.settings_service import load_settings, update_settings
from taiji.tools.file_parser import parse_file_to_text

logger = logging.getLogger("RAG")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4
RRF_K = 60
HYBRID_CANDIDATE_K = 20

_PERSIST_DOCS = "rag_documents.json"
_PERSIST_INDEX_JSON = "rag_index.json"
_PERSIST_INDEX_PKL = "rag_index.pkl"
_PERSIST_EMBEDDINGS = "rag_embeddings.npy"
_PERSIST_BM25_JSON = "rag_bm25.json"
_PERSIST_BM25_PKL = "rag_bm25.pkl"

_CJK_BLOCK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f.!?;\uff1b])\s+")


def _fallback_tokens(text: str) -> List[str]:
    """Return deterministic local tokens without external tokenizers."""

    tokens: List[str] = []
    for piece in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", str(text).lower()):
        if _CJK_BLOCK_RE.fullmatch(piece):
            chars = [ch for ch in piece if ch.strip()]
            tokens.extend(chars)
            if len(piece) >= 2:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
            if len(piece) <= 4:
                tokens.append(piece)
        else:
            tokens.append(piece)
    return tokens


class RAGConfig:
    """RAG retrieval configuration persisted into the unified settings store."""

    _instance: "RAGConfig | None" = None
    _lock = threading.Lock()

    DEFAULTS = {
        "enable_hybrid": True,
        "enable_reranker": True,
        "enable_query_rewrite": False,
        "dense_weight": 0.6,
        "bm25_weight": 0.4,
        "candidate_k": 20,
        "reranker_model": DEFAULT_RERANKER_MODEL,
    }

    def __new__(cls) -> "RAGConfig":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._config = dict(cls.DEFAULTS)
                instance._load_from_settings()
                cls._instance = instance
            return cls._instance

    def _load_from_settings(self) -> None:
        try:
            data = load_settings()
            for key in self.DEFAULTS:
                settings_key = f"rag_{key}"
                if settings_key in data:
                    self._config[key] = data[settings_key]
        except Exception as exc:
            logger.warning("Failed to load RAG settings: %s", exc)

    def save_to_settings(self) -> None:
        try:
            updates = {f"rag_{key}": value for key, value in self._config.items()}
            update_settings(updates)
        except Exception as exc:
            logger.warning("Failed to save RAG settings: %s", exc)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    def to_dict(self) -> dict:
        return dict(self._config)

    def update(self, updates: dict) -> None:
        for key, value in updates.items():
            if key in self.DEFAULTS:
                self._config[key] = value
        self.save_to_settings()


class BM25Index:
    """Simple BM25 index with local tokenization fallback."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: List[Counter[str]] = []
        self.idf: Dict[str, float] = {}
        self._stopwords = self._default_stopwords()

    @staticmethod
    def _default_stopwords() -> set[str]:
        return {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "from",
            "by",
            "and",
            "or",
            "but",
            "not",
            "no",
            "if",
            "then",
            "that",
            "this",
            "it",
            "its",
            "as",
            "how",
            "what",
            "when",
            "where",
            "who",
            "which",
        }

    def _normalize_tokens(self, tokens: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        for token in tokens:
            value = str(token).strip().lower()
            if not value or value in self._stopwords:
                continue
            if _CJK_BLOCK_RE.fullmatch(value):
                cleaned.append(value)
            elif len(value) > 1:
                cleaned.append(value)
        return cleaned

    def _tokenize(self, text: str) -> List[str]:
        lowered = str(text).lower()
        try:
            import jieba

            return self._normalize_tokens(jieba.lcut(lowered))
        except ImportError:
            return self._normalize_tokens(_fallback_tokens(lowered))

    def build(self, chunks: List[Tuple[str, str, int]]) -> None:
        self.doc_count = len(chunks)
        self.avg_dl = 0.0
        self.doc_lengths = []
        self.doc_freqs = []
        self.idf = {}

        df_counter: Counter[str] = Counter()
        for _, chunk_text, _ in chunks:
            tokens = self._tokenize(chunk_text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_lengths.append(len(tokens))
            for term in freq:
                df_counter[term] += 1

        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)
        for term, df in df_counter.items():
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for doc_idx in range(self.doc_count):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]
            freq = self.doc_freqs[doc_idx]
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = freq.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_dl, 1.0)
                )
                score += idf * numerator / max(denominator, 1e-10)
            if score > 0:
                scores.append((doc_idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": [dict(freq) for freq in self.doc_freqs],
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        obj = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        obj.doc_count = int(data.get("doc_count", 0))
        obj.avg_dl = float(data.get("avg_dl", 0.0))
        obj.doc_lengths = [int(item) for item in data.get("doc_lengths", [])]
        obj.doc_freqs = [Counter(freq) for freq in data.get("doc_freqs", [])]
        obj.idf = {str(key): float(value) for key, value in data.get("idf", {}).items()}
        return obj


class CrossEncoderReranker:
    """Optional cross-encoder reranker with lazy loading."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or RAGConfig().get(
            "reranker_model",
            DEFAULT_RERANKER_MODEL,
        )
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name, device="cpu")
            except Exception as exc:
                logger.warning("Failed to load reranker model %s: %s", self._model_name, exc)
                self._model = None

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    def rerank(self, query: str, passages: List[str], top_k: int = DEFAULT_TOP_K) -> List[Tuple[int, float]]:
        if not passages:
            return []

        self._load_model()
        if self._model is None:
            return [(index, 0.0) for index in range(min(top_k, len(passages)))]

        try:
            pairs = [(query, passage) for passage in passages]
            scores = self._model.predict(pairs)
            scored = [(index, float(score)) for index, score in enumerate(scores)]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]
        except Exception as exc:
            logger.warning("Rerank failed: %s", exc)
            return [(index, 0.0) for index in range(min(top_k, len(passages)))]


class RAGKnowledgeBase:
    """Semantic knowledge base with persistence and local fallback embeddings."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self.documents: Dict[str, str] = {}
        self.chunks: List[Tuple[str, str, int]] = []
        self.embeddings: Any | None = None
        self._embedder: Any | None = None
        self._embed_dim = 0
        self.persist_dir = persist_dir
        self._bm25_index: Optional[BM25Index] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._rag_config = RAGConfig()

        if self.persist_dir and self._has_persisted_state(self.persist_dir):
            self._load()

    @staticmethod
    def _has_persisted_state(persist_dir: str) -> bool:
        return any(
            os.path.exists(os.path.join(persist_dir, name))
            for name in (
                _PERSIST_DOCS,
                _PERSIST_INDEX_JSON,
                _PERSIST_INDEX_PKL,
                _PERSIST_EMBEDDINGS,
                _PERSIST_BM25_JSON,
                _PERSIST_BM25_PKL,
            )
        )

    def _invalidate_index(self, reset_embedder: bool = False) -> None:
        self.chunks = []
        self.embeddings = None
        self._bm25_index = None
        if reset_embedder:
            self._embedder = None
            self._embed_dim = 0

    def _fallback_embed_texts(self, texts: List[str]) -> Any:
        import numpy as np

        dim = 256
        vectors: List[Any] = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for token in _fallback_tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec)

        self._embed_dim = dim
        if not vectors:
            return np.zeros((0, dim), dtype=np.float32)
        return np.vstack(vectors)

    @memory_guarded(min_avail_pct=0.15, on_critical="raise")
    def _get_embedder(self) -> Any | None:
        if self._embedder is not None:
            return self._embedder

        model_name = os.environ.get("TAIJI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                model_name,
                device="cpu",
                cache_folder=os.environ.get("TAIJI_CACHE_DIR"),
            )
            self._embed_dim = int(self._embedder.get_sentence_embedding_dimension())
            return self._embedder
        except Exception as exc:
            if model_name != FALLBACK_EMBEDDING_MODEL:
                logger.warning("Primary embedding model failed (%s), trying fallback", exc)
                try:
                    from sentence_transformers import SentenceTransformer

                    self._embedder = SentenceTransformer(
                        FALLBACK_EMBEDDING_MODEL,
                        device="cpu",
                        cache_folder=os.environ.get("TAIJI_CACHE_DIR"),
                    )
                    self._embed_dim = int(self._embedder.get_sentence_embedding_dimension())
                    return self._embedder
                except Exception as fallback_exc:
                    logger.warning("Fallback embedding model failed: %s", fallback_exc)
            else:
                logger.warning("Embedding model failed: %s", exc)

        self._embedder = None
        self._embed_dim = 256
        return None

    def _build_embeddings(self, texts: List[str], batch_size: int = 32) -> Any:
        import numpy as np

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        try:
            embedder = self._get_embedder()
        except MemoryError as exc:
            logger.warning("Embedding model skipped under memory pressure: %s", exc)
            return self._fallback_embed_texts(texts)

        if embedder is None:
            return self._fallback_embed_texts(texts)

        try:
            embeddings = embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if hasattr(embeddings, "shape") and len(embeddings.shape) == 2:
                self._embed_dim = int(embeddings.shape[1])
            return embeddings
        except Exception as exc:
            logger.warning("Embedding generation failed, using local fallback: %s", exc)
            return self._fallback_embed_texts(texts)

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        if not text or not text.strip():
            return []

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
        if not paragraphs:
            return []

        merged: List[str] = []
        buffer = ""
        for paragraph in paragraphs:
            candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
            if buffer and len(candidate) > chunk_size and len(buffer) >= chunk_size // 2:
                merged.append(buffer)
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            merged.append(buffer)

        chunks: List[str] = []
        for paragraph in merged:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
                continue

            sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(paragraph) if item.strip()]
            if not sentences:
                sentences = [paragraph]

            current = ""
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    if current:
                        chunks.append(current)
                        current = ""
                    for start in range(0, len(sentence), chunk_size):
                        piece = sentence[start : start + chunk_size].strip()
                        if piece:
                            chunks.append(piece)
                    continue

                candidate = sentence if not current else f"{current} {sentence}"
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence

            if current:
                chunks.append(current)

        if overlap <= 0 or len(chunks) <= 1:
            return [chunk for chunk in chunks if chunk.strip()]

        overlapped: List[str] = [chunks[0]]
        for chunk in chunks[1:]:
            previous = overlapped[-1]
            prefix = previous[-overlap:] if len(previous) > overlap else previous
            merged_chunk = f"{prefix}{chunk}".strip()
            overlapped.append(merged_chunk or chunk)
        return [chunk for chunk in overlapped if chunk.strip()]

    def add_file(self, file_path: str) -> None:
        filename = os.path.basename(file_path)
        text = parse_file_to_text(file_path)
        if not text.strip():
            logger.warning("Skipping empty file: %s", filename)
            return
        self.documents[filename] = text
        self._invalidate_index()
        self._save()

    def add_text(self, filename: str, text: str) -> None:
        if not text.strip():
            logger.warning("Skipping empty text: %s", filename)
            return
        self.documents[filename] = text
        self._invalidate_index()
        self._save()

    def remove_file(self, filename: str) -> None:
        self.documents.pop(filename, None)
        self._invalidate_index()
        self._save()

    def get_doc_names(self) -> list:
        return list(self.documents.keys())

    def clear(self) -> None:
        self.documents = {}
        self._invalidate_index(reset_embedder=True)
        if self.persist_dir:
            for name in (
                _PERSIST_DOCS,
                _PERSIST_INDEX_JSON,
                _PERSIST_INDEX_PKL,
                _PERSIST_EMBEDDINGS,
                _PERSIST_BM25_JSON,
                _PERSIST_BM25_PKL,
            ):
                with suppress(OSError):
                    os.remove(os.path.join(self.persist_dir, name))

    def rebuild_index(self) -> str:
        import numpy as np

        self.chunks = []
        self.embeddings = None
        self._bm25_index = None

        all_chunk_texts: List[str] = []
        for filename, text in self.documents.items():
            file_chunks = self._chunk_text(text)
            for index, chunk in enumerate(file_chunks):
                self.chunks.append((filename, chunk, index))
                all_chunk_texts.append(chunk)

        if not all_chunk_texts:
            self._save()
            return "Knowledge base is empty."

        watchdog = MemoryWatchdog()
        can_proceed, message = watchdog.can_proceed(min_avail_pct=0.12)
        if not can_proceed:
            self._save()
            logger.warning("RAG rebuild aborted before embedding build: %s", message)
            return f"Not enough memory to build index. {message}"

        embed_dim = self._embed_dim or 384
        can_build, build_message = MemoryWatchdog.can_build_embeddings(
            len(all_chunk_texts),
            embed_dim,
        )
        if not can_build:
            self._save()
            logger.warning("RAG rebuild aborted by embedding memory estimate: %s", build_message)
            return build_message

        start_time = time.time()
        batch_size = 32
        try:
            try:
                embedder = self._get_embedder()
            except MemoryError as exc:
                logger.warning("Embedding model skipped during rebuild: %s", exc)
                embedder = None

            if embedder is None:
                self.embeddings = self._fallback_embed_texts(all_chunk_texts)
            else:
                all_embeddings: List[Any] = []
                for start in range(0, len(all_chunk_texts), batch_size):
                    if start and watchdog.status.level >= 3:
                        self.embeddings = None
                        self._save()
                        return (
                            "Index build interrupted by memory pressure "
                            f"after {start}/{len(all_chunk_texts)} chunks."
                        )
                    batch = all_chunk_texts[start : start + batch_size]
                    batch_embeddings = embedder.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    all_embeddings.append(batch_embeddings)

                self.embeddings = (
                    np.vstack(all_embeddings)
                    if all_embeddings
                    else np.zeros((0, self._embed_dim or 0), dtype=np.float32)
                )

            if hasattr(self.embeddings, "shape") and len(self.embeddings.shape) == 2:
                self._embed_dim = int(self.embeddings.shape[1])
        except Exception as exc:
            self.embeddings = None
            self._save()
            logger.error("Embedding build failed: %s", exc)
            return f"Failed to build embeddings: {exc}"

        try:
            self._bm25_index = BM25Index()
            self._bm25_index.build(self.chunks)
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)
            self._bm25_index = None

        self._save()
        elapsed = time.time() - start_time
        embedding_dim = self.embeddings.shape[1] if getattr(self.embeddings, "ndim", 0) == 2 else 0
        return (
            f"Index built for {len(self.documents)} docs, "
            f"{len(self.chunks)} chunks, dim {embedding_dim} in {elapsed:.1f}s."
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, str, float]]:
        import numpy as np

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        try:
            query_vec = self._build_embeddings([query])
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        if getattr(query_vec, "shape", (0,))[0] == 0:
            return []

        scores = np.dot(self.embeddings, query_vec[0])
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[str, str, float]] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0.0:
                continue
            filename, chunk_text, _ = self.chunks[int(index)]
            results.append((filename, chunk_text, score))
        return results

    def _dense_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        import numpy as np

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        try:
            query_vec = self._build_embeddings([query])
        except Exception as exc:
            logger.warning("Dense query embedding failed: %s", exc)
            return []

        if getattr(query_vec, "shape", (0,))[0] == 0:
            return []

        scores = np.dot(self.embeddings, query_vec[0])
        top_k = min(top_k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[int, float]] = []
        for index in indices:
            score = float(scores[index])
            if score <= 0.0:
                continue
            results.append((int(index), score))
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int | None = None,
    ) -> List[Tuple[str, str, float]]:
        if candidate_k is None:
            candidate_k = int(self._rag_config.get("candidate_k", HYBRID_CANDIDATE_K))

        dense_results = self._dense_search(query, top_k=candidate_k)
        bm25_results: List[Tuple[int, float]] = []
        if self._bm25_index is not None:
            try:
                bm25_results = self._bm25_index.search(query, top_k=candidate_k)
            except Exception as exc:
                logger.warning("BM25 search failed: %s", exc)

        if not dense_results and not bm25_results:
            return []
        if not bm25_results:
            return [
                (self.chunks[index][0], self.chunks[index][1], score)
                for index, score in dense_results[:top_k]
            ]
        if not dense_results:
            return [
                (self.chunks[index][0], self.chunks[index][1], score)
                for index, score in bm25_results[:top_k]
            ]

        rrf_scores: Dict[int, float] = {}
        for rank, (index, _) in enumerate(dense_results):
            rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (index, _) in enumerate(bm25_results):
            rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)

        results: List[Tuple[str, str, float]] = []
        for index, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            if 0 <= index < len(self.chunks):
                filename, chunk_text, _ = self.chunks[index]
                results.append((filename, chunk_text, score))
        return results

    def _get_reranker(self) -> Optional[CrossEncoderReranker]:
        if not self._rag_config.get("enable_reranker", True):
            return None
        if self._reranker is None:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as exc:
                logger.warning("Failed to initialize reranker: %s", exc)
                self._reranker = None
        return self._reranker

    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[str, str, float]]:
        if len(results) <= 1:
            return results[:top_k]

        reranker = self._get_reranker()
        if reranker is None or not reranker.is_available():
            return results[:top_k]

        passages = [text for _, text, _ in results]
        reranked = reranker.rerank(query, passages, top_k=top_k)

        output: List[Tuple[str, str, float]] = []
        for original_index, rerank_score in reranked:
            if 0 <= original_index < len(results):
                filename, chunk_text, _ = results[original_index]
                output.append((filename, chunk_text, rerank_score))
        return output

    def search_with_fallback(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fallback_keyword: bool = True,
    ) -> List[str]:
        config = self._rag_config
        use_hybrid = bool(config.get("enable_hybrid", True)) and self._bm25_index is not None
        use_reranker = bool(config.get("enable_reranker", True))
        candidate_k = int(config.get("candidate_k", HYBRID_CANDIDATE_K))
        rerank_top_k = candidate_k if use_reranker else top_k

        if use_hybrid:
            results = self.hybrid_search(query, top_k=rerank_top_k, candidate_k=candidate_k)
        else:
            results = self.search(query, top_k=rerank_top_k)

        if use_reranker and len(results) > 1:
            results = self._rerank_results(query, results, top_k=top_k)

        if not results and fallback_keyword:
            results = self._keyword_search(query, top_k=top_k)

        return [f"[{filename}] (score: {score:.3f})\n{text}" for filename, text, score in results]

    def get_rag_config(self) -> dict:
        return self._rag_config.to_dict()

    def update_rag_config(self, updates: dict) -> None:
        self._rag_config.update(updates)
        if "reranker_model" in updates:
            self._reranker = None

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        try:
            import jieba

            query_terms = {term for term in jieba.lcut(query) if str(term).strip()}
        except ImportError:
            query_terms = set(_fallback_tokens(query))

        if not query_terms:
            return []

        results: List[Tuple[str, str, float]] = []
        for filename, chunk_text, _ in self.chunks:
            haystack = chunk_text.lower()
            score = 0.0
            for term in query_terms:
                term_text = str(term).strip().lower()
                if not term_text:
                    continue
                score += haystack.count(term_text)
            if score > 0.0:
                results.append((filename, chunk_text, score))

        results.sort(key=lambda item: item[2], reverse=True)
        return results[:top_k]

    def set_persist_dir(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    def _save(self) -> None:
        if not self.persist_dir:
            return

        os.makedirs(self.persist_dir, exist_ok=True)
        safe_json_save(os.path.join(self.persist_dir, _PERSIST_DOCS), self.documents)
        safe_json_save(
            os.path.join(self.persist_dir, _PERSIST_INDEX_JSON),
            {"chunks": self.chunks, "embed_dim": self._embed_dim},
        )

        embeddings_path = os.path.join(self.persist_dir, _PERSIST_EMBEDDINGS)
        if self.embeddings is not None:
            import numpy as np

            np.save(embeddings_path, self.embeddings)
        else:
            with suppress(OSError):
                os.remove(embeddings_path)

        bm25_path = os.path.join(self.persist_dir, _PERSIST_BM25_JSON)
        if self._bm25_index is not None:
            safe_json_save(bm25_path, self._bm25_index.to_dict())
        else:
            with suppress(OSError):
                os.remove(bm25_path)

    def _load(self) -> bool:
        if not self.persist_dir:
            return False

        try:
            doc_path = os.path.join(self.persist_dir, _PERSIST_DOCS)
            index_json_path = os.path.join(self.persist_dir, _PERSIST_INDEX_JSON)
            index_pkl_path = os.path.join(self.persist_dir, _PERSIST_INDEX_PKL)
            embeddings_path = os.path.join(self.persist_dir, _PERSIST_EMBEDDINGS)
            bm25_json_path = os.path.join(self.persist_dir, _PERSIST_BM25_JSON)
            bm25_pkl_path = os.path.join(self.persist_dir, _PERSIST_BM25_PKL)

            self.documents = safe_json_load(doc_path, default={})

            if os.path.exists(index_json_path):
                index_data = safe_json_load(index_json_path, default={})
                self.chunks = index_data.get("chunks", [])
                self._embed_dim = int(index_data.get("embed_dim", 0) or 0)
            elif os.path.exists(index_pkl_path):
                logger.warning("Loading legacy pickle index, migrating to JSON...")
                with open(index_pkl_path, "rb") as handle:
                    index_data = pickle.load(handle)
                self.chunks = index_data.get("chunks", [])
                self._embed_dim = int(index_data.get("embed_dim", 0) or 0)
                # 迁移：保存为 JSON 格式，删除 pickle 文件
                safe_json_save(index_json_path, index_data)
                try:
                    os.remove(index_pkl_path)
                except OSError:
                    pass

            if os.path.exists(embeddings_path):
                import numpy as np

                self.embeddings = np.load(embeddings_path, allow_pickle=False)
                if getattr(self.embeddings, "ndim", 0) == 2 and self._embed_dim == 0:
                    self._embed_dim = int(self.embeddings.shape[1])

            if self.embeddings is not None and len(self.chunks) != int(self.embeddings.shape[0]):
                logger.warning("Persisted RAG state is inconsistent, dropping embeddings")
                self.embeddings = None

            bm25_data: dict | None = None
            if os.path.exists(bm25_json_path):
                bm25_data = safe_json_load(bm25_json_path, default={})
            elif os.path.exists(bm25_pkl_path):
                logger.warning("Loading legacy pickle BM25 index, migrating to JSON...")
                with open(bm25_pkl_path, "rb") as handle:
                    bm25_data = pickle.load(handle)
                # 迁移：保存为 JSON 格式，删除 pickle 文件
                if bm25_data:
                    safe_json_save(bm25_json_path, bm25_data)
                try:
                    os.remove(bm25_pkl_path)
                except OSError:
                    pass
            if bm25_data:
                self._bm25_index = BM25Index.from_dict(bm25_data)
            else:
                self._bm25_index = None
            return True
        except Exception as exc:
            logger.warning("Failed to load RAG persistence: %s", exc)
            self.documents = {}
            self._invalidate_index(reset_embedder=True)
            return False
