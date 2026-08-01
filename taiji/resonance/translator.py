"""Token translator and tokenizer hub for domain-specific tokenizers.

P7 架构：每个 neuron 使用域专用 tokenizer（zh=20k / en=16k / code=12k / math=10k），
vocab 大幅缩小，独立 lm_head 参数量可控（5-10M / neuron）。

TokenizerHub 管理热插拔域 tokenizer —— 新增域 tokenizer 不影响任何已有 neuron 或共振场。

Based on the three-layer architecture:
- Layer 1: Domain tokenizer (10k-20k) — per-neuron I/O + lm_head 对齐
- Layer 2: Resonance field (4096-dim) — completely independent of tokenizers
- (旧) Layer 0: General tokenizer (256K) — 仅用于向后兼容旧 ckpt
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch


class TokenizerHub:
    """Central registry for all domain tokenizers (multi-modal aware).

    Supports hot-swap: adding a new domain tokenizer does not affect
    any existing neurons or the resonance field.

    P7: 域专用 tokenizer 是 per-neuron lm_head 的 vocab 对齐来源。
    每 neuron 在 forward 前用自己的域 tokenizer encode 文本 → input_ids →
    neuron.embedding(input_ids) → shared_emb。

    多模态扩展（P8 预留）：
        modality 维度独立于 domain。文本域（zh/en/code/math）继续用
        SentencePiece；图像/音频域用 VQ-VAE/EnCodec codebook，通过
        register_modality() 注册专用编码器，encode/decode 按 modality 分发。

    内部键格式：(domain, modality)，默认 modality="text"。
    向后兼容：旧调用 encode(text, domain="zh") 等价于 modality="text"。

    Usage:
        hub = TokenizerHub()
        hub.register_domain("code", code_tokenizer)
        ids = hub.encode_tensor("fn main() {}", domain="code")  # → torch.tensor
        text = hub.decode(ids.tolist(), domain="code")

        # 多模态（P8）：
        hub.register_modality("image", vqvae_codec)
        ids = hub.encode(image_tensor, domain="general", modality="image")
    """

    # 默认模态
    DEFAULT_MODALITY = "text"

    def __init__(self, general_tokenizer=None):
        """Args:
            general_tokenizer: the general 256K tokenizer (I/O protocol). Optional.
                P7 推荐不传，让 hub 完全基于域 tokenizer 工作。
        """
        # 文本域 tokenizer：{domain: tokenizer}
        self.tokenizers: Dict[str, object] = {}
        self.general_tokenizer = general_tokenizer

        # 多模态编码器（P8 预留）：{modality: encoder}
        # text 模态走 self.tokenizers，不在这里注册
        self.modal_encoders: Dict[str, object] = {}

        # Register general tokenizer if provided
        if general_tokenizer is not None:
            self.tokenizers["general"] = general_tokenizer

    def register_domain(self, domain: str, domain_tokenizer) -> None:
        """Register a new domain tokenizer (hot-swap, text modality).

        Does not affect any existing neurons or tokenizers.

        Args:
            domain: domain name (e.g., "zh", "code", "rust").
            domain_tokenizer: SentencePiece processor for this domain.
        """
        self.tokenizers[domain] = domain_tokenizer
        print(f"[TokenizerHub] registered {domain} tokenizer (vocab={self.vocab_size(domain)})")

    def register_modality(self, modality: str, encoder) -> None:
        """P8: 注册非文本模态编码器（图像/音频/视频）。

        多模态编码器与文本域 tokenizer 正交：
        - 文本：register_domain("zh", sp_zh) → encode(text, domain="zh", modality="text")
        - 图像：register_modality("image", vqvae) → encode(img, domain="general", modality="image")

        编码器接口契约（P8 实现时需满足）：
            encoder.encode(raw_input) -> list[int]      # 离散化为 token id 序列
            encoder.decode(ids: list[int]) -> raw_output
            encoder.vocab_size() -> int
            encoder.eos_token_id() -> int  (可选，无则返回 -1)

        Args:
            modality: 模态名（"image"/"audio"/"video"）。
            encoder: 编码器实例（VQ-VAE / EnCodec 等）。
        """
        self.modal_encoders[modality] = encoder
        print(f"[TokenizerHub] registered {modality} modality encoder")

    def get_tokenizer(self, domain: str):
        """Get tokenizer for a domain. Falls back to general if domain not found.

        Args:
            domain: domain name.

        Returns:
            Tokenizer instance or None.
        """
        if domain in self.tokenizers:
            return self.tokenizers[domain]
        return self.tokenizers.get("general")

    def encode(self, text: str, domain: str = "general", modality: str = None) -> list[int]:
        """Encode input using the appropriate domain tokenizer or modality encoder.

        Args:
            text: input text (text modality) or raw tensor (non-text modality).
            domain: domain name (falls back to "general").
            modality: 模态（None 默认 "text"）。非文本模态走 modal_encoders。

        Returns:
            list of token IDs.
        """
        modality = modality or self.DEFAULT_MODALITY

        # 非文本模态：走 modal_encoders
        if modality != self.DEFAULT_MODALITY:
            enc = self.modal_encoders.get(modality)
            if enc is None:
                raise ValueError(f"No encoder for modality '{modality}'")
            return enc.encode(text)

        # 文本模态：走域 tokenizer
        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}' and no general fallback")
        return tok.encode(text)

    def encode_tensor(
        self,
        text: str,
        domain: str = "general",
        device: Optional[torch.device] = None,
        modality: str = None,
    ) -> torch.Tensor:
        """P7: encode text to torch.tensor [1, L] for direct feed into neuron.

        Args:
            text: input text.
            domain: domain name.
            device: torch device (default: cpu).
            modality: 模态（None 默认 "text"）。

        Returns:
            input_ids: [1, L] long tensor.
        """
        ids = self.encode(text, domain=domain, modality=modality)
        if not ids:
            ids = [0]  # 防止空 tensor
        return torch.tensor([ids], dtype=torch.long, device=device or torch.device("cpu"))

    def decode(self, ids: list[int], domain: str = "general", modality: str = None) -> str:
        """Decode token IDs back to output using domain tokenizer or modality encoder.

        Args:
            ids: list of token IDs.
            domain: domain name.
            modality: 模态（None 默认 "text"）。非文本模态走 modal_encoders。

        Returns:
            decoded text string (text modality) or raw output (non-text modality).
        """
        modality = modality or self.DEFAULT_MODALITY

        # 非文本模态：走 modal_encoders
        if modality != self.DEFAULT_MODALITY:
            enc = self.modal_encoders.get(modality)
            if enc is None:
                raise ValueError(f"No encoder for modality '{modality}'")
            return enc.decode(ids)

        # 文本模态：走域 tokenizer
        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}' and no general fallback")
        return tok.decode(ids)

    def vocab_size(self, domain: str = "general", modality: str = None) -> int:
        """返回域 tokenizer 或模态编码器的 vocab size。

        Args:
            domain: domain name.
            modality: 模态（None 默认 "text"）。
        """
        modality = modality or self.DEFAULT_MODALITY

        if modality != self.DEFAULT_MODALITY:
            enc = self.modal_encoders.get(modality)
            if enc is None:
                raise ValueError(f"No encoder for modality '{modality}'")
            if callable(getattr(enc, "vocab_size", None)):
                return int(enc.vocab_size())
            raise AttributeError(f"Encoder for '{modality}' has no vocab_size()")

        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}'")
        # SentencePiece: vocab_size() 是方法
        if callable(getattr(tok, "vocab_size", None)):
            return int(tok.vocab_size())
        if hasattr(tok, "GetPieceSize"):
            return int(tok.GetPieceSize())
        # fallback: 尝试直接属性
        if hasattr(tok, "vocab_size"):
            return int(getattr(tok, "vocab_size"))
        raise AttributeError(
            f"Tokenizer for '{domain}' has neither vocab_size nor GetPieceSize"
        )

    def eos_token_id(self, domain: str = "general", modality: str = None) -> int:
        """返回域 tokenizer 或模态编码器的 EOS token id.

        Args:
            domain: domain name.
            modality: 模态（None 默认 "text"）。
        """
        modality = modality or self.DEFAULT_MODALITY

        if modality != self.DEFAULT_MODALITY:
            enc = self.modal_encoders.get(modality)
            if enc is None:
                raise ValueError(f"No encoder for modality '{modality}'")
            # 多模态编码器可能没有 EOS 概念，返回 -1
            for attr in ("eos_token_id", "eos_id"):
                v = getattr(enc, attr, None)
                if callable(v):
                    v = v()
                if isinstance(v, int) and v >= 0:
                    return int(v)
            return -1  # 无 EOS

        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}'")
        # SentencePiece processor
        if hasattr(tok, "eos_id"):
            eid = tok.eos_id()
            if eid is not None and eid >= 0:
                return int(eid)
        # 兼容 TaijiNativeTokenizerV2 等包装器
        for attr in ("eos_token_id", "eos_id"):
            v = getattr(tok, attr, None)
            if callable(v):
                v = v()
            if isinstance(v, int) and v >= 0:
                return int(v)
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                return int(v.item())
        # fallback：SentencePiece 默认 </s>=1
        return 1

    def list_domains(self) -> list[str]:
        """List all registered domains (excluding 'general' fallback)."""
        return [d for d in self.tokenizers.keys() if d != "general"]

    def list_modalities(self) -> list[str]:
        """P8: 列出所有已注册的非文本模态。"""
        return list(self.modal_encoders.keys())

    @classmethod
    def load_default_domains(
        cls,
        domains_dir: str = None,
        general_tokenizer=None,
    ) -> "TokenizerHub":
        """P7: 从 taiji/domains/ 加载默认 4 个域 tokenizer (zh/en/code/math).

        目录结构：
            taiji/domains/zh/sp_zh.model
            taiji/domains/en/sp_en.model
            taiji/domains/code/sp_code.model
            taiji/domains/math/sp_math.model

        Args:
            domains_dir: 域 tokenizer 根目录。None 时自动推断为
                         taiji/domains/
            general_tokenizer: 可选的通用 tokenizer（向后兼容）。

        Returns:
            TokenizerHub 实例，已注册 zh/en/code/math 4 个域。
        """
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError as e:
            raise ImportError(
                "sentencepiece 未安装。请运行: pip install sentencepiece"
            ) from e

        if domains_dir is None:
            # taiji/resonance/translator.py → taiji/domains/
            here = os.path.dirname(os.path.abspath(__file__))
            domains_dir = os.path.normpath(
                os.path.join(here, "..", "domains")
            )

        if not os.path.isdir(domains_dir):
            raise FileNotFoundError(
                f"域 tokenizer 目录不存在: {domains_dir}"
            )

        hub = cls(general_tokenizer=general_tokenizer)

        # 域 → 文件名映射
        domain_files = {
            "zh": "sp_zh.model",
            "en": "sp_en.model",
            "code": "sp_code.model",
            "math": "sp_math.model",
        }

        loaded = []
        missing = []
        for domain, fname in domain_files.items():
            path = os.path.join(domains_dir, domain, fname)
            if os.path.exists(path):
                sp = SentencePieceProcessor()
                sp.Load(path)
                hub.register_domain(domain, sp)
                loaded.append(domain)
            else:
                missing.append(f"{domain}({path})")

        if missing:
            print(f"[TokenizerHub] WARNING: 缺失域 tokenizer: {missing}")
        if not loaded:
            raise FileNotFoundError(
                f"未在 {domains_dir} 下找到任何域 tokenizer。"
                f"预期文件: {list(domain_files.values())}"
            )

        # P7: 注册 general 域（复用 en tokenizer，同 16K vocab）
        if "en" in hub.tokenizers:
            hub.register_domain("general", hub.tokenizers["en"])
            loaded.append("general")

        print(f"[TokenizerHub] loaded {len(loaded)} domain tokenizers: {loaded}")
        return hub


# ============================================================================
# Token alignment utility: domain token ↔ general token position mapping
# ============================================================================

def _get_token_spans(sp, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Encode text and track character spans for each token.

    Handles SentencePiece's "▁" prefix (U+2581) which represents a space.
    The raw piece string is used to track character offsets in the original text.

    Args:
        sp: SentencePieceProcessor.
        text: raw input text.

    Returns:
        (token_ids, spans) where spans[i] = (char_start, char_end) in original text.
    """
    pieces = sp.encode(text, out_type=str)  # list of piece strings
    ids = sp.encode(text)  # list of token IDs
    text_len = len(text)

    spans = []
    pos = 0
    for piece in pieces:
        # SentencePiece 的 ▁ (U+2581) 代表空格：
        #   - 独立 "▁" → 原始文本中的一个空格字符（缩进、词间分隔）
        #   - "▁word"  → 词界标记（前导空格），word 是实际内容
        if piece == "▁":
            # 独立空格 token：对应文本中的一个空格字符
            # 修复前 clean="" 导致零长度 span，缩进信息丢失
            span_len = 1
            spans.append((pos, pos + span_len))
            pos += span_len
        elif piece.startswith("▁"):
            # ▁word：前导空格是词界标记
            # 仅当当前位置确实是空格时才 skip（避免开头 ▁word 误 skip）
            if 0 < pos < text_len and text[pos] == " ":
                pos += 1  # skip the space separator
            clean = piece[1:]
            span_len = len(clean)
            spans.append((pos, pos + span_len))
            pos += span_len
        else:
            clean = piece
            span_len = len(clean)
            spans.append((pos, pos + span_len))
            pos += span_len

    return ids, spans


def build_position_alignment(
    text: str,
    domain_sp,
    general_sp,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build general→domain token position alignment for a single text.

    Given a text encoded by both domain and general tokenizers, returns
    the alignment from each general token position to its corresponding
    domain token position.

    Alignment rule: for each general token at position j, find the domain
    token whose character span has the maximum overlap with the general
    token's character span.

    Args:
        text: raw input text.
        domain_sp: domain-specific SentencePieceProcessor.
        general_sp: general 256K SentencePieceProcessor.

    Returns:
        (general_ids, domain_targets) where:
        - general_ids: [L_g] general token IDs
        - domain_targets: [L_g] domain token IDs, -100 for unaligned positions
    """
    domain_ids, domain_spans = _get_token_spans(domain_sp, text)
    general_ids, general_spans = _get_token_spans(general_sp, text)

    L_g = len(general_ids)
    domain_targets = torch.full((L_g,), -100, dtype=torch.long)

    for j, (g_start, g_end) in enumerate(general_spans):
        best_i = -1
        best_overlap = 0
        for i, (d_start, d_end) in enumerate(domain_spans):
            overlap_start = max(g_start, d_start)
            overlap_end = min(g_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i

        if best_i >= 0:
            domain_targets[j] = domain_ids[best_i]

    return torch.tensor(general_ids, dtype=torch.long), domain_targets


def batch_align_and_embed(
    texts: List[str],
    domain_sp,
    general_sp,
    shared_embedding: torch.nn.Embedding,
    pad_token_id: int = 0,
    max_seq_len: int = 128,
    answer_marker: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-align domain texts to general tokens and produce padded embeddings + targets.

    This is the main training entry point: given a batch of texts from a domain,
    encode them with the general tokenizer (for shared embedding lookup) and
    align domain token targets to general token positions.

    Args:
        texts: list of raw text strings.
        domain_sp: domain-specific SentencePieceProcessor.
        general_sp: general 256K SentencePieceProcessor.
        shared_embedding: nn.Embedding(256000, 512) shared across all neurons.
        pad_token_id: padding token ID for both tokenizers.
        max_seq_len: 最大序列长度（截断）。
        answer_marker: SFT 分隔符（如 "答："）。传入时额外返回 sft_mask，
            只保留 marker 之后的 token（answer 部分）计入 loss。
            不传时保持原行为（返回 3 元组），向后兼容。

    Returns:
        不传 answer_marker:
            (shared_emb, domain_targets, attention_mask)
        传 answer_marker:
            (shared_emb, domain_targets, attention_mask, sft_mask)
        - shared_emb: [B, L_max, base_embed_dim] from shared embedding table
        - domain_targets: [B, L_max] domain token IDs (aligned), -100 for pad/unaligned
        - attention_mask: [B, L_max] bool (True=valid, False=pad)
        - sft_mask: [B, L_max] bool (True=answer token, False=question/pad)
    """
    all_general_ids = []
    all_targets = []
    # S3: SFT answer 起始 token index（general token 维度），None 表示无分隔符
    answer_starts: List[Optional[int]] = []

    for text in texts:
        g_ids, d_targets = build_position_alignment(text, domain_sp, general_sp)
        # 截断到最大序列长度（防止长文本拖慢训练）
        if max_seq_len > 0 and len(g_ids) > max_seq_len:
            g_ids = g_ids[:max_seq_len]
            d_targets = d_targets[:max_seq_len]
        all_general_ids.append(g_ids)
        all_targets.append(d_targets)

        # S3: 计算 answer 起始 token index
        if answer_marker is not None:
            marker_idx = text.find(answer_marker)
            if marker_idx == -1:
                # 无分隔符，整个文本视为 answer（mask 全 True）
                answer_starts.append(0)
            else:
                # prefix 含分隔符本身（分隔符属于 question，不计入 loss）
                prefix_with_marker = text[:marker_idx + len(answer_marker)]
                prefix_ids = general_sp.encode(prefix_with_marker)
                # 截断到 max_seq_len
                start = min(len(prefix_ids), max_seq_len) if max_seq_len > 0 else len(prefix_ids)
                answer_starts.append(start)
        else:
            answer_starts.append(None)

    # Pad to max length (capped by max_seq_len)
    max_len = max(len(ids) for ids in all_general_ids)
    B = len(texts)

    padded_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    padded_targets = torch.full((B, max_len), -100, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    # S3: SFT mask
    sft_mask = None
    if answer_marker is not None:
        sft_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for b in range(B):
        L = len(all_general_ids[b])
        padded_ids[b, :L] = all_general_ids[b]
        padded_targets[b, :L] = all_targets[b]
        mask[b, :L] = True

        # S3: answer 部分 mask 为 True
        if sft_mask is not None and answer_starts[b] is not None:
            start = answer_starts[b]
            if start < L:
                sft_mask[b, start:L] = True

    # Embed
    shared_emb = shared_embedding(padded_ids)  # [B, L_max, base_embed_dim]

    if sft_mask is not None:
        return shared_emb, padded_targets, mask, sft_mask
    return shared_emb, padded_targets, mask
