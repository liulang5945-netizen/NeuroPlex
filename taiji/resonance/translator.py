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

import json
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

    # ---- 词库实时编辑（C25，2026-08-09 用户决策：容量不限 + 实时编辑 → 不需要热插拔）----

    def to_editable(self, domain: str, ext_path: Optional[str] = None) -> "EditableVocabulary":
        """把域 tokenizer 升级为可实时编辑词表（幂等：已可编辑则返回自身）。

        Args:
            domain: 域名。
            ext_path: 扩展区持久化路径（如 taiji/domains/zh/sp_zh_ext.json）。
                      存在则自动加载已追加 token；None 仅包装不持久化。

        Returns:
            EditableVocabulary 实例（同时写回 hub 注册表，替换原 tokenizer）。
        """
        tok = self.get_tokenizer(domain)
        if tok is None:
            raise ValueError(f"No tokenizer for domain '{domain}'")
        if isinstance(tok, EditableVocabulary):
            if ext_path is not None and os.path.exists(ext_path):
                tok.load_ext(ext_path)
            return tok
        ev = EditableVocabulary(tok, ext_path=ext_path)
        self.tokenizers[domain] = ev
        print(f"[TokenizerHub] {domain} tokenizer 已升级为可实时编辑（base={ev.base_vocab}）", flush=True)
        return ev

    def add_tokens(self, domain: str, pieces: List[str], ext_path: Optional[str] = None) -> List[int]:
        """实时给域词表追加 token（不存在的才追加，返回各 piece 的 token id）。

        新 token 的 id ≥ base vocab → 下游对齐/转译表（fingerprint 的
        vocab_size 变化）自动失效重建；neuron lm_head 需配套 resize
        （见 resize_lm_head_for_vocab；Embedding 权重矩阵用 resize_linear_for_vocab）。

        Args:
            domain: 域名。
            pieces: 要追加的 piece 文本列表（与 SP 的 ▁ 词界约定一致）。
            ext_path: 追加后持久化路径（None 仅内存生效）。

        Returns:
            每个 piece 对应的 token id（base 已存在返回 base id）。
        """
        ev = self.to_editable(domain, ext_path=ext_path)
        ids = ev.add_tokens(pieces)
        if ext_path is not None:
            ev.save_ext(ext_path)
        return ids

    def unregister_domain(self, domain: str) -> bool:
        """运行时移除一张域词表（集合级编辑；general fallback 不受影响）。"""
        if domain not in self.tokenizers or domain == "general":
            return False
        del self.tokenizers[domain]
        print(f"[TokenizerHub] unregistered {domain} tokenizer", flush=True)
        return True


class EditableVocabulary:
    """可实时编辑词表：包装 SentencePieceProcessor，运行时增删 token。

    背景（C25 词库收敛，2026-08-09 用户决策）：
    - 词库不做限制（容量不限）+ 支持实时编辑 → 不再需要"热插拔"机制。
    - SentencePiece 模型静态不可变 → 本类在其上叠加"扩展区"：
      运行时 add_tokens 追加 piece（id = base_vocab + i），encode/decode
      在扩展区与 base SP 之间自动合并。

    设计（上限最高）：
    - 扩展区以 piece 文本（可读、可编辑）为键，与 SP 的 ▁ 词界约定一致；
      追加的 piece 若 base 已含则直接复用 base id（不重复占位）
    - encode：扩展区前缀树最长匹配 + 剩余片段走 SP（贪心，base 语义不变；
      新词注入必然改变切分，属预期）
    - decode / id_to_piece / piece_to_id / vocab_size 合并扩展区 → 下游
      对齐/转译表（tokenizer_fingerprint 的 vocab_size 变化）自动失效重建
    - 持久化：扩展区存 JSON（如 sp_zh_ext.json），可热加载还原
    - 接口兼容 SP：GetPieceSize / id_to_piece / encode / decode / eos_id 等

    用法：
        ev = EditableVocabulary(sp)
        nid = ev.add_token("▁量子计算")        # 返回新 token id（≥ base）
        ids = ev.encode("量子计算前沿")          # 新词优先整词编码
        ev.save_ext("taiji/domains/zh/sp_zh_ext.json")
        ev2 = EditableVocabulary(sp, ext_path=...)  # 还原

    tokenizer_fingerprint(sp) 兼容本类：GetPieceSize / id_to_piece 已实现
    → 加 token 后指纹变化 → build_logits_alignment_matrix 缓存自动重建。
    """

    def __init__(self, sp, ext_pieces: Optional[List[str]] = None,
                 ext_path: Optional[str] = None):
        self._sp = sp
        self._ext_pieces: List[str] = []
        self._ext_id: Dict[str, int] = {}
        self._trie = None
        self.ext_path = ext_path
        if ext_path and os.path.exists(ext_path):
            self.load_ext(ext_path)
        if ext_pieces:
            self.add_tokens(ext_pieces)

    # ---- 基础属性 ----

    @property
    def base_vocab(self) -> int:
        """base SP 原始 vocab 大小（不可变区）。"""
        if callable(getattr(self._sp, "GetPieceSize", None)):
            return int(self._sp.GetPieceSize())
        return int(self._sp.vocab_size())

    @property
    def ext_pieces(self) -> List[str]:
        return list(self._ext_pieces)

    # ---- 实时编辑 ----

    def add_token(self, piece: str) -> int:
        """追加一个 token。base 已含则返回 base id；否则分配 ext id。"""
        if piece in self._ext_id:
            return self._ext_id[piece]
        bid = self._sp.piece_to_id(piece)
        if bid != self._sp.unk_id():
            return int(bid)  # base 已含，直接复用（不重复占位）
        new_id = self.base_vocab + len(self._ext_pieces)
        self._ext_pieces.append(piece)
        self._ext_id[piece] = new_id
        self._trie = None  # 失效，下次 encode 重建
        return new_id

    def add_tokens(self, pieces: List[str]) -> List[int]:
        """批量追加，返回各 piece 的 token id（保持输入顺序）。"""
        return [self.add_token(p) for p in pieces]

    def remove_token(self, piece: str) -> bool:
        """移除一个扩展 token（⚠️ ext id 会重排——仅限未绑定 neuron 的编辑态使用）。

        已持久化 / 已被 neuron lm_head 引用时移除会导致 id 错位，
        建议通过词表重建（重训 SP + 清空 ext）而非运行时移除。
        """
        if piece not in self._ext_id:
            return False
        self._ext_pieces.remove(piece)
        del self._ext_id[piece]
        self._ext_id = {p: self.base_vocab + i for i, p in enumerate(self._ext_pieces)}
        self._trie = None
        print(f"[EditableVocabulary] removed token '{piece}'（ext 区 id 已重排）", flush=True)
        return True

    # ---- 编码 / 解码 ----

    def _build_trie(self) -> None:
        root = {}
        for idx, piece in enumerate(self._ext_pieces):
            node = root
            for ch in piece:
                node = node.setdefault(ch, {})
            node[""] = self.base_vocab + idx  # 终端标记 → token id
        self._trie = root

    def encode(self, text) -> List[int]:
        """编码：扩展区前缀树最长匹配 + 剩余片段走 base SP。"""
        if isinstance(text, list):
            ids: List[int] = []
            for s in text:
                ids.extend(self.encode(s))
            return ids
        if not text:
            return []
        if not self._ext_pieces:
            return self._sp.encode(text)
        if self._trie is None:
            self._build_trie()
        ids = []
        buf: List[str] = []      # 未命中的普通字符缓冲
        n = len(text)
        i = 0
        while i < n:
            node = self._trie
            match_id = -1
            match_len = 0
            j = i
            while j < n and text[j] in node:
                node = node[text[j]]
                j += 1
                if "" in node:
                    match_id = node[""]
                    match_len = j - i
            if match_id >= 0:
                if buf:
                    ids.extend(self._sp.encode("".join(buf)))
                    buf = []
                ids.append(match_id)
                i += match_len
            else:
                buf.append(text[i])
                i += 1
        if buf:
            ids.extend(self._sp.encode("".join(buf)))
        return ids

    def decode(self, ids) -> str:
        """解码：ext id → piece 文本；base id → SP decode。"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out: List[str] = []
        base = self.base_vocab
        n_ext = len(self._ext_pieces)
        for i in ids:
            i = int(i)
            if i >= base and (i - base) < n_ext:
                out.append(self._ext_pieces[i - base])
            else:
                out.append(self._sp.decode([i]))
        return "".join(out)

    # ---- 兼容 SP 接口（对齐表 / 转译表 / fingerprint 依赖） ----

    def GetPieceSize(self) -> int:
        return self.base_vocab + len(self._ext_pieces)

    def vocab_size(self) -> int:
        return self.GetPieceSize()

    def id_to_piece(self, i: int) -> str:
        i = int(i)
        base = self.base_vocab
        if i >= base and (i - base) < len(self._ext_pieces):
            return self._ext_pieces[i - base]
        return self._sp.id_to_piece(i)

    def piece_to_id(self, piece: str) -> int:
        if piece in self._ext_id:
            return self._ext_id[piece]
        return self._sp.piece_to_id(piece)

    def eos_id(self) -> int:
        return int(self._sp.eos_id())

    def pad_id(self) -> int:
        return int(self._sp.pad_id())

    def unk_id(self) -> int:
        return int(self._sp.unk_id())

    def bos_id(self) -> int:
        return int(self._sp.bos_id())

    # ---- 持久化 ----

    def save_ext(self, path: Optional[str] = None) -> str:
        """扩展区持久化到 JSON（默认 self.ext_path）。"""
        save_path = path or self.ext_path
        if not save_path:
            raise ValueError("EditableVocabulary.save_ext 需要 path")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"ext_pieces": self._ext_pieces, "base_vocab": self.base_vocab},
                      f, ensure_ascii=False, indent=2)
        self.ext_path = save_path
        print(f"[EditableVocabulary] 已保存 {len(self._ext_pieces)} 个扩展 token 到 {save_path}",
              flush=True)
        return save_path

    def load_ext(self, path: Optional[str] = None) -> int:
        """热加载扩展区（合并进现有扩展区，自动跳过重复）。"""
        load_path = path or self.ext_path
        if not load_path or not os.path.exists(load_path):
            return 0
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pieces = data.get("ext_pieces", [])
        added = 0
        for p in pieces:
            if p not in self._ext_id and self._sp.piece_to_id(p) == self._sp.unk_id():
                self._ext_pieces.append(p)
                self._ext_id[p] = self.base_vocab + len(self._ext_pieces) - 1
                added += 1
        self._trie = None
        if added:
            print(f"[EditableVocabulary] 已加载 {added} 个扩展 token（{load_path}）", flush=True)
        return added

    def __getattr__(self, name):
        """其余属性透传 base SP（保持全接口兼容）。"""
        return getattr(self._sp, name)


def resize_linear_for_vocab(linear, new_vocab: int, init_std: float = 0.02):
    """把 nn.Linear(hidden, old_vocab) resize 到 new_vocab（词表实时编辑配套）。

    新增行用既有行的均值 + 小噪声初始化（比零初始化更快进入可用区），
    旧行权重与 bias 原样保留——已学 token 的输出不受影响。

    Args:
        linear: nn.Linear（lm_head / embedding 权重矩阵形态均可，要求 out_features=vocab）。
        new_vocab: 目标 vocab 大小（含扩展区）。
        init_std: 新增行初始化噪声标准差。

    Returns:
        新的 nn.Linear（old_vocab >= new_vocab 时返回原对象）。
    """
    old_vocab = linear.out_features
    if old_vocab >= new_vocab:
        return linear
    new_linear = torch.nn.Linear(linear.in_features, new_vocab,
                                 bias=linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.data[:old_vocab] = linear.weight.data
        if old_vocab > 0:
            mean = linear.weight.data.mean(dim=0, keepdim=True)
        else:
            mean = torch.zeros(1, linear.in_features)
        new_linear.weight.data[old_vocab:] = (
            mean + torch.randn(new_vocab - old_vocab, linear.in_features) * init_std
        )
        if linear.bias is not None:
            new_linear.bias.data[:old_vocab] = linear.bias.data
    return new_linear


def resize_lm_head_for_vocab(neuron, new_vocab: int) -> bool:
    """把 neuron 的域 lm_head resize 到 new_vocab（词库实时编辑配套）。

    仅处理 neuron.lm_head（域词表头）；judge_lm_head（general 256K）不随
    域词表扩展。shared+delta 低秩头（lm_head_rank>0）不 resize（v1 仅完整头）。

    Args:
        neuron: ResonanceNeuron。
        new_vocab: 目标 vocab 大小（含扩展区）。

    Returns:
        是否执行了 resize。
    """
    head = getattr(neuron, "lm_head", None)
    if head is None or getattr(neuron, "lm_head_rank", 0) > 0:
        return False
    if head.out_features >= new_vocab:
        return False
    neuron.lm_head = resize_linear_for_vocab(head, new_vocab)
    print(f"[resize_lm_head_for_vocab] {neuron.config.neuron_id} lm_head "
          f"{head.out_features} → {new_vocab}", flush=True)
    return True


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
    answer_marker_mode: str = "first",
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
        answer_marker_mode: T4 answer 起点模式。
            "first"（默认，向后兼容）：第一个 marker 之后全部为 answer
            "last"（多轮精确 masking）：最后一个 marker 之后为 answer，
                前序轮次的 question/answer 作为纯上下文（不计 loss）

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

    # EOS 注入（2026-08-04）：训练时在序列末尾追加 EOS token，
    # 让模型学会在 answer 结束时自然停止生成，而非依赖 max_tokens 强制截断。
    # 不加 EOS 的后果：模型永不输出 EOS，生成时只能靠 max_tokens 截断 + 跑偏兜底，
    # 导致长序列语义崩坏。
    try:
        general_eos = general_sp.eos_id()
    except Exception:
        general_eos = 1  # SentencePiece 默认 </s>=1
    try:
        domain_eos = domain_sp.eos_id()
    except Exception:
        domain_eos = 1

    for text in texts:
        g_ids, d_targets = build_position_alignment(text, domain_sp, general_sp)
        # 追加 EOS（让模型学会在 answer 末尾自然停止）
        g_ids = torch.cat([g_ids, torch.tensor([general_eos])])
        d_targets = torch.cat([d_targets, torch.tensor([domain_eos])])
        # 截断到最大序列长度（保留末尾 EOS：截到 max_seq_len-1 + EOS）
        if max_seq_len > 0 and len(g_ids) > max_seq_len:
            g_ids = torch.cat([g_ids[:max_seq_len-1], torch.tensor([general_eos])])
            d_targets = torch.cat([d_targets[:max_seq_len-1], torch.tensor([domain_eos])])
        all_general_ids.append(g_ids)
        all_targets.append(d_targets)

        # S3: 计算 answer 起始 token index
        if answer_marker is not None:
            if answer_marker_mode == "last":
                # T4: 最后一个 marker（多轮精确 masking，前序轮次为纯上下文）
                marker_idx = text.rfind(answer_marker)
            else:
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


# ============================================================================
# 词库转译（跨 vocab 对齐）：domain token → target domain token
# ============================================================================

class AlignmentRules:
    """可编辑/可拓展的词库转译规则层（人工覆盖自动构建的映射）。

    背景：自动转译（piece→text→encode）对多数 token 有效，但新增特殊神经元
    时（如 biology/legal 域），其专业术语 token 自动映射可能退化（byte fallback
    或 <unk>）。本规则层允许人工指定映射，且可增量扩展。

    设计（上限更高）：
    - 匹配键用 source piece **文本**（tokenizer 无关、可读、可编辑），
      不用 token id（随 tokenizer 重训漂移，脆弱）
    - 支持域特定规则 + 全局规则（source_domain="*"）
    - 每次增删递增 version，驱动下游对齐矩阵缓存自动失效
    - 持久化 JSON（默认 taiji/domains/alignment_rules.json），可热加载

    Usage:
        rules = AlignmentRules()                      # 空规则
        rules = AlignmentRules(path="...json")        # 加载已有规则
        rules.add_override("code", "<0x0A>", ["▁\n"]) # 人工指定映射
        rules.add_override("*", "term_x", ["term_x"]) # 全局规则
        rules.save()
    """

    def __init__(self, rules_path: Optional[str] = None):
        self.overrides: Dict[str, Dict[str, List[str]]] = {}
        self.rules_path = rules_path
        self.version: int = 0  # 增删时递增，用于下游缓存失效
        if rules_path and os.path.exists(rules_path):
            self.load(rules_path)

    def add_override(
        self, source_domain: str, source_piece: str,
        target_pieces: List[str],
    ) -> None:
        """人工指定 source token → target token(s) 映射。

        Args:
            source_domain: 源域（"code"/"math"...）；"*" 表示全局规则。
            source_piece: 源 token 的 piece 文本（id_to_piece 返回值）。
            target_pieces: 目标 piece 文本列表（构建时 encode 成 target ids）。
        """
        self.overrides.setdefault(source_domain, {})[source_piece] = list(target_pieces)
        self.version += 1

    def remove_override(self, source_domain: str, source_piece: str) -> bool:
        """移除一条人工规则。返回是否移除成功。"""
        d = self.overrides.get(source_domain)
        if d is None or source_piece not in d:
            return False
        del d[source_piece]
        if not d:
            self.overrides.pop(source_domain, None)
        self.version += 1
        return True

    def get(
        self, source_domain: Optional[str], source_piece: str,
    ) -> Optional[List[str]]:
        """查规则：先域特定，后全局（"*"）。返回 target_pieces 或 None。"""
        if source_domain is not None:
            hit = self.overrides.get(source_domain, {}).get(source_piece)
            if hit is not None:
                return hit
        return self.overrides.get("*", {}).get(source_piece)

    def to_dict(self) -> Dict:
        return {
            "overrides": self.overrides,
            "version": self.version,
            "note": "词库转译人工规则：{source_domain: {source_piece: [target_piece...]}}；"
                    "source_domain='*' 为全局规则；piece 文本见 id_to_piece。",
        }

    def save(self, path: Optional[str] = None) -> str:
        """持久化到 JSON（默认 self.rules_path，需在 __init__ 或此处指定）。"""
        save_path = path or self.rules_path
        if not save_path:
            raise ValueError("AlignmentRules.save 需要 path（构造时未指定 rules_path）")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        self.rules_path = save_path
        print(f"[AlignmentRules] 已保存 {len(self.overrides)} 条规则到 {save_path}", flush=True)
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        """从 JSON 热加载（合并进现有规则，version 取 max）。"""
        load_path = path or self.rules_path
        if not load_path or not os.path.exists(load_path):
            return
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = data.get("overrides", {})
        for dom, rules in loaded.items():
            self.overrides.setdefault(dom, {}).update(rules)
        self.version = max(self.version, int(data.get("version", 0)) + 1)
        self.rules_path = load_path
        print(f"[AlignmentRules] 已加载 {len(loaded)} 条规则（{load_path}）", flush=True)


def tokenizer_fingerprint(sp) -> tuple:
    """Tokenzier 指纹：用于对齐表缓存失效判断（词库热插拔后自动重建）。

    用 (vocab_size, 首/中/尾 piece) 抽样，覆盖绝大多数 tokenizer 变更
    （重训词表 / 增删 token / 换模型），成本 O(1)。

    Args:
        sp: SentencePieceProcessor（或任何提供 id_to_piece/GetPieceSize 的对象）。

    Returns:
        tuple 指纹。
    """
    size = sp.GetPieceSize() if hasattr(sp, "GetPieceSize") else 0
    if size <= 0:
        return (id(sp), 0, ())
    try:
        sample = (sp.id_to_piece(0), sp.id_to_piece(size // 2), sp.id_to_piece(size - 1))
    except Exception:
        return (id(sp), size, ())
    return (size, sample)


def build_domain_to_domain_alignment(
    source_sp,
    target_sp,
    source_domain: Optional[str] = None,
    overrides: Optional[AlignmentRules] = None,
) -> Tuple[List[List[int]], int]:
    """词库转译：构建 source domain vocab → target domain vocab 的 token 对齐表。

    对每个 source token，取其 piece 文本（byte fallback 的 <0x..> 先 decode 成
    真实字节再转译，保留换行等语义），再用 target tokenizer 重新编码得到目标
    token id 列表。空映射用 target pad_id 兜底。

    可编辑层：overrides（AlignmentRules）中匹配的 source piece 跳过自动转译，
    改用人工指定的 target piece 文本编码（新增特殊神经元时补充专业术语映射）。

    用于跨 vocab logits 融合：把 source neuron 的 logits 投影到 target 域空间，
    与 S6 词库转译（domain→general）同源，只是目标空间换为任意 target domain。

    Args:
        source_sp: 源域 tokenizer。
        target_sp: 目标域 tokenizer。
        source_domain: 源域名（overrides 域特定规则匹配用；None 时仅全局规则）。
        overrides: 可编辑规则层（可选）。

    Returns:
        (alignment, source_vocab_size)
        alignment[i] = [target_token_ids...]（空 → [pad_id]）
    """
    vocab_size = source_sp.GetPieceSize() if hasattr(source_sp, "GetPieceSize") else 0
    pad_id = target_sp.pad_id() if hasattr(target_sp, "pad_id") else 0
    alignment: List[List[int]] = []
    for src_id in range(vocab_size):
        piece = source_sp.id_to_piece(src_id)
        manual = None
        if overrides is not None:
            manual = overrides.get(source_domain, piece)
        if manual is not None:
            # 人工规则：target_piece 文本 → target ids（可多段，逐段 encode 拼接）
            tgt_ids: List[int] = []
            for tp in manual:
                tgt_ids.extend(target_sp.encode(tp))
            alignment.append(tgt_ids if tgt_ids else [pad_id])
            continue
        if piece.startswith("<0x") and piece.endswith(">"):
            # byte fallback piece（如 <0x0A>）：decode 成真实字节再 encode，
            # 否则 "<0x0A>" 会被当作 6 个字符编码，换行语义丢失
            text = source_sp.decode([src_id])
        else:
            text = piece
        tgt_ids = target_sp.encode(text)
        alignment.append(tgt_ids if tgt_ids else [pad_id])
    return alignment, vocab_size


def build_logits_alignment_matrix(
    source_sp,
    target_sp,
    source_domain: str,
    target_domain: str,
    cache: Optional[Dict] = None,
    overrides: Optional[AlignmentRules] = None,
    source_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """构建 [V_src, V_tgt] 稀疏 logits 投影矩阵（词库转译），带缓存 + 指纹失效。

    行归一化：一个 source token 映射到 N 个 target token 时，每列权重 1/N，
    保证 logits 尺度守恒（softmax 前线性投影的近似概率转移）。

    缓存失效键：tokenizer 指纹 + overrides 版本（人工规则增删后自动重建）。

    Args:
        source_sp: 源域 tokenizer。
        target_sp: 目标域 tokenizer。
        source_domain: 源域名（缓存键 + overrides 域特定匹配）。
        target_domain: 目标域名（缓存键）。
        cache: 外部缓存 dict；None 时新建（调用方持有以跨步复用）。
              缓存项: {key: {"fp": (src_fp, tgt_fp, rules_ver, src_vocab), "matrix": COO}}。
        overrides: 可编辑规则层（可选；版本参与缓存失效）。
        source_vocab_size: logits 实际 vocab 大小（neuron lm_head）。None 时用
              tokenizer GetPieceSize()。防御 neuron vocab ≠ tokenizer vocab 的场景
              （如 zh neuron 用 20K 词表但标准 zh tokenizer 是 50K）。

    Returns:
        COO 稀疏张量 [V_src, V_tgt]（float32）。
    """
    if cache is None:
        cache = {}
    key = (source_domain, target_domain)
    rules_ver = overrides.version if overrides is not None else 0
    tkn_vocab = source_sp.GetPieceSize() if hasattr(source_sp, "GetPieceSize") else 0
    src_vocab = source_vocab_size if source_vocab_size is not None else tkn_vocab
    fp = (
        tokenizer_fingerprint(source_sp),
        tokenizer_fingerprint(target_sp),
        rules_ver,
        src_vocab,
    )
    cached = cache.get(key)
    if cached is not None and cached["fp"] == fp:
        return cached["matrix"]

    # 遍历 min(logits vocab, tokenizer vocab) 个 token（超出 tokenizer 的 id 无 piece → 零行）
    alignment, _ = build_domain_to_domain_alignment(
        source_sp, target_sp,
        source_domain=source_domain, overrides=overrides,
    )
    tgt_vocab = target_sp.GetPieceSize() if hasattr(target_sp, "GetPieceSize") else 0

    rows, cols, vals = [], [], []
    for src_id in range(min(src_vocab, len(alignment))):
        tgt_ids = alignment[src_id]
        if not tgt_ids:
            continue
        w = 1.0 / len(tgt_ids)
        for t in tgt_ids:
            if t < 0 or t >= tgt_vocab:
                continue
            rows.append(src_id)
            cols.append(t)
            vals.append(w)

    matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        size=(src_vocab, tgt_vocab),
    )
    cache[key] = {"fp": fp, "matrix": matrix}
    return matrix
