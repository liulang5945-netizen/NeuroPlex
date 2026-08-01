"""训练脚本共享工具函数。

从 train_neuron.py、train_standard_leader.py、train_cortex_joint.py 抽取，
供所有活跃训练/评估脚本复用。

包含：
- 常量（路径、vocab 配置）
- tokenizer 加载（domain/general）
- 数据加载（domain texts、simple_zh texts、all texts）
- shared_embedding 管理（create/load/save）
- SequentialSampler（顺序 epoch 采样）
"""
from __future__ import annotations

import math
import os
import random
import json
from typing import List

import sentencepiece as spm
import torch
import torch.nn as nn

from taiji.resonance.config import DOMAIN_VOCAB_SIZES, GENERAL_TOKENIZER_DOMAIN
# P0 硬编码修复：路径/维度/ID 集中管理，从 experiment_config 导入
from scripts.training.experiment_config import (
    DATA_DIR_STR as DATA_DIR,
    OUTPUT_DIR_STR as OUTPUT_DIR,
    DOMAIN_TOKENIZER_DIR_STR as DOMAIN_TOKENIZER_DIR,
    SHARED_EMBEDDING_PATH_STR as SHARED_EMBEDDING_PATH,
    SIMPLE_ZH_DIR_STR as SIMPLE_ZH_DIR,
    GENERAL_VOCAB_SIZE,
    SHARED_EMBED_DIM,
)


# ── HuggingFace 数据源映射（供 load_domain_texts 使用）──────────────────────
DOMAIN_DATA_SOURCES = {
    "zh": [
        {"dataset": "shibing624/alpaca-zh", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
        {"dataset": "qingyue2019/wikipedia-zh-cn-20240820", "config": "default", "split": "train",
         "text_fields": ["text"], "max_samples": 5000},
    ],
    "en": [
        {"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
        {"dataset": "wikipedia", "config": "20220301.en", "split": "train",
         "text_fields": ["text"], "max_samples": 5000},
    ],
    "code": [
        {"dataset": "sahil2801/CodeAlpaca-20k", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
    ],
    "math": [
        {"dataset": "openai/gsm8k", "config": "main", "split": "train",
         "text_fields": ["question", "answer"], "max_samples": 8000},
    ],
    "general": [
        {"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
    ],
}


# ── Tokenizer 加载 ────────────────────────────────────────────────────────
def load_domain_tokenizer(domain: str) -> spm.SentencePieceProcessor:
    """Load domain-specific SentencePiece tokenizer.

    general domain reuses en tokenizer.
    """
    if domain == "general":
        actual_domain = GENERAL_TOKENIZER_DOMAIN  # "en"
    else:
        actual_domain = domain

    model_path = os.path.join(DOMAIN_TOKENIZER_DIR, actual_domain, f"sp_{actual_domain}.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Domain tokenizer not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def load_general_tokenizer(general_model_path: str = None) -> spm.SentencePieceProcessor:
    """Load or create the general 256K tokenizer.

    Priority:
    1. Explicit path (general_model_path)
    2. taiji/domains/general/sp_general.model
    3. Fallback: use en tokenizer as general tokenizer (16K vocab)

    Returns:
        SentencePieceProcessor for the general tokenizer.
    """
    if general_model_path and os.path.exists(general_model_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(general_model_path)
        print(f"  General tokenizer loaded: {general_model_path} (vocab={sp.vocab_size()})")
        return sp

    general_path = os.path.join(DOMAIN_TOKENIZER_DIR, "general", "sp_general.model")
    if os.path.exists(general_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(general_path)
        print(f"  General tokenizer loaded: {general_path} (vocab={sp.vocab_size()})")
        return sp

    en_path = os.path.join(DOMAIN_TOKENIZER_DIR, "en", "sp_en.model")
    if os.path.exists(en_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(en_path)
        print(f"  General tokenizer: FALLBACK to en tokenizer (vocab={sp.vocab_size()}). "
              f"Upgrade to 256K tokenizer later for full vocab coverage.")
        return sp

    raise FileNotFoundError(
        "No general tokenizer found. Place sp_general.model in taiji/domains/general/ "
        "or ensure taiji/domains/en/sp_en.model exists as fallback."
    )


# ── 数据加载 ──────────────────────────────────────────────────────────────
def load_domain_texts(domain: str, max_texts: int = 5000) -> List[str]:
    """Load raw text data for a domain from multiple HuggingFace datasets.

    Strategy:
    1. Try cached text file (data/distill/{domain}_texts.jsonl)
    2. Download from each source in DOMAIN_DATA_SOURCES[domain]
    3. Concatenate text fields into single strings
    """
    cache_path = os.path.join(DATA_DIR, f"{domain}_texts.jsonl")

    if os.path.exists(cache_path):
        texts = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_texts:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
        if texts:
            print(f"  Loaded {len(texts)} texts from cache: {cache_path}")
            return texts

    sources = DOMAIN_DATA_SOURCES.get(domain)
    if not sources:
        raise ValueError(f"Unknown domain: {domain}")
    if isinstance(sources, dict):
        sources = [sources]

    all_texts = []
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets library required: pip install datasets")

    for src in sources:
        src_max = src.get("max_samples", max_texts)
        remaining = max_texts - len(all_texts)
        if remaining <= 0:
            break
        limit = min(remaining, src_max)

        print(f"  Downloading {src['dataset']} (target {limit})...")
        try:
            ds = load_dataset(src["dataset"], src["config"], split=src["split"])
            for example in ds:
                if len(all_texts) >= max_texts:
                    break
                parts = []
                for field in src["text_fields"]:
                    val = example.get(field, "")
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                if parts:
                    all_texts.append(" ".join(parts))
                if len(all_texts) >= limit:
                    break
        except Exception as e:
            print(f"  WARN: {src['dataset']} failed: {e}, skipping")

    if all_texts:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for t in all_texts:
                f.write(t + "\n")
        print(f"  Downloaded {len(all_texts)} texts total, cached to {cache_path}")

    return all_texts


def load_all_texts(data_path: str, max_texts: int = 10000000, min_len: int = 10) -> List[str]:
    """加载全部文本（不分割，给 standard 族长独享全部数据）。"""
    print(f"  加载文本: {data_path}", flush=True)
    all_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_texts:
                break
            line = line.strip()
            if len(line) >= min_len:
                all_texts.append(line)

    total_chars = sum(len(t) for t in all_texts)
    print(f"  加载 {len(all_texts)} 条非空文本（上限 {max_texts}）", flush=True)
    print(f"  总字符: {total_chars/1e6:.1f}M, 估计 tokens: {total_chars/1.7/1e6:.0f}M", flush=True)
    return all_texts


def load_simple_zh_texts(data_files: List[str], max_texts: int = 10000000) -> List[str]:
    """从 data/simple_zh/ 加载多个数据文件并合并（简单中文，匹配 36M 能力）。

    替代 load_domain_texts（维基百科数据对 compact 太复杂）。
    """
    texts = []
    for fname in data_files:
        path = os.path.join(SIMPLE_ZH_DIR, fname) if not os.path.isabs(fname) else fname
        if not os.path.exists(path):
            print(f"  ⚠️ 文件不存在: {path}", flush=True)
            continue
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(texts) >= max_texts:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    text = d.get('text', '')
                    if len(text) >= 20:
                        texts.append(text)
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  {fname}: {count} 条", flush=True)
    return texts


# ── Shared embedding 管理 ─────────────────────────────────────────────────
def create_shared_embedding(device: str = "cpu") -> nn.Embedding:
    """Create the shared embedding table (Layer 1: shared sensory).

    All neurons share this single Embedding(256000, 512) table.
    It can be saved/loaded independently and hot-swapped for larger vocabs.
    """
    emb = nn.Embedding(GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM)
    emb.to(device)
    print(f"  Shared embedding: {GENERAL_VOCAB_SIZE} × {SHARED_EMBED_DIM} "
          f"({GENERAL_VOCAB_SIZE * SHARED_EMBED_DIM / 1e6:.0f}M params)")
    return emb


def load_or_create_shared_embedding(device: str = "cpu") -> nn.Embedding:
    """Load existing shared embedding or create a new one."""
    if os.path.exists(SHARED_EMBEDDING_PATH):
        state = torch.load(SHARED_EMBEDDING_PATH, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "weight" in state:
            weight = state["weight"]
        else:
            weight = state
        emb = nn.Embedding(weight.shape[0], weight.shape[1])
        emb.weight.data.copy_(weight)
        emb.to(device)
        print(f"  Loaded shared embedding from {SHARED_EMBEDDING_PATH}: "
              f"{emb.num_embeddings} × {emb.embedding_dim}")
        return emb

    return create_shared_embedding(device)


def save_shared_embedding(emb: nn.Embedding, path: str = None) -> None:
    """Save the shared embedding table for reuse."""
    path = path or SHARED_EMBEDDING_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb.weight.data, path)
    print(f"  Shared embedding saved to {path}")


# ── 采样器 ────────────────────────────────────────────────────────────────
class SequentialSampler:
    """顺序 epoch 采样：shuffle → 顺序遍历 → 重洗，保证每步看到新内容。

    对比随机采样（torch.randint）：
    - 随机采样：8000步×batch4=32K样本，从1.6M池中随机抽，利用率2%
    - 顺序采样：8000步×batch8=64K样本，前64K条全部唯一，利用率100%（无重复）
    """

    def __init__(self, texts: List[str], batch_size: int, seed: int = 42):
        self.texts = texts
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices = list(range(len(texts)))
        self.rng.shuffle(self.indices)
        self.cursor = 0
        self.epoch = 0
        self.n_texts = len(texts)

    def sample_batch(self) -> List[str]:
        """获取下一批，顺序遍历，epoch 结束自动重洗。"""
        if self.cursor + self.batch_size > self.n_texts:
            self.rng.shuffle(self.indices)
            self.cursor = 0
            self.epoch += 1

        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return [self.texts[i] for i in batch_indices]

    @property
    def unique_seen(self) -> int:
        """已看到的唯一文本数（当前 epoch 内）。"""
        return min(self.cursor, self.n_texts)


# ── WSD 学习率调度（P1 硬编码修复：从 5 个文件抽取）─────────────────────
def make_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    warmup_steps: int = 200,
    decay_ratio: float = 0.85,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """WSD（Warmup-Stable-Decay）学习率调度。

    三个阶段：
    1. Warmup: 线性从 0 升到 1.0（前 warmup_steps 步）
    2. Stable: 保持 1.0（warmup_steps 到 decay_start）
    3. Decay:  cosine 衰减到 min_lr_ratio（decay_start 到 num_steps）

    Args:
        optimizer: 优化器
        num_steps: 总训练步数
        warmup_steps: warmup 步数
        decay_ratio: decay 开始位置（占总步数的比例，默认 0.85）
        min_lr_ratio: decay 最低 lr 比例（默认 0.1，即衰减到 10%）

    Returns:
        LambdaLR 调度器
    """
    decay_start = max(warmup_steps + 1, int(num_steps * decay_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        elif step < decay_start:
            return 1.0
        else:
            progress = (step - decay_start) / max(1, num_steps - decay_start)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Muon + AdamW 混合优化器（P1 硬编码修复：从 2 个文件抽取）─────────────
def build_muon_adamw_optimizers(
    params_2d: list,
    params_1d: list,
    lr: float,
    adamw_weight_decay: float = 0.01,
) -> tuple:
    """构建 Muon + AdamW 混合优化器（借鉴 DeepSeek V4 / GLM-5.2）。

    - 2D 权重矩阵（Linear weight）用 Muon：Newton-Schulz 正交化突破 Adam 局部最小值
    - 1D 参数（bias/LayerNorm/scale）用 AdamW：Muon 仅适用于 2D

    Args:
        params_2d: 2D 权重参数列表（Muon 优化）
        params_1d: 1D/0D 参数列表（AdamW 优化）
        lr: 学习率（Muon 和 AdamW 共用）
        adamw_weight_decay: AdamW weight decay

    Returns:
        (muon_optimizer, adamw_optimizer): adamw_optimizer 可能为 None（无 1D 参数时）
    """
    from torch.optim import Muon
    muon_optimizer = Muon(
        params_2d, lr=lr, momentum=0.95,
        nesterov=True, ns_steps=5,
    )
    if len(params_1d) > 0:
        adamw_optimizer = torch.optim.AdamW(params_1d, lr=lr, weight_decay=adamw_weight_decay)
    else:
        adamw_optimizer = None
    return muon_optimizer, adamw_optimizer
