"""实验配置集中管理（2026-08-01 P0 硬编码修复）。

将散落在 8+ 文件中的硬编码集中到单一真相源：
- 路径常量（PROJECT_ROOT 派生，不依赖 cwd）
- 神经元 ID 列表（NEURON_IDS 单一真相源）
- shared_embedding 维度（GENERAL_VOCAB_SIZE / SHARED_EMBED_DIM）
- 域配置（DOMAIN）

所有训练/评估脚本应 import 此模块，而非各自定义常量。

设计原则：
1. 路径用 PROJECT_ROOT 派生，从任何 cwd 启动都正确
2. 神经元 ID 按实验阶段组织（base / dialogue / cross_spec）
3. 保持向后兼容：utils.py 的旧常量仍可使用（通过 re-export）
"""
from __future__ import annotations

import os
from pathlib import Path

# ── 项目根目录（不依赖 cwd）──────────────────────────────────────────────
# experiment_config.py 位于 scripts/training/，项目根在上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 路径常量（PROJECT_ROOT 派生）─────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "distill"
OUTPUT_DIR = PROJECT_ROOT / "data" / "neurons"
DOMAIN_TOKENIZER_DIR = PROJECT_ROOT / "taiji" / "domains"
SHARED_EMBEDDING_PATH = PROJECT_ROOT / "data" / "shared_embedding.pt"
SIMPLE_ZH_DIR = PROJECT_ROOT / "data" / "simple_zh"
LOG_DIR = PROJECT_ROOT / "logs"

# 向后兼容：utils.py 旧代码期望 str 路径，这里提供 str 版本
DATA_DIR_STR = str(DATA_DIR)
OUTPUT_DIR_STR = str(OUTPUT_DIR)
DOMAIN_TOKENIZER_DIR_STR = str(DOMAIN_TOKENIZER_DIR)
SHARED_EMBEDDING_PATH_STR = str(SHARED_EMBEDDING_PATH)
SIMPLE_ZH_DIR_STR = str(SIMPLE_ZH_DIR)

# ── shared_embedding 维度（单一真相源）───────────────────────────────────
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512

# ── 域配置 ───────────────────────────────────────────────────────────────
DEFAULT_DOMAIN = "zh"

# ── 神经元 ID（单一真相源）──────────────────────────────────────────────
# 按实验阶段组织，避免散落在 8+ 文件中复制粘贴

# 基础神经元（百科/作文训练，无对话能力）
ZH_COMPACT_NEURON_IDS = ["zh_aug0", "zh_aug1", "zh_aug2", "zh_aug3"]
ZH_STD_NEURON_ID = "zh_std0"

# 对话版本（已 fine-tune 对话能力）
ZH_COMPACT_DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue",
    "zh_aug2_dialogue", "zh_aug3_dialogue",
]
ZH_STD_DIALOGUE_ID = "zh_std0_dialogue"

# 综合体完整阵容（对话版本：4 compact + 1 standard）
ENSEMBLE_DIALOGUE_IDS = ZH_COMPACT_DIALOGUE_IDS + [ZH_STD_DIALOGUE_ID]

# 综合体基础版本（无对话能力，用于对照实验）
ENSEMBLE_BASE_IDS = ZH_COMPACT_NEURON_IDS + [ZH_STD_NEURON_ID]

# Shared Expert（废弃 2026-08-10：shared_expert 机制从未被 assemble_cortex 启用
# （cortex 创建 Ensemble 未传 shared_expert_id），zh_general 只是被全量扫描误加载的
# 干扰 neuron（中文任务竞争者、训练最弱 PPL 257）。C24 双头后每个 neuron 自带
# judge_lm_head（general 判定空间），single always-active 底座机制冗余。已删除
# data/neurons/neuron_zh_general.pt，装配收敛为 9 阵容（5 对话 + 4 域）。
# SHARED_EXPERT_ID = "zh_general"

# ── 采样参数（生成阶段统一配置）──────────────────────────────────────────
# 散落在 eval_single_dialogue / eval_aug_joint / eval_dialogue / finetune generate_sample
# 统一默认值，函数签名保留参数可覆盖
# 2026-08-03 完整重训后生成质量对比：temp 0.8→0.55, top_k 40→15, rep 1.2→1.4, max 100→60
SAMPLING_TEMPERATURE = 0.55
SAMPLING_TOP_K = 15
SAMPLING_REPETITION_PENALTY = 1.4
SAMPLING_MAX_TOKENS = 60  # 折中默认值（原 single=100, aug_joint=80, dialogue=120）

# ── 对话数据格式："问：{instruction}\n答：{output}"
# 训练时只对 "答：" 之后的 token 计算 loss（answer masking）
# 口径契约单一真相源已上移至核心库 taiji/resonance/dialogue_format.py
# （2026-08-12）：此处 re-export 保持 scripts 侧
# `from scripts.training.experiment_config import build_dialogue_prompt` 等兼容。
from neuroplex.resonance.dialogue_format import (  # noqa: E402
    SFT_ANSWER_MARKER,
    Q_MARKER,
    build_dialogue_prompt,
    dialogue_prompt_requires_guard,
)

# ── 对话训练数据文件列表（S5: 数据扩充）─────────────────────────────────
# 本地已有的对话数据文件（均为 {"text": "问：...\n答：..."} 格式）
# 2026-08-03 数据清洗（clean_dialogue_data.py）：过滤代码/英文密集样本后 *_clean.jsonl
# 原始 *_clean 对应文件约 97.6K 条 → 清洗后 88.7K 条（90.9%）
# ⚠️ 2026-08-12 发现：sft_shared_core/unique 与 alpaca 内容 100% 重复，实际唯一样本仅 44391 条
DIALOGUE_DATA_FILES = [
    "alpaca_zh_sft_clean.jsonl",    # 44391 条（清洗后，唯一对话样本）
    "sft_shared_core_clean.jsonl",  # 13305 条（与 alpaca 重复，保留作冗余）
    "sft_unique_0_clean.jsonl",     # 6236 条（与 alpaca 重复，保留作冗余）
    "sft_unique_1_clean.jsonl",     # 6168 条（与 alpaca 重复，保留作冗余）
    "sft_unique_2_clean.jsonl",     # 6209 条（与 alpaca 重复，保留作冗余）
    "sft_unique_3_clean.jsonl",     # 6215 条（与 alpaca 重复，保留作冗余）
    "sft_unique_4_clean.jsonl",     # 6206 条（与 alpaca 重复，保留作冗余）
    # 2026-08-12 扩充：BelleGroup/train_2M_CN（build_dialogue_extended.py）
    # 150K 下载 → 去重/清洗后新增 123090 条（唯一）
    "dialogue_extended_clean.jsonl",  # 123090 条（唯一，与 alpaca 去重）
]
# 实际唯一对话样本：44391（alpaca）+ 123090（扩充）= 167481 条

# HuggingFace 对话数据源（S5: 可选扩充，需联网下载）
# 用于进一步扩充对话数据多样性（多轮、推理、代码等）
DIALOGUE_HF_SOURCES = [
    {"dataset": "BelleGroup/train_2M_CN", "config": "default", "split": "train",
     "text_fields": ["instruction", "output"], "max_samples": 100000},
    {"dataset": "COIG/COIG", "config": "default", "split": "train",
     "text_fields": ["instruction", "output"], "max_samples": 50000},
]

# ── 评估 prompt（按神经元类型分组）──────────────────────────────────────
# 对话神经元（fine-tune 过对话数据）：用 "问：...答：" 格式匹配训练数据
DIALOGUE_PROMPTS = [
    "问：你好，请介绍一下自己\n答：",
    "问：什么是人工智能？\n答：",
    "问：如何学习编程？\n答：",
    "问：请解释神经网络的工作原理\n答：",
    "问：你最喜欢的颜色是什么？\n答：",
]

# Base 神经元（仅百科/作文训练，无对话能力）：纯问题/续写
BASE_PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
    "深度学习在自然语言处理中的应用",
    "请解释神经网络的工作原理",
    "在公园里，阳光透过",
]


def get_neuron_path(neuron_id: str) -> str:
    """获取神经元 checkpoint 路径。

    Args:
        neuron_id: 神经元 ID（如 "zh_aug0_dialogue"）

    Returns:
        checkpoint 文件绝对路径
    """
    return str(OUTPUT_DIR / f"neuron_{neuron_id}.pt")


def get_cross_spec_weights_path(stage: str = "dialogue") -> str:
    """获取跨规格投影层权重路径。

    Args:
        stage: "dialogue"=对话训练权重, "cross_spec"=simple_zh训练权重
    """
    if stage == "dialogue":
        return str(OUTPUT_DIR / "cross_spec_dialogue.pt")
    elif stage == "cross_spec":
        return str(OUTPUT_DIR / "cross_spec_finetuned.pt")
    else:
        raise ValueError(f"Unknown stage: {stage}")


def ensure_dirs() -> None:
    """确保关键目录存在。"""
    for d in [OUTPUT_DIR, LOG_DIR, SIMPLE_ZH_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
