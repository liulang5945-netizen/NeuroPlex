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

# Shared Expert（如有）
SHARED_EXPERT_ID = "zh_general"

# ── 采样参数（生成阶段统一配置）──────────────────────────────────────────
# 散落在 eval_single_dialogue / eval_aug_joint / eval_dialogue / finetune generate_sample
# 统一默认值，函数签名保留参数可覆盖
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_K = 40
SAMPLING_REPETITION_PENALTY = 1.2
SAMPLING_MAX_TOKENS = 100  # 折中默认值（原 single=100, aug_joint=80, dialogue=120）

# ── SFT 对话分隔符（S3: answer masking）─────────────────────────────────
# 对话数据格式："问：{instruction}\n答：{output}"
# 训练时只对 "答：" 之后的 token 计算 loss（answer masking）
SFT_ANSWER_MARKER = "答："

# ── 对话训练数据文件列表（S5: 数据扩充）─────────────────────────────────
# 本地已有的对话数据文件（均为 {"text": "问：...\n答：..."} 格式）
# 合并后约 97K 条，是 alpaca_zh_sft 单文件的 2 倍
DIALOGUE_DATA_FILES = [
    "alpaca_zh_sft.jsonl",    # 48818 条（alpaca-zh SFT 格式）
    "sft_shared_core.jsonl",  # 14629 条（共享核心对话）
    "sft_unique_0.jsonl",     # 6827 条（aug0 独有）
    "sft_unique_1.jsonl",     # 6827 条（aug1 独有）
    "sft_unique_2.jsonl",     # 6827 条（aug2 独有）
    "sft_unique_3.jsonl",     # 6827 条（aug3 独有）
    "sft_unique_4.jsonl",     # 6827 条（aug4 独有）
]

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
