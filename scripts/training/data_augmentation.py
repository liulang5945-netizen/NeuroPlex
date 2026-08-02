"""T4: 数据增强模块（训练数据多样性）。

三策略（按上限排序）：
1. 在线模板改写（规则驱动，CPU 快，每 epoch 变体不同）：
   - 同义替换：常用问候/请求词互换（你好→您好/嗨/你好呀）
   - 句式包装：question 前加礼貌语/澄清语前缀
   只改写 question，保留 answer 不变（防止破坏答案质量）
2. 多轮拼接（在线组合）：
   - 从对话池取 1-2 条作为前序轮次，拼接到当前对话前
   - 格式: "问：Q0\n答：A0\n问：Q1\n答：A1"
   - SFT masking 用 answer_marker_mode="last" 只对最终轮 answer 计 loss
3. 神经元改写（离线，可选）：
   - 用本地对话神经元生成 paraphrase（上限最高，CPU 慢）
   - 独立脚本 generate_augmented_data.py 预生成增强数据文件

用法（在线增强）：
    from scripts.training.data_augmentation import augment_dialogue_text, multi_turn_concatenate

    rng = random.Random(epoch_seed)
    # 1. 单条改写
    aug = augment_dialogue_text(text, rng, rewrite_prob=0.5)
    # 2. 多轮拼接（context_pool 是对话池）
    aug = multi_turn_concatenate(text, context_pool, rng, extra_turns=(1, 2))
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

# ── 对话格式常量（与 experiment_config.SFT_ANSWER_MARKER 一致）──
Q_MARKER = "问："
A_MARKER = "答："

# ── 同义替换表（中文常用表达，保语义）──
SYNONYM_MAP = {
    "你好": ["您好", "嗨", "你好呀", "你好啊"],
    "你好吗": ["您好吗", "最近怎么样", "过得怎么样"],
    "请": ["麻烦", "劳驾", "帮忙"],
    "告诉我": ["讲讲", "介绍一下", "说给我听"],
    "什么是": ["请问什么是", "解释一下什么是", "介绍一下什么是"],
    "如何": ["怎么", "怎样", "怎么做才能"],
    "为什么": ["为啥", "什么原因", "为什么会"],
    "谢谢": ["多谢", "感谢", "谢谢啦"],
    "你好，请问": ["您好，请问", "嗨，想问", "你好，我想问"],
}

# ── question 前缀包装（礼貌语/澄清语）──
QUESTION_PREFIXES = [
    "请问", "我想知道", "你能帮我吗，", "麻烦你", "帮我看看",
    "不好意思，想请问", "想了解一下", "能告诉我吗？",
]

# ── question 后缀包装 ──
QUESTION_SUFFIXES = [
    "", "", "", "", "谢谢", "谢谢啦", "麻烦了", "可以吗",
]

# ── 系统/角色前缀（多轮上下文增强）──
ROLE_PREFIXES = ["", "", "我是学生，", "我在学习，", "我很好奇，"]


def parse_dialogue(text: str) -> Optional[Tuple[str, str]]:
    """解析对话文本为 (question, answer)。

    格式: "问：Q\n答：A"（第一个问/答标记切分）。
    无有效格式时返回 None（跳过增强）。
    """
    q_idx = text.find(Q_MARKER)
    a_idx = text.find(A_MARKER)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None
    question = text[q_idx + len(Q_MARKER):a_idx].strip()
    answer = text[a_idx + len(A_MARKER):].strip()
    if len(question) < 2 or len(answer) < 2:
        return None
    return question, answer


def rewrite_question(question: str, rng: random.Random, prefix_prob: float = 0.3,
                     suffix_prob: float = 0.2, synonym_prob: float = 0.5) -> str:
    """改写 question（同义替换 + 前缀/后缀包装），保留语义。

    Args:
        question: 原始 question
        rng: 随机数生成器（可种子化，epoch 间变化）
        prefix_prob: 加前缀概率
        suffix_prob: 加后缀概率
        synonym_prob: 同义替换概率

    Returns:
        改写后的 question
    """
    q = question
    # 1. 同义替换（遍历替换表，按概率替换）
    for src, variants in SYNONYM_MAP.items():
        if src in q and rng.random() < synonym_prob:
            q = q.replace(src, rng.choice(variants), 1)

    # 2. 前缀包装（避免重复：question 本身以 "请问" 开头时跳过）
    if rng.random() < prefix_prob and not q.startswith(("请问", "您", "嗨", "你好")):
        q = rng.choice(QUESTION_PREFIXES) + q

    # 3. 后缀包装
    suffix = rng.choice(QUESTION_SUFFIXES)
    if suffix and rng.random() < suffix_prob:
        q = q + suffix

    return q


def augment_dialogue_text(text: str, rng: random.Random, rewrite_prob: float = 0.5,
                          prefix_prob: float = 0.3, suffix_prob: float = 0.2,
                          synonym_prob: float = 0.5) -> str:
    """增强单条对话：改写 question，保留 answer 不变。

    Args:
        text: 原始对话文本（"问：...\n答：..."）
        rng: 随机数生成器
        rewrite_prob: 是否执行改写的总概率（0-1）
        prefix_prob/suffix_prob/synonym_prob: 传给 rewrite_question

    Returns:
        增强后的对话文本；格式无效时原样返回
    """
    if rng.random() >= rewrite_prob:
        return text
    parsed = parse_dialogue(text)
    if parsed is None:
        return text
    question, answer = parsed
    # 角色前缀增强（少量样本加）
    role = rng.choice(ROLE_PREFIXES) if rng.random() < 0.15 else ""
    new_q = rewrite_question(
        question, rng, prefix_prob, suffix_prob, synonym_prob,
    )
    return f"{Q_MARKER}{role}{new_q}\n{A_MARKER}{answer}"


def multi_turn_concatenate(
    text: str,
    context_pool: List[str],
    rng: random.Random,
    extra_turns: Tuple[int, int] = (1, 2),
    prob: float = 0.4,
) -> str:
    """多轮拼接：从对话池取 1-2 条作为前序轮次，拼接到当前对话前。

    生成格式（多轮上下文）：
        "问：Q0\n答：A0\n问：Q1\n答：A1"
    配合 translator answer_marker_mode="last"，SFT 只对最终轮 answer 计 loss，
    前序轮次作为纯上下文（模型学习多轮连贯性）。

    Args:
        text: 当前对话（最终轮）
        context_pool: 对话池（从中取前序轮次）
        rng: 随机数生成器
        extra_turns: 前序轮次数范围 (min, max)
        prob: 是否执行拼接的概率

    Returns:
        多轮对话文本；条件不满足时原样返回
    """
    if rng.random() >= prob or not context_pool:
        return text
    parsed = parse_dialogue(text)
    if parsed is None:
        return text
    question, answer = parsed

    n_turns = rng.randint(extra_turns[0], extra_turns[1])
    n_turns = min(n_turns, len(context_pool))
    if n_turns < 1:
        return text

    # 从池中随机取前序轮次（避免取到自身）
    pool = [t for t in context_pool if t != text]
    if not pool:
        return text
    prev_turns = rng.sample(pool, min(n_turns, len(pool)))

    parts = []
    for prev in prev_turns:
        prev_parsed = parse_dialogue(prev)
        if prev_parsed is None:
            continue
        pq, pa = prev_parsed
        parts.append(f"{Q_MARKER}{pq}\n{A_MARKER}{pa}")

    if not parts:
        return text
    # 最终轮
    parts.append(f"{Q_MARKER}{question}\n{A_MARKER}{answer}")
    return "\n".join(parts)


def generate_neuron_augmented_data(
    texts: List[str],
    neuron,
    shared_emb,
    domain_sp,
    general_sp,
    rng: random.Random,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_k: int = 40,
    max_samples: int = 200,
) -> List[str]:
    """神经元改写（离线可选）：用本地对话神经元生成 question 的 paraphrase。

    上限最高：神经元的改写反映其真实语言能力分布，与训练分布一致。
    CPU 慢：每样本需一次自回归前向生成，建议离线预生成（max_samples 控制规模）。

    Args:
        texts: 原始对话列表
        neuron: 已训练对话神经元（eval 模式）
        shared_emb: shared embedding（nn.Embedding）
        domain_sp: domain tokenizer
        general_sp: general tokenizer
        rng: 随机数生成器
        max_new_tokens: 生成 paraphrase 的最大 token 数
        temperature/top_k: 采样参数
        max_samples: 最多改写样本数

    Returns:
        增强后的对话文本列表（仅含成功生成的）
    """
    import torch
    import torch.nn.functional as F
    from taiji.resonance.translator import build_position_alignment

    augmented = []
    neuron.eval()
    candidates = rng.sample(texts, min(max_samples, len(texts)))
    pad_id = 0

    def _tokenize(text: str) -> List[int]:
        g_ids, _ = build_position_alignment(text, domain_sp, general_sp)
        return g_ids

    def _sample_next(logits: torch.Tensor) -> int:
        """top-k 采样下一个 token。"""
        if temperature != 1.0:
            logits = logits / temperature
        k = min(top_k, logits.shape[-1])
        topk_vals, topk_idx = torch.topk(logits, k)
        probs = F.softmax(topk_vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).item()
        return topk_idx[sampled].item()

    for text in candidates:
        parsed = parse_dialogue(text)
        if parsed is None:
            continue
        question, answer = parsed
        # 以 question 为 prompt 生成 paraphrase
        prompt = f"{Q_MARKER}{question}\n{A_MARKER}"
        gen_ids = _tokenize(prompt)
        try:
            for _ in range(max_new_tokens):
                ids_t = torch.tensor([gen_ids], dtype=torch.long)
                emb = shared_emb(ids_t)  # [1, L, base_embed_dim]
                with torch.no_grad():
                    result = neuron.forward(emb, return_logits=True)
                logits = result["logits"][0, -1]  # [vocab]
                next_tok = _sample_next(logits)
                gen_ids.append(next_tok)
                if next_tok == pad_id:
                    break
        except Exception:
            continue  # 生成失败跳过该样本

        # 解码 paraphrase（domain_sp 解码）
        try:
            generated_text = domain_sp.decode(gen_ids)
        except Exception:
            continue
        # 提取 "答：" 之后的内容
        a_idx = generated_text.find(A_MARKER)
        paraphrase = generated_text[a_idx + len(A_MARKER):].strip() if a_idx != -1 else ""
        if len(paraphrase) < 4:
            continue
        augmented.append(f"{Q_MARKER}{question}\n{A_MARKER}{paraphrase}")

    return augmented


# ── 便捷 API：数据增强器（供训练脚本接入）──
class DialogueAugmenter:
    """在线对话数据增强器（模板改写 + 多轮拼接）。

    用法：
        augmenter = DialogueAugmenter(rewrite_prob=0.5, multi_turn_prob=0.4)
        # 每 epoch 开始设置种子（保证 epoch 间变化、epoch 内可复现）
        augmenter.set_epoch(epoch)
        # 对 batch 应用
        batch_aug = [augmenter.augment(t) for t in batch_texts]

    设计：
    - epoch_seed 使不同 epoch 产生不同变体（数据多样性随训练递增）
    - 多轮拼接需要 context_pool（整个对话池），由训练脚本传入
    """

    def __init__(self, rewrite_prob: float = 0.5, multi_turn_prob: float = 0.4,
                 prefix_prob: float = 0.3, suffix_prob: float = 0.2,
                 synonym_prob: float = 0.5):
        self.rewrite_prob = rewrite_prob
        self.multi_turn_prob = multi_turn_prob
        self.prefix_prob = prefix_prob
        self.suffix_prob = suffix_prob
        self.synonym_prob = synonym_prob
        self._rng = random.Random(42)
        self._epoch = 0
        self._context_pool: List[str] = []

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch（种子 = 42 + epoch，保证每 epoch 变体不同）。"""
        self._epoch = epoch
        self._rng = random.Random(42 + epoch * 1000)

    def set_context_pool(self, pool: List[str]) -> None:
        """设置多轮拼接的上下文池（通常是全部对话）。"""
        self._context_pool = pool

    def augment(self, text: str) -> str:
        """增强单条对话（多轮拼接 + 模板改写）。"""
        # 多轮拼接优先（产生上下文，改写在后）
        out = multi_turn_concatenate(
            text, self._context_pool, self._rng,
            prob=self.multi_turn_prob,
        )
        # 只对最终轮 question 做模板改写
        # 简化：若拼接成功则不改写最终轮（保持前序轮次纯净），否则改写
        if out == text:
            out = augment_dialogue_text(
                text, self._rng, self.rewrite_prob,
                self.prefix_prob, self.suffix_prob, self.synonym_prob,
            )
        return out
