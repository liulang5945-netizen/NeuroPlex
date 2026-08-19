"""Regression tests for the dialogue leader quality signal."""

import sentencepiece as spm
import torch

from neuroplex.brain.cortex import Cortex
from neuroplex.resonance.translator import build_position_alignment


class _Hub:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tokenizer(self, domain):
        return self.tokenizer if domain in {"zh", "general"} else None


def test_leader_quality_uses_general_to_domain_position_alignment():
    general = spm.SentencePieceProcessor(
        model_file="neuroplex/domains/general/sp_general.model",
    )
    zh = spm.SentencePieceProcessor(
        model_file="neuroplex/domains/zh/sp_zh.model",
    )
    prompt = "问：你好，请介绍一下自己\n答："
    general_ids, aligned_targets = build_position_alignment(prompt, zh, general)

    # 这条样本在两套词表中的长度不同；逐位置硬配会得到错误排序。
    assert len(general_ids) != len(zh.encode(prompt))

    vocab = zh.GetPieceSize()
    good = torch.zeros(1, len(general_ids), vocab)
    bad = torch.zeros_like(good)
    raw_targets = zh.encode(prompt)
    for pos in range(1, len(general_ids)):
        aligned = int(aligned_targets[pos])
        if aligned >= 0:
            good[0, pos - 1, aligned] = 10.0
        if pos < len(raw_targets):
            bad[0, pos - 1, int(raw_targets[pos])] = 10.0

    cortex = object.__new__(Cortex)
    cortex.device = "cpu"
    cortex._tokenizer_hub = _Hub(zh)
    cortex._general_sp = general
    quality = cortex._nll_quality_from_round1_logits(
        {"round1_logits": {"good": good, "bad": bad}},
        prompt,
        "zh",
    )

    assert quality["good"] > quality["bad"]
