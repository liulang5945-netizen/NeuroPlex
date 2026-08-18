import pytest
import torch

from scripts.training._eval_cross_domain_collab import _resolve_generation_tokenizer


class _Tokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def GetPieceSize(self) -> int:
        return self._vocab_size


def test_generation_decoder_follows_shared_general_output_vocab() -> None:
    target = _Tokenizer(12_000)
    general = _Tokenizer(256_000)
    logits = torch.empty(1, 1, 256_000)

    assert _resolve_generation_tokenizer(logits, target, general) is general


def test_generation_decoder_rejects_an_unknown_output_vocab() -> None:
    target = _Tokenizer(12_000)
    general = _Tokenizer(256_000)
    logits = torch.empty(1, 1, 48_000)

    with pytest.raises(RuntimeError, match="无法确定生成词表"):
        _resolve_generation_tokenizer(logits, target, general)
