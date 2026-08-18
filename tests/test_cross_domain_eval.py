from pathlib import Path
from uuid import uuid4

import pytest
import torch

from scripts.training._eval_cross_domain_collab import _resolve_generation_tokenizer
from scripts.training.train_cross_domain_collab import load_cross_spec_reference


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


def test_anchor_reference_loads_only_matching_cross_spec_projections() -> None:
    source_forward = torch.nn.Linear(2, 2, bias=False)
    source_backward = torch.nn.Linear(2, 2, bias=False)
    target_forward = torch.nn.Linear(2, 2, bias=False)
    target_backward = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        source_forward.weight.fill_(3.0)
        source_backward.weight.fill_(5.0)

    class _Ensemble:
        _cross_spec_projectors = {"hub": target_forward}
        _cross_spec_back_projectors = {"code": target_backward}

    ckpt = Path(__file__).with_name(f".anchor_reference_{uuid4().hex}.pt")
    try:
        torch.save(
            {
                "cross_spec_state": {
                    "forward": {"hub": source_forward.state_dict(), "math": {}},
                    "backward": {"code": source_backward.state_dict()},
                }
            },
            ckpt,
        )

        assert load_cross_spec_reference(_Ensemble(), str(ckpt)) == 2
        assert torch.equal(target_forward.weight, source_forward.weight)
        assert torch.equal(target_backward.weight, source_backward.weight)
    finally:
        ckpt.unlink(missing_ok=True)
