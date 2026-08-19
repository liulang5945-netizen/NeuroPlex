"""Regression tests for the bounded 7.58M micro data pilot."""

from scripts.training.diag_micro_data_ab import (
    _select_hf_for_ratio,
    _valid_first_token_position,
)
from scripts.training.diag_micro_route_canary import ROUTE_MODES


def test_first_token_metric_skips_answer_outside_truncated_window() -> None:
    assert _valid_first_token_position(answer_start=156, seq_len=128) is None
    assert _valid_first_token_position(answer_start=127, seq_len=128) == 126


def test_hf_mix_selection_is_deterministic_and_hits_requested_ratio() -> None:
    current = [f"current-{index}" for index in range(12)]
    hf = [f"hf-{index}" for index in range(20)]
    selected = _select_hf_for_ratio(current, hf, 0.25)
    assert selected == _select_hf_for_ratio(current, hf, 0.25)
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_route_canary_keeps_nine_member_baseline_and_two_topk_modes() -> None:
    assert ROUTE_MODES == (
        "base_9_all",
        "with_micro_10_all",
        "with_micro_10_auto_top1",
        "with_micro_10_auto_top2",
    )
